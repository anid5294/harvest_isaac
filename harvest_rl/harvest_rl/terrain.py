"""Mild rough ground for the orchard yard, built with ``isaaclab.terrains``.

Task 1 of the G1 harvesting instructions: replace the flat ground plane with traversable,
uneven orchard ground (small bumps, gentle undulating slopes) while keeping the tree base and
the robot's stance collision-safe, so no orchard asset ends up buried or floating.

How it fits Isaac Lab 3
-----------------------
* The terrain is an ordinary :class:`~isaaclab.terrains.TerrainImporterCfg` of type
  ``"generator"`` whose single sub-terrain type is :class:`OrchardRoughTerrainCfg`. That
  sub-terrain samples a height field and turns it into a trimesh with Isaac Lab's own
  :func:`~isaaclab.terrains.height_field.utils.convert_height_field_to_mesh`, the same path every
  built-in ``Hf*TerrainCfg`` takes. No mesh is built by hand.
* Isaac Lab 3 clones the environment prims onto a fixed grid (``cloner.grid_transforms``),
  and static per-env assets (the tree) stay where they were cloned. The terrain therefore uses
  one tile per environment, laid out exactly on that grid (tile size = ``env_spacing``), and
  :func:`align_env_origins_to_grid` points ``scene.env_origins`` back at the grid after the
  importer has assigned its own (random) tile origins. Call :func:`finalize_orchard_terrain`
  right before the scene is built so the layout follows the final ``num_envs``.
* Each tile is flat (height exactly 0) inside "keep-out" discs given in env-local
  coordinates: the tree base, the robot's feet, and any ground-standing prop. Outside a disc
  the roughness fades in over ``blend_width`` with a smoothstep, so there is no step at the
  edge. Height 0 is the ground the existing assets were authored on, so nothing moves.

The height-field maths (:func:`orchard_height_field`) is plain numpy and is unit-tested in
``tests/test_terrain.py`` without the simulator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass

SOIL_COLOR = (0.30, 0.22, 0.14)


# ---------------------------------------------------------------------------------------
# Height field (pure numpy)
# ---------------------------------------------------------------------------------------


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def keepout_mask(xs: np.ndarray, ys: np.ndarray, keepouts: Sequence[Sequence[float]], blend_width: float) -> np.ndarray:
    """Roughness weight in [0, 1] on the grid ``xs`` x ``ys`` (tile-centred coordinates).

    0 inside every keep-out disc ``(x, y, radius)``, rising smoothly to 1 at ``radius + blend_width``.
    """
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    weight = np.ones_like(gx)
    for cx, cy, radius in keepouts:
        d = np.hypot(gx - cx, gy - cy)
        weight = np.minimum(weight, _smoothstep((d - radius) / max(blend_width, 1e-6)))
    return weight


def orchard_height_field(
    size: tuple[float, float],
    horizontal_scale: float,
    difficulty: float,
    *,
    bump_amplitude: tuple[float, float],
    bump_spacing: float,
    undulation_amplitude: tuple[float, float],
    undulation_wavelength: tuple[float, float],
    num_undulations: int,
    keepouts: Sequence[Sequence[float]],
    blend_width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Heights [m] on a ``(nx, ny)`` vertex grid covering ``size`` with spacing ``horizontal_scale``.

    Vertex ``(i, j)`` sits at ``(i * h, j * h)``; the tile centre is ``(size / 2)``. Roughness =
    gentle undulations (a few long-wavelength sinusoids, i.e. slopes) + small bumps (random
    heights on a coarse lattice, bilinearly interpolated). Both amplitudes interpolate between
    their (min, max) with ``difficulty``. Keep-out discs are given relative to the tile centre.
    """
    nx = int(round(size[0] / horizontal_scale)) + 1
    ny = int(round(size[1] / horizontal_scale)) + 1
    xs = np.arange(nx) * horizontal_scale
    ys = np.arange(ny) * horizontal_scale
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    lerp = lambda r: r[0] + difficulty * (r[1] - r[0])  # noqa: E731

    # Gentle slopes: sum of plane waves. Peak slope of one wave = 2*pi*A/lambda.
    heights = np.zeros((nx, ny))
    amp_total = lerp(undulation_amplitude)
    for _ in range(num_undulations):
        wavelength = rng.uniform(*undulation_wavelength)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        k = 2.0 * math.pi / wavelength
        heights += (amp_total / num_undulations) * np.sin(k * (gx * math.cos(theta) + gy * math.sin(theta)) + phase)

    # Small bumps: random lattice values, bilinear upsampling (no scipy needed).
    bump = lerp(bump_amplitude)
    if bump > 0.0:
        mx = int(math.ceil(size[0] / bump_spacing)) + 2
        my = int(math.ceil(size[1] / bump_spacing)) + 2
        lattice = rng.uniform(-bump, bump, size=(mx, my))
        u, v = gx / bump_spacing, gy / bump_spacing
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = u - i0, v - j0
        heights += (
            lattice[i0, j0] * (1 - fu) * (1 - fv)
            + lattice[i0 + 1, j0] * fu * (1 - fv)
            + lattice[i0, j0 + 1] * (1 - fu) * fv
            + lattice[i0 + 1, j0 + 1] * fu * fv
        )

    # Flat, height-0 pads where assets touch the ground; roughness fades in around them.
    centred = [(x + size[0] / 2, y + size[1] / 2, r) for x, y, r in keepouts]
    heights *= keepout_mask(xs, ys, centred, blend_width)
    # The tile border is pinned to 0 too, so neighbouring tiles meet without a step.
    border = np.minimum.reduce([
        _smoothstep(gx / blend_width), _smoothstep((size[0] - gx) / blend_width),
        _smoothstep(gy / blend_width), _smoothstep((size[1] - gy) / blend_width),
    ])
    return heights * border


# ---------------------------------------------------------------------------------------
# Isaac Lab sub-terrain
# ---------------------------------------------------------------------------------------


def orchard_rough_terrain(difficulty: float, cfg: OrchardRoughTerrainCfg):
    """Sub-terrain function: ``(difficulty, cfg) -> ([trimesh], origin)``.

    The origin is the tile centre at height 0 (inside the tree keep-out pad), not the
    max-height-near-centre rule of the built-in height-field decorator.
    """
    seed = None if cfg.seed is None else int(cfg.seed)
    # Every tile gets its own draw: the generator calls this once per tile with the same cfg.
    rng = np.random.default_rng(None if seed is None else [seed, _next_tile_counter()])
    heights_m = orchard_height_field(
        cfg.size,
        cfg.horizontal_scale,
        difficulty,
        bump_amplitude=cfg.bump_amplitude,
        bump_spacing=cfg.bump_spacing,
        undulation_amplitude=cfg.undulation_amplitude,
        undulation_wavelength=cfg.undulation_wavelength,
        num_undulations=cfg.num_undulations,
        keepouts=cfg.keepouts,
        blend_width=cfg.blend_width,
        rng=rng,
    )
    heights = np.rint(heights_m / cfg.vertical_scale).astype(np.int16)
    vertices, triangles = convert_height_field_to_mesh(heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold)
    import trimesh  # noqa: PLC0415 (Isaac Lab dependency)

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])
    return [mesh], origin


_TILE_COUNTER = [0]


def _next_tile_counter() -> int:
    _TILE_COUNTER[0] += 1
    return _TILE_COUNTER[0]


@configclass
class OrchardRoughTerrainCfg(HfTerrainBaseCfg):
    """Mild orchard ground: gentle undulations + small bumps, flat pads at the keep-outs."""

    function = orchard_rough_terrain

    bump_amplitude: tuple[float, float] = (0.056, 0.14)
    """Half-range of the small bumps [m] at difficulty 0 and 1."""
    bump_spacing: float = 0.3
    """Distance between independent bump heights [m]."""
    undulation_amplitude: tuple[float, float] = (0.14, 0.35)
    """Total amplitude of the long waves [m] at difficulty 0 and 1."""
    undulation_wavelength: tuple[float, float] = (2.0, 4.0)
    """Wavelength range of the long waves [m]."""
    num_undulations: int = 3
    keepouts: list[tuple[float, float, float]] = MISSING
    """Flat discs ``(x, y, radius)`` [m] relative to the tile centre (= env origin = tree base)."""
    blend_width: float = 0.8
    """Distance over which roughness fades in around a keep-out disc and at the tile border [m]."""


# ---------------------------------------------------------------------------------------
# Importer config and grid alignment
# ---------------------------------------------------------------------------------------


def grid_shape(num_envs: int) -> tuple[int, int]:
    """Rows/cols of ``isaaclab.cloner.grid_transforms`` for ``num_envs`` envs."""
    rows = int(math.ceil(num_envs / math.sqrt(num_envs)))
    cols = int(math.ceil(num_envs / rows))
    return rows, cols


def orchard_terrain_cfg(
    keepouts: Sequence[Sequence[float]],
    *,
    num_envs: int = 1,
    env_spacing: float = 5.0,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
    seed: int | None = 0,
    horizontal_scale: float = 0.1,
    vertical_scale: float = 0.005,
    **sub_terrain_kwargs,
) -> TerrainImporterCfg:
    """A ``TerrainImporterCfg`` with one rough tile per env, aligned to the env grid."""
    rows, cols = grid_shape(num_envs)
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=seed,
            size=(env_spacing, env_spacing),
            border_width=2.0,
            border_height=1.0,
            num_rows=rows,
            num_cols=cols,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            slope_threshold=0.75,
            difficulty_range=difficulty_range,
            curriculum=False,
            use_cache=False,
            sub_terrains={"orchard_rough": OrchardRoughTerrainCfg(proportion=1.0, keepouts=[tuple(k) for k in keepouts], **sub_terrain_kwargs)},
        ),
        collision_group=-1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=SOIL_COLOR, roughness=0.95, metallic=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.9, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
        ),
        debug_vis=False,
    )


def finalize_orchard_terrain(scene_cfg) -> None:
    """Re-shape the terrain grid for the final ``num_envs``/``env_spacing`` (call right before the
    scene is built; scripts and trainers often change ``num_envs`` after the cfg is created)."""
    terrain = getattr(scene_cfg, "terrain", None)
    if terrain is None or terrain.terrain_generator is None:
        return
    rows, cols = grid_shape(scene_cfg.num_envs)
    gen = terrain.terrain_generator
    gen.num_rows, gen.num_cols = rows, cols
    gen.size = (scene_cfg.env_spacing, scene_cfg.env_spacing)
    _TILE_COUNTER[0] = 0


def align_env_origins_to_grid(scene, tolerance: float = 1e-3) -> None:
    """Make ``scene.env_origins`` the cloned env positions (tile centres, z = 0).

    The importer otherwise assigns each env a random tile ("curriculum" origins), which would
    separate the robot (reset to ``env_origins``) from the static tree (cloned on the grid).
    """
    terrain = scene.terrain
    if terrain is None:
        return
    grid = scene.clone_plan.positions.to(terrain.env_origins.device).clone()
    tiles = terrain.terrain_origins.reshape(-1, 3)
    # Every env position must be a tile centre with its pad at height 0.
    d = torch_cdist_xy(grid, tiles)
    nearest = d.min(dim=1)
    if bool((nearest.values > tolerance).any()):
        raise RuntimeError(
            "Terrain tiles do not line up with the env grid; call finalize_orchard_terrain(cfg.scene) before the scene is built."
        )
    grid[:, 2] = tiles[nearest.indices, 2]
    terrain.env_origins = grid


def torch_cdist_xy(a, b):
    import torch  # noqa: PLC0415

    return torch.cdist(a[:, :2].float(), b[:, :2].float())
