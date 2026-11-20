"""Rough orchard terrain: flat pads, mild heights/slopes, tiles on the env grid.

Runs without the simulator (Isaac Lab's terrain modules are plain numpy/trimesh):

    C:\\isaac\\env_isaaclab3\\Scripts\\python.exe -m pytest tests/test_terrain.py
"""

import math

import numpy as np
import pytest
import torch

from harvest_rl.terrain import grid_shape, orchard_height_field, orchard_terrain_cfg

KEEPOUTS = [(0.0, 0.0, 0.45), (0.876, -1.526, 0.45), (0.426, -1.496, 0.20)]
SIZE, H = (5.0, 5.0), 0.1


def heights(difficulty=1.0, seed=0, keepouts=KEEPOUTS):
    cfg = orchard_terrain_cfg(keepouts).terrain_generator.sub_terrains["orchard_rough"]
    return orchard_height_field(
        SIZE, H, difficulty,
        bump_amplitude=cfg.bump_amplitude, bump_spacing=cfg.bump_spacing,
        undulation_amplitude=cfg.undulation_amplitude, undulation_wavelength=cfg.undulation_wavelength,
        num_undulations=cfg.num_undulations, keepouts=keepouts, blend_width=cfg.blend_width,
        rng=np.random.default_rng(seed),
    )


@pytest.mark.parametrize("seed", range(5))
def test_pads_are_flat_at_zero(seed):
    z = heights(seed=seed)
    xs = np.arange(z.shape[0]) * H - SIZE[0] / 2
    ys = np.arange(z.shape[1]) * H - SIZE[1] / 2
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    for cx, cy, r in KEEPOUTS:
        inside = np.hypot(gx - cx, gy - cy) <= r
        assert inside.any()
        assert np.abs(z[inside]).max() == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_bounded_and_traversable(seed):
    z = heights(seed=seed)
    assert np.abs(z).max() < 0.6  # up to ~0.45 m (7x roughness preset)
    slope = max(np.abs(np.diff(z, axis=0)).max(), np.abs(np.diff(z, axis=1)).max()) / H
    assert math.degrees(math.atan(slope)) < 60.0  # 7x preset: a few % of cells exceed the 37 deg slope threshold
    assert np.abs(z).max() > 0.005  # actually rough outside the pads


def test_tile_border_is_flat():
    z = heights()
    for edge in (z[0], z[-1], z[:, 0], z[:, -1]):
        assert np.abs(edge).max() == 0.0


def test_difficulty_scales_roughness():
    assert np.abs(heights(difficulty=0.0)).max() < np.abs(heights(difficulty=1.0)).max()


@pytest.mark.parametrize("num_envs", [1, 4, 7, 64])
def test_tiles_line_up_with_env_grid(num_envs):
    from isaaclab.cloner import grid_transforms
    from isaaclab.terrains import TerrainGenerator

    cfg = orchard_terrain_cfg(KEEPOUTS, num_envs=num_envs, env_spacing=5.0)
    rows, cols = grid_shape(num_envs)
    gen = TerrainGenerator(cfg.terrain_generator, device="cpu")
    tiles = torch.tensor(gen.terrain_origins, dtype=torch.float32).reshape(-1, 3)
    grid, _ = grid_transforms(num_envs, 5.0)
    d = torch.cdist(grid[:, :2], tiles[:, :2]).min(dim=1)
    assert d.values.max().item() < 1e-4
    assert tiles[d.indices, 2].abs().max().item() == 0.0  # every env origin sits on its pad
    assert gen.terrain_origins.shape[:2] == (rows, cols)
