"""G1 apple-harvesting benchmark environment (Isaac Lab 3 ``DirectRLEnv``).

Scene (per env, on its own rough-terrain tile): one tree from the generated library (a
kinematic rigid body, re-posed every reset), ``K`` physical apples hanging on modelled stems,
and the Unitree G1 + Dex3 hands with a basket welded to its torso. The pelvis is fixed (the
base is stabilized); the policy commands bounded *relative* joint-position targets for the left
arm and hand, tracked by the joints' PD controllers.

Every reset:

1. **Tree pose**: target-first sampling. Pick a stem site of this env's tree, a yaw within the
   curriculum's range and a point inside the left hand's workspace (``target_reach_box``, G1
   pelvis frame); solve the trunk translation that puts that site's apple there. Reject
   candidates whose trunk shift exceeds the curriculum bound or whose wood comes within the
   clearance of the robot body or its basket. The first valid candidate wins (fallback: a
   precomputed valid pose for the variant).
2. **Apples**: apple 0 is the target, at the solved site. 0..K-1 more apples go to other random
   sites (distractors; knocking them off is penalized), the rest are parked under the ground.
   Mass from a sampled density and the env's apple size, stem thresholds per apple.
3. **Robot**: home pose plus noise on the controlled joints.

Nothing is teleported or attached during an episode: apples move only through contact and the
stem model's forces (``stem.py``), and the stem breaks only on its failure criterion.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from ..robot import G1_DEX3
from ..terrain import align_env_origins_to_grid, finalize_orchard_terrain
from .assets_gen import APPLE_HALF_EXTENTS, STEM_LENGTH, ensure_tree_library
from .env_cfg import LEFT_ARM_LINKS, LEFT_FINGERTIPS, LEFT_HAND, ROBOT_POS, ROBOT_YAW, G1HarvestEnvCfg
from .stem import StemModel
from .task import FAILURE_NAMES, HarvestRules, TargetHarvestTracker

APPLE_VOLUME = 4.0 / 3.0 * math.pi * APPLE_HALF_EXTENTS[0] * APPLE_HALF_EXTENTS[1] * APPLE_HALF_EXTENTS[2]
PARK_Z = -3.0


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack((z, z, torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)), dim=-1)


def _rotz(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate (..., 3) vectors about z by ``yaw`` (broadcast over the leading dims)."""
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    x, y = v[..., 0:1], v[..., 1:2]
    rx, ry = c * x - s * y, s * x + c * y
    return torch.cat((rx, ry, v[..., 2:3].expand_as(rx)), dim=-1)


class G1HarvestEnv(DirectRLEnv):
    cfg: G1HarvestEnvCfg
    FAILURE_NAMES = FAILURE_NAMES

    # ====================================================================================
    # Setup
    # ====================================================================================

    def __init__(self, cfg: G1HarvestEnvCfg, render_mode: str | None = None, **kwargs):
        finalize_orchard_terrain(cfg.scene)
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        align_env_origins_to_grid(self.scene)
        self._robot = self.scene["robot"]
        self._tree = self.scene["tree"]
        self._apples = [self.scene[f"apple_{k}"] for k in range(self.cfg.num_apples)]
        self._stems = [self.scene[f"stem_{k}"] for k in range(self.cfg.num_apples)]
        self._apple_contact = self.scene["apple_contact"]
        self._arm_contact = self.scene["arm_contact"]

    def _configure_gym_env_spaces(self):
        self._init_buffers()
        n = self.num_ctrl
        self.cfg.action_space = n
        # joint pos/vel/last action, apple vs palm + 3 tips, palm quat, basket drop point vs palm,
        # apple in basket frame, apple velocity, attached, 7 finger forces, 4 wrist/elbow torques
        self.cfg.observation_space = 3 * n + 3 + 9 + 4 + 3 + 3 + 3 + 1 + 7 + 4
        self.cfg.state_space = self.cfg.observation_space + 14
        super()._configure_gym_env_spaces()
        self._prev_actions = torch.zeros_like(self.actions)

    def _init_buffers(self):
        c, dev, n, K = self.cfg, self.device, self.num_envs, self.cfg.num_apples
        r = self._robot
        # --- joints and links
        self.ctrl_ids, self.ctrl_names = r.find_joints(c.controlled_joints, preserve_order=True)
        self.num_ctrl = len(self.ctrl_ids)
        lim = r.data.soft_joint_pos_limits.torch[:, self.ctrl_ids]
        self._lower, self._upper = lim[..., 0], lim[..., 1]
        self._targets = r.data.default_joint_pos.torch[:, self.ctrl_ids].clone()
        self._hand_ids = r.find_joints(LEFT_HAND, preserve_order=True)[0]
        self._torque_ids = r.find_joints(["left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
                                          "left_wrist_yaw_joint"], preserve_order=True)[0]
        self._palm_id = r.find_bodies(["left_hand_palm_link"])[0][0]
        self._tip_ids = r.find_bodies(LEFT_FINGERTIPS, preserve_order=True)[0]
        self._basket_id = r.find_bodies(["basket_link"])[0][0]
        self._parent_id = r.find_bodies([c.basket.parent_body])[0][0]
        # apple_contact filters: 0 = basket, 1.. = robot.py contact bodies (left hand 7, right hand 7)
        cb = G1_DEX3.contact_bodies
        self._thumb_cols = [1 + cb.index(b) for b in G1_DEX3.grasp_thumb_bodies]
        self._finger_cols = [1 + cb.index(b) for b in G1_DEX3.grasp_finger_bodies]
        self._left_hand_cols = self._thumb_cols + self._finger_cols

        # --- robot frame (constant: the pelvis is fixed)
        self._root_pos = torch.tensor(ROBOT_POS, device=dev).repeat(n, 1)
        self._root_yaw = torch.full((n,), ROBOT_YAW, device=dev)
        b = c.target_reach_box
        self._reach_lo = torch.tensor([b[0][0], b[1][0], b[2][0]], device=dev)
        self._reach_hi = torch.tensor([b[0][1], b[1][1], b[2][1]], device=dev)

        # --- tree library tensors (variants in cfg.tree_pool, in that order)
        lib = {v["name"]: v for v in ensure_tree_library()["variants"]}
        pool = [lib[name] for name in c.tree_pool]
        self.tree_names = [v["name"] for v in pool]
        s_max = max(len(v["sites"]) for v in pool)
        p_max = max(len(v["branch_points"]) for v in pool)
        V = len(pool)
        self._sites = torch.zeros(V, s_max, 3, device=dev)
        self._site_mask = torch.zeros(V, s_max, dtype=torch.bool, device=dev)
        self._bpts = torch.zeros(V, p_max, 4, device=dev)
        self._bpts_mask = torch.zeros(V, p_max, dtype=torch.bool, device=dev)
        for i, v in enumerate(pool):
            self._sites[i, : len(v["sites"])] = torch.tensor(v["sites"], device=dev)
            self._site_mask[i, : len(v["sites"])] = True
            self._bpts[i, : len(v["branch_points"])] = torch.tensor(v["branch_points"], device=dev)
            self._bpts_mask[i, : len(v["branch_points"])] = True
        lib_index = {v["index"]: i for i, v in enumerate(pool)}
        self.tree_variant = self._read_prototype_ids("Tree", "harvest:tree_variant", lambda x: lib_index[int(x)], V)

        # --- apples
        scale = torch.ones(n, K, device=dev)
        for k in range(K):
            scale[:, k] = self._read_prototype_ids(f"Apple_{k}", "harvest:apple_scale", float, None, default=1.0)
        self.apple_scale = scale
        self.apple_half_height = APPLE_HALF_EXTENTS[2] * scale
        self.apple_mass = torch.zeros(n, K, device=dev)
        self.apple_active = torch.zeros(n, K, dtype=torch.bool, device=dev)
        self._default_mass = [a.data.body_mass.torch.clone() for a in self._apples]
        self._default_inertia = [a.data.body_inertia.torch.clone() for a in self._apples]
        self.stems = StemModel(n, K, c.stem, dev)
        self.apple_external_force = torch.zeros(n, K, 3, device=dev)
        """Extra world-frame force on each apple [N], every physics step (tests, perturbations)."""
        self.apple_external_torque = torch.zeros(n, K, 3, device=dev)
        self.target_rest_height = torch.zeros(n, device=dev)
        self.tree_pose = torch.zeros(n, 3, device=dev)
        """(x, y, yaw) of the trunk base in the env frame."""
        self.detach_step = torch.full((n,), -1, dtype=torch.long, device=dev)

        # --- task, curriculum, reward bookkeeping
        r_half = (c.basket.inner_size[0] / 2, c.basket.inner_size[1] / 2)
        self.harvest = TargetHarvestTracker(n, HarvestRules(basket_half_inner=r_half), dev)
        self.curriculum_level = float(c.tree_pose_variation)
        self._recent_success: deque[float] = deque(maxlen=c.curriculum_window)
        self._phi_prev = torch.zeros(n, 3, device=dev)
        self._phi_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._paid = {k: torch.zeros(n, dtype=torch.bool, device=dev)
                      for k in ("grasp", "detach", "place", "release", "carry")}
        self._grasp_hold = torch.zeros(n, dtype=torch.long, device=dev)
        self._success = torch.zeros(n, dtype=torch.bool, device=dev)
        self._was_attached = torch.ones(n, K, dtype=torch.bool, device=dev)
        self._basket_touched = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ep = {k: torch.zeros(n, device=dev) for k in (
            "reached", "max_finger_force", "impact_damage", "wood_steps", "basket_steps", "knocked_off")}
        self._terms: dict[str, torch.Tensor] = {}

        # --- fallback poses (one valid pose + target site per variant, at full variation)
        self._fallback_pose = torch.zeros(V, 3, device=dev)
        self._fallback_site = torch.zeros(V, dtype=torch.long, device=dev)
        for i in range(V):
            ids = torch.full((1,), i, dtype=torch.long, device=dev)
            ok = False
            for _ in range(20):
                pose, site, valid = self._sample_tree_pose(ids, level=1.0, candidates=256)
                if bool(valid[0]):
                    ok = True
                    break
            if not ok:
                raise RuntimeError(f"Tree variant '{self.tree_names[i]}' has no collision-free pose with a site in reach.")
            self._fallback_pose[i], self._fallback_site[i] = pose[0], site[0]
        counts = torch.bincount(self.tree_variant, minlength=V).tolist()
        print(f"[G1Harvest] trees per variant: {dict(zip(self.tree_names, counts))}; "
              f"apple scales: {sorted(set(round(x, 3) for x in self.apple_scale.flatten().tolist()))}")

    def _read_prototype_ids(self, prim: str, attr: str, convert, num_variants, default=None) -> torch.Tensor:
        """Which heterogeneous prototype each env got, from a custom attribute on its prim."""
        import isaaclab.sim as sim_utils  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        out = []
        for i in range(self.num_envs):
            p = stage.GetPrimAtPath(f"/World/envs/env_{i}/{prim}")
            a = p.GetAttribute(attr) if p and p.IsValid() else None
            if a is not None and a.IsValid() and a.Get() is not None:
                out.append(convert(a.Get()))
            elif num_variants is not None:
                out.append(i % num_variants)  # sequential clone strategy
            else:
                out.append(default)
        dtype = torch.long if num_variants is not None else torch.float32
        return torch.tensor(out, dtype=dtype, device=self.device)

    # ====================================================================================
    # Frames
    # ====================================================================================

    def _to_robot(self, p: torch.Tensor, env_ids=None) -> torch.Tensor:
        """Env-local points (m, ..., 3) -> G1 pelvis frame."""
        rp = self._root_pos if env_ids is None else self._root_pos[env_ids]
        yaw = self._root_yaw if env_ids is None else self._root_yaw[env_ids]
        shape = (-1,) + (1,) * (p.dim() - 2) + (3,)
        return _rotz(p - rp.view(shape), -yaw.view(shape[:-1]))

    def _from_robot(self, p: torch.Tensor, env_ids) -> torch.Tensor:
        shape = (-1,) + (1,) * (p.dim() - 2) + (3,)
        return _rotz(p, self._root_yaw[env_ids].view(shape[:-1])) + self._root_pos[env_ids].view(shape)

    def basket_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Actual basket frame (floor top centre) in env coordinates and its (x, y, z, w) orientation,
        read from the simulated ``basket_link`` every call."""
        pos = self._robot.data.body_pos_w.torch[:, self._basket_id] - self.scene.env_origins
        return pos, self._robot.data.body_quat_w.torch[:, self._basket_id]

    def basket_target(self) -> torch.Tensor:
        """Where a carried apple should be released: above the basket centre, in env coordinates."""
        pos, quat = self.basket_pose()
        up = torch.zeros_like(pos)
        up[:, 2] = self.apple_half_height[:, 0] + 0.07
        return pos + quat_apply(quat, up)

    # ====================================================================================
    # Reset-time sampling
    # ====================================================================================

    def _shift_max(self, level: float) -> float:
        lo, hi = self.cfg.tree_shift_range
        return lo + level * (hi - lo)

    def _sample_tree_pose(self, env_variant: torch.Tensor, level: float, candidates: int | None = None):
        """Target-first tree poses for ``len(env_variant)`` envs.

        Returns ``(pose (m, 3) = x, y, yaw), target site index (m,), valid (m,)``.
        """
        c, dev = self.cfg, self.device
        T = candidates or c.pose_candidates
        m = len(env_variant)
        sites = self._sites[env_variant]  # (m, S, 3)
        smask = self._site_mask[env_variant]
        # Apple centre (nominal size) of each site, tree frame; only sites whose height fits the box.
        centre = sites.clone()
        centre[..., 2] -= STEM_LENGTH + APPLE_HALF_EXTENTS[2]
        z_rel = centre[..., 2] - ROBOT_POS[2]
        zmask = smask & (z_rel > self._reach_lo[2]) & (z_rel < self._reach_hi[2])
        has = zmask.any(-1)
        weights = torch.where(has.unsqueeze(-1), zmask.float(), smask.float()) + 1e-12
        site = torch.multinomial(weights, T, replacement=True)  # (m, T)
        c_site = torch.gather(centre, 1, site.unsqueeze(-1).expand(-1, -1, 3))  # (m, T, 3)
        yaw = (torch.rand(m, T, device=dev) * 2 - 1) * c.tree_yaw_max * level
        p_r = self._reach_lo + torch.rand(m, T, 3, device=dev) * (self._reach_hi - self._reach_lo)
        p_env = _rotz(p_r, torch.full((m, T), ROBOT_YAW, device=dev)) + torch.tensor(ROBOT_POS, device=dev)
        rot_c = _rotz(c_site, yaw)
        # Solve the trunk shift, clamped to the allowed disc (with a zero bound the site must
        # already be in reach); the reach box check below decides.
        t_xy = p_env[..., :2] - rot_c[..., :2]
        norm = torch.linalg.norm(t_xy, dim=-1, keepdim=True)
        t_xy = t_xy * torch.clamp(self._shift_max(level) / norm.clamp_min(1e-9), max=1.0)
        ok = zmask.gather(1, site)
        # Exact reach-box check after the solve (z comes from the site itself).
        apple_env = torch.cat((rot_c[..., :2] + t_xy, rot_c[..., 2:3]), -1)
        rr = _rotz(apple_env - torch.tensor(ROBOT_POS, device=dev), torch.full((m, T), -ROBOT_YAW, device=dev))
        ok &= ((rr >= self._reach_lo) & (rr <= self._reach_hi)).all(-1)
        # Wood clearance from the robot body and the basket (checked in chunks).
        ok &= self._clear_of_robot(env_variant, t_xy, yaw)
        first = torch.where(ok.any(-1), ok.float().argmax(-1), torch.zeros(m, dtype=torch.long, device=dev))
        ar = torch.arange(m, device=dev)
        pose = torch.stack((t_xy[ar, first, 0], t_xy[ar, first, 1], yaw[ar, first]), -1)
        return pose, site[ar, first], ok.any(-1)

    def _clear_of_robot(self, env_variant, t_xy, yaw, chunk=64):
        c, dev = self.cfg, self.device
        m, T = yaw.shape
        basket_r = torch.tensor(c.basket.offset, device=dev) + torch.tensor((-0.0039635, 0.0, 0.044), device=dev)
        out = torch.ones(m, T, dtype=torch.bool, device=dev)
        root = torch.tensor(ROBOT_POS, device=dev)
        for s in range(0, m, chunk):
            v = env_variant[s : s + chunk]
            pts, msk = self._bpts[v], self._bpts_mask[v]  # (c, P, 4)
            p = _rotz(pts[:, None, :, :3], yaw[s : s + chunk, :, None])  # (c, T, P, 3)
            p = p + torch.cat((t_xy[s : s + chunk], torch.zeros_like(t_xy[s : s + chunk, :, :1])), -1)[:, :, None]
            pr = _rotz(p - root, torch.full(p.shape[:-1], -ROBOT_YAW, device=dev))
            rad = pts[:, None, :, 3]
            bx, by = c.robot_clearance_box
            body = (pr[..., 0].abs() - rad < bx) & (pr[..., 1].abs() - rad < by) & (pr[..., 2] + root[2] < c.robot_clearance_height)
            basket = torch.linalg.norm(pr - basket_r, dim=-1) - rad < c.basket_clearance_radius
            hit = (body | basket) & msk[:, None, :]
            out[s : s + chunk] = ~hit.any(-1)
        return out

    def _reset_scene_objects(self, env_ids: torch.Tensor):
        c, dev = self.cfg, self.device
        m, K = len(env_ids), c.num_apples
        origins = self.scene.env_origins[env_ids]
        variant = self.tree_variant[env_ids]

        # 1. tree pose
        pose, site, valid = self._sample_tree_pose(variant, self.curriculum_level)
        pose = torch.where(valid.unsqueeze(-1), pose, self._fallback_pose[variant])
        site = torch.where(valid, site, self._fallback_site[variant])
        self.tree_pose[env_ids] = pose
        tree_root = torch.zeros(m, 7, device=dev)
        tree_root[:, :2] = origins[:, :2] + pose[:, :2]
        tree_root[:, 2] = origins[:, 2]
        tree_root[:, 3:] = _yaw_quat(pose[:, 2])
        self._tree.write_root_pose_to_sim_index(root_pose=tree_root, env_ids=env_ids)
        self._tree.write_root_velocity_to_sim_index(root_velocity=torch.zeros(m, 6, device=dev), env_ids=env_ids)

        # 2. apples: target at `site`, distractors at other sites clear of the robot
        sites = _rotz(self._sites[variant], pose[:, 2:3]) + torch.cat((pose[:, :2], torch.zeros(m, 1, device=dev)), -1).unsqueeze(1)
        smask = self._site_mask[variant].clone()
        sr = self._to_robot(sites, env_ids)
        r_apple = max(APPLE_HALF_EXTENTS) * 1.15
        bx, by = c.robot_clearance_box
        body_clear = (sr[..., 0].abs() > bx + r_apple) | (sr[..., 1].abs() > by + r_apple) | (sr[..., 2] + ROBOT_POS[2] > c.robot_clearance_height)
        smask &= body_clear
        smask[torch.arange(m, device=dev), site] = False
        lo, hi = c.num_apples_range
        n_active = torch.randint(lo, hi + 1, (m,), device=dev).clamp(max=K)
        w = smask.float() + 1e-12
        others = torch.multinomial(w, max(K - 1, 1), replacement=False) if K > 1 else torch.zeros(m, 0, dtype=torch.long, device=dev)
        chosen = torch.cat((site.unsqueeze(-1), others[:, : K - 1]), -1)  # (m, K)
        active = torch.arange(K, device=dev).unsqueeze(0) < n_active.unsqueeze(-1)
        if K > 1:  # a distractor only where a valid free site was drawn
            active[:, 1:] &= torch.gather(smask, 1, chosen[:, 1:])
        active[:, 0] = True
        anchor = torch.gather(sites, 1, chosen.unsqueeze(-1).expand(-1, -1, 3))  # (m, K, 3)
        half = self.apple_half_height[env_ids]
        park = torch.zeros(m, K, 3, device=dev)
        park[..., 2] = PARK_Z - 0.3 * torch.arange(K, device=dev)
        anchor = torch.where(active.unsqueeze(-1), anchor, park + torch.tensor([0.0, 0.0, 0.2], device=dev))
        center = anchor.clone()
        center[..., 2] -= STEM_LENGTH + half
        self.apple_active[env_ids] = active
        self.target_rest_height[env_ids] = center[:, 0, 2]

        # mass from density x volume(size); stem thresholds per apple
        def U(rng, shape):
            return rng[0] + torch.rand(shape, device=dev) * (rng[1] - rng[0])

        mass = (U(c.apple_density_range, (m, K)) * APPLE_VOLUME * self.apple_scale[env_ids] ** 3).clamp(*c.apple_mass_range)
        self.apple_mass[env_ids] = mass
        big = 1e6 if c.stems_unbreakable else 1.0
        self.stems.set_apples(
            env_ids, anchor, center, mass, half, torch.full((m, K), STEM_LENGTH, device=dev),
            U(c.stem_tensile_range, (m, K)) * big, U(c.stem_bending_range, (m, K)) * big,
            U(c.stem_twist_range, (m, K)) * big, torch.tensor(c.sim.gravity, device=dev),
        )
        # inactive apples: unbreakable (they just hang under the ground)
        self.stems.f_tensile[env_ids] = torch.where(active, self.stems.f_tensile[env_ids], torch.full_like(mass, 1e6))
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        for k, (apple, stem) in enumerate(zip(self._apples, self._stems)):
            pose_k = torch.cat((center[:, k] + origins, ident.expand(m, 4)), -1)
            apple.write_root_pose_to_sim_index(root_pose=pose_k, env_ids=env_ids)
            apple.write_root_velocity_to_sim_index(root_velocity=torch.zeros(m, 6, device=dev), env_ids=env_ids)
            ratio = (mass[:, k] / self._default_mass[k][env_ids, 0]).unsqueeze(-1)
            apple.set_masses_index(masses=mass[:, k : k + 1].contiguous(), env_ids=env_ids)
            apple.set_inertias_index(inertias=(self._default_inertia[k][env_ids] * ratio.unsqueeze(-1)).contiguous(), env_ids=env_ids)
            stem_pose = torch.cat((anchor[:, k] - torch.tensor([0.0, 0.0, STEM_LENGTH / 2], device=dev) + origins, ident.expand(m, 4)), -1)
            stem.write_root_pose_to_sim_index(root_pose=stem_pose, env_ids=env_ids)
        self.apple_external_force[env_ids] = 0.0
        self.apple_external_torque[env_ids] = 0.0
        self.detach_step[env_ids] = -1

    # ====================================================================================
    # Actions and stem physics
    # ====================================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-1.0, 1.0)
        # Bounded relative targets: each step moves a target by at most action_scale [rad].
        self._targets[:] = torch.clamp(self._targets + self.cfg.action_scale * self.actions, self._lower, self._upper)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._targets, joint_ids=self.ctrl_ids)
        origins = self.scene.env_origins.unsqueeze(1)
        d = [a.data for a in self._apples]
        pos = torch.stack([x.root_pos_w.torch for x in d], 1) - origins
        quat = torch.stack([x.root_quat_w.torch for x in d], 1)
        lin = torch.stack([x.root_lin_vel_w.torch for x in d], 1)
        ang = torch.stack([x.root_ang_vel_w.torch for x in d], 1)
        force, torque, newly = self.stems.step(pos, quat, lin, ang)
        if bool(newly[:, 0].any()):
            self.detach_step[newly[:, 0]] = self.episode_length_buf[newly[:, 0]]
        force = force + self.apple_external_force
        torque = torque + self.apple_external_torque
        for k, apple in enumerate(self._apples):
            apple.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force[:, k : k + 1], torques=torque[:, k : k + 1], is_global=True
            )

    # ====================================================================================
    # State
    # ====================================================================================

    def _apple_forces(self) -> torch.Tensor:
        """(n, K, 1 + 14, 3) contact forces on each apple: basket, then each hand link."""
        return self._apple_contact.data.normal_force_matrix_w.torch

    def _arm_forces(self) -> torch.Tensor:
        """(n, links, 2) force magnitudes of the left arm/hand links on the tree and the basket."""
        return torch.linalg.norm(self._arm_contact.data.normal_force_matrix_w.torch, dim=-1)

    def get_state(self) -> dict[str, torch.Tensor]:
        """Everything about the robot, apples, basket and task, batched (env frame, (x, y, z, w))."""
        o = self.scene.env_origins
        r = self._robot.data
        a = self._apples[0].data
        bpos, bquat = self.basket_pose()
        forces = self._apple_forces()
        mag = torch.linalg.norm(forces, dim=-1)
        arm = self._arm_forces()
        h = self.harvest
        apples = torch.stack([x.data.root_pos_w.torch for x in self._apples], 1) - o.unsqueeze(1)
        return {
            "ctrl_joint_pos": r.joint_pos.torch[:, self.ctrl_ids],
            "ctrl_joint_vel": r.joint_vel.torch[:, self.ctrl_ids],
            "ctrl_joint_target": self._targets,
            "palm_pos": r.body_pos_w.torch[:, self._palm_id] - o,
            "palm_quat": r.body_quat_w.torch[:, self._palm_id],
            "fingertip_pos": r.body_pos_w.torch[:, self._tip_ids] - o.unsqueeze(1),
            "root_pos": r.root_pos_w.torch - o,
            "root_quat": r.root_quat_w.torch,
            "basket_pos": bpos,
            "basket_quat": bquat,
            "basket_target": self.basket_target(),
            "apple_pos": a.root_pos_w.torch - o,  # target apple
            "apple_quat": a.root_quat_w.torch,
            "apple_lin_vel": a.root_lin_vel_w.torch,
            "apple_ang_vel": a.root_ang_vel_w.torch,
            "apple_attached": self.stems.attached[:, 0],
            "all_apple_pos": apples,
            "all_apple_attached": self.stems.attached,
            "apple_active": self.apple_active,
            "apple_mass": self.apple_mass,
            "apple_scale": self.apple_scale,
            "stem_damage": self.stems.damage,  # failure index D (>= 1 breaks after hold_steps)
            "stem_axial": self.stems.axial,
            "stem_bending": self.stems.bending,
            "stem_torsion": self.stems.torsion,
            "finger_contact_force": mag[:, 0, 1:],  # target apple vs robot.py contact bodies
            "basket_contact_force": mag[:, 0, 0],
            "arm_wood_force": arm[..., 0],
            "arm_basket_force": arm[..., 1],
            "tree_pose": self.tree_pose,
            "tree_variant": self.tree_variant,
            "hand_distance": h.hand_distance,
            "grasp_contact": h.grasp_contact,
            "carried": h.carried,
            "released": h.released,
            "placed": h.placed,
            "success": self._success,
            "failure": h.failure,
            "episode_step": self.episode_length_buf,
        }

    # ====================================================================================
    # Task update, observations, rewards, dones
    # ====================================================================================

    def _update_task(self):
        o = self.scene.env_origins
        r, a = self._robot.data, self._apples[0].data
        apple = a.root_pos_w.torch - o
        bpos, bquat = self.basket_pose()
        forces = self._apple_forces()[:, 0]
        mag = torch.linalg.norm(forces, dim=-1)
        up = quat_apply(bquat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(bpos))
        self.harvest.update(
            detached=~self.stems.attached[:, 0],
            apple_pos=apple,
            apple_rest_height=self.target_rest_height,
            ground_height=torch.zeros(self.num_envs, device=self.device),
            apple_speed=torch.linalg.norm(a.root_lin_vel_w.torch, dim=-1),
            apple_mass=self.apple_mass[:, 0],
            apple_half_height=self.apple_half_height[:, 0],
            apple_in_basket=quat_apply_inverse(bquat, apple - bpos),
            basket_normal_force=(forces[:, 0] * up).sum(-1),
            palm_pos=r.body_pos_w.torch[:, self._palm_id] - o,
            thumb_forces=mag[:, self._thumb_cols],
            finger_forces=mag[:, self._finger_cols],
            hand_joint_pos=r.joint_pos.torch[:, self._hand_ids],
            finite=torch.isfinite(r.joint_pos.torch).all(-1) & torch.isfinite(a.root_pos_w.torch).all(-1),
        )
        h = self.harvest
        self._grasp_hold = torch.where(h.grasp_contact & self.stems.attached[:, 0], self._grasp_hold + 1, torch.zeros_like(self._grasp_hold))
        mode = self.cfg.success_mode
        if mode == "grasp":
            self._success = (self._grasp_hold >= self.cfg.grasp_success_steps) & (h.failure == 0)
        elif mode == "carry":
            self._success = h.carried & (h.failure == 0)
        elif mode == "place":
            self._success = h.success
        else:
            raise ValueError(f"Unknown success_mode {mode!r}")

    def _get_observations(self) -> dict:
        o = self.scene.env_origins
        r, a = self._robot.data, self._apples[0].data
        rq = r.root_quat_w.torch
        ids = self.ctrl_ids

        def rf(v):  # world vector -> pelvis frame
            return quat_apply_inverse(rq, v)

        apple = a.root_pos_w.torch - o
        palm = r.body_pos_w.torch[:, self._palm_id] - o
        tips = r.body_pos_w.torch[:, self._tip_ids] - o.unsqueeze(1)
        bpos, bquat = self.basket_pose()
        forces = torch.linalg.norm(self._apple_forces()[:, 0], dim=-1)
        actor = torch.cat((
            r.joint_pos.torch[:, ids] - r.default_joint_pos.torch[:, ids],
            0.1 * r.joint_vel.torch[:, ids],
            self.actions,
            rf(apple - palm),
            quat_apply_inverse(rq.repeat_interleave(len(self._tip_ids), 0),
                               (apple.unsqueeze(1) - tips).reshape(-1, 3)).reshape(self.num_envs, -1),
            quat_mul(_conj(rq), r.body_quat_w.torch[:, self._palm_id]),
            rf(self.basket_target() - palm),
            quat_apply_inverse(bquat, apple - bpos),
            rf(a.root_lin_vel_w.torch),
            self.stems.attached[:, 0:1].float(),
            torch.clamp(forces[:, self._left_hand_cols], 0.0, 10.0) / 10.0,
            torch.clamp(r.applied_torque.torch[:, self._torque_ids], -20.0, 20.0) / 20.0,
        ), dim=-1)
        st = self.stems
        h = self.harvest
        privileged = torch.cat((
            torch.clamp(st.damage[:, 0:1], 0.0, 2.0),
            torch.clamp(st.axial[:, 0:1] / st.f_tensile[:, 0:1].clamp_max(1e3), 0, 2),
            torch.clamp(st.bending[:, 0:1] / st.m_bend_max[:, 0:1].clamp_max(1e3), 0, 2),
            torch.clamp(st.torsion[:, 0:1] / st.t_twist_max[:, 0:1].clamp_max(1e3), 0, 2),
            self.apple_mass[:, 0:1] / 0.3,
            torch.clamp(st.f_tensile[:, 0:1], 0, 40) / 40.0,
            torch.clamp(st.m_bend_max[:, 0:1], 0, 1),
            torch.clamp(st.t_twist_max[:, 0:1], 0, 0.1) * 10.0,
            h.grasp_contact.float().unsqueeze(-1),
            h.carried.float().unsqueeze(-1),
            h.released.float().unsqueeze(-1),
            h.placed.float().unsqueeze(-1),
            (apple[:, 2:3] - self.target_rest_height.unsqueeze(-1)) * 10.0,
            torch.clamp(self._arm_forces().amax(1)[:, 0:1], 0, 20) / 20.0,
        ), dim=-1)
        return {"policy": actor, "critic": torch.cat((actor, privileged), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        c, h, dev = self.cfg, self.harvest, self.device
        o = self.scene.env_origins
        r, a = self._robot.data, self._apples[0].data
        apple = a.root_pos_w.torch - o
        attached = self.stems.attached[:, 0]
        ok = h.failure == 0
        grasp_pt = r.body_pos_w.torch[:, self._tip_ids].mean(1) - o
        d_reach = torch.linalg.norm(grasp_pt - apple, dim=-1)
        bpos, bquat = self.basket_pose()
        rest_spot = bpos + quat_apply(bquat, torch.stack((torch.zeros_like(d_reach), torch.zeros_like(d_reach),
                                                          self.apple_half_height[:, 0]), -1))
        d_basket = torch.linalg.norm(apple - rest_spot, dim=-1)
        lift = torch.where(attached, torch.zeros_like(d_reach),
                           torch.clamp((apple[:, 2] - self.target_rest_height) / 0.06, 0.0, 1.0))
        phi = torch.stack((1.0 - torch.tanh(d_reach / 0.1), lift, 1.0 - torch.tanh(d_basket / 0.2)), -1)
        delta = torch.where(self._phi_valid.unsqueeze(-1), phi - self._phi_prev, torch.zeros_like(phi))
        self._phi_prev[:] = phi
        self._phi_valid[:] = True
        gripping = h.grasp_contact.float()

        def once(name, cond):
            new = cond & ~self._paid[name]
            self._paid[name] |= cond
            return new.float()

        # safety / damage proxies
        forces = torch.linalg.norm(self._apple_forces(), dim=-1)  # (n, K, 15)
        hand_f = forces[:, :, 1:8] * self.apple_active.unsqueeze(-1)
        excess = torch.relu(hand_f - c.finger_force_safe).sum((1, 2))
        touch_basket = forces[:, 0, 0] > 0.05
        first_touch = touch_basket & ~self._basket_touched & ~attached
        self._basket_touched |= touch_basket
        speed = torch.linalg.norm(a.root_lin_vel_w.torch, dim=-1)
        impact = first_touch.float() * torch.relu(speed - c.impact_speed_safe)
        arm = self._arm_forces()
        wood = (arm[..., 0] > c.wood_contact_threshold).any(-1)
        basket_hit = (arm[..., 1] > c.basket_contact_threshold).any(-1)
        knocked = self._was_attached[:, 1:] & ~self.stems.attached[:, 1:] & self.apple_active[:, 1:]
        self._was_attached[:] = self.stems.attached

        terms = {
            "reach": c.w_reach * delta[:, 0] * attached.float(),
            "grasp": c.w_grasp * once("grasp", h.grasp_contact & attached),
            "hold_grasp": c.w_hold_grasp * (h.grasp_contact & attached).float(),
            "detach": c.w_detach * once("detach", ~attached & ok),
            "lift": c.w_lift * delta[:, 1] * gripping,
            "transport": c.w_transport * delta[:, 2] * gripping * (~attached).float(),
            "place": c.w_place * once("place", h.carried & h.placed),
            "release": c.w_release * once("release", h.released),
            "success": c.w_success * self._success.float(),
            "time": torch.full_like(d_reach, c.w_time),
            "failure": c.w_failure * (~ok).float(),
            "finger_force": c.w_finger_force * excess,
            "impact": c.w_impact * impact,
            "wood_collision": c.w_wood_collision * wood.float(),
            "basket_collision": c.w_basket_collision * basket_hit.float(),
            "knock_off": c.w_knock_off * knocked.float().sum(-1),
            "action_rate": c.w_action_rate * ((self.actions - self._prev_actions) ** 2).sum(-1),
            "joint_vel": c.w_joint_vel * (r.joint_vel.torch[:, self.ctrl_ids] ** 2).sum(-1),
        }
        # episode metrics
        ep = self._ep
        ep["reached"] = torch.maximum(ep["reached"], (d_reach < 0.05).float())
        ep["max_finger_force"] = torch.maximum(ep["max_finger_force"], hand_f[:, 0].amax(-1))
        ep["impact_damage"] = torch.maximum(ep["impact_damage"], (impact > 0).float())
        ep["wood_steps"] += wood.float()
        ep["basket_steps"] += basket_hit.float()
        ep["knocked_off"] += knocked.float().sum(-1)

        reward = torch.zeros(self.num_envs, device=dev)
        for name, value in terms.items():
            reward += value
            self._terms.setdefault(name, torch.zeros(self.num_envs, device=dev))
            self._terms[name] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_task()
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.terminate_on_success:
            terminated |= self._success
        if self.cfg.terminate_on_failure:
            terminated |= self.harvest.failure != 0
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ====================================================================================
    # Reset
    # ====================================================================================

    def _log_episodes(self, env_ids: torch.Tensor):
        ran = env_ids[self.episode_length_buf[env_ids] > 0]
        log = self.extras.setdefault("log", {})
        if len(ran) == 0:
            return
        h = self.harvest
        succ = self._success[ran].float()
        log["Metrics/success"] = succ.mean().item()
        log["Metrics/reach"] = self._ep["reached"][ran].mean().item()
        log["Metrics/grasp"] = self._paid["grasp"][ran].float().mean().item()
        log["Metrics/detach_with_grasp"] = self._paid["detach"][ran].float().mean().item()
        log["Metrics/carried"] = h.carried[ran].float().mean().item()
        log["Metrics/placed"] = self._paid["place"][ran].float().mean().item()
        log["Metrics/released"] = self._paid["release"][ran].float().mean().item()
        log["Metrics/failed_grasp"] = ((self._ep["reached"][ran] > 0) & ~self._paid["grasp"][ran]).float().mean().item()
        log["Safety/max_finger_force"] = self._ep["max_finger_force"][ran].mean().item()
        log["Safety/bruise_rate"] = (self._ep["max_finger_force"][ran] > self.cfg.finger_force_safe).float().mean().item()
        log["Safety/impact_damage_rate"] = self._ep["impact_damage"][ran].mean().item()
        log["Safety/wood_collision_steps"] = self._ep["wood_steps"][ran].mean().item()
        log["Safety/basket_collision_steps"] = self._ep["basket_steps"][ran].mean().item()
        log["Safety/knocked_off_apples"] = self._ep["knocked_off"][ran].mean().item()
        for code, name in enumerate(FAILURE_NAMES[1:], start=1):
            log[f"Failure/{name}"] = (h.failure[ran] == code).float().mean().item()
        for i, name in enumerate(self.tree_names):
            sel = self.tree_variant[ran] == i
            if bool(sel.any()):
                log[f"Tree/{name}_success"] = succ[sel].mean().item()
        log["Curriculum/tree_pose_variation"] = self.curriculum_level
        log["Physics/apple_mass"] = self.apple_mass[ran, 0].mean().item()
        log["Physics/active_apples"] = self.apple_active[ran].float().sum(-1).mean().item()
        for name, total in self._terms.items():
            log[f"Episode_Reward/{name}"] = total[ran].mean().item()
        # curriculum on tree pose variation
        self._recent_success.extend(succ.tolist())
        c = self.cfg
        if c.curriculum and len(self._recent_success) >= c.curriculum_window:
            rate = sum(self._recent_success) / len(self._recent_success)
            if rate > c.curriculum_up and self.curriculum_level < 1.0:
                self.curriculum_level = min(1.0, self.curriculum_level + c.curriculum_step)
                self._recent_success.clear()
            elif rate < c.curriculum_down and self.curriculum_level > 0.0:
                self.curriculum_level = max(0.0, self.curriculum_level - c.curriculum_step)
                self._recent_success.clear()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._log_episodes(env_ids)
        super()._reset_idx(env_ids)
        k, dev = len(env_ids), self.device
        origins = self.scene.env_origins[env_ids]
        r = self._robot
        root = r.data.default_root_pose.torch[env_ids].clone()
        root[:, :3] += origins
        r.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        r.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=dev), env_ids=env_ids)
        q = r.data.default_joint_pos.torch[env_ids].clone()
        if self.cfg.arm_reset_noise > 0.0:
            qc = q[:, self.ctrl_ids] + (torch.rand(k, self.num_ctrl, device=dev) * 2 - 1) * self.cfg.arm_reset_noise
            q[:, self.ctrl_ids] = torch.clamp(qc, self._lower[env_ids], self._upper[env_ids])
        r.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        r.set_joint_position_target_index(target=q, env_ids=env_ids)
        self._targets[env_ids] = q[:, self.ctrl_ids]

        self._reset_scene_objects(env_ids)

        self.harvest.reset(env_ids)
        self.actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._phi_valid[env_ids] = False
        for paid in self._paid.values():
            paid[env_ids] = False
        self._grasp_hold[env_ids] = 0
        self._success[env_ids] = False
        self._was_attached[env_ids] = True
        self._basket_touched[env_ids] = False
        for buf in list(self._ep.values()) + list(self._terms.values()):
            buf[env_ids] = 0.0


def _conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)
