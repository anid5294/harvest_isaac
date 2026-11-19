"""Orchard harvesting environment: pick the apple off the tree, put it in the basket.

A replica of harvest_isaac's OrchardBench pick task on Isaac Lab 3.0, runnable with many
parallel environments. It provides the simulation, the robot interface, cameras and the task
state/outcome. It deliberately does NOT define a reward or a policy; fill in
:meth:`OrchardEnv._get_rewards` (and change :meth:`OrchardEnv._get_observations` if you want
different inputs).

Task outcome
------------
Success and failure follow harvest_isaac exactly (see :mod:`harvest_rl.task_logic`, which is
tested step-for-step against the original code): the stem must break during a sustained
opposing grip, the apple must be carried (lifted 6 cm, 10 steps), released (hand open, no
contact, palm > 18 cm away) and rest in the basket for 30 steps. Breaking the stem without a
grip, dropping the apple below 0.40 m, or a non-finite state is a failure.

Stem model
----------
The apple is a free rigid body (sphere). While attached, a spring-damper "stem" pulls it
toward its rest position on the branch, every physics step (120 Hz): stiff along the stem,
softer sideways, with a weak torsional spring. The stem snaps when the elastic force exceeds
``stem_break_force`` (17.3 N from tree.json) or, optionally, when the apple has been rotated
more than ``stem_twist_break_angle``. After that the apple is a normal free body. The robot
has to actually pull or twist it off through contact; nothing is teleported or attached.
harvest_isaac used a native PhysX breakable joint (same 17.3 N) instead, but that can only be
rebuilt by editing the USD stage, which does not work with many parallel environments.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_rotate_inverse

from .compat import LAB3, as_torch, asset_call, backend_quat_xyzw, canonical_quat, default_root_pose
from .orchard_env_cfg import BASKET_INNER_SIZE, OrchardEnvCfg
from .task_logic import FAILURE_NAMES, HarvestParams, HarvestTracker


class OrchardEnv(DirectRLEnv):
    cfg: OrchardEnvCfg

    FAILURE_NAMES = FAILURE_NAMES
    """Names for the integer ``failure`` codes in :meth:`get_state`."""

    # ------------------------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = self.scene["robot"]
        self._apple = self.scene["apple"]
        self._apple_contact = self.scene["apple_contact"]
        self.camera_names = []
        if self.cfg.cameras:
            self.camera_names = ["cam_left_high"] + [cam.name for cam in self.cfg.robot.wrist_cameras]

    def _configure_gym_env_spaces(self):
        # Runs once the simulation has started, so joint/body names can be resolved here
        # and the action/observation sizes follow whatever robot is plugged in.
        self._init_buffers()
        n = self.num_controlled_joints
        self.cfg.action_space = n
        if self.cfg.observation_mode == "contract":
            self.cfg.observation_space = 3 * n
        elif self.cfg.observation_mode == "state":
            self.cfg.observation_space = 3 * n + 3 * len(self.tracked_body_names) + 7
        else:
            raise ValueError(f"Unknown observation_mode: {self.cfg.observation_mode!r}")
        super()._configure_gym_env_spaces()
        self._prev_actions = torch.zeros_like(self.actions)

    def _init_buffers(self):
        spec = self.cfg.robot
        n, dev = self.num_envs, self.device

        self.ctrl_joint_ids, self.ctrl_joint_names = self._robot.find_joints(
            spec.controlled_joints, preserve_order=True
        )
        self.num_controlled_joints = len(self.ctrl_joint_ids)
        self.joint_names = list(self._robot.joint_names)
        self.tracked_body_ids, self.tracked_body_names = self._robot.find_bodies(
            spec.tracked_bodies, preserve_order=True
        )
        self.contact_body_names = list(spec.contact_bodies)
        missing = [b for b in spec.grasp_thumb_bodies + spec.grasp_finger_bodies if b not in self.contact_body_names]
        if missing:
            raise ValueError(f"grasp bodies must also be listed in contact_bodies: {missing}")
        # Columns of the contact matrix: 0 = basket floor, 1 + i = contact_bodies[i].
        self._thumb_cols = [1 + self.contact_body_names.index(b) for b in spec.grasp_thumb_bodies]
        self._finger_cols = [1 + self.contact_body_names.index(b) for b in spec.grasp_finger_bodies]
        self._grasp_palm_id = self._robot.find_bodies([spec.grasp_palm_body])[0][0]
        self._grasp_hand_joint_ids, _ = self._robot.find_joints(spec.grasp_hand_joints, preserve_order=True)

        limits = as_torch(self._robot.data.soft_joint_pos_limits)[:, self.ctrl_joint_ids]
        self._ctrl_lower, self._ctrl_upper = limits[..., 0], limits[..., 1]
        self._joint_targets = as_torch(self._robot.data.default_joint_pos)[:, self.ctrl_joint_ids].clone()

        # Apple / stem geometry in env-local coordinates.
        def vec(values):
            return torch.tensor(values, dtype=torch.float32, device=dev).repeat(n, 1)

        self._apple_center = vec(self.cfg.fruit.center)
        self._stem_anchor = vec(self.cfg.fruit.anchor)
        axis = self._stem_anchor - self._apple_center
        self._stem_axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        self._basket_pos = vec(self.cfg.basket_position)
        self._identity_quat = vec(backend_quat_xyzw((0.0, 0.0, 0.0, 1.0)))

        # Spring constants, and the spring rest point that makes the sagged apple sit at its
        # tree.json position: rest = center - K^-1 * m * g.
        m = self.cfg.fruit.mass
        c = self.cfg
        self._k_ax, self._k_lat = c.stem_axial_stiffness, c.stem_lateral_stiffness
        self._c_ax = 2.0 * c.stem_damping_ratio * (self._k_ax * m) ** 0.5
        self._c_lat = 2.0 * c.stem_damping_ratio * (self._k_lat * m) ** 0.5
        inertia = 0.4 * m * self.cfg.fruit.radius**2
        self._k_rot = c.stem_twist_stiffness
        self._c_rot = 2.0 * c.stem_damping_ratio * (self._k_rot * inertia) ** 0.5
        weight = torch.tensor(self.cfg.sim.gravity, device=dev) * m
        w_ax = (weight * self._stem_axis).sum(-1, keepdim=True) * self._stem_axis
        self._stem_rest = self._apple_center - (w_ax / self._k_ax + (weight - w_ax) / self._k_lat)

        # Stem state.
        self.apple_attached = torch.ones(n, dtype=torch.bool, device=dev)
        self.stem_break_force = torch.full((n,), c.stem_break_force, device=dev)
        self.stem_tension = torch.zeros(n, device=dev)
        self.stem_twist = torch.zeros(n, device=dev)
        self.detach_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.break_tension = torch.zeros(n, device=dev)
        self.break_threshold = torch.zeros(n, device=dev)
        self.break_external_force = torch.zeros(n, device=dev)
        self.apple_external_force = torch.zeros(n, 3, device=dev)
        """Extra world-frame force on the apple [N], applied every physics step (tests, wind, perturbations)."""

        # harvest_isaac's success/failure bookkeeping.
        self.harvest = HarvestTracker(
            n,
            HarvestParams(
                apple_radius=self.cfg.fruit.radius,
                apple_mass=self.cfg.fruit.mass,
                apple_rest_height=self.cfg.fruit.center[2],
                basket_center=self.cfg.basket_position,
                basket_half_inner=(BASKET_INNER_SIZE[0] / 2 - 0.01, BASKET_INNER_SIZE[1] / 2 - 0.01),
                gravity=-self.cfg.sim.gravity[2],
            ),
            dev,
        )

    # ------------------------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions[:] = self.actions
        self.actions[:] = actions.clamp(-self.cfg.clip_actions, self.cfg.clip_actions)
        mode = self.cfg.action_mode
        if mode == "absolute":
            targets = self._ctrl_lower + (self.actions + 1.0) * 0.5 * (self._ctrl_upper - self._ctrl_lower)
        elif mode == "offset":
            home = as_torch(self._robot.data.default_joint_pos)[:, self.ctrl_joint_ids]
            targets = home + self.cfg.action_scale * self.actions
        elif mode == "delta":
            targets = self._joint_targets + self.cfg.action_scale * self.actions
        else:
            raise ValueError(f"Unknown action_mode: {mode!r}")
        self._joint_targets[:] = torch.clamp(targets, self._ctrl_lower, self._ctrl_upper)

    def _apply_action(self):
        # Called every physics step (decimation times per env.step).
        asset_call(self._robot, "set_joint_position_target", target=self._joint_targets, joint_ids=self.ctrl_joint_ids)
        self._apply_stem()

    def home_action(self) -> torch.Tensor:
        """The action that holds the robot's home pose under the current ``action_mode``."""
        if self.cfg.action_mode != "absolute":
            return torch.zeros(self.num_envs, self.num_controlled_joints, device=self.device)
        home = as_torch(self._robot.data.default_joint_pos)[:, self.ctrl_joint_ids]
        return 2.0 * (home - self._ctrl_lower) / (self._ctrl_upper - self._ctrl_lower) - 1.0

    # ------------------------------------------------------------------------------------
    # Stem
    # ------------------------------------------------------------------------------------

    def _apply_stem(self):
        origins = self.scene.env_origins
        data = self._apple.data
        pos = as_torch(data.root_pos_w) - origins
        vel = as_torch(data.root_lin_vel_w)
        quat = as_torch(data.root_quat_w)
        ang_vel = as_torch(data.root_ang_vel_w)

        axis = self._stem_axis
        disp = pos - self._stem_rest
        disp_ax = (disp * axis).sum(-1, keepdim=True) * axis
        vel_ax = (vel * axis).sum(-1, keepdim=True) * axis
        elastic = -(self._k_ax * disp_ax + self._k_lat * (disp - disp_ax))
        force = elastic - (self._c_ax * vel_ax + self._c_lat * (vel - vel_ax))

        rot = axis_angle_from_quat(quat_mul(quat, quat_conjugate(self._identity_quat)))
        torque = -self._k_rot * rot - self._c_rot * ang_vel

        attached = self.apple_attached
        self.stem_tension = torch.where(attached, torch.linalg.norm(elastic, dim=-1), torch.zeros_like(self.stem_tension))
        self.stem_twist = torch.where(attached, torch.linalg.norm(rot, dim=-1), torch.zeros_like(self.stem_twist))
        snapped = attached & (self.stem_tension > self.stem_break_force)
        if self.cfg.stem_twist_break_angle is not None:
            snapped |= attached & (self.stem_twist > self.cfg.stem_twist_break_angle)
        if snapped.any():
            self.break_tension[snapped] = self.stem_tension[snapped]
            self.break_threshold[snapped] = self.stem_break_force[snapped]
            self.break_external_force[snapped] = torch.linalg.norm(self.apple_external_force[snapped], dim=-1)
            self.apple_attached &= ~snapped
            self.detach_step[snapped] = self.episode_length_buf[snapped]

        on = self.apple_attached.unsqueeze(-1).float()
        force_w = force * on + self.apple_external_force
        torque_w = torque * on
        if LAB3:
            self._apple.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force_w.unsqueeze(1), torques=torque_w.unsqueeze(1), is_global=True
            )
        else:
            # Lab 2's wrench setter accepts body-frame values only.
            self._apple.set_external_force_and_torque(
                forces=quat_rotate_inverse(quat, force_w).unsqueeze(1),
                torques=quat_rotate_inverse(quat, torque_w).unsqueeze(1),
            )

    # ------------------------------------------------------------------------------------
    # Task outcome (harvest_isaac logic)
    # ------------------------------------------------------------------------------------

    def _contact_forces(self) -> torch.Tensor:
        """Normal-force vectors (num_envs, 1 + num_contact_bodies, 3): basket floor, then each contact body."""
        matrix = self._apple_contact.data.normal_force_matrix_w if LAB3 else self._apple_contact.data.force_matrix_w
        return as_torch(matrix)[:, 0]

    def _update_task(self):
        """Advance the harvest bookkeeping by one control step (called from _get_dones)."""
        origins = self.scene.env_origins
        robot, apple = self._robot.data, self._apple.data
        forces = self._contact_forces()
        magnitudes = torch.linalg.norm(forces, dim=-1)
        self.harvest.update(
            detached=~self.apple_attached,
            apple_pos=as_torch(apple.root_pos_w) - origins,
            apple_speed=torch.linalg.norm(as_torch(apple.root_lin_vel_w), dim=-1),
            palm_pos=as_torch(robot.body_pos_w)[:, self._grasp_palm_id] - origins,
            thumb_forces=magnitudes[:, self._thumb_cols],
            finger_forces=magnitudes[:, self._finger_cols],
            basket_normal_force_z=forces[:, 0, 2],
            hand_joint_pos=as_torch(robot.joint_pos)[:, self._grasp_hand_joint_ids],
            finite=torch.isfinite(as_torch(robot.joint_pos)).all(-1) & torch.isfinite(as_torch(apple.root_state_w)).all(-1),
        )

    # ------------------------------------------------------------------------------------
    # State and cameras
    # ------------------------------------------------------------------------------------

    # Constants (they never change during an episode), kept out of get_state().

    @property
    def joint_lower(self) -> torch.Tensor:
        """(num_envs, n) lower soft limit of each controlled joint [rad]; action -1 maps here."""
        return self._ctrl_lower

    @property
    def joint_upper(self) -> torch.Tensor:
        """(num_envs, n) upper soft limit of each controlled joint [rad]; action +1 maps here."""
        return self._ctrl_upper

    @property
    def joint_home(self) -> torch.Tensor:
        """(num_envs, n) home pose of each controlled joint [rad] (the reset pose)."""
        return as_torch(self._robot.data.default_joint_pos)[:, self.ctrl_joint_ids]

    @property
    def basket_pos(self) -> torch.Tensor:
        """(num_envs, 3) basket floor centre, env-local [m]."""
        return self._basket_pos

    @property
    def stem_anchor(self) -> torch.Tensor:
        """(num_envs, 3) where the stem meets the branch, env-local [m]."""
        return self._stem_anchor

    def get_state(self) -> dict[str, torch.Tensor]:
        """Everything that changes about the robot, apple and task, batched over envs.

        Positions are in each env's local frame (the tree's frame, metres, Z up);
        quaternions are (x, y, z, w). Shapes are (num_envs, ...).
        """
        origins = self.scene.env_origins
        robot, apple = self._robot.data, self._apple.data
        ids, bids = self.ctrl_joint_ids, self.tracked_body_ids
        contact = torch.linalg.norm(self._contact_forces(), dim=-1)
        h = self.harvest
        return {
            # robot
            "joint_pos": as_torch(robot.joint_pos),  # all joints, order: self.joint_names
            "joint_vel": as_torch(robot.joint_vel),
            "ctrl_joint_pos": as_torch(robot.joint_pos)[:, ids],  # controlled joints, order: self.ctrl_joint_names
            "ctrl_joint_vel": as_torch(robot.joint_vel)[:, ids],
            "ctrl_joint_target": self._joint_targets,
            "root_pos": as_torch(robot.root_pos_w) - origins,
            "root_quat": canonical_quat(as_torch(robot.root_quat_w)),
            "body_pos": as_torch(robot.body_pos_w)[:, bids] - origins.unsqueeze(1),  # order: self.tracked_body_names
            "body_quat": canonical_quat(as_torch(robot.body_quat_w)[:, bids]),
            "body_lin_vel": as_torch(robot.body_lin_vel_w)[:, bids],
            "body_ang_vel": as_torch(robot.body_ang_vel_w)[:, bids],
            # apple and stem
            "apple_pos": as_torch(apple.root_pos_w) - origins,
            "apple_quat": canonical_quat(as_torch(apple.root_quat_w)),
            "apple_lin_vel": as_torch(apple.root_lin_vel_w),
            "apple_ang_vel": as_torch(apple.root_ang_vel_w),
            "apple_attached": self.apple_attached,
            "break_tension": self.break_tension,
            "break_threshold": self.break_threshold,
            "break_external_force": self.break_external_force,
            "stem_tension": self.stem_tension,  # current elastic stem force [N]
            "stem_twist": self.stem_twist,  # current apple rotation from rest [rad]
            "detach_step": self.detach_step,
            # contacts (normal force magnitudes [N])
            "finger_contact_force": contact[:, 1:1 + len(self.contact_body_names)],
            "wood_contact_force": contact[:, 1 + len(self.contact_body_names):].sum(-1),
            "basket_contact_force": contact[:, 0],
            # harvest_isaac task status
            "hand_distance": h.hand_distance,  # grasping palm to apple [m]
            "opposing_contact": h.opposing_contact,  # thumb and a finger both pressing
            "grasp_contact": h.grasp_contact,  # opposing contact for >= 5 steps
            "carried": h.carried,
            "released": h.released,
            "hand_open": h.opened,
            "basket_supported": h.supported,  # basket floor carries >= 40 % of the apple's weight
            "apple_settled_in_basket": h.settled,
            "placed_steps": h.hold_steps,
            "success": h.success,
            "failure": h.failure,  # index into OrchardEnv.FAILURE_NAMES
            "episode_step": self.episode_length_buf,
        }

    def get_images(self) -> dict[str, torch.Tensor]:
        """Latest camera images: ``{name: (num_envs, 480, 640, 3) uint8}``, plus
        ``cam_left_high_depth`` (num_envs, 480, 640, 1) in metres. Empty if cameras are off."""
        images = {}
        for name in self.camera_names:
            output = self.scene[name].data.output
            images[name] = output["rgb"]
            if "distance_to_image_plane" in output:
                images[f"{name}_depth"] = output["distance_to_image_plane"]
        return images

    # ------------------------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Default policy input (see ``observation_mode``). Change freely; keep the size in
        _configure_gym_env_spaces in sync."""
        robot = self._robot.data
        ids = self.ctrl_joint_ids
        # harvest_isaac's PreviewPolicyCfg: joint_pos_rel, joint_vel_rel, last_action.
        terms = [
            as_torch(robot.joint_pos)[:, ids] - as_torch(robot.default_joint_pos)[:, ids],
            as_torch(robot.joint_vel)[:, ids] - as_torch(robot.default_joint_vel)[:, ids],
            self.actions,
        ]
        if self.cfg.observation_mode == "state":
            origins = self.scene.env_origins
            apple_pos = as_torch(self._apple.data.root_pos_w) - origins
            palms = as_torch(robot.body_pos_w)[:, self.tracked_body_ids] - origins.unsqueeze(1)
            terms += [
                (palms - apple_pos.unsqueeze(1)).flatten(1),
                apple_pos - self._basket_pos,
                as_torch(self._apple.data.root_lin_vel_w),
                self.apple_attached.float().unsqueeze(-1),
            ]
        return {"policy": torch.cat(terms, dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        """TODO(you): your reward. Use ``self.get_state()`` for everything you need.

        Useful pieces: ``body_pos`` (palms) vs ``apple_pos``, ``finger_contact_force``,
        ``opposing_contact``/``grasp_contact``, ``stem_tension``, ``apple_attached``,
        ``carried``, ``released``, ``apple_pos`` vs ``basket_pos``, ``success``, ``failure``,
        ``self.actions``/``self._prev_actions``.
        """
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_task()
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.terminate_on_success:
            terminated |= self.harvest.success
        if self.cfg.terminate_on_failure:
            terminated |= self.harvest.failure != 0
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        log = self.extras.setdefault("log", {})
        log["Metrics/success"] = self.harvest.success[env_ids].float().mean().item()
        log["Metrics/carried"] = self.harvest.carried[env_ids].float().mean().item()
        log["Metrics/apple_detached"] = (~self.apple_attached[env_ids]).float().mean().item()
        for code, name in enumerate(FAILURE_NAMES[1:], start=1):
            log[f"Failure/{name}"] = (self.harvest.failure[env_ids] == code).float().mean().item()

        super()._reset_idx(env_ids)
        origins = self.scene.env_origins[env_ids]
        k = len(env_ids)

        # Robot: home pose (+ optional noise), zero velocity, fixed root at its stance.
        joint_pos = as_torch(self._robot.data.default_joint_pos)[env_ids].clone()
        if self.cfg.reset_joint_noise > 0.0:
            joint_pos += (torch.rand_like(joint_pos) * 2.0 - 1.0) * self.cfg.reset_joint_noise
            limits = as_torch(self._robot.data.soft_joint_pos_limits)[env_ids]
            joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
        root_pose = default_root_pose(self._robot.data)[env_ids].clone()
        root_pose[:, :3] += origins
        asset_call(self._robot, "write_root_pose_to_sim", root_pose=root_pose, env_ids=env_ids)
        asset_call(self._robot, "write_root_velocity_to_sim", root_velocity=torch.zeros(k, 6, device=self.device), env_ids=env_ids)
        asset_call(self._robot, "write_joint_position_to_sim", position=joint_pos, env_ids=env_ids)
        asset_call(self._robot, "write_joint_velocity_to_sim", velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
        asset_call(self._robot, "set_joint_position_target", target=joint_pos, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_pos[:, self.ctrl_joint_ids]

        # Apple: back on its stem, at rest.
        apple_pose = torch.cat((self._apple_center[env_ids] + origins, self._identity_quat[env_ids]), dim=-1)
        asset_call(self._apple, "write_root_pose_to_sim", root_pose=apple_pose, env_ids=env_ids)
        asset_call(self._apple, "write_root_velocity_to_sim", root_velocity=torch.zeros(k, 6, device=self.device), env_ids=env_ids)

        # Stem and task bookkeeping.
        self.apple_attached[env_ids] = True
        self.break_tension[env_ids] = 0.0
        self.break_threshold[env_ids] = 0.0
        self.break_external_force[env_ids] = 0.0
        if self.cfg.stem_break_force_range is not None:
            lo, hi = self.cfg.stem_break_force_range
            self.stem_break_force[env_ids] = torch.empty(k, device=self.device).uniform_(lo, hi)
        self.stem_tension[env_ids] = 0.0
        self.stem_twist[env_ids] = 0.0
        self.detach_step[env_ids] = -1
        self.apple_external_force[env_ids] = 0.0
        self.harvest.reset(env_ids)
        self.actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
