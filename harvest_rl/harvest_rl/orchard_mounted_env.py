"""Explicit parked mounted single-fruit task, prepared offline, runtime untested.

Legacy configs/classes/registration are untouched. configure_parked_mounted_case
returns a private copy only when enabled. The lazy class factory requires Isaac
already launched by a separately approved runner. Raw calibrated shape-contact
and safety readers are REQUIRED; basket-root forces are not relabelled floor.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .harvest_cycle import BasketProposal, FruitTarget, HarvestCycle
from .harvest_cycle_adapter import (TASK_NAME, HarvestCycleAdapter, ParkedTaskSignals,
                                   motion_at_frame_origin)
from .mounted_basket_verifier import BasketGeometry, FruitGeometry, MountedBasketVerifier, rotate
from .orchard_mounted_basket import build_basket_plan, spawn_mounted_basket

LEGACY_BASKET_ENTITIES = ("basket_floor", "basket_wall_0", "basket_wall_1",
                         "basket_wall_2", "basket_wall_3", "pedestal")


def configure_parked_mounted_case(cfg, spec, target, *, contact_reader=None,
                                 safety_reader=None, prior_failures=None,
                                 runtime_review_reference=None, admission_reference=None,
                                 admission_sha256=None, other_contact_ids=()):
    """Opt-in private scene/config derivative; no USD or simulator imports/writes.

    Preview needs no runtime permission. Missing reviewed admission/readers stops
    the lazy runtime class before it calls the legacy constructor. Call existing
    configure_orchard_case FIRST if using OrchardSpec: its stationary basket
    adaptation must not run again after this new-task opt-in.
    """
    plan = build_basket_plan(spec)
    if plan is None:
        return cfg
    if not isinstance(target, FruitTarget) or target.fruit_id != cfg.fruit.id or target.mass_kg != cfg.fruit.mass:
        raise ValueError("Exact current native fruit identity/mass must match the target")
    if getattr(cfg, "mounted_spec", None) is not None:
        raise ValueError("Do not reconfigure an already mounted scene")
    if cfg.scene.num_envs != 1 or cfg.action_mode != "delta":
        raise ValueError("Prepare exactly one existing delta-action scene")
    if cfg.scene.robot.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0') != spec.robot_prim_path:
        raise ValueError("Mounted parent must bind this exact scene's robot root")
    if prior_failures is not None and (type(prior_failures) is not int or prior_failures < 0):
        raise ValueError("Historical task failure count must be explicit and nonnegative")
    original_filters = cfg.scene.apple_contact.filter_prim_paths_expr
    if len(original_filters) < 15:
        raise ValueError("Preserve the existing basket +14 hand contact prefix")
    result = copy.deepcopy(cfg)
    for name in LEGACY_BASKET_ENTITIES:
        if not hasattr(result.scene, name):
            raise ValueError("Unexpected legacy basket/reset layout: " + name)
        setattr(result.scene, name, None)
    # Column0 is now generic basket-body contact for legacy acquisition plumbing;
    # independent named success NEVER uses it as classified floor support.
    result.scene.apple_contact.filter_prim_paths_expr = [plan["body_path"], *original_filters[1:]]
    result.scene.num_envs = 1
    result.cameras = False
    camera_names = ['cam_left_high', *(camera.name for camera in getattr(result.robot, 'wrist_cameras', ()))]
    for name in camera_names:
        if hasattr(result.scene, name):
            setattr(result.scene, name, None)
    result.sim.device = 'cpu'
    result.episode_length_s = 30.
    result.terminate_on_success = result.terminate_on_failure = False
    for name in ("apple_offset_xy", "apple_offset_z", "basket_offset_xy"):
        setattr(result, name, 0.)
    result.stem_break_force_range = None
    # New-task self-collision is explicit; no changes to the original robot cfg.
    result.robot.articulation.spawn.articulation_props.enabled_self_collisions = True
    result.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    result.mounted_spec, result.mounted_plan, result.mounted_target = spec, plan, target
    result.mounted_contact_reader, result.mounted_safety_reader = contact_reader, safety_reader
    result.mounted_prior_failures = prior_failures
    result.mounted_runtime_review_reference = runtime_review_reference
    result.mounted_admission_reference = admission_reference
    result.mounted_admission_sha256 = admission_sha256
    result.mounted_other_contact_ids = tuple(other_contact_ids)
    return result


def validate_parked_runtime_inputs(cfg):
    """Fail before constructor/physics; this function grants no authorization."""
    spec = getattr(cfg, "mounted_spec", None)
    if spec is None or build_basket_plan(spec) is None:
        raise ValueError("Explicit enabled mounted configuration required")
    if (not callable(cfg.mounted_contact_reader) or not callable(cfg.mounted_safety_reader)
            or not cfg.mounted_runtime_review_reference or not cfg.mounted_admission_reference):
        raise ValueError("Require reviewed runtime/admission and real classified-contact/safety readers")
    if type(cfg.mounted_prior_failures) is not int or cfg.mounted_prior_failures >= 3:
        raise ValueError("Missing inherited counts or stopped reproduction group; human review required")
    if (cfg.scene.num_envs != 1 or cfg.sim.device != 'cpu' or cfg.episode_length_s != 30.
            or cfg.cameras or cfg.terminate_on_success or cfg.terminate_on_failure
            or any(getattr(cfg.scene, name) is not None for name in LEGACY_BASKET_ENTITIES)
            or not cfg.robot.articulation.spawn.articulation_props.enabled_self_collisions
            or not cfg.scene.robot.spawn.articulation_props.enabled_self_collisions):
        raise ValueError("Mounted scene/reset/self-collision contract changed")
    if (cfg.mounted_plan != build_basket_plan(cfg.mounted_spec)
            or cfg.scene.apple_contact.filter_prim_paths_expr[0] != cfg.mounted_plan['body_path']):
        raise ValueError("Basket plan/contact target changed")
    return _read_admission(cfg)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _hashed_json(reference):
    path = Path(reference['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError("Changed native admission evidence: " + str(path))
    return json.loads(raw)


def _read_admission(cfg):
    """Specific parked proof artifacts, never admission from nonempty booleans."""
    admission = _hashed_json(dict(path=cfg.mounted_admission_reference, sha256=cfg.mounted_admission_sha256))
    binding = dict(task_name=TASK_NAME, basket_plan_sha256=_digest(cfg.mounted_plan),
                   target=asdict(cfg.mounted_target), scene_binding_sha256=_digest(parked_scene_binding(cfg)))
    if (admission.get('schema') != 'parked_mounted_admission_v1'
            or any(admission.get(k) != v for k, v in binding.items())
            or admission.get('runtime_review_reference') != cfg.mounted_runtime_review_reference
            or admission.get('prior_task_failures') != cfg.mounted_prior_failures
            or admission.get('preserved_stops') != dict(smoke=3, historical_1002=3, calibration_failed=1)):
        raise ValueError("Missing exact parked scene/target/review/history binding")
    for category, predicates in (
        ('native_geometry', ('reset_collision_free', 'placement_withdrawal_clear', 'self_collision_enabled', 'fixed_base_stable')),
        ('shape_contacts', ('physical_initialization_valid', 'positive_contact_and_separation', 'floor_wall_identity_distinguished'))):
        proof = _hashed_json(admission['proofs'][category])
        if (proof.get('category') != category or proof.get('passed') is not True
                or any(proof.get(k) != v for k, v in binding.items())
                or any(proof.get(k) is not True for k in predicates)):
            raise ValueError("Unmet native parked admission: " + category)
        # A flag/reference alone is not a physical record. Require the retained
        # trace bytes as well; their scientific interpretation still needs review.
        trace = proof['trace']
        if hashlib.sha256(Path(trace['path']).read_bytes()).hexdigest() != trace['sha256']:
            raise ValueError("Missing/changed physical admission trace")
    # Preserve and explicitly review all stopped group bytes; never rewrite them.
    parents = {key: _hashed_json(admission['parent_ledgers'][key])
               for key in ('smoke', 'historical_1002', 'calibration_failed')}
    if (parents['smoke'].get('smoke_failures') != 3
            or parents['smoke'].get('historical_1002_failures') != 3
            or parents['calibration_failed'].get('failures') != 1
            or parents['calibration_failed'].get('status') != 'failed_stop_no_retry'):
        raise ValueError("Historical stopped counts/status changed")
    return admission


def parked_scene_binding(cfg):
    """Freeze requested source/stance/reset context, not a measured snapshot."""
    return dict(robot_position_m=list(cfg.scene.robot.init_state.pos),
        robot_orientation_backend=list(cfg.scene.robot.init_state.rot),
        default_joint_position=copy.deepcopy(getattr(cfg.scene.robot.init_state, 'joint_pos', {})),
        fruit_center_m=list(cfg.fruit.center), fruit_anchor_m=list(cfg.fruit.anchor),
        fruit_half_extents_m=list(cfg.fruit.half_extents), fruit_mass_kg=cfg.fruit.mass,
        reset_seed=getattr(cfg, 'seed', None), arm_reset_noise_rad=getattr(cfg, 'arm_reset_noise', 0.),
        policy_frame_yaw_deg=getattr(cfg, 'policy_frame_yaw_deg', 0.),
        physics_dt_s=getattr(cfg.sim, 'dt', None), decimation=getattr(cfg, 'decimation', None),
        gravity_w_m_s2=getattr(cfg.sim, 'gravity', None), action_scale=getattr(cfg, 'action_scale', None),
        stem_parameters={name: getattr(cfg, name, None) for name in (
            'stem_break_force', 'stem_break_force_range', 'stem_axial_stiffness', 'stem_lateral_stiffness',
            'stem_damping_ratio', 'stem_twist_stiffness', 'stem_twist_break_angle')},
        native_priming_control_steps=1 if getattr(cfg.mounted_contact_reader, 'native_interval_contract', None) == 'mounted_native_window_v1' else 0,
        native_orchard_provenance=copy.deepcopy(getattr(cfg, 'orchard_case', None)))


def _row(value):
    value = value.torch if hasattr(value, "torch") else value
    result = value[0]
    return result.detach().cpu().tolist() if hasattr(result, "detach") else result.tolist()


def native_com_offset(data):
    """Accept explicit N×1×3 or N×3 COM buffers, never guess missing offsets."""
    value = _row(data.com_pos_b)
    if len(value) == 1 and isinstance(value[0], (tuple, list)):
        value = value[0]
    if (len(value) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("Require measured finite three-vector COM offset")
    return tuple(value)


def make_parked_mounted_env_class():
    """Import only after AppLauncher; returns an explicit opt-in class, no launch."""
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaacsim.core.utils.stage import get_current_stage
    from .baseline.pick_place_env import PickPlaceEnv
    from .compat import as_torch, canonical_quat
    from .orchard_env import OrchardEnv
    from .pluck.env import PluckEnv

    class ParkedMountedOrchardEnv(PluckEnv):
        mounted_task_contract = TASK_NAME

        def __init__(self, cfg, *args, **kwargs):
            self._mounted_admission = validate_parked_runtime_inputs(cfg)
            self._mounted_started = False
            self.mounted_timeout_reached = False
            self._mounted_adapter = None
            self._native_prime_attempted = self._native_prime_active = self._native_primed = False
            super().__init__(cfg, *args, **kwargs)

        def _setup_scene(self):
            # RandomApple's setup expects the removed kinematic basket parts.
            OrchardEnv._setup_scene(self)
            self._stem = self.scene["stem"]
            self._basket_parts = []  # Stem reset remains initial-only; basket is NEVER pose-followed.
            plan = spawn_mounted_basket(get_current_stage(), self.cfg.mounted_spec)
            self._mounted_plan = plan
            self._mounted_body = RigidObject(RigidObjectCfg(prim_path=plan['body_path'], spawn=None))
            self.scene.rigid_objects['mounted_basket'] = self._mounted_body
            if self._native_reader() is not None:
                self._native_reader().attach(self, get_current_stage())

        def _native_reader(self):
            reader = self.cfg.mounted_contact_reader
            return reader if getattr(reader, 'native_interval_contract', None) == 'mounted_native_window_v1' else None

        def step(self, actions):
            reader = self._native_reader()
            if not self._mounted_started and not self._native_prime_active:
                raise RuntimeError("Start the learned skill or use the explicit native zero-action prime")
            if reader is not None:
                reader.begin_interval(self)
            try:
                result = super().step(actions)
                if reader is not None:reader.finish_interval(self)
                return result
            except BaseException as exc:
                if reader is not None and reader.buffer is not None:
                    reader.buffer.fail(f"Native step incomplete: {type(exc).__name__}: {exc}")
                raise

        def prime_native_telemetry(self):
            """One declared physical zero-delta hold, NOT executed by importing.

            An approved runner must count it in its bounds and save actual init
            state afterwards. Never resets/teleports fruit or scripts reaching.
            """
            if (self._native_reader() is None or self._mounted_started or self._native_prime_attempted
                    or int(self.episode_length_buf[0]) != 0):
                raise RuntimeError("Native telemetry prime is one-shot, before any skill/action")
            self._native_prime_attempted = self._native_prime_active = True
            try:
                self.step(torch.zeros((1, 14), device=self.device))
                self._native_primed = True
            finally:
                self._native_prime_active = False

        def close(self):
            cfg = getattr(self, 'cfg', None)
            reader = getattr(cfg, 'mounted_contact_reader', None)
            if getattr(reader, 'native_interval_contract', None) == 'mounted_native_window_v1':
                reader.close()
            super().close()

        @property
        def basket_pos(self):
            # Actual B origin = inner floor surface centre; no old stationary pose.
            return as_torch(self._mounted_body.data.root_pos_w) - self.scene.env_origins

        def _get_dones(self):
            self._update_task()  # Keep native physical grasp/detachment/carry/failure.
            self.mounted_timeout_reached = bool(self.episode_length_buf[0] >= self.max_episode_length-1)
            # All exits are reported through the named task adapter; prevent any
            # DirectRLEnv autoreset from respawning fruit/erasing deposit evidence.
            zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return zeros, zeros.clone()

        def _reset_idx(self, env_ids):
            if self._mounted_started or self._native_prime_attempted:
                raise RuntimeError("No world reset/respawn after mounted skill starts")
            super()._reset_idx(env_ids)

        def _pre_physics_step(self, actions):
            if self._native_prime_active and bool((as_torch(actions) != 0).any()):
                raise RuntimeError("Telemetry priming may only hold actual reset targets with zero delta")
            if self._mounted_started and (self._mounted_adapter.stopped
                    or self._mounted_adapter.last_sample != int(self.episode_length_buf[0])):
                raise RuntimeError("Require a fresh mounted skill action; no stepping after stop/reset")
            super()._pre_physics_step(actions)

        def _mounted_motion(self, time_s):
            data = self._mounted_body.data
            # Lab2 root pose is link frame; native velocity is COM velocity.
            # The authored COM offset converts it to B-origin point velocity.
            com_offset = native_com_offset(data)
            if math.dist(com_offset, self._mounted_plan['center_of_mass_b_m']) > 1e-5:
                raise ValueError("Native basket COM differs from authored dry-only model")
            return motion_at_frame_origin(time_s=time_s,
                position_w_m=_row(data.root_pos_w),
                orientation_xyzw=_row(canonical_quat(as_torch(data.root_quat_w))),
                com_linear_velocity_w_m_s=_row(data.root_lin_vel_w),
                angular_velocity_w_rad_s=_row(data.root_ang_vel_w),
                center_of_mass_b_m=com_offset)

        def _mounted_read(self, _env):
            sample = int(self.episode_length_buf[0])
            time_s = sample*self.step_dt
            motion = self._mounted_motion(time_s)
            # Raw reader must supply shape-classified contact forces and actual
            # minimum hand-surface separation; no force-matrix relabelling here.
            frame = self.cfg.mounted_contact_reader(self, sample, time_s, motion)
            if frame.sample != sample or frame.time_s != time_s or frame.basket_id != self._mounted_plan['body_path']:
                raise ValueError("Raw mounted telemetry sample/body mismatch")
            if frame.calibration_reference != self._mounted_admission['proofs']['shape_contacts']['sha256']:
                raise ValueError("Reader is not bound to the admitted physical contact mapping")
            if (math.dist(frame.basket.position_w_m, motion.position_w_m) > 1e-5
                    or math.dist(frame.basket.linear_velocity_w_m_s, motion.linear_velocity_w_m_s) > 1e-5
                    or math.dist(frame.basket.angular_velocity_w_rad_s, motion.angular_velocity_w_rad_s) > 1e-5
                    or min(math.dist(frame.basket.orientation_xyzw, motion.orientation_xyzw),
                           math.dist(frame.basket.orientation_xyzw, tuple(-x for x in motion.orientation_xyzw))) > 1e-5):
                raise ValueError("Reader supplied a different basket pose/twist")
            if len(frame.fruits) != 1:
                raise ValueError("Parked integration supports exactly one native fruit")
            fruit = frame.fruits[0]
            q = _row(canonical_quat(as_torch(self._apple.data.root_quat_w)))
            apple_motion = motion_at_frame_origin(time_s=time_s,
                position_w_m=_row(self._apple.data.root_pos_w), orientation_xyzw=q,
                com_linear_velocity_w_m_s=_row(self._apple.data.root_lin_vel_w),
                angular_velocity_w_rad_s=_row(self._apple.data.root_ang_vel_w),
                center_of_mass_b_m=native_com_offset(self._apple.data))
            if (fruit.target_id != self.cfg.mounted_target.target_id or fruit.body_id != self.cfg.scene.apple.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0')
                    or math.dist(fruit.motion.position_w_m, _row(self._apple.data.root_pos_w)) > 1e-5
                    or math.dist(fruit.motion.linear_velocity_w_m_s, apple_motion.linear_velocity_w_m_s) > 1e-5
                    or math.dist(fruit.motion.angular_velocity_w_rad_s, _row(self._apple.data.root_ang_vel_w)) > 1e-5
                    or min(math.dist(fruit.motion.orientation_xyzw, q), math.dist(fruit.motion.orientation_xyzw, tuple(-x for x in q))) > 1e-5):
                raise ValueError("Reader fruit identity/native-centred YCB pose mismatch")
            safety = self.cfg.mounted_safety_reader(self, sample, time_s)
            if (safety.get('sample') != sample or safety.get('time_s') != time_s or not safety.get('source_reference')
                    or any(type(safety.get(k)) is not bool for k in ('fallen', 'unsafe_collision'))):
                raise ValueError("Require fresh physical safety telemetry")
            s = self.get_state()
            flag = lambda name: bool(s[name][0])
            norm = lambda value: float(torch.linalg.norm(as_torch(value)[0]))
            signals = ParkedTaskSignals(sample, time_s, self.cfg.mounted_target.target_id,
                attached=flag('apple_attached'), sustained_grasp=flag('grasp_contact'),
                opposing_grasp=flag('opposing_contact'), carried=flag('carried'), released=flag('released'),
                failure_code=int(s['failure'][0]), extra_apple_assistance=norm(self.apple_external_force) > 0.,
                fallen=safety['fallen'], unsafe_collision=safety['unsafe_collision'],
                base_linear_speed_m_s=norm(self._robot.data.root_lin_vel_w),
                base_angular_speed_rad_s=norm(self._robot.data.root_ang_vel_w),
                time_limit_reached=self.mounted_timeout_reached)
            return frame, signals

        def make_cycle_adapter(self, learned_policy):
            if self._mounted_started:
                raise ValueError("One adapter/skill only; no implicit reset/retry")
            reader = self._native_reader()
            if reader is not None and not self._native_primed:
                raise ValueError("Explicit measured telemetry prime/init snapshot required before the skill")
            ids, target = self._mounted_plan['contact_ids'], self.cfg.mounted_target
            geometry = BasketGeometry(ids['basket_body_id'], ids['floor_contact_id'],
                tuple(ids['wall_contact_ids']), tuple(self.cfg.mounted_spec.inner_size_m))
            apple_path = self.cfg.scene.apple.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0')
            verifier = MountedBasketVerifier(geometry,
                (FruitGeometry(target.target_id, apple_path, tuple(self.cfg.fruit.half_extents), target.mass_kg),),
                hand_body_ids=reader.hand_actor_ids if reader is not None else tuple(self.cfg.mounted_spec.robot_prim_path+'/'+n for n in self.contact_body_names),
                other_body_ids=tuple(dict.fromkeys((ids['basket_body_id'], *self.cfg.mounted_other_contact_ids,
                                                  *(reader.other_actor_ids if reader is not None else ())))))
            proposal = BasketProposal(inner_size_m=tuple(self.cfg.mounted_spec.inner_size_m),
                wall_thickness_m=self.cfg.mounted_spec.wall_thickness_m, dry_mass_kg=self.cfg.mounted_spec.dry_mass_kg)
            cycle = HarvestCycle((target,), basket=proposal, prior_failures=self.cfg.mounted_prior_failures)
            cycle.select(target.target_id, admitted=bool(self.cfg.mounted_admission_reference), stance_stable=True)
            self._mounted_adapter = HarvestCycleAdapter(self, learned_policy, cycle, verifier,
                reader=self._mounted_read, observation_builder=PickPlaceEnv._get_observations)
            self._mounted_started = True
            return self._mounted_adapter

    return ParkedMountedOrchardEnv


def _dense_spawn_mounted_basket(env):
    """Scene-setup callback; physical constraint, never per-step pose following."""
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaacsim.core.utils.stage import get_current_stage
    if getattr(env.cfg, 'dense_camera_joint_filters', False):
        from .camera_joint_filters import apply_camera_joint_filters
        env._camera_joint_filter_proof = apply_camera_joint_filters(
            get_current_stage(), env.cfg.mounted_spec.robot_prim_path)
    plan = spawn_mounted_basket(get_current_stage(), env.cfg.mounted_spec)
    env._mounted_plan = plan
    env._mounted_body = RigidObject(RigidObjectCfg(prim_path=plan['body_path'], spawn=None))
    env.scene.rigid_objects['mounted_basket'] = env._mounted_body


def _dense_measured_basket_position(env):
    from .compat import as_torch
    return as_torch(env._mounted_body.data.root_pos_w)-env.scene.env_origins


def configure_dense_mounted_case(cfg, spec, *, feasible_target_ids=(), diagnostic_target_ids=(), contact_reader,
        safety_reader, prior_failures, runtime_review_reference, contact_review,
        max_task_seconds, history_review_reference=None, camera_joint_filters=False):
    """Opt-in first integrated diagnostic, not fabricated native admission.

    All native fruit remain in cfg.dense_inventory. Capacity covers precisely the
    frozen feasible subset before any outcomes. Runtime authorization is checked
    by the concrete bounded runner before SDK launch; this config records its
    reference and leaves new geometry/contact/mount validation explicitly false.
    """
    from .orchard_mounted_basket import inventory_capacity_plan
    plan = build_basket_plan(spec)
    if plan is None or not getattr(cfg, 'dense_inventory', None):
        raise ValueError('Require enabled mount and complete physical inventory')
    if (cfg.scene.num_envs != 1 or cfg.action_mode != 'delta'
            or cfg.scene.robot.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0') != spec.robot_prim_path):
        raise ValueError('Require exact robot, one parked delta-action world')
    if (not callable(contact_reader) or not callable(safety_reader) or not runtime_review_reference
            or type(prior_failures) is not int or prior_failures < 0
            or not math.isfinite(max_task_seconds) or max_task_seconds <= 0):
        raise ValueError('Require reviewed bounded runtime/readers/history')
    if prior_failures >= 3 and not history_review_reference:
        raise ValueError('Stopped reproduction history requires explicit human review reference')
    if getattr(cfg, 'dense_setup_callback', None) is not None:
        raise ValueError('Refusing to replace an existing dense scene setup callback')
    review = copy.deepcopy(contact_review)
    if (not isinstance(review, dict) or review.get('scope') != 'runtime_test_unvalidated'
            or review.get('runtime_review_reference') != runtime_review_reference
            or review.get('raw_reader_contract') != 'mounted_native_window_v1'
            or review.get('report_impulse_frame') != 'world'
            or type(review.get('report_impulse_on_actor0_sign')) is not int
            or review['report_impulse_on_actor0_sign'] not in (-1, 1)
            or len(review.get('calibration_reference', '')) != 64):
        raise ValueError('Explicit measured contact convention and unvalidated diagnostic scope required')
    if feasible_target_ids and diagnostic_target_ids:
        raise ValueError('Do not relabel unresolved diagnostic targets as feasible')
    ids = tuple(str(i) for i in (feasible_target_ids or diagnostic_target_ids))
    rows = {str(row['target_id']): row for row in cfg.dense_inventory}
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(rows):
        raise ValueError('Frozen feasible IDs must be distinct members of full physical inventory')
    capacity = inventory_capacity_plan(spec, [dict(target_id=i,
        half_extents_m=rows[i]['fruit']['half_extents'], mass_kg=rows[i]['fruit']['mass']) for i in ids])
    if not capacity['fits']:
        raise ValueError('Mounted basket lacks geometric capacity for complete frozen feasible inventory')
    result = copy.deepcopy(cfg)
    if type(camera_joint_filters) is not bool:
        raise ValueError('Explicit boolean camera joint-filter opt-in required')
    result.dense_camera_joint_filters = camera_joint_filters
    for name in LEGACY_BASKET_ENTITIES:
        if not hasattr(result.scene, name):
            raise ValueError('Unexpected legacy stand basket layout: '+name)
        setattr(result.scene, name, None)
    for name, sensor in vars(result.scene).items():
        if name == 'apple_contact' or name.startswith('dense_contact_'):
            filters = list(sensor.filter_prim_paths_expr)
            if len(filters) < 15:
                raise ValueError('Missing original basket/hand sensor prefix')
            sensor.filter_prim_paths_expr = [plan['body_path'], *filters[1:]]
    result.robot.articulation.spawn.articulation_props.enabled_self_collisions = True
    result.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    result.cameras = False
    for name in ['cam_left_high', *(c.name for c in getattr(result.robot, 'wrist_cameras', ()))]:
        if hasattr(result.scene, name):setattr(result.scene, name, None)
    result.scene.num_envs, result.sim.device = 1, 'cpu'
    result.episode_length_s = float(max_task_seconds)
    result.terminate_on_success = result.terminate_on_failure = False
    result.mounted_spec, result.mounted_plan = spec, plan
    result.mounted_contact_reader, result.mounted_safety_reader = contact_reader, safety_reader
    result.mounted_prior_failures = prior_failures
    result.mounted_runtime_review_reference = runtime_review_reference
    result.mounted_history_review_reference = history_review_reference
    result.dense_feasible_target_ids = ids if feasible_target_ids else ()
    result.dense_requested_target_ids = ids
    capacity['target_scope'] = 'feasible' if feasible_target_ids else 'unresolved_diagnostic'
    result.dense_capacity_plan = capacity
    result.dense_contact_review = review
    result.dense_setup_callback = _dense_spawn_mounted_basket
    result.dense_basket_position_reader = _dense_measured_basket_position
    result.dense_integration_scope = 'runtime_test_unvalidated'
    result.dense_mount_validated = result.dense_physics_validated = False
    return result


def make_dense_mounted_env_class():
    """Lazy all-fruit mounted scene; no simulator launch or default mutation."""
    import torch
    from isaacsim.core.utils.stage import get_current_stage
    from .compat import as_torch, canonical_quat
    from .orchard_dense_env import make_dense_orchard_env_class
    Dense = make_dense_orchard_env_class()

    class DenseMountedOrchardEnv(Dense):
        def __init__(self, cfg, *args, **kwargs):
            if (getattr(cfg, 'dense_integration_scope', None) != 'runtime_test_unvalidated'
                    or cfg.dense_setup_callback is not _dense_spawn_mounted_basket
                    or cfg.mounted_plan != build_basket_plan(cfg.mounted_spec)
                    or not cfg.dense_capacity_plan['fits']
                    or not cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
                    or any(getattr(cfg.scene, key) is not None for key in LEGACY_BASKET_ENTITIES)):
                raise ValueError('Explicit full-inventory mounted diagnostic configuration required')
            self._dense_native_prime_attempted = False
            super().__init__(cfg, *args, **kwargs)
            try:
                cfg.mounted_contact_reader.attach(self, get_current_stage())
            except BaseException:
                self.close()
                raise

        def step(self, actions):
            reader = self.cfg.mounted_contact_reader
            reader.begin_interval(self)
            try:
                result = super().step(actions)
                reader.finish_interval(self)
                return result
            except BaseException as exc:
                if reader.buffer is not None:
                    reader.buffer.fail(f'Dense mounted interval incomplete: {type(exc).__name__}: {exc}')
                raise

        def prime_native_telemetry(self):
            """One counted zero-delta physical hold, retaining clock and all fruit."""
            if self._dense_native_prime_attempted or int(self.episode_length_buf[0]) != 0:
                raise RuntimeError('Dense telemetry prime is one-shot before any skill')
            self._dense_native_prime_attempted = True
            self.begin_dense_episode()
            self.step(torch.zeros((1, 14), device=self.device))

        def _mounted_motion(self, time_s):
            data = self._mounted_body.data
            com = native_com_offset(data)
            if math.dist(com, self._mounted_plan['center_of_mass_b_m']) > 1e-5:
                raise ValueError('Native basket COM differs from dry-only physical model')
            return motion_at_frame_origin(time_s=time_s, position_w_m=_row(data.root_pos_w),
                orientation_xyzw=_row(canonical_quat(as_torch(data.root_quat_w))),
                com_linear_velocity_w_m_s=_row(data.root_lin_vel_w),
                angular_velocity_w_rad_s=_row(data.root_ang_vel_w), center_of_mass_b_m=com)

        def read_mounted_telemetry(self):
            sample = int(self.episode_length_buf[0]); time_s = sample*self.step_dt
            motion = self._mounted_motion(time_s)
            frame = self.cfg.mounted_contact_reader(self, sample, time_s, motion)
            safety = self.cfg.mounted_safety_reader(self, sample, time_s)
            if (frame.sample != sample or frame.time_s != time_s
                    or frame.basket_id != self._mounted_plan['body_path']
                    or safety.get('sample') != sample or safety.get('time_s') != time_s
                    or safety.get('source_reference') != frame.calibration_reference):
                raise ValueError('Stale or mismatched complete-inventory mounted telemetry')
            return frame, safety

        def close(self):
            cfg = getattr(self, 'cfg', None)
            reader = getattr(cfg, 'mounted_contact_reader', None)
            if reader is not None:reader.close()
            super().close()

    return DenseMountedOrchardEnv
