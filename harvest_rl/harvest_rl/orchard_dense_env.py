"""Opt-in continuous full-fruit orchard environment, lazy SDK construction.

One physical body/contact sensor/stem/tracker per source fruit, including
unreachable and unresolved fruit. Target selection is a read-view switch, never
an environment reset. Mounted retained-ID verification remains authoritative
for deposits; the legacy tracker supplies unchanged acquisition/carry evidence.
"""
from __future__ import annotations

from dataclasses import replace

# Stem functions rebind some tensors, hence save after EVERY fruit operation.
FRUIT_BUFFERS = ('_apple_center', '_stem_anchor', '_stem_axis', '_stem_rest',
    '_c_ax', '_c_lat', '_c_rot', 'apple_attached', 'stem_break_force',
    'stem_tension', 'stem_twist', 'detach_step', 'break_tension',
    'break_threshold', 'break_external_force', 'apple_external_force', 'harvest')


class DenseInventoryMixin:
    """Small view/dispatch seam, independently testable without Isaac or torch."""
    def _save_dense_view(self):
        record = self._dense_records[self.active_target_id]
        for name in FRUIT_BUFFERS:
            record['buffers'][name] = getattr(self, name)

    def _bind_dense_view(self, target_id):
        target_id = str(target_id)
        record = self._dense_records[target_id]
        self.active_target_id = target_id
        self._apple, self._apple_contact = record['object'], record['contact']
        self.cfg.fruit = record['fruit']
        self.cfg.stem_break_force = record['fruit'].stem_break_force
        for name, value in record['buffers'].items():
            setattr(self, name, value)

    def select_target(self, fruit_id):
        """Keep measured joints, targets, action history, clock and all fruit intact."""
        target_id = str(fruit_id)
        if target_id not in self._dense_records:
            raise KeyError('Target not present in physical inventory: ' + target_id)
        if getattr(self, '_dense_in_interval', False):
            raise RuntimeError('Do not switch targets during a physical interval')
        self._save_dense_view()
        self._bind_dense_view(target_id)

    def _for_each_dense_fruit(self, operation):
        active = self.active_target_id
        self._save_dense_view()
        try:
            for target_id in self._dense_records:
                self._bind_dense_view(target_id)
                try:
                    operation()
                finally:
                    self._save_dense_view()
        finally:
            self._bind_dense_view(active)

    def dense_fruit_inventory(self):
        self._save_dense_view()
        return tuple(dict(target_id=k, fruit_id=r['fruit'].id,
                          body_id=r['body_id'], object=r['object'],
                          half_extents_m=tuple(r['fruit'].half_extents), mass_kg=r['fruit'].mass,
                          attached=bool(r['buffers']['apple_attached'][0]))
                     for k, r in self._dense_records.items())

    def fruit_state(self, fruit_id):
        active = self.active_target_id
        self._save_dense_view()
        try:
            self._bind_dense_view(str(fruit_id))
            return self.get_state()
        finally:
            self._bind_dense_view(active)

    def inventory_states(self):
        return {target_id: self.fruit_state(target_id) for target_id in self._dense_records}

    def begin_dense_episode(self):
        if self._dense_started:
            raise RuntimeError('Dense continuous episode already started')
        self._dense_started = True


def make_dense_orchard_env_class(base_class=None):
    """Only call after AppLauncher; no registry/default/controller changes."""
    import torch
    from isaaclab.envs import DirectRLEnv
    from .compat import as_torch, asset_call, backend_quat_xyzw, default_root_pose
    from .orchard_env import OrchardEnv
    from .pluck.env import PluckEnv
    from .tree import Fruit
    if base_class is None:
        base_class = PluckEnv

    class DenseOrchardEnv(DenseInventoryMixin, base_class):
        def __init__(self, cfg, *args, **kwargs):
            if not getattr(cfg, 'dense_inventory', None) or cfg.scene.num_envs != 1:
                raise ValueError('Explicit full dense inventory required')
            # Correct each visual/collidable stem cylinder orientation before scene spawn.
            for index,row in enumerate(cfg.dense_inventory):
                fid=row['fruit_id']; dx,dy,dz=cfg.dense_stem_directions[str(fid)]
                quat=(1.,0.,0.,0.) if dz < -1.+1e-10 else (-dy,dx,0.,1.+dz)
                norm=sum(v*v for v in quat)**.5
                stem=getattr(cfg.scene,'stem' if index==0 else f'dense_stem_{fid}')
                stem.init_state.rot=backend_quat_xyzw(tuple(v/norm for v in quat))
            self._dense_started = self._dense_in_interval = False
            self.dense_timeout_reached = False
            self._dense_reset_count = 0
            super().__init__(cfg, *args, **kwargs)

        def _setup_scene(self):
            OrchardEnv._setup_scene(self)
            self._dense_assets = {}
            for index, row in enumerate(self.cfg.dense_inventory):
                fid = row['fruit_id']
                key = str(fid)
                self._dense_assets[key] = dict(
                    object=self.scene['apple' if index == 0 else f'dense_apple_{fid}'],
                    contact=self.scene['apple_contact' if index == 0 else f'dense_contact_{fid}'])
            callback = self.cfg.dense_setup_callback
            if callback is not None:
                callback(self)

        def _init_buffers(self):
            self._dense_records = {}
            for row in self.cfg.dense_inventory:
                key = str(row['fruit_id'])
                f = Fruit(**{k: tuple(v) if isinstance(v, list) else v for k,v in row['fruit'].items()})
                self.cfg.fruit, self.cfg.stem_break_force = f, f.stem_break_force
                self._apple = self._dense_assets[key]['object']
                self._apple_contact = self._dense_assets[key]['contact']
                OrchardEnv._init_buffers(self)
                self._dense_records[key] = dict(**self._dense_assets[key], fruit=f,
                    body_id=f'/World/envs/env_0/Apple_{f.id}',
                    buffers={name: getattr(self, name) for name in FRUIT_BUFFERS})
            self._bind_dense_view(self.cfg.dense_initial_target_id)
            self._hand_ids = self._grasp_hand_joint_ids
            self.counts = {}

        @property
        def basket_pos(self):
            reader = self.cfg.dense_basket_position_reader
            return reader(self) if reader is not None else self._basket_pos

        def _apply_stem(self):
            self._for_each_dense_fruit(lambda: OrchardEnv._apply_stem(self))

        def _update_task(self):
            # Preserve native acquisition criteria independently for every fruit.
            # Mounted floor/wall/support identity is checked by the existing
            # independent retained-ID verifier, not this singleton basket column.
            self._basket_pos = self.basket_pos
            def update():
                b = self._basket_pos
                self.harvest.p = replace(self.harvest.p,
                    basket_center=(b[:,0],b[:,1],b[:,2]))
                OrchardEnv._update_task(self)
            self._for_each_dense_fruit(update)

        def _get_dones(self):
            self._update_task()
            self.dense_timeout_reached = bool(self.episode_length_buf[0] >= self.max_episode_length-1)
            zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return zeros, zeros.clone()  # No DirectRLEnv autoreset between picks.

        def _get_rewards(self):
            # Inference integration only, not a new training reward/termination.
            return torch.zeros(self.num_envs, device=self.device)

        def step(self, actions):
            if not self._dense_started:
                raise RuntimeError('Call begin_dense_episode before physical actions')
            if self.dense_timeout_reached:
                raise RuntimeError('No steps after dense episode timeout')
            self._dense_in_interval = True
            try:
                if self.cfg.dense_before_step is not None:
                    self.cfg.dense_before_step(self)
                result = super().step(actions)
                if self.cfg.dense_after_step is not None:
                    self.cfg.dense_after_step(self)
                return result
            finally:
                self._dense_in_interval = False

        def _reset_idx(self, env_ids):
            if self._dense_started:
                raise RuntimeError('World reset/fruit respawn forbidden during dense harvest')
            if self._dense_reset_count:
                raise RuntimeError('One initial dense reset only; start a new reviewed episode explicitly')
            self._dense_reset_count += 1
            # Initial-only robot reset + every physical fruit, original attachment
            # and tracker. No RandomApple relocation or basket-following writer.
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            DirectRLEnv._reset_idx(self, env_ids)
            origins = self.scene.env_origins[env_ids]
            q = as_torch(self._robot.data.default_joint_pos)[env_ids].clone()
            initial = getattr(self.cfg,'dense_initial_joint_positions',None)
            if initial is not None:
                from .dense_stance import initial_reset_positions
                q = initial_reset_positions(q,list(self._robot.joint_names),initial)
            pose = default_root_pose(self._robot.data)[env_ids].clone()
            pose[:,:3] += origins
            asset_call(self._robot,'write_root_pose_to_sim',root_pose=pose,env_ids=env_ids)
            asset_call(self._robot,'write_root_velocity_to_sim',root_velocity=torch.zeros(len(env_ids),6,device=self.device),env_ids=env_ids)
            asset_call(self._robot,'write_joint_position_to_sim',position=q,env_ids=env_ids)
            asset_call(self._robot,'write_joint_velocity_to_sim',velocity=torch.zeros_like(q),env_ids=env_ids)
            asset_call(self._robot,'set_joint_position_target',target=q,env_ids=env_ids)
            self._joint_targets[env_ids] = q[:,self.ctrl_joint_ids]
            def reset_fruit():
                apple_pose = torch.cat((self._apple_center[env_ids]+origins,self._identity_quat[env_ids]),dim=-1)
                asset_call(self._apple,'write_root_pose_to_sim',root_pose=apple_pose,env_ids=env_ids)
                asset_call(self._apple,'write_root_velocity_to_sim',root_velocity=torch.zeros(len(env_ids),6,device=self.device),env_ids=env_ids)
                self.apple_attached[env_ids] = True
                for name in ('stem_tension','stem_twist','break_tension','break_threshold','break_external_force','apple_external_force'):
                    getattr(self,name)[env_ids] = 0.
                self.detach_step[env_ids] = -1
                self.harvest.reset(env_ids)
            self._for_each_dense_fruit(reset_fruit)
            self.actions[env_ids] = self._prev_actions[env_ids] = 0.
    return DenseOrchardEnv
