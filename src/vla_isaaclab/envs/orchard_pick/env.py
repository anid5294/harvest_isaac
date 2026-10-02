"""Single-attempt environment preserving terminal fruit poses and commands."""

from isaaclab.envs import ManagerBasedRLEnv


class OrchardPickEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("Orchard pick requires one environment")
        self._inside_step = False
        self._finished = False
        super().__init__(cfg, **kwargs)
        if getattr(cfg, "cpu_harvest_diagnostics", False):
            from .diagnostics import install
            install(self)

    def orchard_physics_diagnostics(self):
        diagnostics = getattr(self, "_orchard_physics_diagnostics", None)
        return diagnostics.snapshot() if diagnostics is not None else None

    def _reset_idx(self, env_ids):
        # ManagerBasedRLEnv ordinarily resets before returning the terminal step.
        # Preserve this attempt for videos, diagnostics, and final commanded targets.
        if not self._inside_step:
            self._finished = False
            super()._reset_idx(env_ids)

    def step(self, action):
        if self._finished:
            raise RuntimeError("Episode finished; explicitly reset before another attempt")
        self._inside_step = True
        try:
            result = super().step(action)
            self._finished = bool(result[2][0]) or bool(result[3][0])
            return result
        finally:
            self._inside_step = False

    def close(self):
        self._orchard_contact_report_subscription = None
        self._orchard_physics_step_subscription = None
        self.orchard_break_subscription = None
        super().close()
