"""Base environment configuration shared by previews and manipulation tasks."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .managers import EventsCfg


@configclass
class VLAEnvCfg(ManagerBasedRLEnvCfg):
    """Common simulation timing and material settings."""

    decimation: int = 4
    episode_length_s: float = 10.0
    task_instruction: str = ""
    camera_eye: tuple[float, float, float] = (0.35, 1.90, 1.85)
    camera_target: tuple[float, float, float] = (-0.15, -0.30, 0.72)
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 0.8
        self.sim.physics_material.dynamic_friction = 0.65
        self.sim.physics_material.restitution = 0.02
        self.sim.physics_material.friction_combine_mode = "max"
        self.rerender_on_reset = True
        self.wait_for_textures = True
        self.viewer.eye = self.camera_eye
        self.viewer.lookat = self.camera_target
