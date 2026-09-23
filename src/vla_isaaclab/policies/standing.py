"""Policy that holds the G1 at its configured default upper-body pose."""

from .joint_limits import joint_targets_to_normalized, resolved_action_joints


class StandingPolicy:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        _, self.joint_names, self.joint_ids = resolved_action_joints(env)

    def reset(self):
        pass

    def compute(self, step: int = 0):
        del step
        targets = self.robot.data.default_joint_pos[:, self.joint_ids]
        return joint_targets_to_normalized(self.robot, self.joint_ids, targets)
