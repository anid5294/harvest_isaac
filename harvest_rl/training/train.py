import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app          # must start the simulator before the imports below

import gymnasium as gym
import torch
import harvest_rl
from harvest_rl.orchard_env_cfg import OrchardRLEnvCfg

cfg = OrchardRLEnvCfg()
cfg.scene.num_envs = args.num_envs
env = gym.make(harvest_rl.RL_TASK_ID, cfg=cfg)
u = env.unwrapped                    # the OrchardEnv itself

obs, info = env.reset()
while app.is_running():
    s = u.get_state()                # dict of GPU tensors, shape (num_envs, ...)
    print("state: ", s)
    print(f"")
    # action = u.home_action()         # (num_envs, 43) in [-1, 1]  <- replace with your policy(obs["policy"])
    # print("action: ", action)
    action =  torch.zeros(1, 43, device="cuda") # action shape is [1, 43]
    print(action)
    right_arm_joint_values = action[:, 22:29]
    right_hand_joint_values = action[:, 36:43]
    
    current_operating_joint_state = torch.cat((right_arm_joint_values, right_hand_joint_values), axis=1) 
    print(type(current_operating_joint_state))
    print(current_operating_joint_state.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    

env.close()
app.close()