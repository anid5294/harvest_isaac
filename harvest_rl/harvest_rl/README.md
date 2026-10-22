# harvest_rl

A simulation environment for training a humanoid to pick an apple off a tree and put it
in a basket. Built on **Isaac Lab 3.0 / Isaac Sim 6.1 (PhysX)** and many environments run
in parallel. It provides the scene, the robot interface and the full task state. **It
contains no reward, policy or loss**: those are yours.

The tree, apple and basket come from the teammate's
[`harvest_isaac` `feature/isaac-orchard-preview`](https://github.com/anid5294/harvest_isaac/tree/feature/isaac-orchard-preview)
branch (OrchardBench tree, seed 42). That repo is left untouched.

## Run it

Double-click `C:\isaac\start_isaaclab.bat`, then:

```bat
:: self-test (no window): apple hangs, stem snaps at 17.3 N, basket + contacts, reset
isaaclab.bat -p C:\isaac\harvest_rl\scripts\check_env.py

:: look at it (robot holds still, state printed every second)
isaaclab.bat -p C:\isaac\harvest_rl\scripts\explore.py --viz kit

:: watch the stem: a test force ramps up until the apple snaps off
isaaclab.bat -p C:\isaac\harvest_rl\scripts\explore.py --viz kit --pull
```

No window is the default in Isaac Lab 3.0; `--viz kit` opens one.

## Where your work goes

| You want to... | Edit |
| --- | --- |
| Plug in your humanoid | `harvest_rl/robot.py` (USD/URDF path, PD gains, home pose, which joints are actions, hand/finger link names, where it stands) |
| Define the reward | `OrchardEnv._get_rewards` in `harvest_rl/orchard_env.py` (returns zeros now) |
| Change the policy inputs | `OrchardEnv._get_observations` (and its size in `_configure_gym_env_spaces`) |
| Change timing, stem physics, action mode, terminations | `harvest_rl/orchard_env_cfg.py` (`OrchardEnvCfg`) |

### Using it from your own training loop

```python
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)   # adds --viz etc.
app = AppLauncher(parser.parse_args()).app   # start the simulator before other imports

import gymnasium as gym
import harvest_rl
from harvest_rl.orchard_env_cfg import OrchardEnvCfg

cfg = OrchardEnvCfg()
cfg.scene.num_envs = 256
env = gym.make(harvest_rl.TASK_ID, cfg=cfg)

obs, info = env.reset()
while True:
    action = my_policy(obs["policy"])                 # (num_envs, 31), values in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    state = env.unwrapped.get_state()                 # dict of batched tensors, see below
```

Everything is a batched torch tensor on the GPU. Environments that finish are reset
automatically inside `step()` (Isaac Lab convention). It is a standard Isaac Lab
`DirectRLEnv`, so Isaac Lab's rsl_rl / skrl / rl_games / SB3 wrappers should accept it if
you want one of those libraries (not tested here).

## Interface

**Action** (31-D for the placeholder robot): joint position targets for
`robot.controlled_joints` (waist, both arms, both Dex3 hands). `action_mode`:

- `"absolute"` (default): [-1, 1] maps onto each joint's soft limits. Same convention as
  harvest_isaac's 43-D contract. Use `env.unwrapped.home_action()` to hold the home pose.
- `"offset"`: home pose + `action_scale * action` [rad].
- `"delta"`: previous target + `action_scale * action` [rad per step].

Physics runs at 120 Hz, with the policy acting at 30 Hz (decimation 4). Episodes last 20 s.

**State**: `env.unwrapped.get_state()` returns a dict. Positions are in each env's local
frame (the tree's frame, metres, Z up). Quaternions are **(x, y, z, w)**.

| Key | Meaning |
| --- | --- |
| `joint_pos`, `joint_vel` | all robot joints (names: `env.unwrapped.joint_names`) |
| `ctrl_joint_pos/vel/target` | the controlled joints (`ctrl_joint_names`) |
| `root_pos`, `root_quat` | robot base |
| `body_pos/quat/lin_vel/ang_vel` | tracked links, i.e. both palms (`tracked_body_names`) |
| `apple_pos/quat/lin_vel/ang_vel` | the apple |
| `apple_attached`, `detach_step` | stem still intact; step it snapped (-1 if not) |
| `stem_tension`, `stem_twist` | current stem force [N], apple rotation [rad] |
| `finger_contact_force` | apple contact force per finger link [N] (`contact_body_names`) |
| `basket_contact_force` | apple pressing on the basket floor [N] |
| `hand_distance`, `opposing_contact`, `grasp_contact`, `hand_open` | grasp progress (grasping hand) |
| `carried`, `released`, `basket_supported`, `apple_settled_in_basket`, `placed_steps` | harvest stages |
| `success`, `failure` | outcome; failure names in `env.unwrapped.FAILURE_NAMES` |

**Constants** are properties on `env.unwrapped`, not in the state: `joint_lower`,
`joint_upper`, `joint_home` (controlled joints), `basket_pos`, `stem_anchor`,
`stem_break_force`.

**Default observation** (106-D): normalized controlled-joint positions, their velocities,
the last action, palm-to-apple vectors, apple position relative to the basket, apple
velocity, and the attached flag.

**Episode ends** on success (apple resting in the basket for 15 steps), on the apple
hitting the ground, or on timeout. Each can be switched off in the config.

## The stem

The apple is the **YCB `013_apple`** scan at real size (7.5 × 7.5 × 7.2 cm, textured,
convex-hull collider, 160 g from tree.json). It hangs from fruit 28's stem anchor in tree.json,
so its centre sits 6 mm lower than tree.json's 3 cm sphere; it still clears the nearest branch
by 4.9 mm. The 39 decorative apples on the tree are the same YCB mesh (visual only).

The apple is a free rigid body. While attached, a spring-damper holds it
on the branch every physics step: stiff along the stem (1000 N/m), softer sideways
(300 N/m) and weak in twist. It **snaps when the elastic force exceeds 17.3 N** (the
`break_force_n` in tree.json), or when the apple has been rotated more than 90° (twist-off
picking; set `stem_twist_break_angle = None` to disable). After that the apple falls
normally. The robot has to actually pull or twist it off; nothing is teleported.
`stem_break_force_range` randomizes the threshold per episode.

harvest_isaac used a native PhysX breakable joint instead. That joint can only be rebuilt by
deleting and recreating it on the USD stage, which works with one environment but not with
hundreds in parallel. That is why this environment uses the spring model.

`env.unwrapped.apple_external_force` (num_envs × 3, N) adds an external force on the
apple every physics step, for tests, wind or perturbations.

## Layout

```
harvest_rl/
  robot.py              # robot plug-in (placeholder: G1 29-DoF + Dex3, fixed pelvis)
  orchard_env_cfg.py    # scene, timing, stem, actions, terminations
  orchard_env.py        # the environment: stem model, state, obs, (empty) reward, resets
  tree.py               # reads tree.json (stem anchor, mass, break force) + YCB apple size
  task_logic.py         # harvest_isaac's success/failure rules (batched)
  assets/
    tree_orchardbench_seed42/   # tree.json; isaac/tree.usda (her original, sphere apples)
                                #            isaac/tree_ycb.usda (used: YCB visual apples)
    ycb_013_apple/              # YCB apple: apple_visual.usda, apple_physics.usda, texture
    robot_g1_placeholder/       # G1 USD from harvest_isaac
scripts/
  check_env.py          # headless self-test
  explore.py            # viewer / state printout, --pull stem demo
  build_ycb_apple.py    # rebuilds the YCB apple USDs and tree_ycb.usda from the YCB download
tests/
  test_task_logic.py    # task_logic vs harvest_isaac's original code
```

## Known limitations

- **Robot stance and reach are not validated.** The robot stands where harvest_isaac
  placed it (0.43 m from the apple). In the home pose the left palm is about 0.35 m from
  the apple. Retune `stance_offset` for your robot.
- **One target apple** (fruit 28). It is the only fruit that `tree.usda` leaves out and the
  only one clear of branches. The other 39 apples are visual only; at YCB size, 15 of them
  poke into branches (visual only, so it has no physical effect).
- **The tree and basket are fixed.** There is no randomization yet besides the stem break
  force and `reset_joint_noise`.
- **Leaves are visual only**; only the wood collides.
