# AGENTS.md

Read this file before editing or running the simulator.

## Environment

- Activate a developer-owned Conda environment first, or set `VLA_ISAACLAB_ENV`.
- Shared Isaac Lab: `${ISAACLAB_ROOT:-/media/data-ssd/software/IsaacLab-v2.0.2}`.
- Required Isaac Lab tag: `v2.0.2`; Isaac Sim: 4.5.0.0.
- Do not clone, checkout, pull, or modify the shared Isaac Lab dependency.
- Do not use sudo, alter drivers/system CUDA, or touch another user's files.

Before simulator work:

```bash
conda activate <developer-env>
source scripts/activate.sh
check_install_environment
```

## Safety

- Preserve unrelated work and generated outputs.
- Do not reset, clean, stash, or commit unless explicitly asked.
- Never teleport, parent, kinematically move, or invisibly attach an object.
- Never claim task success unless its named success termination fires.
- Never save or label a failed manipulation as a successful demonstration.
- If the same task reproduction fails three times, stop and request human inspection.

## Architecture

Complete environments are registered Gym IDs. There is no independent
World/Object/Task/Expert/Controller registry and no custom joint ActionTerm.

```text
Gym ID -> EnvCfg -> Scene + Isaac Lab Managers -> normalized action -> robot
                     ↑
               optional scripted policy
```

- `envs/common/`: reusable G1 and physical scene/config helpers.
- `envs/<task>/env_cfg.py`: concrete robot, object, camera, action and managers.
- `envs/<task>/mdp/`: command, observation, reward, event and termination terms.
- `policies/`: optional scripted strategy and action generation.
- `recording/`: HDF5 staging and contract-aligned LeRobot v3/v2.1 export; v3 is default.

Use Isaac Lab `JointPositionToLimitsActionCfg` for the 43-D normalized action.
Its mapping is `-1=soft lower limit`, `0=midpoint`, `+1=soft upper limit`.
Policies that calculate physical joint targets must invert that exact mapping.
The action term follows the USD's internal joint order because Isaac Lab v2.0.2
does not expose `preserve_order` on this action config. Dataset recording must
explicitly remap state and processed targets into `g1_29body_dex3_43d_v1` order.
The only retained custom control algorithm is bounded DLS IK inside the scripted
sugar-box policy.

Object and task-specific configs stay in their concrete environment. Do not
recreate top-level `robots/`, `objects/`, `worlds/`, `sensors/`, `controllers/`,
or `experts/` component layers.

Camera placement is owned by the concrete scene EnvCfg. Shared intrinsics and
modalities are owned by `envs/common/scene.py`.

## Reference behavior

- Preview IDs: YCB, dinnerware, microwave; 240 control steps at 30 Hz.
- Sugar-box ID: `VLA-YCBSugarBox-G1-JointPos-v0`.
- Sugar-box table contains only `004_sugar_box`.
- Validated success occurs at step 961 using physical contact and three fingers.
- Preserve the calibrated phase thresholds, pose, grasp, and success criteria
  unless the user explicitly requests behavioral changes.

Useful commands:

```bash
./scripts/run_env.sh --headless --physics-only --list-tasks
./scripts/run_env.sh --headless --physics-only \
  --task VLA-ScenePreview-YCB-G1-v0 --steps 240
./scripts/run_env.sh --headless --physics-only \
  --task VLA-YCBSugarBox-G1-JointPos-v0 --steps 1200
```
