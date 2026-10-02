# Isolated stem diagnosis and CPU manipulation baseline

These additions do not change production stem mechanics, tree geometry, shared
Isaac Lab, drivers, G1 assets, or the normalized 43-D interface. No simulator
results or repeatable manipulation success are claimed from local static tests.

## Activate once on Songkhla

```bash
cd /home/vlakbnn/anikad/harvest_isaac
source /home/vlakbnn/miniconda3/etc/profile.d/conda.sh
conda activate /home/vlakbnn/anikad/.conda/envs/harvest-isaac
export VLA_ISAACLAB_ENV="$CONDA_PREFIX"
export ISAACLAB_ROOT=/media/data-ssd/software/IsaacLab-v2.0.2
source scripts/activate.sh
check_install_environment
export STEM_RUN="$PWD/outputs/stem_diagnostic_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$STEM_RUN"
printf '%s\n' "$STEM_RUN"
```

Use a fresh process for each variant. Commands below capture combined logs;
inspect the **first** CUDA/PhysX error, not only the JSON or process exit.
The minimal scene has no ground: a released apple falls freely during the
measurement window rather than landing at the production floor height.
All step counts in this diagnostic are physics substeps at 120 Hz, not the
production runner's 30 Hz control steps.

## Minimal experiment matrix

1. CPU baseline, late-created native joint:

```bash
python scripts/debug/minimal_stem_physx.py --headless --device cpu \
  --joint-lifecycle late-create --break-mode breakable --collisions on \
  --force-n 34.682148892074274 --passive-steps 120 --loaded-steps 120 \
  --post-break-steps 240 --report "$STEM_RUN/01_cpu.json" \
  > "$STEM_RUN/01_cpu.log" 2>&1
printf 'CPU process exit: %s\n' "$?"
cat "$STEM_RUN/01_cpu.json"
```

2. Same variant on GPU:

```bash
CUDA_LAUNCH_BLOCKING=1 python scripts/debug/minimal_stem_physx.py --headless --device cuda:0 \
  --joint-lifecycle late-create --break-mode breakable --collisions on \
  --force-n 34.682148892074274 --passive-steps 120 --loaded-steps 120 \
  --post-break-steps 240 --report "$STEM_RUN/02_gpu.json" \
  > "$STEM_RUN/02_gpu.log" 2>&1
printf 'GPU process exit: %s\n' "$?"
cat "$STEM_RUN/02_gpu.json"
```

Only if CPU is healthy and GPU fails, continue:

3. GPU, both break thresholds effectively infinite:

```bash
CUDA_LAUNCH_BLOCKING=1 python scripts/debug/minimal_stem_physx.py --headless --device cuda:0 \
  --joint-lifecycle late-create --break-mode unbreakable --collisions on \
  --force-n 34.682148892074274 --report "$STEM_RUN/03_unbreakable.json" \
  > "$STEM_RUN/03_unbreakable.log" 2>&1
printf 'Unbreakable process exit: %s\n' "$?"
cat "$STEM_RUN/03_unbreakable.json"
```

4. GPU, joint created before simulation initialization:

```bash
CUDA_LAUNCH_BLOCKING=1 python scripts/debug/minimal_stem_physx.py --headless --device cuda:0 \
  --joint-lifecycle precreate --break-mode breakable --collisions on \
  --force-n 34.682148892074274 --report "$STEM_RUN/04_precreate.json" \
  > "$STEM_RUN/04_precreate.log" 2>&1
printf 'Precreate process exit: %s\n' "$?"
cat "$STEM_RUN/04_precreate.json"
```

5. Only if still needed, same failing late-created variant without either
body's collision geometry enabled:

```bash
CUDA_LAUNCH_BLOCKING=1 python scripts/debug/minimal_stem_physx.py --headless --device cuda:0 \
  --joint-lifecycle late-create --break-mode breakable --collisions off \
  --force-n 34.682148892074274 --report "$STEM_RUN/05_collision_off.json" \
  > "$STEM_RUN/05_collision_off.log" 2>&1
printf 'Collision-off process exit: %s\n' "$?"
cat "$STEM_RUN/05_collision_off.json"
```

Inspect each complete log and use this focused search to find candidate faults:

```bash
rg -n -i 'fail to launch|cuda.*error|cuda.*fail|illegal memory|device.*assert|fatal|traceback' "$STEM_RUN"/*.log
```

`rg` exit 1 means no matching lines, not proof that the backend is healthy.
Missing JSON, nonzero process exit, nonfinite state, missing expected break/fall,
or native physics errors all require investigation. Do not count frozen finite
GPU state as healthy merely because an unbreakable joint did not move.

| Outcome | Next decision |
|---|---|
| CPU baseline fails | Stop matrix; inspect harness/scene setup before blaming GPU. |
| CPU and GPU minimal baseline pass | Full-task managers, sensors, other collision pairs, or stepping differences remain; do not call GPU fixed. |
| GPU breakable fails, unbreakable stays healthy with clean logs | Break transition is implicated, not proven as the only cause. |
| Precreate passes where late-create fails | Joint-creation lifecycle is implicated; validate a minimal scene-construction patch before production changes. |
| Only collision-off passes | Investigate stem–apple collision/pair handling; tree collider tuning remains irrelevant. |
| All GPU variants fail under load | Investigate wrench/backend path next; do not infer threshold-specific failure. |

No force ladder is requested yet. The script supports `force-only` and
`torque-only` for later discrimination. The production joint's 0.35 Nm limit
and 0.03 m moment arm mean its default horizontal load tests both thresholds.
Break timestamps are callback-observation substeps; delayed callback delivery
cannot reveal the exact internal solver instant.

## Separate full-collider CPU production reference

This uses the literal checked-in asset and existing probe, not the minimal test:

```bash
python scripts/check_orchard_stem.py --headless --device cpu \
  --orchardbench-asset "$PWD/assets/orchard/orchardbench_seed42/isaac" \
  --seed 42 --report "$STEM_RUN/production_cpu_full.json" \
  > "$STEM_RUN/production_cpu_full.log" 2>&1
printf 'Production stem process exit: %s\n' "$?"
python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r,indent=2)); ok=r.get("passed") is True and r.get("passive_stable") is True and 0 in r.get("broken_indices",[]) and r.get("fall_after_force_clear_m",0)>=0.05 and r.get("robot_grasp_claim") is False and r.get("harvest_success_claim") is False; sys.exit(0 if ok else 1)' "$STEM_RUN/production_cpu_full.json"
```

Require process exit 0 **and** the checker exit 0, native break and physical fall.
This is direct external loading, not a grasp or training demonstration.

## CPU free-apple demo

Separate Gym ID: `VLA-FreeApplePickPlace-G1-JointPos-v0`. One physical apple
rests on a narrow support. There is no tree or stem joint. The left arm and
Dex3 hand manipulate it; legs/right arm stay at their configured targets.
Control uses a contact/state-driven FSM, conservative pose interpolation and
the existing bounded damped least-squares IK, then the unchanged normalized
43-D action mapping. Grasp geometry remains an **unvalidated calibration** until
the videos and measured contacts establish otherwise.

State sequence:

```text
SETTLE -> PREGRASP -> APPROACH -> CLOSE -> VERIFY_GRASP
       -> LIFT -> TRANSPORT -> LOWER -> RELEASE -> RETREAT -> VERIFY_PLACEMENT -> DONE
Any timeout, nonfinite state or unsafe contact loss -> FAIL
```

Physics-only first:

```bash
export DEMO_RUN="$PWD/outputs/free_apple_demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEMO_RUN"
./scripts/run_env.sh --headless --physics-only --device cpu \
  --task VLA-FreeApplePickPlace-G1-JointPos-v0 --policy pick-place \
  --seed 42 --steps 2700 --report-dir "$DEMO_RUN/physics"
printf 'Free demo runner exit: %s\n' "$?"
python scripts/check_free_pick_place_run.py "$DEMO_RUN/physics/validation.json"
```

Then cameras/MP4s, only after reviewing the physics result:

```bash
./scripts/run_env.sh --headless --device cpu \
  --task VLA-FreeApplePickPlace-G1-JointPos-v0 --policy pick-place \
  --seed 42 --steps 2700 --report-dir "$DEMO_RUN/render" \
  --camera-videos "$DEMO_RUN/videos"
printf 'Rendered demo runner exit: %s\n' "$?"
python scripts/check_free_pick_place_run.py "$DEMO_RUN/render/validation.json" \
  --videos "$DEMO_RUN/videos"
```

Expected outputs: `validation.json`, per-control-step `trajectory.jsonl`, and
`external.mp4`, `left_wrist.mp4`, `right_wrist.mp4`. On a failure inspect the first
failed state, pose error, measured finger forces and object motion; do not repeat
an unchanged failed run three times. Runner/checker 0 requires actual named
success and evidence for grasp, lift, carry, release and basket support—not
simply reaching a controller phase or waypoint.

Only after repeated physical success, optional contract-aligned recording:

```bash
./scripts/run_env.sh --headless --device cpu \
  --task VLA-FreeApplePickPlace-G1-JointPos-v0 --policy pick-place \
  --seed 42 --steps 2700 --record-format lerobot --episodes 1 \
  --dataset-name free_apple_cpu_verified --report-dir "$DEMO_RUN/recording"
```

Do not use `--include-failed-episodes` for imitation-training data. Free-object
success is not a harvesting success. The next attached-apple step is to reuse
the validated grasp, insert load/wait-for-native-break states before lift, and
require contact preceding the CPU joint-break event; no manual detachment or
grip-assist force has been introduced here.
