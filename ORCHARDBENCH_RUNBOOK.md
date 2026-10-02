# OrchardBench single-tree handoff

## What is and is not complete

Implemented: pinned offline generator, canonical topology/local frames, checked-in
seed-42 asset, separate render/collision USD, actual G1 scale renders, one-fruit
Isaac integration, three-camera profile, passive checker and independent stem
load test. The decision and candidate audit are in `ORCHARDBENCH_DECISION.md` and
`ORCHARD_SOURCE_AUDIT.md`; exact provenance is under
`assets/orchard/orchardbench_seed42/SOURCE.md`.

**Not complete:** Isaac rendering/contacts on Songkhla have not been executed
from this workspace. No successful G1 grasp/detach/carry/place has been measured
and no harvest video exists. The default policy deliberately holds the robot
still. The old scripted orchard controller is not calibrated for this tree;
do not run it and interpret its exit status as a working new policy. No RL/MPC
harvesting policy is supplied. Gates 3–5 remain open. Five-seed render review,
rows and locomotion are deferred until the required earlier gates pass.

## 1. Local tests and exact Git handoff

Recorded local evidence: all four Blender views were generated and visually
inspected; the real G1 USD imported as 274 objects at scale 1. The final review
is in `outputs/orchardbench/seed42/review_final/`. Skeleton dimensions are
2.5368 × 2.7822 × 2.4463 m; visible USD surface bounds span approximately
2.6293 × 2.7912 × 2.5282 m. USD traversal confirmed exactly 168 static wood
colliders and no duplicate visual target fruit. Targeted offline checks: 43
passed, one optional source-regeneration test skipped by default; that test
also passed when supplied the pinned source and NumPy interpreter. Full test
discovery on system Python could not import the unrelated bounded-IK test
because Torch is not installed locally. No Isaac physics checks were run.

Run in the Mac terminal. These commands do not stage the unrelated pre-existing
README, ROADMAP, Arena script or older planning changes. `orchard.sensor.kit` and
`render_smoke.py` were already untracked but are required by the current runner;
include them so a fresh lab checkout can run the documented camera test.

```bash
cd /Users/anikadixit/Desktop/ctrl_learning/harvest_isaac
git branch --show-current
git status --short
python3 -m unittest discover -s tests -p 'test_orchard*.py' -v
python3 -m unittest discover -s tests -p test_external_tree.py -v
git diff --check

git add ORCHARDBENCH_DECISION.md ORCHARD_SOURCE_AUDIT.md ORCHARDBENCH_RUNBOOK.md
git add assets/orchard/orchardbench_seed42
git add scripts/export_orchardbench_tree.py scripts/build_orchardbench_usd.py
git add scripts/render_tree_review.py scripts/check_orchard_stem.py
git add scripts/run_env.py scripts/check_orchard_run.py scripts/preview_views.py
git add src/vla_isaaclab/envs/orchard_pick/external_tree.py
git add src/vla_isaaclab/envs/orchard_pick/env_cfg.py src/vla_isaaclab/envs/orchard_pick/mdp.py
git add src/vla_isaaclab/recording/frame.py
git add tests/test_external_tree.py tests/test_orchardbench_export.py
git add configs/orchard.sensor.kit scripts/render_smoke.py
git diff --cached --check
git diff --cached --stat
```

Review the staged list, especially anything you had staged before this task.
Then commit/push on the existing `feature/isaac-orchard-preview` branch:

```bash
git commit -m "Add pinned OrchardBench single-tree asset and Isaac physical-fruit review"
git push origin feature/isaac-orchard-preview
git rev-parse HEAD
```

No generated MP4s, `.blend` files, temporary source checkout or Conda environment
need to be committed. The canonical JSON, Apache license, USD and manifest are
small checked-in assets; the lab does **not** need Newton, Warp or Blender to
use them. The commands above are instructions, not commits already made here.

## 2. Update and activate on Songkhla

Use the existing installation. Do not delete/recreate it or change shared Lab.
You may start from `(base)` or another prompt: explicit activation selects the
right environment. Do not paste a standalone `/bin/bash` before the block.

```bash
cd /home/vlakbnn/anikad/harvest_isaac
git status --short
git branch --show-current
git pull --ff-only origin feature/isaac-orchard-preview
git rev-parse HEAD

source /home/vlakbnn/miniconda3/etc/profile.d/conda.sh
conda activate /home/vlakbnn/anikad/.conda/envs/harvest-isaac
export VLA_ISAACLAB_ENV="$CONDA_PREFIX"
export ISAACLAB_ROOT=/media/data-ssd/software/IsaacLab-v2.0.2
source scripts/activate.sh
command -v python
check_install_environment
git -C "$ISAACLAB_ROOT" describe --tags --exact-match

python -m unittest discover -s tests -p 'test_orchard*.py' -v
python -m unittest discover -s tests -p test_external_tree.py -v
./scripts/run_env.sh --headless --physics-only --list-tasks

export ORCHARD_ASSET="$PWD/assets/orchard/orchardbench_seed42/isaac"
export ORCHARD_RUN="$PWD/outputs/orchardbench/lab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ORCHARD_RUN"
printf '%s\n' "$ORCHARD_RUN"
```

If `git status` shows local work or the branch differs, stop before pulling;
preserve/resolve those changes deliberately. Do not reset, clean or overwrite.
The Lab tag must be `v2.0.2`. Compare the printed commit with the Mac commit.

## 3. Passive physics gate — no rendering

```bash
./scripts/run_env.sh --headless --physics-only \
  --task VLA-OrchardPick-G1-JointPos-v0 \
  --orchard-tree-model orchardbench --orchardbench-asset "$ORCHARD_ASSET" \
  --policy standing --seed 42 --steps 120 \
  --report-dir "$ORCHARD_RUN/passive"
printf 'runner exit: %s\n' "$?"

python scripts/check_orchard_run.py "$ORCHARD_RUN/passive/validation.json" --passive
printf 'passive checker exit: %s\n' "$?"
```

Expected: passive checker **0**, no broken joints, target drift <=15 mm, speed
<=0.035 m/s. Runner **2** is expected for a passive orchard run because named
harvest success did not occur. Runner 1/139, missing report or checker 1 are not
passes. Do not use `set -e` across the expected runner 2. Inspect the report and
trajectory; repeat only after a concrete fix, not three identical failed runs.

## 4. Independent physical detachment probe

Run only after passive hold passes:

```bash
python scripts/check_orchard_stem.py --headless \
  --orchardbench-asset "$ORCHARD_ASSET" --seed 42 \
  --report "$ORCHARD_RUN/stem_probe.json"
printf 'stem probe process exit: %s\n' "$?"
python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r,indent=2)); sys.exit(0 if r.get("passed") is True else 1)' "$ORCHARD_RUN/stem_probe.json"
```

Require both process exit 0 and report `passed: true`. The probe holds the fruit
under gravity for 120 steps, applies a direct external load, requires a native
joint break, removes the load and measures downward motion. **This is a stem
mechanism test, not a G1 grasp or harvesting demonstration.** It must never enter
the successful-training dataset. Break force comes from the pinned source;
no strength reduction was used to conceal missing finger contact.

## 5. Sensor rendering, then three videos

```bash
./scripts/run_env.sh --headless --render-smoke \
  --report-dir "$ORCHARD_RUN/render_smoke"
printf 'render smoke exit: %s\n' "$?"
```

Only if that succeeds, render the actual tree and robot:

```bash
./scripts/run_env.sh --headless \
  --task VLA-OrchardPick-G1-JointPos-v0 \
  --orchard-tree-model orchardbench --orchardbench-asset "$ORCHARD_ASSET" \
  --policy standing --seed 42 --steps 240 \
  --report-dir "$ORCHARD_RUN/render" \
  --camera-videos "$ORCHARD_RUN/videos"
printf 'render runner exit: %s\n' "$?"

python scripts/check_orchard_run.py "$ORCHARD_RUN/render/validation.json" \
  --passive --videos "$ORCHARD_RUN/videos"
printf 'scene/video checker exit: %s\n' "$?"
```

Expected files: `external.mp4`, `left_wrist.mp4`, `right_wrist.mp4`; 640×480,
30 fps, 240 frames each. `ffprobe` must already be available for the video
checker. Runner 2 still means no harvest; checker 0 validates only passive
scene stability/video structure. Actually watch the videos: check tree scale,
branch alignment, G1 body/branch clearance, selected apple and all camera views.
The candidate robot stance is not a proven collision-free reach configuration.
If the minimal renderer crashes, stop the orchard render attempt and preserve
its logs. Xvfb is not evidence that Hydra initialization was fixed.

For an interactive lab display, use the same task/model/asset with `--steps 0`
and omit `--headless`; it is still a standing inspection, not a pick controller.

## 6. Copy results to the Mac

Run on the Mac. Enter your existing SSH alias/address and the exact absolute
`ORCHARD_RUN` path printed on Songkhla. No guessed host address is required.

```bash
cd /Users/anikadixit/Desktop/ctrl_learning/harvest_isaac
printf 'Songkhla SSH alias/address: '
read -r ORCHARD_HOST
printf 'Absolute Songkhla ORCHARD_RUN path: '
read -r ORCHARD_REMOTE_RUN
mkdir -p outputs/songkhla_orchardbench
scp -r "${ORCHARD_HOST}:${ORCHARD_REMOTE_RUN}/." outputs/songkhla_orchardbench/
open outputs/songkhla_orchardbench/videos/external.mp4
open outputs/songkhla_orchardbench/videos/left_wrist.mp4
open outputs/songkhla_orchardbench/videos/right_wrist.mp4
```

## 7. Optional exact source regeneration and local geometry views

Not required for the lab tests: the prepared asset is already committed.
Use a Python with NumPy and `pxr`; on this Mac Blender's bundled Python supplies
both without altering the Isaac installation.

```bash
cd /Users/anikadixit/Desktop/ctrl_learning/harvest_isaac
export ORCHARDBENCH_SOURCE="$(mktemp -d /tmp/orchardbench-source.XXXXXX)"
git clone https://github.com/humphreymunn/orchardbench.git "$ORCHARDBENCH_SOURCE"
git -C "$ORCHARDBENCH_SOURCE" checkout --detach 6313313db8b1a7d23fb2cc3afd67cac46f29399a
export ORCHARDBENCH_PYTHON=/Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13
"$ORCHARDBENCH_PYTHON" scripts/export_orchardbench_tree.py \
  --source "$ORCHARDBENCH_SOURCE" --seed 42 \
  --output outputs/orchardbench/reproduced/tree.json
"$ORCHARDBENCH_PYTHON" scripts/build_orchardbench_usd.py \
  --tree outputs/orchardbench/reproduced/tree.json \
  --output outputs/orchardbench/reproduced/isaac --fruit-id 28
python3 -m unittest discover -s tests -p test_orchardbench_export.py -v

/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python scripts/render_tree_review.py -- \
  --tree assets/orchard/orchardbench_seed42/tree.json \
  --output "outputs/orchardbench/review_$(date +%Y%m%d_%H%M%S)" \
  --robot-usd assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd
```

The renderer produces `front.png`, `side.png`, `three_quarter.png`,
`g1_scale.png`, `review.json`, and `tree_review.blend`. These are geometry/scale
diagnostics, not Isaac camera frames. To inspect collision geometry, import
`assets/orchard/orchardbench_seed42/isaac/collision_review.usda` into Blender or
Isaac; it references the exact collider meshes used by the task.

## Harvesting gate still required

There is no truthful copy-pastable command for an already-validated harvesting
policy: none exists for this asset yet. After the lab import/hold/load/video gates
pass, implement/test contact-based control (e.g. constrained reach optimization
plus learned grasp/pull control) without changing morphology or adding grip
springs. Require sustained opposing-finger contact, native break, carry, release
and named basket success. Preserve the current 43-D action/recording mapping.
Only successful measured trials may be exported as demonstrations. A command
that simply selects the old `--policy orchard` is not that solution.
