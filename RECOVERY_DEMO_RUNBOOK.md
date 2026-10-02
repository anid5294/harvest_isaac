# Recovered visual deliverables and CPU harvest trial

The four-seed comparison and orchard preview were generated locally in Blender.
They are visual geometry/layout reviews, **not Isaac simulation results**. Trees
remain sparse OrchardBench prototypes, not photorealistic mature commercial
canopies. Native geometry is unscaled; the preview has two rows of four trees,
4 m center spacing and 1.15 m managed strips. There is no orchard locomotion or
multi-tree physics in this preview.

## Local artifacts and reproduction

- `outputs/orchardbench/stochastic_tree_grid/grid_2x2.png`: seeds 1, 7, 21, 42,
  reading left-to-right, top-to-bottom; same camera and scale, 800 px per tile.
- That directory also contains four seed JSON files, individual PNGs, source
  license and `metadata.json`.
- `outputs/orchardbench/orchard_preview/orchard_overview_cycles.png`
- `outputs/orchardbench/orchard_preview/orchard_alley_eye_level.png`
- Preview `metadata.json` records placement and scope.

Reproduce on this Mac (Blender is installed at the path below). The export
requires NumPy, supplied by Blender's bundled Python. Use the existing pinned
source checkout; do not modify it. These commands replace the generated images
in these two output directories, not source assets.

```bash
cd /Users/anikadixit/Desktop/ctrl_learning/harvest_isaac
/Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 \
  scripts/render_orchard_deliverables.py export \
  --source /private/tmp/harvest-orchardbench-review-20261001
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python scripts/render_orchard_deliverables.py -- render --mode grid --resolution 800
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
  --python scripts/render_orchard_deliverables.py -- render --mode orchard --resolution 900
open outputs/orchardbench/stochastic_tree_grid/grid_2x2.png
open outputs/orchardbench/orchard_preview/orchard_overview_cycles.png
```

If the temporary source checkout has expired, create a fresh local checkout and
substitute its path for `--source` (never use the shared Isaac Lab checkout):

```bash
git clone https://github.com/humphreymunn/orchardbench.git /tmp/harvest-orchardbench-source
git -C /tmp/harvest-orchardbench-source checkout --detach 6313313db8b1a7d23fb2cc3afd67cac46f29399a
```

## Publish code, not generated outputs

No commit or push was performed by the agent. The commands below include the
recovered free-apple fallback and diagnostic work on which the current runner
depends, but omit unrelated README, roadmap, Arena and commercial-planner edits.
Review the staged diff before committing; do not use `git add .` in this dirty
checkout. Outputs are gitignored and are reproduced/copied separately.

```bash
cd /Users/anikadixit/Desktop/ctrl_learning/harvest_isaac
git status --short
git branch --show-current
git add RECOVERY_DEMO_RUNBOOK.md STEM_DIAGNOSTICS_AND_CPU_DEMO.md \
  scripts/render_orchard_deliverables.py scripts/run_env.py \
  scripts/check_cpu_harvest_run.py scripts/check_free_pick_place_run.py \
  scripts/check_orchard_run.py scripts/preview_views.py \
  scripts/debug/minimal_stem_physx.py \
  src/vla_isaaclab/envs/__init__.py src/vla_isaaclab/envs/free_pick_place \
  src/vla_isaaclab/policies/scripted_pick_place.py \
  src/vla_isaaclab/policies/cpu_harvest.py \
  src/vla_isaaclab/policies/cpu_harvest_fsm.py \
  src/vla_isaaclab/recording/frame.py \
  tests/test_free_pick_place_env.py tests/test_free_pick_place_report.py \
  tests/test_minimal_stem_physx.py tests/test_cpu_harvest_report.py \
  tests/test_cpu_harvest.py
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Add stochastic tree previews and CPU attached-apple harvest trial"
git push origin feature/isaac-orchard-preview
```

Confirm the branch printed above is `feature/isaac-orchard-preview` before
committing. Review and unstage any already-staged unrelated files first.

## Songkhla: update and activate

Run in a terminal on Songkhla, not inside Python. Starting in `(base)` or another
environment is fine: the explicit activation below selects the intended one.
If Git reports local changes that prevent the update, stop and preserve them;
do not reset or clean the checkout.

```bash
cd /home/vlakbnn/anikad/harvest_isaac
git status --short
git switch feature/isaac-orchard-preview
git pull --ff-only origin feature/isaac-orchard-preview
source /home/vlakbnn/miniconda3/etc/profile.d/conda.sh
conda activate /home/vlakbnn/anikad/.conda/envs/harvest-isaac
export VLA_ISAACLAB_ENV="$CONDA_PREFIX"
export ISAACLAB_ROOT=/media/data-ssd/software/IsaacLab-v2.0.2
source scripts/activate.sh
check_install_environment
python -m unittest discover -s tests -p 'test_cpu_harvest*.py' -v
export HARVEST_RUN="$PWD/outputs/cpu_harvest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$HARVEST_RUN"
printf '%s\n' "$HARVEST_RUN"
```

## CPU attached-apple trial

This uses the checked-in OrchardBench seed-42 asset and unchanged native stem
mechanics. It is not the free-apple fallback. The entire physics episode uses
CPU; rendering still needs the lab's working Isaac rendering stack.

First confirm passive attachment. A passive run need not meet the runner's
active-harvest success predicate; use the explicit `--passive` checker here.

```bash
./scripts/run_env.sh --headless --physics-only --device cpu \
  --task VLA-OrchardPick-G1-JointPos-v0 --policy standing \
  --orchard-tree-model orchardbench --seed 42 --steps 120 \
  --report-dir "$HARVEST_RUN/passive" > "$HARVEST_RUN/passive.log" 2>&1
printf 'Passive runner exit: %s\n' "$?"
python scripts/check_orchard_run.py "$HARVEST_RUN/passive/validation.json" --passive
```

Only after that check passes:

```bash
./scripts/run_env.sh --headless --physics-only --device cpu \
  --task VLA-OrchardPick-G1-JointPos-v0 --policy cpu-harvest \
  --orchard-tree-model orchardbench --seed 42 --steps 2700 \
  --report-dir "$HARVEST_RUN/physics" > "$HARVEST_RUN/physics.log" 2>&1
printf 'Active runner exit: %s\n' "$?"
python scripts/check_cpu_harvest_run.py "$HARVEST_RUN/physics/validation.json"
```

Require **both active runner and checker exit 0**, named success and measured
evidence. Missing JSON, nonfinite state or native errors are failures. Review
`validation.json`, `trajectory.jsonl` and the complete log. A reached waypoint
or close palm-to-target error does not demonstrate finger/apple contact.

After physics passes, run the same attempt with camera videos:

```bash
./scripts/run_env.sh --headless --device cpu \
  --task VLA-OrchardPick-G1-JointPos-v0 --policy cpu-harvest \
  --orchard-tree-model orchardbench --seed 42 --steps 2700 \
  --report-dir "$HARVEST_RUN/render" --camera-videos "$HARVEST_RUN/videos" \
  > "$HARVEST_RUN/render.log" 2>&1
printf 'Rendered runner exit: %s\n' "$?"
python scripts/check_cpu_harvest_run.py "$HARVEST_RUN/render/validation.json" \
  --videos "$HARVEST_RUN/videos"
```

Expected MP4s: `external.mp4`, `left_wrist.mp4`, `right_wrist.mp4`. Inspect the
grasp, detachment, transport, release and physical settling. Do not collect
training demonstrations until those checks succeed repeatedly. Do not change
stem strength merely because grasp/pull times out. Stop after three identical
failures and inspect the first failed transition/contact geometry.

## Copy the lab result to this Mac

In a **Mac** terminal, enter the SSH alias that you normally use and the exact
`HARVEST_RUN` path printed on Songkhla. No SSH address has been assumed:

```bash
cd /Users/anikadixit/Desktop/ctrl_learning/harvest_isaac
printf 'Songkhla SSH alias or user@host: '
read -r SONGKHLA_HOST
printf 'Absolute HARVEST_RUN path printed on Songkhla: '
read -r SONGKHLA_RUN
mkdir -p outputs/songkhla_cpu_harvest
scp -r "${SONGKHLA_HOST}:${SONGKHLA_RUN}/." outputs/songkhla_cpu_harvest/
open outputs/songkhla_cpu_harvest/videos/external.mp4
open outputs/songkhla_cpu_harvest/videos/left_wrist.mp4
```

## Limits / next handoff

Local verification: 11 CPU-policy/checker tests, four recovered free-demo tests
and three recovered minimal-stem tests pass. Changed Python entry points compile
and `git diff --check` passes. Four generated branch-graph hashes are distinct.
Main additions are `render_orchard_deliverables.py`, `cpu_harvest.py`,
`cpu_harvest_fsm.py`, `check_cpu_harvest_run.py`, their tests and this runbook;
`run_env.py` selects the policy and records the physics device. The interrupted
free-demo, camera-profile and minimal-stem diagnostic work was preserved.

Local checks cover code and evidence logic, not Isaac execution. The next
engineer should run the passive and CPU active commands, then inspect measured
finger forces and apple-in-palm coordinates at the first failing phase before
calibrating the provisional grasp pose. Native break timestamps in controller
diagnostics are observation/control-step times, not the exact solver instant.
The CPU policy cannot react inside a 30 Hz control interval. GPU joint failure
investigation is deliberately deferred; shared Isaac Lab remains untouched.
