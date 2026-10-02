# CPU attached-apple approach: next evidence gate

The uploaded run is a failed harvest, not a successful detachment demo. It
stayed in approach, had 59 mm position error at step 200, never recorded finger
contact, and broke at step 203. The requested waist target and measured waist
position differed by about 0.35 rad. This does not establish unreachable
geometry, excessive orientation constraint, or a collision as the cause.

The next patch instruments that distinction; it does **not** claim to fix the
physical grasp. No geometry, initial robot pose, stem force/torque threshold,
collision shape, action mapping or success criterion changes were made.
The sugar-box controller is unchanged.

## Changes

- `policies/cpu_harvest.py`: every-step requested/measured palm pose,
  orientation error, effective limits, bounded-IK correction, normalized action,
  remapped target, processed action and articulation joint target. Post-step
  tracking is distinct from the command. Terminal policy diagnostics now
  reflect environment failure/break even without another controller update.
- `envs/orchard_pick/diagnostics.py`, `env.py`, `mdp.py`: opt-in native contact
  reports for robot rigid bodies, apples and stems; actor/collider paths,
  contact impulse vectors (N s), counted physics callbacks and native joint
  break observations. The last 1200 step/event records are retained. Contact
  reporting threshold is set to zero for observation, **not** the break threshold.
- `scripts/run_env.py`: diagnostics flag, output files, post-step sampling and
  opt-in orientation weighting (baseline remains 0.20).
- `tests/test_orchard_physics_diagnostics.py`: pure ring-buffer/error tests.

Callbacks are observations, not exact solver timestamps. Their ordering is
preserved in the event stream; do not infer the precise internal break substep
from callback count alone. Apple velocity is still sampled in the 30 Hz task
trajectory, not a native substep velocity trace. Collision reports can identify
loads missing from the original seven-finger filter. A silent report does not
prove contact absence until native reporting has been confirmed working.

## Publish only this patch on the Mac

The preceding CPU controller work is already committed as `00d8ba0` in this
checkout. Preserve unrelated dirty README/roadmap/Arena changes.

```bash
cd /Users/anikadixit/Desktop/ctrl_learning/harvest_isaac
git branch --show-current
git add CPU_APPROACH_DIAGNOSTICS.md scripts/run_env.py \
  src/vla_isaaclab/policies/cpu_harvest.py \
  src/vla_isaaclab/envs/orchard_pick/diagnostics.py \
  src/vla_isaaclab/envs/orchard_pick/env.py \
  src/vla_isaaclab/envs/orchard_pick/mdp.py \
  tests/test_orchard_physics_diagnostics.py
git diff --cached --check
git diff --cached
git commit -m "Instrument CPU orchard approach tracking and native contact events"
git push origin feature/isaac-orchard-preview
```

Verify the branch and staged diff before committing. The agent has not committed
or pushed anything.

## One short reproduction on Songkhla

```bash
cd /home/vlakbnn/anikad/harvest_isaac
git status --short
git pull --ff-only origin feature/isaac-orchard-preview
source /home/vlakbnn/miniconda3/etc/profile.d/conda.sh
conda activate /home/vlakbnn/anikad/.conda/envs/harvest-isaac
export VLA_ISAACLAB_ENV="$CONDA_PREFIX"
export ISAACLAB_ROOT=/media/data-ssd/software/IsaacLab-v2.0.2
source scripts/activate.sh
check_install_environment
export APPROACH_RUN="$PWD/outputs/cpu_approach_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$APPROACH_RUN"
./scripts/run_env.sh --headless --physics-only --device cpu \
  --task VLA-OrchardPick-G1-JointPos-v0 --policy cpu-harvest \
  --cpu-harvest-diagnostics --orchard-tree-model orchardbench \
  --seed 42 --steps 240 --report-dir "$APPROACH_RUN/physics" \
  > "$APPROACH_RUN/physics.log" 2>&1
printf 'Diagnostic runner exit: %s\n' "$?"
python scripts/check_cpu_harvest_run.py "$APPROACH_RUN/physics/validation.json"
tar -czf "$APPROACH_RUN/evidence.tar.gz" -C "$APPROACH_RUN" physics physics.log
printf 'Return this archive: %s/evidence.tar.gz\n' "$APPROACH_RUN"
```

An unsuccessful 240-step diagnostic should not pass harvest validation. Require
`control_trace.jsonl` and `physics_diagnostics.json` in addition to
`trajectory.jsonl` and `validation.json`. Native diagnostic API failures are
explicit errors; preserve the full log rather than silently treating absent
reports as zero contact. If this reproduces the same premature break, return
the archive rather than running the identical attempt again.

## Render the same short attempt (not training-data collection)

```bash
./scripts/run_env.sh --headless --device cpu \
  --task VLA-OrchardPick-G1-JointPos-v0 --policy cpu-harvest \
  --cpu-harvest-diagnostics --orchard-tree-model orchardbench \
  --seed 42 --steps 240 --report-dir "$APPROACH_RUN/render" \
  --camera-videos "$APPROACH_RUN/videos" > "$APPROACH_RUN/render.log" 2>&1
printf 'Rendered diagnostic exit: %s\n' "$?"
```

Videos: `external.mp4`, `left_wrist.mp4`, `right_wrist.mp4` in the videos folder.
Rendering can be done even for a failed diagnostic; it must not be labeled a
successful demonstration. Prefer this rendered short attempt instead of another
physics-only repeat if Songkhla rendering is already known to work.

## Decision after evidence

1. If processed actions differ from requested joint targets, fix mapping first.
2. If applied targets agree but measured joints lag while native contacts load
   the hand/tree, adjust the approach clearance/waypoint or fixed robot pose.
3. If no obstructing contact is observed and the pose trade-off dominates,
   compare one short run with `--cpu-harvest-orientation-weight 0.05`. This is
   an explicit experiment, not an automatically selected fix. It still uses
   the existing 15-degree orientation arrival gate and all physical grasp gates.
4. Only after approach and measured grasp work, run the full 2700-step attempt.

```bash
./scripts/run_env.sh --headless --device cpu \
  --task VLA-OrchardPick-G1-JointPos-v0 --policy cpu-harvest \
  --cpu-harvest-diagnostics --orchard-tree-model orchardbench \
  --seed 42 --steps 2700 --report-dir "$APPROACH_RUN/full" \
  --camera-videos "$APPROACH_RUN/full_videos" > "$APPROACH_RUN/full.log" 2>&1
printf 'Full runner exit: %s\n' "$?"
python scripts/check_cpu_harvest_run.py "$APPROACH_RUN/full/validation.json" \
  --videos "$APPROACH_RUN/full_videos"
```

Use the same explicitly validated controller settings in the full run if an
experiment changed them. Success still requires native attachment/break,
pre-break sustained opposing contact, contacted carry travel, release followed
by supported stable placement, and the named success termination. Both runner
and checker must pass. No successful end-to-end CPU harvest has yet been
demonstrated by the available runtime evidence.
