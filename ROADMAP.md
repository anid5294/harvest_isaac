# G1 orchard harvesting roadmap

The first deliverable is a verified tabletop apple pick and place. The end task is a standing G1 with two Dex3 hands that approaches a randomized tree, detaches one apple through contact, and deposits it in a supported basket. Only successful episodes become training demonstrations. The data profile is `g1_29body_dex3_43d_v1` in LeRobot v2.1.

## 1. Verify the present baseline

Run `scripts/arena_apple_pick.py` on Beijing with the pinned Arena checkout and inspect `result.json` and both MP4s. Require `verified_pick_and_place: true`, `success_phase` of `retreat` or `hold`, a sustained lift near the hand, and visible release onto the plate. Repeat with at least five fixed seeds and preserve failed runs separately. The earlier 355-step result reached an Arena success termination during `lower`; it predates the stricter post-release check and is not proof of a completed place.

This script currently provides a 23-value WBC/PINK input and debug videos. It is a manipulation calibration trial, not a 43-channel demonstration dataset or a learned policy.

## 2. Build the four-view recording path

Keep the head camera as `observation.images.cam_left_high`. Mount RGB cameras to the left and right wrist links, near the back/inside of each wrist as shown in the lab photos, using keys `cam_left_wrist` and `cam_right_wrist`. Use a fixed front/top external view for `cam_right_high`. Save measured intrinsics, link-frame extrinsics, orientation, serial/asset IDs, and a calibration revision. Photo-based offsets are provisional; compare sample frames with the physical cameras before calling them calibrated.

Record all four as real sensor images at 640x480 RGB. Debug video recorder frame labels are not dataset timestamps. Capture camera acquisition times and joint feedback on the same simulation clock. Build a 30 Hz reference grid; associate observations within 15 ms; record the last joint-position command actually active at each reference instant. Preserve raw clock times and target history. Reject episodes with missing/duplicated frames, wrong channel shape, or timing violations.

At startup, assert that the articulation has the exact 43 simulator joint names and build an explicit permutation into the canonical order. Read `robot.data.joint_pos` for measured state and the applied `robot.data.joint_pos_target` for action after the WBC/PINK action term has run. Confirm on Beijing that these are the actual final 43 targets and that no separate actuator layer rewrites them. Never turn the 23-value WBC input into the 43-value action field.

Stream each attempt to an HDF5 source archive with episode ID, seed, controller configuration, scene parameters, success/failure, sensor timestamps, joint targets, and images. Finalize only after successful file close. Export successful sources through `recording/lerobot_v21.py`; validate Parquet fixed-size float32[43], four AV1/yuv420p MP4s with exactly one frame per row, 30 Hz timestamps, per-episode stats, and `meta/collection.json`. Test the result with the project's pinned LeRobot v2.1 loader. Failed sources remain archived and are excluded from the default imitation dataset.

## 3. Turn the tabletop trial into an orchard task

First generate one deterministic, physically supported tree and basket. Use a small set of reachable apples as separate rigid bodies connected to branches by breakable physical stem joints. Establish the release threshold with a force ramp and verify that gravity holds every fruit before contact, only the grasped fruit detaches, and detached fruit collides with the hand, ground, and basket. Use simplified collision shapes; keep high-detail tree meshes visual only. Build a basket with a base and four collision walls, then verify a free apple settles inside it. Record detach, lift, basket contact, release, and hand retreat as separate predicates. A branch or apple changing pose because of a script override is a failed physics test.

Port the tabletop grasp controller to the nearest reachable tree apple. Add reachability and trunk/branch collision checks. Keep the robot pelvis supported/fixed during initial arm and hand calibration; release that constraint only after balance or locomotion control is tested. For full-body harvesting, test walking to a tree, stopping stably, grasping, detaching, placing, and retreating in that order. A standing fixed-pelvis manipulation episode is not a locomotion demonstration.

Use OrchardBench as a procedural tree and detachment reference, not as a drop-in Isaac physics implementation. A procedural local tree avoids unresolved redistribution terms for external scans. Evaluate MFO visual meshes only after checking each mesh's license; treat OSU TreeSim license as unresolved until confirmed. O3DE ROSConDemo contains assets under mixed licenses, including noncommercial terms, so select individual assets only after tracing their license files.

## 4. Randomize and collect

Parameterize tree seed, row spacing, branch layout, apple count/size/color/mass/stem strength, target fruit, robot start, basket pose, lighting, and camera exposure. Bound every sample by the real robot's reachable workspace and reject impossible layouts before simulation. Use a deterministic seed per attempt and save all sampled values in per-episode collection metadata. Keep the same English instruction, such as “Pick an apple from the tree and place it in the basket.”

Start with 1, then 10, then 50 successful episodes. At each gate inspect video and numeric traces, audit at least one failure, and check that the success predicates survive randomization. A resumable runner should count successes rather than attempts, bound the maximum attempts, never overwrite sources, and report success rate by seed and failure reason. Collect 500 successful episodes only after the 50-episode gate and a storage estimate for four 640x480 streams. Build one LeRobot v2.1 dataset with 500 successful episodes and retain the failure archive separately.

## 5. Train and evaluate

Confirm the pinned LeRobot loader can read all 500 episodes and inspect state/action ordering and synchronized frames. A separate 28-channel arm/hand projection can be derived from columns 15:43 for the existing pi05 recipe. Full-body training needs a 43-channel policy interface, adapted checkpoint input/output projections, new normalization statistics, and an explicit choice for state conditioning. Validate action readback in simulation before hardware. Evaluate on held-out tree seeds and report detach, basket deposit, fall, collision, and whole-task success rates; compare against the scripted controller. Do not treat a visualized rollout or a dataset export as a trained-policy success.

## Beijing gate for the current commit

```bash
cd /home/vlakbnn/anikad/harvest_isaac
git switch feature/isaac-orchard-preview
git pull --ff-only origin feature/isaac-orchard-preview
git log -1 --oneline

cd /home/vlakbnn/anikad/deps/IsaacLab-Arena-native
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python \
  /home/vlakbnn/anikad/harvest_isaac/scripts/arena_apple_pick.py \
  --headless --num_envs 1 --presets physx \
  --output /home/vlakbnn/anikad/harvest_isaac/outputs/arena_apple_pick

cd /home/vlakbnn/anikad/harvest_isaac
python3 -m json.tool outputs/arena_apple_pick/result.json
find outputs/arena_apple_pick -maxdepth 1 -name '*.mp4' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort
```

Run with a new `--output` directory for each additional seed so evidence is retained. A shell exit code of 2 means the stricter trial did not verify pick and place; inspect its result and latest overview/head videos before changing controller parameters.
