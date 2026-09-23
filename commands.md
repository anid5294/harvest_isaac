# Common Commands

```bash
cd <vla_isaaclab-checkout>
conda activate <developer-env>
source scripts/activate.sh
check_install_environment
```

List registered environments:

```bash
./scripts/run_env.sh --headless --physics-only --list-tasks
```

Preview G1, the white table, four YCB objects, and the RGB-D cameras:

```bash
./scripts/run_env.sh --headless \
  --task VLA-ScenePreview-YCB-G1-v0 \
  --steps 240 \
  --preview-video outputs/previews/vla_isaaclab_scene_preview.mp4
```

Record and inspect an HDF5 dataset:

```bash
./scripts/record_env.sh --dataset-name environment_smoke
python scripts/inspect_dataset.py outputs/datasets/environment_smoke.hdf5
```

Record a successful sugar-box demonstration as a LeRobot v3 dataset
(one episode, up to 1,200 steps):

```bash
./scripts/record_lerobot.sh \
  --dataset-name sugar_box_demo \
  --lerobot-version 3 \
  --episodes 1 \
  --steps 1200
```

`--lerobot-version` accepts `3` or `2.1` and defaults to `3`.

The dataset is written to `outputs/lerobot/sugar_box_demo/`. Inspect and replay it with:

```bash
python scripts/inspect_lerobot.py outputs/lerobot/sugar_box_demo
./scripts/replay_lerobot.sh outputs/lerobot/sugar_box_demo --episode 0 --headless
```
