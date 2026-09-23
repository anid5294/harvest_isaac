#!/bin/bash
set -e
source "$(dirname "$0")/activate.sh"
check_install_environment
cd "$VLA_ISAACLAB_PROJECT"
exec python scripts/run_env.py \
  --headless \
  --record-format lerobot \
  --lerobot-version 3 \
  --task VLA-YCBSugarBox-G1-JointPos-v0 \
  --episodes 1 \
  --steps 1200 \
  --dataset-name vla_ycb_sugar_box \
  "$@"
