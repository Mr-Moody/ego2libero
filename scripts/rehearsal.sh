#!/usr/bin/env bash
# Rehearse everything after retargeting on jittered scripted demos: replay each (Run A source),
# multiply them (Run B source), export both LeRobot datasets. Rebuilds from scratch on rerun.
# Spec: docs/superpowers/specs/2026-10-12-scripted-rehearsal-design.md
#   COUNT=25 PER_DEMO=16 WORKERS=4 scripts/rehearsal.sh      (make rehearsal)
# Each generate worker renders MuJoCo images; use WORKERS=2 on a 16 GB machine.
set -euo pipefail
unset PYTHONPATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Own data root, so default `e2l generate run` / `e2l train export` on real demos never pick up
# the stand-ins. Every e2l command below reads and writes under it via E2L_DATA.
export E2L_DATA="${E2L_DATA:-$PWD/data/rehearsal}"
DATA="$E2L_DATA"
COUNT="${COUNT:-25}"
PER_DEMO="${PER_DEMO:-16}"
WORKERS="${WORKERS:-4}"
e2l() { uv run e2l "$@"; }

uv run python scripts/scripted_segments.py --count "$COUNT"
mapfile -t ids < <(for ((i = 0; i < COUNT; i++)); do printf '20261003_scripted_%03d\n' "$i"; done)

rm -rf "$DATA/generated/rehearsal_replay" "$DATA/lerobot/rehearsal_a" "$DATA/lerobot/rehearsal_b"
for i in "${!ids[@]}"; do
  e2l sim replay "${ids[$i]}" --run-id rehearsal_replay --episode-index $((i % 50))
done
ok=$(grep -l '"success": true' "$DATA"/generated/rehearsal_replay/*.json | wc -l)
echo "raw replay: $ok / $COUNT succeeded"

e2l generate run "${ids[@]}" --run-id rehearsal_gen --overwrite --set "episodes_per_demo=$PER_DEMO" \
  --set "workers=$WORKERS"

e2l train export rehearsal_replay \
  --set "dataset_root=$DATA/lerobot/rehearsal_a" --set dataset_repo_id=local/rehearsal_a
e2l train export rehearsal_replay rehearsal_gen \
  --set "dataset_root=$DATA/lerobot/rehearsal_b" --set dataset_repo_id=local/rehearsal_b
