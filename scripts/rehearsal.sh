#!/usr/bin/env bash
# Rehearse everything after retargeting on jittered scripted demos: replay each (Run A source),
# multiply them (Run B source), export both LeRobot datasets. Rebuilds from scratch on rerun.
# Spec: docs/superpowers/specs/2026-10-12-scripted-rehearsal-design.md
#   COUNT=25 PER_DEMO=16 scripts/rehearsal.sh      (make rehearsal)
set -euo pipefail
unset PYTHONPATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
COUNT="${COUNT:-25}"
PER_DEMO="${PER_DEMO:-16}"
e2l() { uv run e2l "$@"; }

uv run python scripts/scripted_segments.py --count "$COUNT"
mapfile -t ids < <(for ((i = 0; i < COUNT; i++)); do printf '20261003_scripted_%03d\n' "$i"; done)

rm -rf data/generated/rehearsal_replay data/lerobot/rehearsal_a data/lerobot/rehearsal_b
for i in "${!ids[@]}"; do
  e2l sim replay "${ids[$i]}" --run-id rehearsal_replay --episode-index $((i % 50))
done
ok=$(grep -l '"success": true' data/generated/rehearsal_replay/*.json | wc -l)
echo "raw replay: $ok / $COUNT succeeded"

e2l generate run "${ids[@]}" --run-id rehearsal_gen --overwrite --set "episodes_per_demo=$PER_DEMO"

e2l train export rehearsal_replay \
  --set dataset_root=data/lerobot/rehearsal_a --set dataset_repo_id=local/rehearsal_a
e2l train export rehearsal_replay rehearsal_gen \
  --set dataset_root=data/lerobot/rehearsal_b --set dataset_repo_id=local/rehearsal_b
