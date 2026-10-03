#!/usr/bin/env bash
# Prepare a linked worktree: own venv on /mnt/data (workspace packages are editable installs, so
# sharing the main .venv would import the main checkout's code), shared data/ and models/.
# Run from inside the worktree:  bash scripts/worktree_setup.sh
set -euo pipefail
unset PYTHONPATH

top=$(git rev-parse --show-toplevel)
main=$(dirname "$(cd "$(git rev-parse --git-common-dir)" && pwd -P)")
[[ "$top" != "$main" ]] || { echo "run this inside a linked worktree, not $main" >&2; exit 1; }
python3 "$top/scripts/check_git_names.py" branch

venv=/mnt/data/venvs/ego2libero-wt/$(basename "$top")
mkdir -p "$venv" "$main/data" "$main/models"
for link in ".venv:$venv" "data:$main/data" "models:$main/models"; do
  [[ -e "$top/${link%%:*}" ]] || ln -s "${link#*:}" "$top/${link%%:*}"
done
cd "$top"
uv sync --all-groups
uv run pytest -q
