SHELL := /bin/bash
# ROS Humble (sourced in ~/.zshrc) puts python3.10 site-packages on PYTHONPATH; keep it out.
unexport PYTHONPATH
export MUJOCO_GL ?= egl
export PYOPENGL_PLATFORM ?= egl

HAND_MODEL_URL := https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
E2L := uv run e2l

.PHONY: setup lint format test models gpu smoke normalise perception segment retarget sim-check \
	generate export finetune baseline eval report

## Environment ------------------------------------------------------------------------------
setup:            ## install every group (laptop); Spark: uv sync --group train --group dev
	uv sync --all-groups
	uv run pre-commit install

models: models/hand_landmarker.task

models/hand_landmarker.task:
	mkdir -p models
	curl -fL -o $@ $(HAND_MODEL_URL)

## Quality ----------------------------------------------------------------------------------
lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

gpu:              ## fails unless torch sees a CUDA device
	uv run python -c "import torch; assert torch.cuda.is_available(), \"no CUDA\"; print(torch.__version__, torch.cuda.get_device_name(0))"

smoke: lint test gpu sim-check baseline  ## full toolchain check from a fresh clone

## Stages (DEMO=<demo_id>, RUN=<run_id>) ----------------------------------------------------
normalise:
	$(E2L) perception normalise data/raw

perception:
	$(E2L) perception run $(DEMO)

segment:
	$(E2L) segment run $(DEMO)

retarget:
	$(E2L) retarget run $(DEMO)

sim-check:
	$(E2L) sim check

generate:
	$(E2L) generate run

export:
	$(E2L) train export

finetune:
	$(E2L) train finetune

baseline:
	$(E2L) eval baseline

eval:
	$(E2L) eval rollout

report:
	$(E2L) eval report
