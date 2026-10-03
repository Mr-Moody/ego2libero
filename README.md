# ego2libero

Turning a handful of phone videos of my own hand into a LIBERO policy: recover the wrist
trajectory and grip from each video, multiply every demo into many simulated episodes at new
object poses, keep the ones that succeed, and fine-tune SmolVLA on them.

> Work in progress. Sections marked TODO are filled in as results come in.

## Overview

TODO: task, headline result (success rate vs number of human demos, with and without
augmentation), one figure, one rollout video.

## Setup

```bash
git clone <repo> && cd ego2libero
make setup            # uv sync --all-groups + pre-commit hooks (Python 3.12, CUDA GPU)
make models           # MediaPipe hand_landmarker.task
make smoke            # lint, tests, CUDA check, LIBERO render check, pretrained baseline
```

DGX Spark / training-only machines: `uv sync --group train --group dev`.

## Pipeline

| Stage | Command | Output |
| --- | --- | --- |
| Capture | `scripts/make_markers.py`, `scripts/calibrate_phone.py` | printable markers, `data/calib/` |
| Perception | `e2l perception normalise`, `e2l perception run <demo>` | table-frame hand + objects |
| Segment | `e2l segment run <demo>` | object-centric hand segments |
| Retarget | `e2l retarget run <demo>` | Panda end-effector segments |
| Generate | `e2l generate run` | successful simulated episodes |
| Train | `e2l train export`, `e2l train finetune` | LeRobot dataset, SmolVLA checkpoint |
| Eval | `e2l eval rollout`, `e2l eval ablate`, `e2l eval report` | success rates, figures |

## Results

TODO

## Design choices

TODO

## What worked and what didn't

TODO
