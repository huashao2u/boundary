# MCAgent Boundary

`mcagent_boundary` is now a single repository with a `src/` layout.

The previously separate shared layer has been folded in-repo as `src/mcagent_core`, so there is
no cross-repository code dependency during normal development.

## Layout

- package code: `src/mcagent_boundary/` and `src/mcagent_core/`
- shared assets: `dataset/` and `models/`
- runtime outputs: `./artifacts/`

## Quick Start

```bash
cd /media/songyl/mcagent_boundary
conda run -n salra python src/mcagent_boundary/scripts/00_inspect_legacy.py
conda run -n salra python src/mcagent_boundary/scripts/01_build_adapters.py --limit-per-dataset 1
conda run -n salra python src/mcagent_boundary/scripts/02_rollout_all.py --limit-per-dataset 1
conda run -n salra python src/mcagent_boundary/scripts/03_teacher_label_boundary.py
conda run -n salra python src/mcagent_boundary/scripts/04_make_pairs.py
conda run -n salra python src/mcagent_boundary/scripts/05_optional_warmup.py
conda run -n salra python src/mcagent_boundary/scripts/06_train_dpo.py --smoke
conda run -n salra python src/mcagent_boundary/scripts/07_eval.py --limit-per-dataset 1
```

The repository ships with:

- `dataset/` copied locally from the legacy project
- `models/` symlinked to `/media/songyl/SALRA/models`
- a single `pyproject.toml` that exposes both `mcagent_boundary` and `mcagent_core`

## Rollout backend

The default rollout backend is `hf` (the student model, `qwen2.5-7b-instruct`),
configured in `src/mcagent_boundary/configs/rollout.yaml`. Available options:

- `hf` — **default**; runs the student model for every `natural_action` decision.
- `heuristic` — oracle-style smoke policy (reads gold labels). Useful for
  ablations, debugging, and pipeline smoke tests; **not** for mainline experiment
  runs. A warning is logged whenever it becomes active.
- `auto` — prefer `hf`, but fall back to `heuristic` (with a loud warning) if
  the student model assets at `paths.model_root` are missing. Convenient for
  CI / smoke runs on machines without the full checkpoint.

If `backend: hf` is configured but the checkpoint is absent on disk,
`generate_rollouts()` automatically demotes the run to `heuristic` and emits a
warning — the pipeline still finishes, but the output should not be treated as
a mainline experiment artifact.

