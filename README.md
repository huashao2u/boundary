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
