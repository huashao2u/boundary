#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

DATASETS="${DATASETS:-gsm8k,math,in3,mintqa,or_bench}"
BACKEND="${BACKEND:-vllm}"
TEACHER_WORKERS="${TEACHER_WORKERS:-8}"
TEACHER_RPM_LIMIT="${TEACHER_RPM_LIMIT:-240}"
INCLUDE_CLEAR_ANSWER="${INCLUDE_CLEAR_ANSWER:-1}"
STOP_AFTER="${STOP_AFTER:-teacher}"
SELECTION_PRESET="${SELECTION_PRESET:-v023_full_rollout}"
RUN_ID="${RUN_ID:-v023_full_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-artifacts_${RUN_ID}}"
EMAIL_TO="${EMAIL_TO:-}"

mkdir -p "$RUN_DIR"

notify() {
  local subject="$1"
  local body="$2"
  local args=(send_email.py --subject "$subject" --body "$body")
  if [[ -n "$EMAIL_TO" ]]; then
    args+=(--to "$EMAIL_TO")
  fi
  if ! python3 "${args[@]}"; then
    echo "[warn] failed to send email: $subject" >&2
  fi
}

line_counts() {
  python3 - "$RUN_DIR" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
names = [
    "all_rollouts.jsonl",
    "boundary_candidates.jsonl",
    "clear_answer_anchors.jsonl",
    "clear_external_anchors.jsonl",
    "teacher_labels.jsonl",
]
for name in names:
    path = run_dir / name
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        print(f"{name}: {count}")
PY
}

teacher_source_counts() {
  python3 - "$RUN_DIR" <<'PY'
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sys

path = Path(sys.argv[1]) / "teacher_labels.jsonl"
if not path.exists():
    raise SystemExit(0)
counts = Counter()
datasets = Counter()
with path.open("r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        counts[str(row.get("source", "unknown"))] += 1
        datasets[str(row.get("dataset", "unknown"))] += 1
print("teacher source counts:")
for key, value in sorted(counts.items()):
    print(f"  {key}: {value}")
print("teacher labels by dataset:")
for key, value in sorted(datasets.items()):
    print(f"  {key}: {value}")
PY
}

run_stage() {
  local stage_name="$1"
  shift
  local started_at
  local finished_at
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$started_at] START ${stage_name}"
  "$@"
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local body
  body="[$finished_at] END ${stage_name}
run_dir: ${RUN_DIR}
datasets: ${DATASETS}
selection_preset: ${SELECTION_PRESET}
backend: ${BACKEND}
$(line_counts)"
  if [[ "$stage_name" == "03_teacher_label_boundary" ]]; then
    body="${body}
$(teacher_source_counts)"
  fi
  echo "$body"
  notify "[boundary] ${stage_name} finished (${RUN_ID})" "$body"
}

on_error() {
  local exit_code=$?
  local failed_at
  failed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  notify "[boundary] pipeline failed (${RUN_ID})" "Pipeline failed at ${failed_at} with exit code ${exit_code}.

run_dir: ${RUN_DIR}
datasets: ${DATASETS}
backend: ${BACKEND}

Recent artifact counts:
$(line_counts)
"
  exit "$exit_code"
}
trap on_error ERR

if [[ "$STOP_AFTER" != "rollout" && "$STOP_AFTER" != "mining" && "$STOP_AFTER" != "teacher" ]]; then
  echo "STOP_AFTER must be one of: rollout, mining, teacher" >&2
  exit 2
fi

if [[ "$STOP_AFTER" == "teacher" ]]; then
  python3 - <<'PY'
from __future__ import annotations

import os
from mcagent_boundary.config import load_boundary_config

config = load_boundary_config()
teacher = config.get("teacher", {})
api_key_env = str(teacher.get("api_key_env", "POE_API_KEY"))
if not (os.getenv(api_key_env) or teacher.get("api_key")):
    raise SystemExit(
        f"Missing LLM teacher credentials. Set {api_key_env} or teacher.api_key "
        "in configs/default.local.yaml before running strict teacher labeling."
    )
PY
fi

notify "[boundary] full pipeline started (${RUN_ID})" "Starting full five-dataset pipeline.

run_dir: ${RUN_DIR}
datasets: ${DATASETS}
selection_preset: ${SELECTION_PRESET}
backend: ${BACKEND}
stop_after: ${STOP_AFTER}
teacher_workers: ${TEACHER_WORKERS}
teacher_rpm_limit: ${TEACHER_RPM_LIMIT}

Stages:
1. 01_build_adapters
2. 02_rollout_all
3. 02b_mine_boundary $([[ "$STOP_AFTER" == "rollout" ]] && echo "(skipped)")
4. 03_teacher_label_boundary strict LLM labeling $([[ "$STOP_AFTER" != "teacher" ]] && echo "(skipped)")"

run_stage \
  "01_build_adapters" \
  python3 src/mcagent_boundary/scripts/01_build_adapters.py \
    --full-dataset \
    --selection-preset "$SELECTION_PRESET"

run_stage \
  "02_rollout_all" \
  python3 src/mcagent_boundary/scripts/02_rollout_all.py \
    --backend "$BACKEND" \
    --datasets "$DATASETS" \
    --full-dataset \
    --selection-preset "$SELECTION_PRESET" \
    --output-dir "$RUN_DIR"

if [[ "$STOP_AFTER" == "rollout" ]]; then
  notify "[boundary] rollout pipeline finished (${RUN_ID})" "Finished adapter build and rollout/mining only.

run_dir: ${RUN_DIR}
datasets: ${DATASETS}
selection_preset: ${SELECTION_PRESET}
backend: ${BACKEND}

Artifact counts:
$(line_counts)

Teacher labeling was skipped."
  echo "Pipeline stopped after rollout. Outputs are in ${RUN_DIR}"
  exit 0
fi

run_stage \
  "02b_mine_boundary" \
  python3 src/mcagent_boundary/scripts/02b_mine_boundary.py \
    --input-dir "$RUN_DIR" \
    --output-dir "$RUN_DIR"

if [[ "$STOP_AFTER" == "mining" ]]; then
  notify "[boundary] mining pipeline finished (${RUN_ID})" "Finished adapter build, rollout, and mining.

run_dir: ${RUN_DIR}
datasets: ${DATASETS}
selection_preset: ${SELECTION_PRESET}
backend: ${BACKEND}

Artifact counts:
$(line_counts)

Teacher labeling was skipped."
  echo "Pipeline stopped after mining. Outputs are in ${RUN_DIR}"
  exit 0
fi

teacher_args=(
  src/mcagent_boundary/scripts/03_teacher_label_boundary.py
  --input-dir "$RUN_DIR"
  --output-dir "$RUN_DIR"
  --strict-teacher
  --resume-existing
  --teacher-workers "$TEACHER_WORKERS"
  --teacher-rpm-limit "$TEACHER_RPM_LIMIT"
)
if [[ "$INCLUDE_CLEAR_ANSWER" == "1" ]]; then
  teacher_args+=(--include-clear-answer)
fi

run_stage \
  "03_teacher_label_boundary" \
  python3 "${teacher_args[@]}"

echo "Pipeline finished. Outputs are in ${RUN_DIR}"
