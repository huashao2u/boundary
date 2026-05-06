#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/media/boundary"
LS_BIN="${LS_BIN:-${PROJECT_ROOT}/.venvs/label-studio/bin/label-studio}"
LS_DATA_DIR="${LS_DATA_DIR:-${PROJECT_ROOT}/.label_studio_data_v113}"
LS_PORT="${LS_PORT:-8080}"
LS_INTERNAL_HOST="${LS_INTERNAL_HOST:-127.0.0.1}"
LS_LOG_LEVEL="${LS_LOG_LEVEL:-INFO}"

mkdir -p "${LS_DATA_DIR}"

SECRET_KEY_FILE="${LS_DATA_DIR}/.secret_key"
if [[ ! -f "${SECRET_KEY_FILE}" ]]; then
  python3 -c 'import secrets; print(secrets.token_urlsafe(50))' > "${SECRET_KEY_FILE}"
  chmod 600 "${SECRET_KEY_FILE}"
fi

export DEBUG=false
export LATEST_VERSION_CHECK=false
export SECRET_KEY
SECRET_KEY="$(cat "${SECRET_KEY_FILE}")"

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="${PROJECT_ROOT}"

cmd=(
  "${LS_BIN}"
  start
  --no-browser
  --internal-host "${LS_INTERNAL_HOST}"
  --port "${LS_PORT}"
  --data-dir "${LS_DATA_DIR}"
  --log-level "${LS_LOG_LEVEL}"
)

if [[ -n "${LS_USERNAME:-}" && -n "${LS_PASSWORD:-}" ]]; then
  cmd+=(--username "${LS_USERNAME}" --password "${LS_PASSWORD}")
fi

echo "Starting Label Studio on ${LS_INTERNAL_HOST}:${LS_PORT}"
echo "Data dir: ${LS_DATA_DIR}"
echo "Review tasks: ${PROJECT_ROOT}/artifacts_v023_full_20260430T153529Z/human_review/label_studio_pair_review_100_per_dataset.json"
echo "Label config: ${PROJECT_ROOT}/artifacts_v023_full_20260430T153529Z/human_review/label_studio_pair_review_config.xml"
exec "${cmd[@]}"
