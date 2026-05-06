#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOP_AFTER=mining exec "${SCRIPT_DIR}/run_full_until_teacher.sh" "$@"
