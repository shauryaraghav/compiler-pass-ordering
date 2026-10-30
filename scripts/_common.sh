#!/usr/bin/env bash
# Shared by all scripts: move to repo root, load config, activate venv if present.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f config/experiment.env ]; then set -a; . config/experiment.env; set +a
else set -a; . config/experiment.env.example; set +a; fi
POLYBENCH_DIR="${POLYBENCH_DIR/#\$HOME/$HOME}"; export POLYBENCH_DIR
[ -f "$VENV_DIR/bin/activate" ] && . "$VENV_DIR/bin/activate" || true
