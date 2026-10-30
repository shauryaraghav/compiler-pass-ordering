#!/usr/bin/env bash
# One-time setup: Python venv + dependencies, then PolyBench/C. Run from anywhere.
. "$(dirname "$0")/_common.sh"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
bash scripts/get_polybench.sh
echo
for t in clang opt llc; do command -v $t >/dev/null || echo "MISSING: $t  ->  sudo apt install clang llvm"; done
clang --version | head -1 || true
echo "Setup finished. Next: bash scripts/smoke_test.sh"
