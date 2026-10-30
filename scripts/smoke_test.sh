#!/usr/bin/env bash
# Quick end-to-end check: 1 program, 12 evaluations per method, writes ./results_smoke
. "$(dirname "$0")/_common.sh"
python run_all.py smoke
python run_all.py analyze --out results_smoke
