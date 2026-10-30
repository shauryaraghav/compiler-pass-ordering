#!/usr/bin/env bash
# Summary tables, graphs, and the statistics used in the paper (no LLVM needed, works on the shipped results/).
. "$(dirname "$0")/_common.sh"
python run_all.py analyze --out "$OUT_DIR"
python analysis/extra_stats.py --results "$OUT_DIR"
