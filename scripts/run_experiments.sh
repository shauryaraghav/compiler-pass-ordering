#!/usr/bin/env bash
# Full experiment with the paper settings (BUDGET/SEEDS from config). Resumable; safe to Ctrl+C and re-run.
# 5 programs x 5 methods x SEEDS searches; a few hours on a laptop. Do not run other heavy jobs meanwhile (timing noise).
. "$(dirname "$0")/_common.sh"
python run_all.py run --budget "$BUDGET" --seeds "$SEEDS" --out "$OUT_DIR"
