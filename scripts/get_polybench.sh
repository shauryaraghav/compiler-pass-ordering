#!/usr/bin/env bash
# Dataset step: download PolyBench/C 4.2.1 (the benchmark programs used as the "dataset").
. "$(dirname "$0")/_common.sh"
if [ -d "$POLYBENCH_DIR/utilities" ]; then echo "PolyBench already at $POLYBENCH_DIR"; exit 0; fi
git clone https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git "$POLYBENCH_DIR"
echo "PolyBench/C 4.2.1 -> $POLYBENCH_DIR"
