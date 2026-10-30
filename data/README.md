# Datasets

This project has no training data. The "dataset" is the five PolyBench/C 4.2.1 kernels
(GEMM, Jacobi-2D, Floyd-Warshall, Covariance, 2MM), compiled with the parameters in
`config/benchmarks.json` (also listed in Table 2 of the paper).

- **Download:** `bash scripts/get_polybench.sh` clones
  <https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git> into `$POLYBENCH_DIR` (default `~/PolyBenchC-4.2.1`).
  The upstream suite is also at <https://sourceforge.net/projects/polybench/>.
- **Preprocessing:** done inside `run_all.py` (`base_ir`): each program is compiled with
  `clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S` to LLVM IR, plus a `-DPOLYBENCH_DUMP_ARRAYS` variant
  used for output-correctness hashing. `setup()` also builds `polybench.o` and filters the pass list against the installed `opt`.
- **Measured data:** all experiment outputs are in `results/` (see the main README).
