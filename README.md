# Compiler pass ordering with Bayesian optimization

Compares -O3, random search, a genetic algorithm, vanilla BO (Optuna TPE) and a proposed
GP-based BO (sequence-aware kernel + warm start) on 5 PolyBench/C programs with LLVM 18.

## Requirements
- Linux or WSL (Ubuntu 24.04), `sudo apt install clang llvm python3-venv`
- PolyBench/C 4.2.1 cloned to ~/PolyBenchC-4.2.1:
  `git clone https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git ~/PolyBenchC-4.2.1`
- `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## Run
    python run_all.py smoke                          # quick end-to-end test
    python run_all.py run --budget 20 --seeds 2      # settings used for the reported results
    python run_all.py analyze                        # summary.csv, per_program.csv, per_run.csv, 4 PNG graphs

The run is resumable. Output goes to ./results (smoke test: ./results_smoke).

## Files
- run_all.py: evaluator, all methods, logging, analysis and plots
- dashboard.py: Streamlit dashboard over everything in results/
- results/: all measured outputs (CSV, JSON and the 4 exported PNG graphs)

## Dashboard

    pip install -r requirements.txt
    streamlit run dashboard.py          # opens http://localhost:8501

`results/` already contains every file the dashboard reads, so all sections are filled in. The sidebar
shows a checklist of the 12 result files (6 CSV, 2 JSON, 4 PNG). If one is missing, only the section that
needs it shows a note saying which file to copy in.

Sections: headline numbers, the six parameters, beat rate / validity / failed builds, comparison table,
convergence (speedup, per-program runtime, share of searches beating -O3), speedup by program, spread across
runs, runtime of the best binary, best pass sequence, every metric by program, -O3 and unoptimized
references, search timing vs verified result, pass usage, search space and warm start, per-seed table, the
four exported graphs, findings, and the full evaluation log.
