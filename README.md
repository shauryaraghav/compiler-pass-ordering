# Compiler Pass Ordering with Sequence-Aware Bayesian Optimization

Companion repository of the manuscript *"Compiler Pass Ordering with Sequence-Aware Bayesian Optimization"*
(S. Raghav, M. Tripathi, Y. Reddy, P. Mahto, B. M., VIT Vellore).

It compares `-O3`, Random Search, a Genetic Algorithm, TPE (Optuna) and a proposed Gaussian-Process Bayesian optimizer
(sequence-aware kernel + Expected Improvement + cross-program warm start, plus a no-warm-start ablation) on five PolyBench/C 4.2.1
programs with LLVM 18. Each searched sequence of 15 passes (25 passes + `none`) **replaces** the IR-level `-O3` pipeline; the best
candidates are checked for output correctness and re-timed (10 runs) before they count.

**Main finding (honest summary):** under a 20-evaluation budget no method reaches `-O3` on average (verified geometric-mean speedup 0.917-0.952),
and the proposed method is not significantly better than any baseline. It does need fewer failed builds than TPE and less tuning time.

## Repository layout

| Path | Content |
|---|---|
| `paper/` | LaTeX manuscript (`main.tex`, `sections/`, `tables/`, `figures/`, `references.tex`), `manuscript_latex.zip` (Overleaf-ready), `main.pdf`, and the original PDF |
| `run_all.py` | evaluator, all 5 search methods, correctness check, logging, analysis + plots |
| `analysis/extra_stats.py` | bootstrap CIs, Wilcoxon tests, failure and tuning-cost tables (paper Tables 5-10) |
| `dashboard.py`, `.streamlit/` | Streamlit dashboard over `results/` |
| `results/` | all measured outputs (CSV, JSON, PNG) - see `results/README.md` |
| `config/` | `experiment.env.example` (settings template), `benchmarks.json` (benchmark parameters) |
| `scripts/` | setup, dataset download, smoke test, full run, analysis, dashboard, paper build |
| `data/`, `models/` | dataset and model notes (no training data / no trained models in this project) |
| `requirements.txt`, `requirements-system.txt` | Python and system dependencies |

## System requirements

- To **run experiments**: Linux or WSL2 (the paper used Ubuntu 24.04 on WSL2), LLVM/Clang 18 (`clang`, `opt`, `llc`; paper: 18.1.3), `git`, Python 3.10+.
  A multi-core laptop is fine; the programs run single-threaded. For stable timings close other heavy programs while running.
- To **only analyze results / run the dashboard / build the paper**: any OS with Python 3.10+ (and a LaTeX install for the paper).

## Installation

    git clone https://github.com/shauryaraghav/compiler-pass-ordering.git
    cd compiler-pass-ordering
    sudo apt install clang llvm git python3-venv build-essential     # Ubuntu 24.04 / WSL2
    cp config/experiment.env.example config/experiment.env           # optional: edit paths and settings
    bash scripts/setup.sh        # venv + pip install -r requirements.txt + clones PolyBench/C 4.2.1

Manual equivalent:

    python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
    git clone https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1.git ~/PolyBenchC-4.2.1

## Configuration

| Setting | Where | Default |
|---|---|---|
| PolyBench location | `POLYBENCH_DIR` (env var or `config/experiment.env`) | `~/PolyBenchC-4.2.1` |
| Evaluations per search / seeds | `BUDGET`, `SEEDS` (or `--budget`, `--seeds`) | 20 / 5 (paper) |
| Output folder | `OUT_DIR` (or `--out`) | `results` |
| Sequence length, timing repeats, timeout, penalty, GA/TPE/GP settings | constants at the top of `run_all.py` | see paper Table 4 |
| Benchmarks and their `-D` parameters | `PROGRAMS` in `run_all.py` (copy in `config/benchmarks.json`) | paper Table 2 |

## Execution

    bash scripts/smoke_test.sh        # 1 program, 12 evals per method (minutes); output in results_smoke/
    bash scripts/run_experiments.sh   # full paper experiment, resumable (hours); output in results/
    bash scripts/analyze.sh           # summary.csv, per_program.csv, per_run.csv, 4 PNGs, statistics CSVs
    bash scripts/run_dashboard.sh     # http://localhost:8501

Equivalent direct commands: `python run_all.py smoke|run|analyze [--budget N --seeds N --out DIR]`, `python analysis/extra_stats.py --results results`.
`run` skips finished (program, method, seed) combinations, so an interrupted run continues where it stopped.
Warning: re-running the full experiment into `results/` appends to / continues the shipped data; use `--out results_new` (or `OUT_DIR=results_new`) to produce a fresh run.

## Results

The shipped `results/` folder holds the data behind every table of the manuscript (125 searches, 2,500 evaluations). `results/README.md` maps files to tables
and lists the headline numbers. `bash scripts/analyze.sh` regenerates the derived tables and graphs without LLVM.

## Manuscript

`paper/manuscript_latex.zip` contains everything needed to compile on Overleaf (upload the zip as a new project; compiler: pdfLaTeX, main file `main.tex`).
All content (text, 10 tables, 2 algorithms, 19 equations, Fig. 1 drawn in TikZ, 30 references) is self-contained - there are no external image files.
Locally: `bash scripts/build_paper.sh`.

The LaTeX source in this repository was **reconstructed from the PDF** and uses the plain `article` class (two columns) so that it compiles anywhere.
`paper/compiler_pass_ordering_original.pdf` is the original IEEE Access-formatted PDF. To get the exact journal layout, create a project from the
official IEEE Access template on Overleaf and paste in the files from `sections/`, `tables/`, `figures/` and `references.tex` (they use only standard packages).
If you still have the original Overleaf project, export it and replace `paper/` with it.

## Reproducibility notes

- The paper used one machine under WSL2 without CPU pinning or frequency control, and the CPU model was not recorded; absolute runtimes will differ on other hardware.
- Timing noise is large relative to the differences between methods (median verified speedup is 0.999-1.001 for all methods).
- `proposed_nowarm` and the 20 evaluations / 5 seeds defaults were added to `run_all.py` to match the paper's protocol; the end-to-end LLVM run of this revised script has not been re-executed
  by the person who prepared this repository (no LLVM toolchain available there). The analysis code reproduces the shipped summary files exactly, and the search methods were exercised on a toy objective.
