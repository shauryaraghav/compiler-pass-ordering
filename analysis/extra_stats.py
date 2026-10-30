#!/usr/bin/env python3
"""
Post-hoc statistics used in the manuscript (Tables 5-10).

Reads the files written by `run_all.py run` / `run_all.py analyze` in --results and writes:
    verified_vs_search.csv      Table 5 + search-time columns (bootstrap CI, median, beat rates)
    per_benchmark_verified.csv  Table 7 (mean +- std over seeds, per benchmark)
    failure_rate.csv            Table 8 (failed evaluations)
    tuning_time_breakdown.csv   Table 8 (compile / execute / other seconds per search)
    paired_tests.csv            Table 9 (proposed vs each other method, Wilcoxon)
    paired_cost_tests.csv       Table 10 (exploratory search-cost comparisons)

Usage:  python analysis/extra_stats.py [--results results] [--boot 5000] [--seed 0]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ORDER = ["random", "ga", "bo_tpe", "proposed", "proposed_nowarm"]
RUNS_PER_TIMING = 3          # search-time timing repeats (RUNS in run_all.py)


def gm(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def boot_ci(x, n, rng):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), size=(n, len(x)))
    g = np.exp(np.log(x)[idx].mean(axis=1))
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def paired(a, b):
    """Two-sided Wilcoxon signed-rank on matched (program, seed) pairs; returns (median diff, p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.all(d == 0):
        return 0.0, 1.0
    return float(np.median(d)), float(wilcoxon(a, b, zero_method="wilcox", method="auto").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R = a.results
    res = pd.read_csv(f"{R}/results.csv")
    runs = pd.read_csv(f"{R}/per_run.csv")
    base = pd.read_csv(f"{R}/baseline.csv").set_index("program")
    methods = [m for m in ORDER if m in set(runs.method)]
    rng = np.random.default_rng(a.seed)
    budget = int(res.eval_idx.max())

    # ---- verified vs search-time summary (Table 5, 6)
    rows = []
    for m in methods:
        r = runs[runs.method == m]
        sp = r.speedup.values
        lo, hi = boot_ci(sp, a.boot, rng)
        g = res[res.method == m].copy()
        g["sp"] = g.program.map(base.runtime) / g.runtime
        best = g.groupby(["program", "seed"]).sp.max()
        rows.append(dict(method=m, runs=len(r), VERIFIED_geomean=gm(sp), VERIFIED_ci_lo=lo,
                         VERIFIED_ci_hi=hi, VERIFIED_median=float(np.nanmedian(sp)),
                         VERIFIED_beat_O3_pct=100 * float(np.mean(sp > 1)),
                         SEARCH_geomean_best=gm(best.values),
                         SEARCH_beat_O3_pct=100 * float(r.reached.mean()),
                         valid_pct=100 * float(r.valid.mean())))
    pd.DataFrame(rows).to_csv(f"{R}/verified_vs_search.csv", index=False)

    # ---- per-benchmark mean +- std over seeds (Table 7)
    pv = runs.pivot_table(index="program", columns="method", values="speedup", aggfunc=["mean", "std"])
    tab = pd.DataFrame({m: [f"{pv['mean'][m][p]:.3f} +- {pv['std'][m][p]:.3f}" for p in pv.index]
                        for m in methods}, index=pv.index)
    tab.reset_index().to_csv(f"{R}/per_benchmark_verified.csv", index=False)

    # ---- failed builds + tuning time breakdown (Table 8)
    fr, tb = [], []
    for m in methods:
        g = res[res.method == m]
        per_search = g.groupby(["program", "seed"])
        failed = per_search.failed.sum().mean()
        fr.append(dict(method=m, failed_evals_per_search=failed, failure_rate_pct=100 * failed / budget))
        ok = g[g.failed == 0]
        compile_s = ok.groupby(["program", "seed"]).compile_time.sum().reindex(per_search.size().index, fill_value=0).mean()
        exec_s = (ok.runtime * RUNS_PER_TIMING).groupby([ok.program, ok.seed]).sum().reindex(per_search.size().index, fill_value=0).mean()
        total = per_search.elapsed.max().mean()
        other = total - compile_s - exec_s
        tb.append(dict(method=m, total=total, compile=compile_s, execute=exec_s, other=other,
                       other_pct=100 * other / total))
    pd.DataFrame(fr).to_csv(f"{R}/failure_rate.csv", index=False)
    pd.DataFrame(tb).to_csv(f"{R}/tuning_time_breakdown.csv", index=False)

    # ---- paired tests (Table 9, Table 10)
    key = ["program", "seed"]
    wide = lambda col: runs.pivot_table(index=key, columns="method", values=col)
    sp_w, ev_w, fl_w, tt_w = wide("speedup"), wide("evals_to_beat"), wide("failed_evals"), wide("tuning_time")
    out = []
    for m in [x for x in methods if x != "proposed"]:
        d, p = paired(sp_w["proposed"], sp_w[m])
        out.append(dict(comparison=f"proposed vs {m}", pairs=len(sp_w), median_diff=d, wilcoxon_p=p))
    pd.DataFrame(out).to_csv(f"{R}/paired_tests.csv", index=False)

    cost = []
    pairs = [("proposed", m) for m in ("random", "ga", "bo_tpe")] + \
            [("proposed_nowarm", m) for m in ("random", "ga", "bo_tpe")]
    for x, y in pairs:
        if x not in methods or y not in methods:
            continue
        row = dict(comparison=f"{x} vs {y}")
        for name, w in (("evals_to_beat", ev_w), ("failed_builds", fl_w), ("tuning_time_s", tt_w)):
            d, p = paired(w[x], w[y])
            row[f"{name}_median_diff"], row[f"{name}_p"] = d, p
        cost.append(row)
    pd.DataFrame(cost).to_csv(f"{R}/paired_cost_tests.csv", index=False)
    print(f"Wrote 6 CSV files to {R}/")


if __name__ == "__main__":
    main()
