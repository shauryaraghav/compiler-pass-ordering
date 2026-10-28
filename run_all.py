#!/usr/bin/env python3
"""
Compiler pass ordering experiment (LLVM 18 + PolyBench/C).

Usage (inside your venv, from your home folder):
    python run_all.py smoke              # tiny end-to-end test (1 program, 12 evals)
    python run_all.py run                # full experiment (resumable, safe to Ctrl+C)
    python run_all.py analyze            # summary table + graphs from results/

Options:  --budget N  --seeds N  --out DIR
Outputs (in --out, default ./results):
    baseline.csv   -O3 and unoptimized reference per program
    results.csv    every single evaluation (method, seed, eval_idx, runtime, ...)
    final.csv      verified best sequence per run, re-measured
    summary.csv, per_program.csv, per_run.csv, *.png   (after `analyze`)
"""
import os, sys, time, json, csv, math, random, hashlib, subprocess, argparse, zlib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
PB = os.path.expanduser("~/PolyBenchC-4.2.1")
PROGRAMS = [
    dict(name="gemm",           dir="linear-algebra/blas/gemm",    flags="-DLARGE_DATASET"),
    dict(name="jacobi-2d",      dir="stencils/jacobi-2d",          flags="-DTSTEPS=200 -DN=1300"),
    dict(name="floyd-warshall", dir="medley/floyd-warshall",       flags="-DN=900"),
    dict(name="covariance",     dir="datamining/covariance",       flags="-DM=800 -DN=900"),
    dict(name="2mm",            dir="linear-algebra/kernels/2mm",  flags="-DNI=500 -DNJ=550 -DNK=700 -DNL=750"),
]
PASSES = ["none", "sroa", "mem2reg", "early-cse", "instcombine", "simplifycfg", "gvn", "licm",
          "loop-rotate", "indvars", "loop-unroll", "loop-vectorize", "slp-vectorizer", "sccp",
          "dce", "adce", "dse", "reassociate", "jump-threading", "correlated-propagation",
          "memcpyopt", "tailcallelim", "loop-deletion", "loop-idiom", "loop-load-elim", "inline"]
L = 15
BUDGET = 50
SEEDS = 5
RUNS = 5
FINAL_RUNS = 5
PENALTY = 1e3
RUN_TIMEOUT = 30
USE_WARM = False    # proposed method: warm start from other programs' best sequences
METHODS = ["random", "ga", "bo_tpe", "proposed"]
OUT, W = "results", "results/work"


# ----------------------------------------------------------------- helpers
def sh(cmd, timeout=None):
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="ignore")[-400:])


def append_row(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(fields)
        w.writerow(row)


LOOP_PASSES = {"loop-rotate", "indvars", "loop-deletion", "loop-idiom"}
LOOP_MSSA_PASSES = {"licm"}


def to_pipeline(seq):
    """Build a correctly nested LLVM 18 pipeline string from a flat pass list."""
    ps = [p for p in seq if p != "none"]
    if not ps:
        return "verify"
    parts, cur, prev_loop = [], [], False

    def flush():
        if cur:
            parts.append("function(" + ",".join(cur) + ")")
            cur.clear()

    for p in ps:
        if p == "inline":
            flush()
            parts.append("cgscc(inline)")
            prev_loop = False
            continue
        is_loop = p in LOOP_PASSES or p in LOOP_MSSA_PASSES
        if is_loop and not prev_loop:
            cur.extend(["loop-simplify", "lcssa"])
        if p in LOOP_MSSA_PASSES:
            cur.append("loop-mssa(" + p + ")")
        elif p in LOOP_PASSES:
            cur.append("loop(" + p + ")")
        else:
            cur.append(p)
        prev_loop = is_loop
    flush()
    return ",".join(parts)


def rand_seq(rng):
    return [rng.choice(PASSES) for _ in range(L)]


def mutate(seq, rng):
    s = list(seq)
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.7:
            s[rng.randrange(L)] = rng.choice(PASSES)
        else:
            i = rng.randrange(L - 1)
            s[i], s[i + 1] = s[i + 1], s[i]
    return s


# ----------------------------------------------------------------- build / measure
_ir_cache = {}


def base_ir(prog, dump=False):
    key = (prog["name"], dump)
    if key not in _ir_cache:
        d = f"{PB}/{prog['dir']}"
        p = f"{W}/base_{prog['name']}{'_dump' if dump else ''}.ll"
        cmd = (["clang", "-O0", "-Xclang", "-disable-O0-optnone", "-emit-llvm", "-S",
                "-I", f"{PB}/utilities", "-I", d] + prog["flags"].split()
               + (["-DPOLYBENCH_DUMP_ARRAYS"] if dump else [])
               + [f"{d}/{prog['name']}.c", "-o", p])
        sh(cmd)
        _ir_cache[key] = p
    return _ir_cache[key]


def build(prog, pipeline, dump=False):
    """opt (your pass sequence) -> llc -O3 (codegen only) -> link. Returns build seconds."""
    t0 = time.perf_counter()
    ir = base_ir(prog, dump)
    sh(["opt", f"-passes={pipeline}", ir, "-S", "-o", f"{W}/opt.ll"], 120)
    sh(["llc", "-O3", "-relocation-model=pic", "-filetype=obj", f"{W}/opt.ll", "-o", f"{W}/opt.o"], 120)
    sh(["clang", f"{W}/opt.o", f"{W}/polybench.o", "-lm", "-o", f"{W}/prog"], 60)
    return time.perf_counter() - t0


def time_run(runs):
    samples = []

    for _ in range(runs):
        t = time.perf_counter()

        subprocess.run(
            [f"{W}/prog"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=RUN_TIMEOUT
        )

        samples.append(time.perf_counter() - t)

    return float(np.median(samples))


def evaluate(prog, pipeline, runs=RUNS):
    ct = build(prog, pipeline)
    size = os.path.getsize(f"{W}/prog")
    return dict(runtime=time_run(runs), size=size, compile_time=ct)


def output_hash(prog, pipeline):
    """Correctness check: hash of the program's dumped arrays."""
    build(prog, pipeline, dump=True)
    r = subprocess.run([f"{W}/prog"], capture_output=True, timeout=120, check=True)
    return hashlib.md5(r.stderr).hexdigest()


# ----------------------------------------------------------------- tracker
RES_FIELDS = ["program", "method", "seed", "eval_idx", "runtime", "size", "compile_time",
              "elapsed", "failed", "seq"]
FIN_FIELDS = ["program", "method", "seed", "valid", "rank_used", "runtime", "size",
              "compile_time", "seq"]


class Tracker:
    """Every method calls this to evaluate a sequence; it logs everything."""

    def __init__(self, prog, method, seed):
        self.prog, self.method, self.seed = prog, method, seed
        self.i, self.t0, self.hist = 0, time.perf_counter(), []

    def __call__(self, seq):
        seq = list(seq)
        failed = 0
        try:
            r = evaluate(self.prog, to_pipeline(seq), RUNS)
        except Exception:
            r = dict(runtime=PENALTY, size=0, compile_time=0.0)
            failed = 1
        self.i += 1
        self.hist.append((r["runtime"], seq))
        append_row(f"{OUT}/results.csv", RES_FIELDS,
                   [self.prog["name"], self.method, self.seed, self.i, r["runtime"], r["size"],
                    r["compile_time"], time.perf_counter() - self.t0, failed, ",".join(seq)])
        return r["runtime"]


# ----------------------------------------------------------------- methods
def run_random(f, rng, **kw):
    while f.i < BUDGET:
        f(rand_seq(rng))


def run_ga(f, rng, **kw):
    pop_size = 10
    pop = []
    for _ in range(min(pop_size, BUDGET)):
        s = rand_seq(rng)
        pop.append((f(s), s))
    while f.i < BUDGET:
        def tour():
            a, b = rng.sample(pop, 2)
            return (a if a[0] <= b[0] else b)[1]
        p1, p2 = tour(), tour()
        cut = rng.randint(1, L - 1)
        child = p1[:cut] + p2[cut:]
        child = [rng.choice(PASSES) if rng.random() < 1.0 / L else g for g in child]
        fit = f(child)
        worst = max(range(len(pop)), key=lambda i: pop[i][0])
        if fit <= pop[worst][0]:
            pop[worst] = (fit, child)


def run_tpe(f, rng, seed=0, **kw):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(trial):
        return f([trial.suggest_categorical(f"p{i}", PASSES) for i in range(L)])

    st = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
    st.optimize(obj, n_trials=BUDGET)


# ---- proposed: GP-BO with a sequence-aware kernel + warm start -----------------
def featurize(seqs):
    P = len(PASSES)
    idx = {p: i for i, p in enumerate(PASSES)}
    X = np.zeros((len(seqs), L * P))     # position-wise one-hot
    G = np.zeros((len(seqs), P * P))     # ordered adjacent pairs (captures order)
    C = np.zeros((len(seqs), P))         # bag of passes
    for a, s in enumerate(seqs):
        ids = [idx[p] for p in s]
        for pos, i in enumerate(ids):
            X[a, pos * P + i] = 1
            C[a, i] += 1
        for u, v in zip(ids, ids[1:]):
            G[a, u * P + v] = 1
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return X, G, C


def kern(F1, F2, ls):
    # exp(-(sum of (1 - PSD similarity)) / ls) = const * exp(sum sim / ls)  -> valid PSD kernel
    X1, G1, C1 = F1
    X2, G2, C2 = F2
    d = (1 - X1 @ X2.T / L) + (1 - G1 @ G2.T / (L - 1)) + (1 - C1 @ C2.T)
    return np.exp(-d / ls)


def gp_fit(F, y, ls, noise=1e-2):
    K = kern(F, F, ls) + noise * np.eye(len(y))
    Lc = np.linalg.cholesky(K)
    alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, y))
    lml = -0.5 * y @ alpha - np.log(np.diag(Lc)).sum()
    return Lc, alpha, lml


def gp_predict(F, Lc, alpha, Fc, ls):
    Ks = kern(Fc, F, ls)
    mu = Ks @ alpha
    v = np.linalg.solve(Lc, Ks.T)
    var = np.maximum(1 - (v ** 2).sum(0), 1e-9)
    return mu, np.sqrt(var)


_erf = np.vectorize(math.erf)


def run_proposed(f, rng, base_rt=1.0, warm=(), **kw):
    init = [list(w) for w in warm][:3]
    while len(init) < min(8, BUDGET):
        init.append(rand_seq(rng))
    seqs, ys = [], []
    for s in init[:BUDGET]:
        ys.append(f(s))
        seqs.append(s)
    while f.i < BUDGET:
        y = np.clip(np.log(np.array(ys) / base_rt), -2.0, 2.5)
        yn = (y - y.mean()) / (y.std() + 1e-9)
        F = featurize(seqs)
        best, best_lml = None, -1e18
        for ls in (0.25, 0.5, 1.0, 2.0):          # pick length-scale by marginal likelihood
            try:
                Lc, alpha, lml = gp_fit(F, yn, ls)
            except np.linalg.LinAlgError:
                continue
            if lml > best_lml:
                best, best_lml = (Lc, alpha, ls), lml
        cands = [rand_seq(rng) for _ in range(300)]
        for o in np.argsort(y)[:5]:               # local search around current best
            cands += [mutate(seqs[o], rng) for _ in range(40)]
        seen = {tuple(s) for s in seqs}
        cands = [c for c in cands if tuple(c) not in seen]
        if best is None or not cands:
            nxt = rand_seq(rng)
        else:
            Lc, alpha, ls = best
            mu, sd = gp_predict(F, Lc, alpha, featurize(cands), ls)
            imp = yn.min() - mu - 0.01
            z = imp / sd
            ei = imp * 0.5 * (1 + _erf(z / math.sqrt(2))) + sd * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
            nxt = cands[int(np.argmax(ei))]
        ys.append(f(nxt))
        seqs.append(nxt)


RUNNERS = dict(random=run_random, ga=run_ga, bo_tpe=run_tpe, proposed=run_proposed)


# ----------------------------------------------------------------- finalize / baselines
def finalize(tr, ref_hash):
    """Re-verify the best sequences for correctness, then re-measure the first valid one."""
    cands, seen = [], set()
    for rt, seq in sorted(tr.hist, key=lambda x: x[0]):
        if rt >= PENALTY:
            break
        k = tuple(seq)
        if k in seen:
            continue
        seen.add(k)
        cands.append(seq)
        if len(cands) >= 5:
            break
    for rank, seq in enumerate(cands):
        pipe = to_pipeline(seq)
        try:
            if output_hash(tr.prog, pipe) != ref_hash:
                continue
            r = evaluate(tr.prog, pipe, FINAL_RUNS)
        except Exception:
            continue
        return dict(valid=1, rank_used=rank, seq=",".join(seq), **r)
    return dict(valid=0, rank_used=-1, runtime=np.nan, size=np.nan, compile_time=np.nan, seq="")


def run_baselines(progs):
    path = f"{OUT}/baseline.csv"
    have = set(pd.read_csv(path).program) if os.path.exists(path) else set()
    for prog in progs:
        if prog["name"] in have:
            continue
        print(f"[baseline] {prog['name']}", flush=True)
        ref = output_hash(prog, "verify")
        r = evaluate(prog, "default<O3>", FINAL_RUNS)
        o3_ok = int(output_hash(prog, "default<O3>") == ref)
        noopt = evaluate(prog, "verify", 3)["runtime"]
        append_row(path, ["program", "runtime", "size", "compile_time", "runtime_noopt",
                          "ref_hash", "o3_valid"],
                   [prog["name"], r["runtime"], r["size"], r["compile_time"], noopt, ref, o3_ok])


# ----------------------------------------------------------------- setup
def setup():
    os.makedirs(W, exist_ok=True)
    sh(["clang", "-c", "-O2", "-I", f"{PB}/utilities", f"{PB}/utilities/polybench.c",
        "-o", f"{W}/polybench.o"])
    global PASSES
    pf = f"{OUT}/passes_ok.json"
    if os.path.exists(pf):
        PASSES = json.load(open(pf))
        return
    ir = base_ir(PROGRAMS[0])
    ok = ["none"]
    for p in PASSES:
        if p == "none":
            continue
        r = subprocess.run(["opt", f"-passes={p}", ir, "-disable-output"], capture_output=True)
        if r.returncode == 0:
            ok.append(p)
        else:
            print(f"[setup] dropping pass not supported by this opt: {p}")
    PASSES = ok
    json.dump(PASSES, open(pf, "w"))
    print(f"[setup] {len(PASSES)} passes in search space")


def purge(prog, method, seed):
    p = f"{OUT}/results.csv"
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    m = (df.program == prog["name"]) & (df.method == method) & (df.seed == seed)
    if m.any():
        df[~m].to_csv(p, index=False)


def cmd_run(a):
    global BUDGET, SEEDS, FINAL_RUNS
    BUDGET, SEEDS = a.budget, a.seeds
    progs = PROGRAMS
    if a.cmd == "smoke":
        BUDGET, SEEDS, FINAL_RUNS, progs = 12, 1, 3, PROGRAMS[:1]
    os.makedirs(OUT, exist_ok=True)
    setup()
    run_baselines(progs)
    base = pd.read_csv(f"{OUT}/baseline.csv").set_index("program")
    fpath = f"{OUT}/final.csv"
    warm_path = f"{OUT}/warm.json"
    for seed in range(SEEDS):
        for prog in progs:
            for method in METHODS:
                n = prog["name"]
                if os.path.exists(fpath):
                    d = pd.read_csv(fpath)
                    if ((d.program == n) & (d.method == method) & (d.seed == seed)).any():
                        continue
                purge(prog, method, seed)
                tr = Tracker(prog, method, seed)
                rng = random.Random(zlib.crc32(f"{method}-{n}-{seed}".encode()))
                warm_db = json.load(open(warm_path)) if os.path.exists(warm_path) else {}
                warm = [s.split(",") for k, s in warm_db.items() if k != n] if USE_WARM else []
                RUNNERS[method](tr, rng, seed=seed, base_rt=float(base.loc[n, "runtime"]), warm=warm)
                fin = finalize(tr, base.loc[n, "ref_hash"])
                append_row(fpath, FIN_FIELDS, [n, method, seed, fin["valid"], fin["rank_used"],
                                               fin["runtime"], fin["size"], fin["compile_time"], fin["seq"]])
                if method == "proposed" and fin["valid"]:
                    warm_db[n] = fin["seq"]
                    json.dump(warm_db, open(warm_path, "w"))
                sp = base.loc[n, "runtime"] / fin["runtime"] if fin["valid"] else float("nan")
                print(f"[{n:14s} {method:9s} seed{seed}] speedup vs O3 = {sp:.3f}  valid={fin['valid']}  "
                      f"failed_evals={sum(1 for rt, _ in tr.hist if rt >= PENALTY)}  ({time.perf_counter() - tr.t0:.0f}s)", flush=True)
    print("done. now run:  python run_all.py analyze" + (f" --out {OUT}" if OUT != "results" else ""))


# ----------------------------------------------------------------- analysis
def gm(x):
    x = x.dropna()
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def cmd_analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = a.out
    res = pd.read_csv(f"{out}/results.csv")
    fin = pd.read_csv(f"{out}/final.csv")
    base = pd.read_csv(f"{out}/baseline.csv").set_index("program")
    B = int(res.eval_idx.max())
    rows = []
    for (p, m, s), g in res.groupby(["program", "method", "seed"]):
        f = fin[(fin.program == p) & (fin.method == m) & (fin.seed == s)]
        if f.empty:
            continue
        f, b = f.iloc[0], base.loc[p]
        beat = g[g.runtime < b["runtime"]]
        ok = bool(f["valid"])
        rows.append(dict(
            program=p,
            method=m,
            seed=s,
            speedup=b["runtime"] / f["runtime"] if ok else np.nan,
            size_ratio=f["size"] / b["size"] if ok else np.nan,
            compile_ratio=f["compile_time"] / b["compile_time"] if ok else np.nan,
            evals_to_beat=int(beat.eval_idx.min()) if len(beat) else B,
            reached=int(len(beat) > 0),
            tuning_time=g.elapsed.max(),
            valid=int(ok),
            failed_evals=int(g.failed.sum()),
            failure_rate=float(g.failed.mean())
        ))
    runs = pd.DataFrame(rows)
    runs.to_csv(f"{out}/per_run.csv", index=False)

    pp = runs.groupby(["method", "program"]).agg(
        speedup=("speedup", "mean"), speedup_std=("speedup", "std"),
        size_ratio=("size_ratio", "mean"), compile_ratio=("compile_ratio", "mean"),
        evals_to_beat=("evals_to_beat", "mean"), tuning_time=("tuning_time", "mean"),
        reached=("reached", "mean"), valid=("valid", "mean"),
        failed_evals=("failed_evals", "mean")).reset_index()
    pp.to_csv(f"{out}/per_program.csv", index=False)

        summ = runs.groupby("method").agg(
        speedup_vs_O3=("speedup", gm),
        evals_to_beat_O3=("evals_to_beat", "mean"),
        tuning_time_s=("tuning_time", "mean"),
        compile_time_vs_O3=("compile_ratio", gm),
        reached_O3_rate=("reached", "mean"),
        valid_rate=("valid", "mean"),
        avg_failed_evals=("failed_evals", "mean"),
        failure_rate=("failure_rate", "mean")
    )

    summ["robustness_std"] = (
        pp.groupby("method")["speedup_std"].mean()
    )
    order = [m for m in METHODS if m in summ.index]
    summ = summ.loc[order]
    summ.to_csv(f"{out}/summary.csv")
    print(summ.round(3).to_string())

    methods = order
    # 1) convergence
    res = res.sort_values(["program", "method", "seed", "eval_idx"]).copy()
    groups = ["program", "method", "seed"]

    res["sp"] = res.apply(
        lambda r: base.loc[r.program, "runtime"] / r.runtime,
        axis=1
    )
    res.loc[res["failed"].astype(bool), "sp"] = np.nan

    res["best_sp"] = res.groupby(groups)["sp"].cummax()
    res["best_sp"] = res.groupby(groups)["best_sp"].ffill()
    plt.figure(figsize=(7, 4.5))
    for m in methods:
        g = res[res.method == m].groupby("eval_idx")["best_sp"].agg(
            ["mean", "std"]
        )
        g["std"] = g["std"].fillna(0)
        plt.plot(g.index, g["mean"], label=m)
        plt.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
    plt.axhline(1.0, color="k", ls="--", lw=0.8, label="-O3")
    plt.xlabel("Evaluations"); plt.ylabel("Best speedup over -O3 so far"); plt.legend()
    plt.title("Convergence (mean over programs and seeds)"); plt.tight_layout()
    plt.savefig(f"{out}/convergence.png", dpi=150); plt.close()

    # 2) per-program speedup
    pv = pp.pivot(index="program", columns="method", values="speedup")[methods]
    sd = pp.pivot(index="program", columns="method", values="speedup_std")[methods]
    ax = pv.plot.bar(yerr=sd, figsize=(8, 4.5), capsize=2)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Speedup over -O3"); ax.set_title("Per-program speedup (mean ± std over seeds)")
    plt.tight_layout(); plt.savefig(f"{out}/per_program_speedup.png", dpi=150); plt.close()

    # 3) robustness boxplot
    plt.figure(figsize=(6, 4.5))
    plt.boxplot([runs[runs.method == m].speedup.dropna() for m in methods])
    plt.xticks(range(1, len(methods) + 1), methods)
    plt.axhline(1.0, color="k", ls="--", lw=0.8)
    plt.ylabel("Speedup over -O3 (all programs, all seeds)"); plt.title("Spread across runs")
    plt.tight_layout(); plt.savefig(f"{out}/speedup_boxplot.png", dpi=150); plt.close()

    # 4) six-metric summary
    cols = [
        ("speedup_vs_O3", "1. Geometric-mean speedup (higher better)"),
        ("evals_to_beat_O3", "2. Evaluations to beat -O3 (lower better)"),
        ("tuning_time_s", "3. Tuning time in seconds (lower better)"),
        ("compile_time_vs_O3", "4. Compile time vs -O3 (lower better)"),
        ("robustness_std", "5. Mean per-program speedup SD (lower better)"),
        ("failure_rate", "6. Failed evaluation rate (lower better)")
    ]

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))

    for ax, (column, title) in zip(axs.ravel(), cols):
        ax.bar(methods, summ[column])
        ax.set_title(title, fontsize=9)

        if column in {"speedup_vs_O3", "compile_time_vs_O3"}:
            ax.axhline(1.0, color="k", ls="--", lw=0.8)

    plt.tight_layout()
    plt.savefig(f"{out}/summary_six_params.png", dpi=150)
    plt.close()
    print(f"\nSaved summary.csv, per_program.csv, per_run.csv and 4 PNG graphs in {out}/")


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run", "analyze"])
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    OUT = a.out or ("results_smoke" if a.cmd == "smoke" else "results")
    W = f"{OUT}/work"
    a.out = OUT
    if a.cmd == "analyze":
        cmd_analyze(a)
    else:
        cmd_run(a)
