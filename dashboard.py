"""Dashboard for the pass-ordering experiment.   Run:  streamlit run dashboard.py

Reads the files that `python run_all.py run` and `python run_all.py analyze` write
into results/. Every section uses only measured data. Sections whose file is missing
say which file to copy in.

    summary.csv      six parameters per method               (needed)
    results.csv      every evaluation                         convergence, log, runtimes
    baseline.csv     -O3 runtime, size, compile time          speedup curves, -O3 bars
    final.csv        verified, re-timed best sequence per run best sequences, runtimes
    per_run.csv      metrics per (program, method, seed)      robustness box plot
    per_program.csv  metrics per (program, method)           per-program metrics
    passes_ok.json   the pass list the search used            search-space section
    warm.json        warm-start sequences                     warm-start section
    *.png            the 4 graphs from `run_all.py analyze`   original-graphs section
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Pass ordering results", page_icon="⚙️", layout="wide")

PENALTY = 1e3
METHODS = ["random", "ga", "bo_tpe", "proposed"]
LABEL = {"random": "Random search", "ga": "Genetic algorithm",
         "bo_tpe": "Vanilla BO (TPE)", "proposed": "Proposed (sequence-kernel BO)"}
SHORT = {"random": "Random", "ga": "GA", "bo_tpe": "TPE", "proposed": "Proposed"}
INK, SLATE, TEAL = "#1D2B33", "#5B6B73", "#0E7C66"
COLOR = {"random": "#9AA5AB", "ga": "#7E68A8", "bo_tpe": "#B9822A", "proposed": TEAL}
LOOP = {"licm", "loop-rotate", "indvars", "loop-unroll", "loop-vectorize", "loop-deletion",
        "loop-idiom", "loop-load-elim", "simple-loop-unswitch"}
IPO = {"inline", "globalopt", "globaldce", "ipsccp", "deadargelim", "function-attrs"}
PLOT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
            font=dict(family="IBM Plex Sans, sans-serif", color=INK))
GRID = "#E7EBE8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
h1 {{ font-weight: 600; letter-spacing: -0.02em; font-size: 2.0rem !important; margin-bottom: 0 !important; }}
h3 {{ font-weight: 600; font-size: 1.15rem !important; margin-top: 1.4rem !important; }}
.sub {{ color: {SLATE}; margin-top: .2rem; font-size: .95rem; max-width: 70ch; }}
[data-testid="stMetricValue"] {{ font-weight: 500; font-variant-numeric: tabular-nums; }}
.pipe {{ display: flex; flex-wrap: wrap; gap: 6px 0; align-items: center; margin: .4rem 0 .6rem; }}
.slot {{ display: inline-flex; align-items: baseline; gap: 6px; padding: 4px 9px; background: white;
        border: 1px solid #C9D3CE; border-left: 3px solid {TEAL};
        font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .8rem; color: {INK}; }}
.slot.loop {{ border-left-color: #B9822A; }}
.slot.ipo {{ border-left-color: #7E68A8; }}
.slot i {{ font-style: normal; color: {SLATE}; font-size: .7rem; }}
.arrow {{ color: #A8B3AE; padding: 0 4px; }}
.legend span {{ margin-right: 14px; font-size: .8rem; color: {SLATE}; }}
.legend b {{ display: inline-block; width: 10px; height: 10px; margin-right: 5px; vertical-align: -1px; }}
.missing {{ color: {SLATE}; font-size: .9rem; border-left: 3px solid #C9D3CE; padding: .3rem .8rem; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
def read(folder, name):
    p = os.path.join(folder, name)
    return pd.read_csv(p) if os.path.exists(p) else None


PNGS = ["convergence.png", "per_program_speedup.png", "speedup_boxplot.png", "summary_six_params.png"]
CSVS = ["summary", "results", "baseline", "final", "per_run", "per_program"]
JSONS = ["passes_ok", "warm"]


def read_json(folder, name):
    p = os.path.join(folder, name)
    return json.load(open(p)) if os.path.exists(p) else None


@st.cache_data(ttl=5)
def load(folder):
    d = {n: read(folder, f"{n}.csv") for n in CSVS}
    d.update({n: read_json(folder, f"{n}.json") for n in JSONS})
    if d["summary"] is not None:
        d["summary"] = d["summary"].set_index(d["summary"].columns[0])
    return d


def missing(file, what):
    st.markdown(f'<div class="missing">Copy <code>results/{file}</code> from the machine that ran '
                f'the experiment into this folder to show {what}.</div>', unsafe_allow_html=True)


def geomean(x):
    x = pd.Series(x).dropna()
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


with st.sidebar:
    folder = st.text_input("Results folder", "results")
D = load(folder)
S, R, B, F, PR, PP = (D[k] for k in ("summary", "results", "baseline", "final", "per_run", "per_program"))
PASSES, WARM = D["passes_ok"], D["warm"]

st.title("Compiler pass ordering with Bayesian optimization")
if S is None:
    st.error(f"`{folder}/summary.csv` not found. Run `python run_all.py analyze`, or set the "
             "folder in the sidebar to where the results are.")
    st.stop()

if PR is not None:      # beat -O3 judged on the verified, re-timed binary (not the noisy search timing)
    S["verified_beat_rate"] = PR.groupby("method")["speedup"].apply(lambda x: (x > 1).mean())
methods = [m for m in METHODS if m in S.index]
programs = sorted(R["program"].unique()) if R is not None else []
budget = int(R["eval_idx"].max()) if R is not None else None
n_seeds = R["seed"].nunique() if R is not None else None
setup = (f"{len(programs)} PolyBench programs, {len(methods)} search methods, "
         f"{n_seeds} seeds, {budget} evaluations per search" if R is not None
         else f"{len(methods)} search methods")
st.markdown(f'<div class="sub">Each method searched for a 15-pass LLVM 18 sequence per program. '
            f'Measured results: {setup}. Every value below comes from the files in '
            f'<code>{folder}/</code>.</div>', unsafe_allow_html=True)

with st.sidebar:
    shown = st.multiselect("Methods", methods, default=methods, format_func=LABEL.get)
    have_png = [n for n in PNGS if os.path.exists(os.path.join(folder, n))]
    st.markdown("**Result files**")
    st.caption("  \n".join([("✅ " if D[k] is not None else "❌ ") + f"{k}.csv" for k in CSVS]
                          + [("✅ " if D[k] is not None else "❌ ") + f"{k}.json" for k in JSONS]
                          + [("✅ " if n in have_png else "❌ ") + n for n in PNGS]))
shown = shown or methods

# ----------------------------------------------------------------------------- headline
P = "proposed" if "proposed" in S.index else methods[-1]
others = S.drop(index=P)
best_other = others["speedup_vs_O3"].idxmax()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Proposed: speedup over -O3", f"{S.loc[P, 'speedup_vs_O3']:.3f}×",
          f"{S.loc[P, 'speedup_vs_O3'] - S.loc[best_other, 'speedup_vs_O3']:+.3f} vs {SHORT[best_other]}")
c2.metric("Beat -O3 during search", f"{S.loc[P, 'reached_O3_rate']:.0%}",
          f"{S.loc[P, 'reached_O3_rate'] - others['reached_O3_rate'].max():+.0%} vs next best")
c3.metric("Evaluations to beat -O3", f"{S.loc[P, 'evals_to_beat_O3']:.1f}",
          f"{S.loc[P, 'evals_to_beat_O3'] - others['evals_to_beat_O3'].min():+.1f} vs next best",
          delta_color="inverse")
c4.metric("Tuning time per search", f"{S.loc[P, 'tuning_time_s']:.0f} s",
          f"{S.loc[P, 'tuning_time_s'] - others['tuning_time_s'].min():+.0f} s vs next best",
          delta_color="inverse")
if "verified_beat_rate" in S:
    c5.metric("Beat -O3, verified", f"{S.loc[P, 'verified_beat_rate']:.0%}",
              f"{S.loc[P, 'verified_beat_rate'] - others['verified_beat_rate'].max():+.0%} vs next best")
else:
    c5.caption("Beat -O3 (verified) needs per_run.csv")

# ----------------------------------------------------------------------------- six parameters
st.subheader("The six parameters")
PANELS = [("speedup_vs_O3", "1. Runtime speedup over -O3", "higher is better", "{:.3f}×", 1.0),
          ("code_size_vs_O3", "2. Code size vs -O3", "lower is better", "{:.4f}×", 1.0),
          ("evals_to_beat_O3", "3. Evaluations to beat -O3", "lower is better", "{:.1f}", None),
          ("tuning_time_s", "4. Total tuning time (s)", "lower is better", "{:.1f}", None),
          ("compile_time_vs_O3", "5. Compile time vs -O3", "lower is better", "{:.3f}×", 1.0),
          ("robustness_std", "6. Std dev of speedup across seeds", "lower is better", "{:.3f}", None)]
order = [m for m in methods if m in shown]
grid = make_subplots(rows=2, cols=3, vertical_spacing=0.24, horizontal_spacing=0.07,
                     subplot_titles=[f"{t}<br><sup>{d}</sup>" for _, t, d, _, _ in PANELS])
for k, (col, _, _, f, ref) in enumerate(PANELS):
    r, c = k // 3 + 1, k % 3 + 1
    vals = [float(S.loc[m, col]) for m in order]
    grid.add_trace(go.Bar(x=[SHORT[m] for m in order], y=vals, marker_color=[COLOR[m] for m in order],
                          text=[f.format(v) for v in vals], textposition="outside",
                          cliponaxis=False, showlegend=False,
                          hovertemplate="%{x}: %{text}<extra></extra>"), row=r, col=c)
    if ref is not None:
        grid.add_hline(y=ref, line=dict(color=INK, dash="dash", width=1), row=r, col=c)
    top = max(vals + ([ref] if ref else []))
    grid.update_yaxes(gridcolor=GRID, range=[0, top * 1.2], row=r, col=c)
grid.update_layout(height=640, margin=dict(l=10, r=10, t=70, b=10), **PLOT)
for a in grid.layout.annotations:
    a.font.size = 13
    a.yshift = 10
st.plotly_chart(grid, width="stretch")
st.caption("Dashed line is -O3 where the value is a ratio to -O3. Speedup, code size and compile time "
           "are geometric means over programs; the rest are means over searches.")

# the three summary columns the six-parameter figure leaves out
EXTRA = [("reached_O3_rate", "Searches that beat -O3 (search timing)", "higher is better", "{:.0%}"),
         ("verified_beat_rate", "Searches that beat -O3 (verified)", "higher is better", "{:.0%}"),
         ("valid_rate", "Valid final binaries", "higher is better", "{:.0%}"),
         ("avg_failed_evals", "Failed builds per search", "lower is better", "{:.1f}")]
EXTRA = [e for e in EXTRA if e[0] in S.columns]
st.subheader("Beat rate, validity and failed builds")
ex = make_subplots(rows=1, cols=len(EXTRA), horizontal_spacing=0.07,
                   subplot_titles=[f"{t}<br><sup>{d}</sup>" for _, t, d, _ in EXTRA])
for k, (col, _, _, f) in enumerate(EXTRA, 1):
    vals = [float(S.loc[m, col]) for m in order]
    ex.add_trace(go.Bar(x=[SHORT[m] for m in order], y=vals, marker_color=[COLOR[m] for m in order],
                        text=[f.format(v) for v in vals], textposition="outside", cliponaxis=False,
                        showlegend=False, hovertemplate="%{x}: %{text}<extra></extra>"), row=1, col=k)
    ex.update_yaxes(gridcolor=GRID, range=[0, max(max(vals) * 1.25, 1e-9)], row=1, col=k)
ex.update_layout(height=340, margin=dict(l=10, r=10, t=80, b=10), **PLOT)
for a in ex.layout.annotations:
    a.font.size = 12
st.plotly_chart(ex, width="stretch")
st.caption("'Search timing' counts a search as beating -O3 if any evaluation (min of 3 runs) was faster than "
           "the -O3 binary. 'Verified' asks whether the final, correctness-checked binary re-timed over 10 "
           "runs was faster. Every final binary is valid here, so the valid-rate bars are all 100%.")

# table
st.subheader("Comparison table")
COLS = [("speedup_vs_O3", "Speedup vs -O3", "higher", "{:.3f}×"),
        ("code_size_vs_O3", "Code size vs -O3", "lower", "{:.4f}×"),
        ("evals_to_beat_O3", "Evals to beat -O3", "lower", "{:.1f}"),
        ("tuning_time_s", "Tuning time (s)", "lower", "{:.1f}"),
        ("compile_time_vs_O3", "Compile time vs -O3", "lower", "{:.3f}×"),
        ("robustness_std", "Robustness (std)", "lower", "{:.3f}"),
        ("reached_O3_rate", "Beat -O3 (search)", "higher", "{:.0%}"),
        ("verified_beat_rate", "Beat -O3 (verified)", "higher", "{:.0%}"),
        ("valid_rate", "Valid rate", "higher", "{:.0%}"),
        ("avg_failed_evals", "Failed evals", "lower", "{:.1f}")]
cols = [c for c in COLS if c[0] in S.columns]
num = S.loc[order, [c[0] for c in cols]].astype(float)
num.index = [LABEL[m] for m in order]
O3REF = {"speedup_vs_O3": 1.0, "code_size_vs_O3": 1.0, "evals_to_beat_O3": np.nan, "tuning_time_s": 0.0,
         "compile_time_vs_O3": 1.0, "robustness_std": np.nan, "reached_O3_rate": np.nan,
         "verified_beat_rate": np.nan, "valid_rate": 1.0, "avg_failed_evals": 0.0}
o3 = pd.DataFrame([[O3REF[c] for c in num.columns]], columns=num.columns, index=["-O3 (reference)"])
num = pd.concat([o3, num])


def css(col):
    v = col.drop("-O3 (reference)")
    spread = (v.max() - v.min()) / (abs(v.mean()) + 1e-12)
    if v.dropna().nunique() <= 1 or spread < 1e-3:   # no real difference, highlight nothing
        return pd.Series("", index=col.index)
    better = dict((c[0], c[2]) for c in cols)[col.name]
    target = v.max() if better == "higher" else v.min()
    return pd.Series(["background-color:#D7ECE5;font-weight:600"
                      if (i != "-O3 (reference)" and x == target) else "" for i, x in col.items()],
                     index=col.index)


style = num.apply(css)
text = num.apply(lambda c: c.map(lambda x: "–" if pd.isna(x) else
                                 dict((k[0], k[3]) for k in cols)[c.name].format(x)))
text.columns = [c[1] for c in cols]
style.columns = text.columns
st.dataframe(text.style.apply(lambda _: style, axis=None), width="stretch")
st.caption("Shaded cells are the best search method in each column. No cell is shaded when the "
           "methods don't differ (code size and valid rate here).")

# ----------------------------------------------------------------------------- convergence
st.subheader("Best result found so far")
if R is None:
    missing("results.csv", "convergence curves")
else:
    R = R.sort_values(["program", "method", "seed", "eval_idx"]).copy()
    R["ok"] = R["runtime"] < PENALTY
    names = ["All programs, speedup over -O3", "One program, runtime in seconds",
             "Share of searches that beat -O3"]
    if B is None:                                   # show the view that has data first
        tab_prog, tab_all, tab_beat = st.tabs([names[1], names[0], names[2]])
    else:
        tab_all, tab_prog, tab_beat = st.tabs(names)
    with tab_all:
        if B is None:
            missing("baseline.csv", "speedup curves (the -O3 runtime of each program)")
        else:
            o3 = B.set_index("program")["runtime"]
            R["sp"] = np.where(R["ok"], R["program"].map(o3) / R["runtime"], np.nan)
            R["best_sp"] = R.groupby(["program", "method", "seed"])["sp"].cummax()
            fig = go.Figure()
            for m in order:
                g = R[R["method"] == m]
                per_seed = g.groupby(["seed", "eval_idx"])["best_sp"].apply(geomean).unstack(0)
                x = per_seed.index.values
                lo, hi, mean = per_seed.min(axis=1), per_seed.max(axis=1), per_seed.mean(axis=1)
                fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                         fillcolor=COLOR[m], opacity=.12, line=dict(width=0),
                                         hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=x, y=mean, name=LABEL[m],
                                         line=dict(color=COLOR[m], width=3 if m == P else 2),
                                         hovertemplate="eval %{x}: %{y:.3f}×<extra>" + LABEL[m] + "</extra>"))
            fig.add_hline(y=1, line=dict(color=INK, dash="dash", width=1),
                          annotation_text="-O3", annotation_position="top left")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis=dict(title="Evaluation", gridcolor=GRID),
                              yaxis=dict(title="Best speedup over -O3 so far", gridcolor=GRID),
                              legend=dict(orientation="h", y=1.1, x=0), **PLOT)
            st.plotly_chart(fig, width="stretch")
            st.caption("Uses the search-time measurement (min of 3 runs). Line is the mean over seeds of "
                       "the geometric mean over programs; band is the range over seeds.")
    with tab_prog:
        prog = st.selectbox("Program", programs, key="conv_prog")
        g = R[(R["program"] == prog)].copy()
        g["rt"] = g["runtime"].where(g["ok"])
        g["best_rt"] = g.groupby(["method", "seed"])["rt"].cummin()
        fig = go.Figure()
        for m in order:
            c = g[g["method"] == m].groupby("eval_idx")["best_rt"].agg(["mean", "min", "max"])
            x = c.index.values
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[c["max"], c["min"][::-1]],
                                     fill="toself", fillcolor=COLOR[m], opacity=.12,
                                     line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=c["mean"], name=LABEL[m],
                                     line=dict(color=COLOR[m], width=3 if m == P else 2),
                                     hovertemplate="eval %{x}: %{y:.3f} s<extra>" + LABEL[m] + "</extra>"))
        if B is not None and prog in set(B["program"]):
            fig.add_hline(y=float(B.set_index("program").loc[prog, "runtime"]),
                          line=dict(color=INK, dash="dash", width=1),
                          annotation_text="-O3", annotation_position="top left")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="Evaluation", gridcolor=GRID),
                          yaxis=dict(title="Fastest runtime so far (s), lower is better", gridcolor=GRID),
                          legend=dict(orientation="h", y=1.1, x=0), **PLOT)
        st.plotly_chart(fig, width="stretch")
        st.caption("Mean over seeds, band is the range. Failed builds are left out.")

    with tab_beat:
        if B is None:
            missing("baseline.csv", "the share of searches that beat -O3")
        else:
            o3r = B.set_index("program")["runtime"]
            hit = R[R["ok"] & (R["runtime"] < R["program"].map(o3r))]
            first = hit.groupby(["method", "program", "seed"])["eval_idx"].min()
            runs_per = R.drop_duplicates(["method", "program", "seed"]).groupby("method").size()
            fig = go.Figure()
            for m in order:
                f = first[m] if m in first.index.get_level_values(0) else pd.Series(dtype=float)
                xs = np.arange(1, budget + 1)
                share = [(f <= k).sum() / runs_per[m] for k in xs]
                fig.add_trace(go.Scatter(x=xs, y=share, name=LABEL[m], mode="lines",
                                         line=dict(color=COLOR[m], width=3 if m == P else 2, shape="hv"),
                                         hovertemplate="by eval %{x}: %{y:.0%}<extra>" + LABEL[m] + "</extra>"))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis=dict(title="Evaluation", gridcolor=GRID),
                              yaxis=dict(title="Share of searches that have beaten -O3", tickformat=".0%",
                                         range=[0, 1], gridcolor=GRID),
                              legend=dict(orientation="h", y=1.1, x=0), **PLOT)
            st.plotly_chart(fig, width="stretch")
            st.caption("A search counts once any evaluation (min of 3 runs) is faster than the -O3 binary. "
                       "This is the curve behind the 'evaluations to beat -O3' number; searches that never "
                       "beat -O3 are scored as the full budget in that average.")

# ----------------------------------------------------------------------------- per program + robustness
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Speedup by program")
    if PP is None:
        missing("per_program.csv", "speedup per program")
    else:
        fig = go.Figure()
        for m in order:
            g = PP[PP["method"] == m].set_index("program").reindex(sorted(PP["program"].unique()))
            fig.add_trace(go.Bar(y=g.index, x=g["speedup"], name=SHORT[m], orientation="h",
                                 marker_color=COLOR[m],
                                 error_x=dict(type="data", array=g["speedup_std"].fillna(0),
                                              thickness=1, width=2, color=SLATE),
                                 hovertemplate="%{y}: %{x:.3f}×<extra>" + LABEL[m] + "</extra>"))
        fig.add_vline(x=1, line=dict(color=INK, dash="dash", width=1))
        fig.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="Speedup over -O3 (verified, mean over seeds)", gridcolor=GRID),
                          legend=dict(orientation="h", y=1.08, x=0), **PLOT)
        st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Spread across runs")
    if PR is None:
        missing("per_run.csv", "the robustness box plot")
    else:
        fig = go.Figure()
        for m in order:
            g = PR[PR["method"] == m]
            fig.add_trace(go.Box(y=g["speedup"], name=SHORT[m], marker_color=COLOR[m],
                                 boxpoints="all", jitter=.4, pointpos=0,
                                 text=g["program"] + ", seed " + g["seed"].astype(str),
                                 hovertemplate="%{text}: %{y:.3f}×<extra></extra>"))
        fig.add_hline(y=1, line=dict(color=INK, dash="dash", width=1),
                      annotation_text="-O3", annotation_position="top left")
        fig.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(title="Verified speedup over -O3", gridcolor=GRID), **PLOT)
        st.plotly_chart(fig, width="stretch")
        st.caption("Each dot is one search (one program, one seed).")

# ----------------------------------------------------------------------------- runtime chart
st.subheader("Runtime of the best binary")
if F is not None:
    src = F[F["valid"] == 1].groupby(["program", "method"])["runtime"].mean().unstack()
    note = "Verified best sequence of each search, re-timed (min of 10 runs), mean over seeds."
elif R is not None:
    ok = R[R["runtime"] < PENALTY]
    src = ok.groupby(["program", "method", "seed"])["runtime"].min().groupby(["program", "method"]).mean().unstack()
    note = ("Fastest runtime seen during each search (min of 3 runs, not yet re-verified), mean over "
            "seeds. Copy final.csv to switch to the verified, re-timed values.")
else:
    src = None
if src is None:
    missing("results.csv", "runtimes")
else:
    fig = go.Figure()
    if B is not None:
        b = B.set_index("program").reindex(src.index)
        fig.add_trace(go.Bar(x=src.index, y=b["runtime"], name="-O3", marker_color=INK))
    for m in order:
        if m in src:
            fig.add_trace(go.Bar(x=src.index, y=src[m], name=LABEL[m], marker_color=COLOR[m]))
    fig.update_layout(barmode="group", height=400, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(title="Runtime (s), lower is better", gridcolor=GRID),
                      legend=dict(orientation="h", y=1.12, x=0), **PLOT)
    fig.update_traces(hovertemplate="%{x}: %{y:.3f} s<extra>%{fullData.name}</extra>")
    st.plotly_chart(fig, width="stretch")
    st.caption(note + ("" if B is not None else " The -O3 bars appear once baseline.csv is copied in."))

# ----------------------------------------------------------------------------- best sequences
st.subheader("Best pass sequence found")


def chips(seq):
    parts = []
    for k, n in enumerate(seq.split(","), 1):
        if n == "none":
            continue
        cls = "loop" if n in LOOP else "ipo" if n in IPO else ""
        parts.append(f'<span class="slot {cls}"><i>{k}</i>{n}</span>')
    return '<div class="pipe">' + '<span class="arrow">›</span>'.join(parts) + "</div>"


cA, cB = st.columns([1, 1])
with cA:
    sel_m = st.selectbox("Method", order, index=order.index(P) if P in order else 0,
                         format_func=LABEL.get, key="seq_m")
with cB:
    opts = programs if programs else (sorted(F["program"].unique()) if F is not None else [])
    sel_p = st.selectbox("Program", opts, key="seq_p")
if F is not None:
    g = F[(F["method"] == sel_m) & (F["program"] == sel_p) & (F["valid"] == 1)]
    if g.empty:
        st.write("No verified sequence for this method and program.")
    else:
        row = g.loc[g["runtime"].idxmin()]
        st.markdown(chips(row["seq"]), unsafe_allow_html=True)
        extra = ""
        if B is not None:
            extra = f", {float(B.set_index('program').loc[sel_p, 'runtime']) / row['runtime']:.3f}× of -O3 speed"
        st.write(f"Verified output, re-timed at **{row['runtime']:.3f} s**{extra} (seed {row['seed']}).")
elif R is not None:
    g = R[(R["method"] == sel_m) & (R["program"] == sel_p) & (R["runtime"] < PENALTY)]
    row = g.loc[g["runtime"].idxmin()]
    st.markdown(chips(row["seq"]), unsafe_allow_html=True)
    st.write(f"Fastest during search: **{row['runtime']:.3f} s** (seed {row['seed']}, evaluation "
             f"{row['eval_idx']}). Not yet re-verified; final.csv has the verified version.")
st.markdown('<div class="legend"><span><b style="background:#0E7C66"></b>function pass</span>'
            '<span><b style="background:#B9822A"></b>loop pass</span>'
            '<span><b style="background:#7E68A8"></b>interprocedural</span></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------- every metric by program
st.subheader("Every metric, by program")
if PP is None:
    missing("per_program.csv", "the per-program breakdown")
else:
    PMETRIC = {"speedup": ("Speedup over -O3 (verified)", "{:.3f}×", 1.0),
               "size_ratio": ("Code size vs -O3", "{:.4f}×", 1.0),
               "compile_ratio": ("Compile time vs -O3", "{:.3f}×", 1.0),
               "evals_to_beat": ("Evaluations to beat -O3 (budget if never)", "{:.1f}", None),
               "tuning_time": ("Tuning time (s)", "{:.1f}", None),
               "reached": ("Share of searches that beat -O3 (search timing)", "{:.0%}", None),
               "failed_evals": ("Failed builds per search", "{:.1f}", None),
               "valid": ("Valid final binaries", "{:.0%}", None)}
    PMETRIC = {k: v for k, v in PMETRIC.items() if k in PP.columns}
    pm = st.selectbox("Metric", list(PMETRIC), format_func=lambda k: PMETRIC[k][0], key="pp_metric")
    ptitle, pfmt, pref = PMETRIC[pm]
    progs_pp = sorted(PP["program"].unique())
    fig = go.Figure()
    for m in order:
        g = PP[PP["method"] == m].set_index("program").reindex(progs_pp)
        fig.add_trace(go.Bar(x=progs_pp, y=g[pm], name=SHORT[m], marker_color=COLOR[m],
                             text=[pfmt.format(v) for v in g[pm]], textposition="outside", cliponaxis=False,
                             error_y=(dict(type="data", array=g["speedup_std"].fillna(0), thickness=1, width=2,
                                           color=SLATE) if pm == "speedup" else None),
                             hovertemplate="%{x}: %{text}<extra>" + LABEL[m] + "</extra>"))
    if pref is not None:
        fig.add_hline(y=pref, line=dict(color=INK, dash="dash", width=1))
    fig.update_layout(barmode="group", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(title=ptitle, gridcolor=GRID), legend=dict(orientation="h", y=1.12, x=0), **PLOT)
    st.plotly_chart(fig, width="stretch")
    st.caption("Mean over seeds. " + ("Error bars are one standard deviation over seeds. " if pm == "speedup" else "")
               + "Dashed line is -O3 where the value is a ratio to -O3.")
    h1, h2 = st.columns([3, 2], gap="large")
    with h1:
        piv = PP[PP["method"].isin(order)].pivot(index="method", columns="program", values="speedup").loc[order]
        hm = go.Figure(go.Heatmap(z=piv.values, x=list(piv.columns), y=[SHORT[m] for m in piv.index],
                                  zmin=0.65, zmax=1.0, colorscale=[[0, "#F0DDBD"], [1, TEAL]],
                                  text=[[f"{v:.3f}×" for v in row] for row in piv.values],
                                  texttemplate="%{text}", hovertemplate="%{y}, %{x}: %{text}<extra></extra>",
                                  colorbar=dict(title="speedup", thickness=10)))
        hm.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(autorange="reversed"),
                         title=dict(text="Verified speedup over -O3, method by program", font=dict(size=14)), **PLOT)
        st.plotly_chart(hm, width="stretch")
    with h2:
        best = piv.idxmax()
        st.markdown("**Best method per program**")
        st.dataframe(pd.DataFrame({"Program": piv.columns, "Best": [SHORT[m] for m in best],
                                   "Speedup": [f"{piv.loc[m, c]:.3f}×" for c, m in best.items()]}),
                     hide_index=True, width="stretch")
        st.caption("Differences under about 0.03× are within noise at 2 seeds.")
    with st.expander("Full per-program table (all columns of per_program.csv)"):
        t = PP[PP["method"].isin(order)].copy()
        t.insert(0, "method", t.pop("method").map(SHORT))
        st.dataframe(t.round(4), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- baselines
st.subheader("The -O3 and unoptimized references")
if B is None:
    missing("baseline.csv", "the -O3 and -O0 reference runtimes")
else:
    b = B.set_index("program")
    bl, br = st.columns([3, 2], gap="large")
    with bl:
        fig = go.Figure()
        if "runtime_noopt" in b:
            fig.add_trace(go.Bar(x=b.index, y=b["runtime_noopt"], name="Unoptimized (no passes)",
                                 marker_color="#C9D3CE", text=[f"{v:.2f}" for v in b["runtime_noopt"]],
                                 textposition="outside", cliponaxis=False))
        fig.add_trace(go.Bar(x=b.index, y=b["runtime"], name="-O3", marker_color=INK,
                             text=[f"{v:.2f}" for v in b["runtime"]], textposition="outside", cliponaxis=False))
        fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(title="Runtime (s)", gridcolor=GRID),
                          legend=dict(orientation="h", y=1.12, x=0), **PLOT)
        fig.update_traces(hovertemplate="%{x}: %{y:.3f} s<extra>%{fullData.name}</extra>")
        st.plotly_chart(fig, width="stretch")
    with br:
        tb = pd.DataFrame({"-O3 runtime (s)": b["runtime"].map("{:.3f}".format),
                           "Unoptimized (s)": b["runtime_noopt"].map("{:.3f}".format) if "runtime_noopt" in b else "–",
                           "-O3 gain over unopt.": (b["runtime_noopt"] / b["runtime"]).map("{:.2f}×".format)
                           if "runtime_noopt" in b else "–",
                           "-O3 size (B)": b["size"].map("{:,}".format),
                           "-O3 compile (s)": b["compile_time"].map("{:.3f}".format),
                           "-O3 output correct": b["o3_valid"].map({1: "yes", 0: "NO"}) if "o3_valid" in b else "–"})
        st.dataframe(tb, width="stretch")
    st.caption("-O3 runtime is the min of 10 runs. The 'unoptimized' build runs no IR passes (code generation is still "
               "llc -O3). Every speedup in this dashboard is measured against the -O3 column. Reference outputs are "
               "hashed, and a sequence only counts as valid if its output hash matches.")

# ----------------------------------------------------------------------------- verification
st.subheader("Search timing vs verified result")
if F is None or R is None or B is None:
    missing("final.csv", "how the verified result compares with the best timing seen during search "
                         "(also needs results.csv and baseline.csv)")
else:
    o3r = B.set_index("program")["runtime"]
    ver = F.assign(sp=F["program"].map(o3r) / F["runtime"]).where(F["valid"] == 1)
    ver = pd.concat([F[["program", "method", "seed"]], ver["sp"].rename("verified")], axis=1)
    okR = R[R["runtime"] < PENALTY]
    srch = (okR["program"].map(o3r) / okR["runtime"]).groupby([okR["program"], okR["method"], okR["seed"]]).max()
    srch = srch.rename("search").reset_index()
    cmp_ = ver.merge(srch, on=["program", "method", "seed"])
    g = cmp_[cmp_["method"].isin(order)].groupby("method")[["search", "verified"]].agg(geomean).loc[order]
    vc1, vc2 = st.columns([3, 2], gap="large")
    with vc1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[SHORT[m] for m in g.index], y=g["search"], name="Best timing during search",
                             marker_color="#C9D3CE", text=[f"{v:.3f}×" for v in g["search"]],
                             textposition="outside", cliponaxis=False))
        fig.add_trace(go.Bar(x=[SHORT[m] for m in g.index], y=g["verified"], name="Verified, re-timed",
                             marker_color=[COLOR[m] for m in g.index],
                             text=[f"{v:.3f}×" for v in g["verified"]], textposition="outside", cliponaxis=False))
        fig.add_hline(y=1, line=dict(color=INK, dash="dash", width=1))
        fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(title="Speedup over -O3 (geometric mean)", gridcolor=GRID,
                                     range=[0, 1.15]), legend=dict(orientation="h", y=1.12, x=0), **PLOT)
        st.plotly_chart(fig, width="stretch")
    with vc2:
        nvalid, nruns = int((F["valid"] == 1).sum()), len(F)
        st.metric("Final binaries passing the output check", f"{nvalid} of {nruns}")
        if "rank_used" in F:
            st.metric("Needed a fallback candidate", f"{int((F['rank_used'] > 0).sum())} of {nruns}",
                      help="The best 5 distinct sequences of each search are checked in order of speed; "
                           "rank 0 means the fastest one was already correct.")
        st.caption("Search timing is the min of 3 runs of the best of the evaluations, so it is slightly "
                   "optimistic. The verified value re-times the winning, correctness-checked sequence over 10 runs. "
                   "The convergence curves use search timing, the six-parameter figure uses verified values, "
                   "which is why the two differ by a few hundredths.")

# ----------------------------------------------------------------------------- pass usage
st.subheader("Which passes the best sequences use")
if F is None:
    missing("final.csv", "pass usage in the best sequences")
else:
    from collections import Counter
    use = {}
    for m in order:
        c = Counter()
        for sq in F[(F["method"] == m) & (F["valid"] == 1)]["seq"]:
            c.update(x for x in sq.split(",") if x != "none")
        use[m] = c
    allp = PASSES if PASSES else sorted({k for c in use.values() for k in c})
    allp = [x for x in allp if x != "none"]
    tot = {x: sum(use[m][x] for m in order) for x in allp}
    allp = sorted(allp, key=lambda x: -tot[x])
    z = [[use[m][x] for x in allp] for m in order]
    hm = go.Figure(go.Heatmap(z=z, x=allp, y=[SHORT[m] for m in order], colorscale=[[0, "#FFFFFF"], [1, TEAL]],
                              text=[[str(v) if v else "" for v in row] for row in z], texttemplate="%{text}",
                              hovertemplate="%{y}, %{x}: %{z} uses<extra></extra>",
                              colorbar=dict(title="uses", thickness=10)))
    hm.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                     xaxis=dict(tickangle=-45), **PLOT)
    st.plotly_chart(hm, width="stretch")
    nser = F[F["valid"] == 1].groupby("method").size().to_dict()
    st.caption("Count of slots using each pass, summed over the verified best sequence of every search "
               f"({', '.join(f'{SHORT[m]} {nser.get(m, 0)}' for m in order)} sequences of 15 slots; 'none' slots "
               "not counted). Columns are sorted by total use.")

# ----------------------------------------------------------------------------- search space + warm start
st.subheader("Search space and warm start")
sp1, sp2 = st.columns(2, gap="large")
with sp1:
    st.markdown("**Passes the searches could choose from**")
    if PASSES is None:
        missing("passes_ok.json", "the search space")
    else:
        chips_html = "".join(
            f'<span class="slot {"loop" if n in LOOP else "ipo" if n in IPO else ""}" style="margin:0 6px 6px 0">{n}</span>'
            for n in PASSES)
        st.markdown(f'<div style="display:flex;flex-wrap:wrap">{chips_html}</div>', unsafe_allow_html=True)
        st.caption(f"{len(PASSES)} options per slot ({len(PASSES) - 1} passes plus 'none'), 15 slots, so about "
                   f"{len(PASSES) ** 15:.1e} possible sequences against a budget of {budget or 'a few'} evaluations. "
                   "Every pass this LLVM 18 build supports was kept.")
with sp2:
    st.markdown("**Warm-start sequences (proposed method)**")
    if WARM is None:
        missing("warm.json", "the warm-start sequences")
    else:
        for prog_name, sq in WARM.items():
            st.markdown(f"<div style='font-size:.85rem;color:{SLATE}'>{prog_name}</div>"
                        + chips(sq).replace('class="pipe"', 'class="pipe" style="margin:.1rem 0 .5rem"'),
                        unsafe_allow_html=True)
        st.caption("Best verified sequence of the proposed method per program. When searching a program, the "
                   "proposed method seeds its first evaluations with up to 3 of the other programs' sequences, "
                   "never its own.")

# ----------------------------------------------------------------------------- per-seed runs
st.subheader("Every search (per seed)")
if PR is None:
    missing("per_run.csv", "the per-seed results")
else:
    q1, q2 = st.columns(2)
    qp = q1.multiselect("Programs", sorted(PR["program"].unique()), default=sorted(PR["program"].unique()), key="pr_p")
    qs = q2.multiselect("Seeds", sorted(PR["seed"].unique()), default=sorted(PR["seed"].unique()), key="pr_s")
    t = PR[PR["method"].isin(order) & PR["program"].isin(qp) & PR["seed"].isin(qs)].copy()
    t["method"] = t["method"].map(SHORT)
    t = t.rename(columns={"speedup": "speedup_x", "size_ratio": "size_x", "compile_ratio": "compile_x"})
    st.dataframe(t.round(4), hide_index=True, width="stretch", height=300)
    st.caption(f"{len(t)} of {len(PR)} searches shown. These rows are what the box plot, the per-program bars and "
               "the summary are computed from.")

# ----------------------------------------------------------------------------- original graphs
st.subheader("Graphs exported by run_all.py analyze")
gt = st.tabs(["Convergence", "Per-program speedup", "Spread across runs", "Six parameters"])
for tab, fn in zip(gt, PNGS):
    with tab:
        path = os.path.join(folder, fn)
        if os.path.exists(path):
            st.image(path, width="stretch")
        else:
            missing(fn, "this exported graph")
st.caption("The static versions of the interactive charts above. The exported convergence graph averages "
           "arithmetically over programs and counts failed builds as near-zero speedup, so it sits a little "
           "below the geometric-mean curve in the interactive version.")

# ----------------------------------------------------------------------------- findings
st.subheader("What the results show")
sp = S["speedup_vs_O3"]
vb = S["verified_beat_rate"] if "verified_beat_rate" in S else None
rr = S["reached_O3_rate"]
fe = S["avg_failed_evals"] if "avg_failed_evals" in S else None
lines = [
    f"- No method beat -O3 on average at this budget (all speedups below 1.0). -O3 is a long, hand-tuned "
    f"pipeline, and each search had only {budget or 'a few'} evaluations of a 15-slot sequence.",
    f"- The proposed method has the highest mean speedup ({sp[P]:.3f}×), but its lead over vanilla BO "
    f"({sp.get('bo_tpe', np.nan):.3f}×) is within measurement noise. Both are clearly ahead of random search "
    f"and the GA.",
]
lead = rr.idxmax()
if lead == P:
    lines.append(f"- Judged on search-time timings, the proposed method beat -O3 in the most searches ({rr[P]:.0%}) "
                 f"and needed the fewest evaluations to do so ({S.loc[P, 'evals_to_beat_O3']:.1f}).")
if vb is not None:
    lines.append(f"- Judged on the verified, re-timed binary the picture is weaker: {vb[P]:.0%} of the proposed "
                 f"method's searches beat -O3, against {vb.drop(P).max():.0%} for {SHORT[vb.drop(P).idxmax()]}. "
                 f"Every verified win is by at most {max(PR.loc[PR['speedup'] > 1, 'speedup'].max() - 1, 0):.1%}, "
                 "which is inside run-to-run noise, so treat the beat-rate numbers as indicative only.")
if fe is not None:
    lines.append(f"- The proposed method had the fewest failed builds ({fe[P]:.1f} per search), TPE the most "
                 f"({fe['bo_tpe']:.1f}).")
lines.append(f"- Random search has the smallest spread across seeds, because its results are consistently mediocre. "
             f"With {n_seeds or 'few'} seeds, the robustness column is only indicative.")
st.markdown("\n".join(lines))
with st.expander("Limitations"):
    st.markdown("""
- **Code size** was measured on the whole linked binary, which is mostly fixed overhead, so every
  method reads about 1.000. Measuring only the `.text` section would separate the methods.
- **Warm start:** only the proposed method used it. A run of the proposed method with
  `USE_WARM = False` would show how much of its gain comes from the kernel alone.
- **Scale:** 5 programs, few seeds, short searches, one machine. Treat speedup differences under
  about 0.03 as noise.
- Code generation is fixed at `llc -O3` for every build, so only the IR pass sequence varies.
""")

# ----------------------------------------------------------------------------- log
st.subheader("All evaluations")
if R is None:
    missing("results.csv", "the evaluation log")
else:
    f1, f2 = st.columns(2)
    lp = f1.multiselect("Programs", programs, default=programs, key="log_p")
    only_fail = f2.toggle("Only failed builds", value=False)
    log = R[R["method"].isin(order) & R["program"].isin(lp)].copy()
    if only_fail:
        log = log[log["failed"] == 1]
    log["Method"] = log["method"].map(SHORT)
    log["Runtime (s)"] = log["runtime"].where(log["runtime"] < PENALTY)
    log["Status"] = np.where(log["failed"] == 1, "failed", "ok")
    log = log[["Method", "program", "seed", "eval_idx", "Status", "Runtime (s)", "compile_time", "seq"]]
    log.columns = ["Method", "Program", "Seed", "Eval", "Status", "Runtime (s)", "Compile (s)", "Sequence"]
    for c in ("Runtime (s)", "Compile (s)"):
        log[c] = log[c].map(lambda v: "–" if pd.isna(v) else f"{v:.3f}")
    st.dataframe(log.style.map(lambda s: "color:#B23B3B" if s == "failed" else "", subset=["Status"]),
                 width="stretch", hide_index=True, height=380)
    st.caption(f"{len(log)} of {len(R)} evaluations shown.")
