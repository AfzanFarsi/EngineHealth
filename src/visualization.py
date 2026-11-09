"""Charts: Plotly figures for the website and Matplotlib PNGs saved to outputs/figures."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config as cfg

# ------------------------------------------------------------- palette ------
PALETTE = {
    "bg": "#0B1220", "card": "#121B2E", "grid": "#23304A", "text": "#E6EDF7", "muted": "#8FA0BC",
    "blue": "#3B82F6", "cyan": "#22D3EE", "green": "#34D399", "amber": "#FBBF24", "red": "#F87171",
    "violet": "#A78BFA",
}
SERIES = [PALETTE["cyan"], PALETTE["blue"], PALETTE["violet"], PALETTE["amber"], PALETTE["green"], PALETTE["red"]]
CATEGORY_COLOURS = {"Healthy": PALETTE["green"], "Warning": PALETTE["amber"], "Critical": PALETTE["red"]}


def _layout(fig: go.Figure, title: str = "", height: int = 380, xtitle: str = "", ytitle: str = "") -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": PALETTE["text"]}, "x": 0.01},
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"], "family": "Inter, Segoe UI, sans-serif", "size": 12},
        margin={"l": 50, "r": 20, "t": 50, "b": 45},
        legend={"orientation": "h", "y": -0.22, "font": {"size": 11}},
        xaxis={"title": xtitle, "gridcolor": PALETTE["grid"], "zerolinecolor": PALETTE["grid"]},
        yaxis={"title": ytitle, "gridcolor": PALETTE["grid"], "zerolinecolor": PALETTE["grid"]},
        hoverlabel={"bgcolor": PALETTE["card"], "font_color": PALETTE["text"]},
    )
    return fig


# ------------------------------------------------------------- plotly -------
def plotly_rul_curve(df: pd.DataFrame, pred_cols: dict[str, str], title: str, low=None, high=None,
                     marker_cycle: int | None = None) -> go.Figure:
    """True RUL (ground truth) and one or more model predictions across the cycles of one engine."""
    fig = go.Figure()
    if low is not None and high is not None:
        fig.add_trace(go.Scatter(x=df["cycle"], y=high, line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df["cycle"], y=low, fill="tonexty", line={"width": 0},
                                 fillcolor="rgba(34,211,238,0.15)", name=f"{int(cfg.INTERVAL_COVERAGE*100)}% interval"))
    fig.add_trace(go.Scatter(x=df["cycle"], y=np.minimum(df["rul_true"], cfg.RUL_CAP), name=f"Ground truth (capped at {cfg.RUL_CAP})",
                             line={"color": PALETTE["text"], "width": 2, "dash": "dot"}))
    for i, (label, col) in enumerate(pred_cols.items()):
        fig.add_trace(go.Scatter(x=df["cycle"], y=df[col], name=label, line={"color": SERIES[i % len(SERIES)], "width": 2.2}))
    if marker_cycle is not None:
        fig.add_vline(x=marker_cycle, line={"color": PALETTE["amber"], "dash": "dash"})
    fig.add_hline(y=cfg.FAULT_THRESHOLD, line={"color": PALETTE["red"], "dash": "dash", "width": 1},
                  annotation_text="critical threshold", annotation_font_color=PALETTE["red"])
    return _layout(fig, title, 400, "Operating cycle", "Remaining useful life (cycles)")


def plotly_scatter_actual_pred(y_true, y_pred, title: str, colour: str | None = None) -> go.Figure:
    lim = float(max(np.max(y_true), np.max(y_pred))) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect prediction",
                             line={"color": PALETTE["muted"], "dash": "dash"}))
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Engines",
                             marker={"color": colour or PALETTE["cyan"], "size": 7, "opacity": 0.75}))
    return _layout(fig, title, 400, "Actual RUL (cycles)", "Predicted RUL (cycles)")


def plotly_model_bars(table: pd.DataFrame, metric: str, title: str, highlight: str | None = None,
                      lower_is_better: bool = True) -> go.Figure:
    t = table.sort_values(metric, ascending=not lower_is_better if False else False).copy()
    t = table.sort_values(metric, ascending=False)
    colours = [PALETTE["cyan"] if m == highlight else PALETTE["blue"] for m in t["model"]]
    fig = go.Figure(go.Bar(x=t[metric], y=t["model"], orientation="h", marker_color=colours,
                           text=[f"{v:.2f}" for v in t[metric]], textposition="outside"))
    fig = _layout(fig, title, 80 + 28 * len(t), metric, "")
    fig.update_layout(margin={"l": 190, "r": 50, "t": 50, "b": 40}, showlegend=False)
    return fig


def plotly_residuals(y_true, y_pred, title: str) -> go.Figure:
    res = np.asarray(y_pred) - np.asarray(y_true)
    fig = go.Figure(go.Histogram(x=res, nbinsx=40, marker_color=PALETTE["blue"]))
    fig.add_vline(x=0, line={"color": PALETTE["amber"], "dash": "dash"})
    return _layout(fig, title, 340, "Prediction error = predicted - actual (cycles)", "Count")


def plotly_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=[[0, PALETTE["red"]], [0.5, PALETTE["card"]], [1, PALETTE["cyan"]]],
                               colorbar={"title": "r"}))
    fig = _layout(fig, title, 520)
    fig.update_yaxes(autorange="reversed")
    return fig


def plotly_confusion(cm: np.ndarray, title: str) -> go.Figure:
    labels = ["No imminent fault", "Imminent fault"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, PALETTE["card"]], [1, PALETTE["blue"]]],
                               showscale=False, text=cm, texttemplate="%{text:,}", textfont={"size": 18}))
    fig = _layout(fig, title, 360, "Predicted", "Actual")
    fig.update_yaxes(autorange="reversed")
    return fig


def plotly_health_donut(counts: dict) -> go.Figure:
    names = ["Healthy", "Warning", "Critical"]
    fig = go.Figure(go.Pie(labels=names, values=[counts.get(n, 0) for n in names], hole=0.65,
                           marker={"colors": [CATEGORY_COLOURS[n] for n in names]}, sort=False,
                           textinfo="label+value"))
    fig = _layout(fig, "", 300)
    fig.update_layout(showlegend=False, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return fig


def plotly_bar_h(df: pd.DataFrame, x: str, y: str, title: str, colour: str | None = None, signed: bool = False) -> go.Figure:
    d = df.iloc[::-1]
    cols = ([PALETTE["cyan"] if v >= 0 else PALETTE["red"] for v in d[x]] if signed else (colour or PALETTE["cyan"]))
    fig = go.Figure(go.Bar(x=d[x], y=d[y], orientation="h", marker_color=cols))
    fig = _layout(fig, title, 60 + 26 * len(d), x, "")
    fig.update_layout(showlegend=False, margin={"l": 130, "r": 20, "t": 50, "b": 40})
    return fig


def plotly_lines(df: pd.DataFrame, x: str, ys: list[str], title: str, ytitle: str = "", height: int = 380) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(ys):
        fig.add_trace(go.Scatter(x=df[x], y=df[c], name=c, mode="lines", line={"color": SERIES[i % len(SERIES)], "width": 2}))
    return _layout(fig, title, height, x.capitalize(), ytitle)


def plotly_beeswarm(sample: pd.DataFrame, top: int = 12) -> go.Figure:
    """SHAP summary ('beeswarm') plot: x = SHAP value, colour = feature value (low->high)."""
    shap_cols = [c for c in sample.columns if c.startswith("shap::")]
    order = sample[shap_cols].abs().mean().sort_values(ascending=False).head(top).index.tolist()[::-1]
    rng = np.random.RandomState(0)
    fig = go.Figure()
    for i, c in enumerate(order):
        f = c.replace("shap::", "val::")
        v = sample[f].to_numpy()
        norm = (v - np.nanpercentile(v, 5)) / (np.nanpercentile(v, 95) - np.nanpercentile(v, 5) + 1e-9)
        fig.add_trace(go.Scattergl(x=sample[c], y=i + rng.uniform(-0.3, 0.3, len(sample)), mode="markers",
                                   marker={"size": 4, "color": np.clip(norm, 0, 1), "colorscale": [[0, PALETTE["blue"]], [1, PALETTE["red"]]],
                                           "opacity": 0.7, "showscale": i == len(order) - 1,
                                           "colorbar": {"title": "feature value", "tickvals": [0, 1], "ticktext": ["low", "high"]}},
                                   hovertemplate="SHAP %{x:.2f}<extra></extra>", showlegend=False))
    fig = _layout(fig, "SHAP summary: how each feature pushes the predicted RUL", 80 + 34 * len(order),
                  "SHAP value (impact on predicted RUL, cycles)", "")
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[o.replace("shap::", "") for o in order])
    fig.add_vline(x=0, line={"color": PALETTE["muted"], "width": 1})
    return fig


# ------------------------------------------------------------ matplotlib ----
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    return plt


def pick_engines(last_cycle_truth: pd.Series, n: int = 4) -> list[int]:
    """Deterministic, representative engines: spread across short ... long remaining life."""
    s = last_cycle_truth.sort_values()
    return [int(s.index[int(q * (len(s) - 1))]) for q in np.linspace(0.05, 0.95, n)]


def save_subset_figures(subset: str) -> list[str]:
    """PNG figures for one subset (used in the README / your report)."""
    plt = _mpl()
    from src import data_loader as dl
    from src.evaluate import regression_metrics

    out = cfg.FIGURES_DIR / subset
    out.mkdir(parents=True, exist_ok=True)
    summ = json.loads((cfg.METRICS_DIR / f"{subset}_summary.json").read_text())
    best = summ["best_model"]
    pc = f"pred::{best}"
    te = pd.read_csv(cfg.PREDICTIONS_DIR / f"{subset}_test_predictions.csv.gz")
    va = pd.read_csv(cfg.PREDICTIONS_DIR / f"{subset}_val_predictions.csv.gz")
    tr = pd.read_csv(cfg.PREDICTIONS_DIR / f"{subset}_train_predictions.csv.gz")
    last = te[te["is_last_cycle"]]
    saved = []

    def save(fig, name):
        fig.tight_layout(); fig.savefig(out / name, dpi=150); plt.close(fig); saved.append(name)

    # 1. actual vs predicted (train / validation / test)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, (lbl, yt, yp) in zip(ax, [
            ("Training (in-sample)", np.minimum(tr["rul_true"], cfg.RUL_CAP), tr["pred"]),
            ("Validation (held-out engines)", va["rul_true_capped"], va[pc]),
            ("Test - last cycle of each engine (official)", last["rul_true"], last[pc])]):
        a.scatter(yt, yp, s=6 if len(yt) > 500 else 28, alpha=0.4 if len(yt) > 500 else 0.8, color="#2563EB")
        lim = [0, max(np.max(yt), np.max(yp)) * 1.05]
        a.plot(lim, lim, "k--", lw=1); a.set_xlabel("Actual RUL (cycles)"); a.set_ylabel("Predicted RUL (cycles)")
        m = regression_metrics(yt, yp)
        a.set_title(f"{lbl}\nRMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}  R²={m['R2']:.2f}", fontsize=10)
    fig.suptitle(f"{subset} - actual vs predicted RUL ({best})", fontweight="bold")
    save(fig, "actual_vs_predicted.png")

    # 2. RUL degradation curves of 4 test engines
    engines = pick_engines(last.set_index("unit")["rul_true"])
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for a, u in zip(ax.ravel(), engines):
        g = te[te["unit"] == u]
        a.plot(g["cycle"], np.minimum(g["rul_true"], cfg.RUL_CAP), "k--", label="Ground truth (capped)")
        a.plot(g["cycle"], g[pc], color="#2563EB", label=f"Predicted ({best})")
        a.set_title(f"Test engine {u}"); a.set_xlabel("Cycle"); a.set_ylabel("RUL (cycles)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"{subset} - RUL prediction curves vs ground truth (test engines)", fontweight="bold")
    save(fig, "rul_curves_test_engines.png")

    # 3. multi-model comparison on one engine
    tt = pd.read_csv(cfg.METRICS_DIR / f"{subset}_regression_test.csv")
    top = [m for m in tt["model"] if "Baseline" not in m][:5]
    u = engines[1]
    g = te[te["unit"] == u]
    fig, a = plt.subplots(figsize=(9, 4.6))
    a.plot(g["cycle"], np.minimum(g["rul_true"], cfg.RUL_CAP), "k--", lw=2, label="Ground truth (capped)")
    for m in top:
        a.plot(g["cycle"], g[f"pred::{m}"], lw=1.3, label=m)
    a.set_xlabel("Cycle"); a.set_ylabel("RUL (cycles)"); a.legend(fontsize=8)
    a.set_title(f"{subset} - top 5 models on test engine {u}")
    save(fig, "model_curves_comparison.png")

    # 4. model comparison bars (validation RMSE + official test RMSE)
    vt = pd.read_csv(cfg.METRICS_DIR / f"{subset}_regression_validation.csv").sort_values("RMSE", ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].barh(vt["model"], vt["RMSE"], color="#60A5FA"); ax[0].set_title("Validation RMSE (held-out engines, lower = better)")
    t2 = tt.sort_values("official_RMSE", ascending=False)
    ax[1].barh(t2["model"], t2["official_RMSE"], color="#2563EB"); ax[1].set_title("Test RMSE (official last-cycle RUL, lower = better)")
    for a in ax: a.set_xlabel("RMSE (cycles)")
    save(fig, "model_comparison.png")

    # 5. residuals
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(va[pc] - va["rul_true_capped"], bins=40, color="#60A5FA"); ax[0].set_title("Validation residuals")
    ax[1].hist(last[pc] - last["rul_true"], bins=25, color="#2563EB"); ax[1].set_title("Test residuals (last cycle)")
    for a in ax: a.axvline(0, color="k", ls="--"); a.set_xlabel("predicted - actual (cycles)")
    save(fig, "residuals.png")

    # 6. correlation heatmap, degradation curves, distributions (raw EDA)
    import seaborn as sns
    raw = dl.add_train_rul(dl.load_train(subset))
    kept = summ["sensors_kept"]
    fig, a = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(raw[kept + ["rul"]].corr(), cmap="coolwarm", center=0, annot=False, ax=a)
    a.set_title(f"{subset} - correlation of kept sensors with RUL (raw values)")
    save(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(3, 3, figsize=(14, 9))
    for a, s in zip(ax.ravel(), kept[:9]):
        for u in raw[cfg.ID_COL].unique()[:5]:
            g = raw[raw[cfg.ID_COL] == u]
            a.plot(g[cfg.CYCLE_COL], g[s], lw=0.8)
        a.set_title(f"{s}: {cfg.SENSOR_DESCRIPTIONS[s][:34]}", fontsize=8); a.set_xlabel("cycle")
    fig.suptitle(f"{subset} - raw sensor degradation trends of 5 training engines", fontweight="bold")
    save(fig, "sensor_degradation.png")

    fig, ax = plt.subplots(3, 5, figsize=(16, 8))
    for a, s in zip(ax.ravel(), kept[:15]):
        a.hist(raw[s], bins=40, color="#60A5FA"); a.set_title(s, fontsize=9)
    fig.suptitle(f"{subset} - sensor distributions (raw)", fontweight="bold")
    save(fig, "sensor_distributions.png")

    # 7. classification confusion matrix (best classifier, all test cycles)
    ct = pd.read_csv(cfg.METRICS_DIR / f"{subset}_classification_test.csv")
    r = ct[(ct["model"] == summ["classification"]["best_model"]) & (ct["evaluation"] == "all test cycles")].iloc[0]
    cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
    fig, a = plt.subplots(figsize=(4.8, 4.2))
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center", fontsize=13)
    a.set_xticks([0, 1], ["No fault", "Imminent fault"]); a.set_yticks([0, 1], ["No fault", "Imminent fault"])
    a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.grid(False)
    a.set_title(f"{subset} - {summ['classification']['best_model']}\n(RUL <= {cfg.FAULT_THRESHOLD}, test)", fontsize=10)
    save(fig, "confusion_matrix.png")

    # 8. SHAP summary (if pre-computed)
    sp = cfg.METRICS_DIR / f"{subset}_shap_sample.csv.gz"
    if sp.exists():
        import shap
        s = pd.read_csv(sp)
        vals = s[[c for c in s.columns if c.startswith("shap::")]]
        feats = s[[c for c in s.columns if c.startswith("val::")]]
        vals.columns = [c.replace("shap::", "") for c in vals.columns]
        feats.columns = [c.replace("val::", "") for c in feats.columns]
        plt.figure(figsize=(9, 6))
        shap.summary_plot(vals.to_numpy(), feats, max_display=15, show=False)
        fig = plt.gcf(); fig.suptitle(f"{subset} - SHAP summary ({summ['best_tree_model_for_shap']})")
        save(fig, "shap_summary.png")
    return saved


def save_cross_subset_figures() -> None:
    plt = _mpl()
    rows = []
    for s in cfg.SUBSETS:
        p = cfg.METRICS_DIR / f"{s}_summary.json"
        if p.exists():
            j = json.loads(p.read_text())
            rows.append({"subset": s, "RMSE": j["test_best"]["official_RMSE"], "MAE": j["test_best"]["official_MAE"],
                         "R2": j["test_best"]["official_R2"], "NASA": j["test_best"]["official_NASA_score"],
                         "best_model": j["best_model"], "clf_F1": j["classification"]["test_all_cycles"]["F1"],
                         "clf_AUC": j["classification"]["test_all_cycles"]["ROC_AUC"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(cfg.METRICS_DIR / "cross_subset_comparison.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (c, t) in zip(ax, [("RMSE", "Official test RMSE (cycles)"), ("R2", "Official test R²"), ("clf_F1", "Fault-risk F1 (all test cycles)")]):
        a.bar(df["subset"], df[c], color=["#2563EB", "#60A5FA", "#1D4ED8", "#93C5FD"][:len(df)]); a.set_title(t)
    fig.suptitle("Best model per C-MAPSS subset", fontweight="bold")
    fig.tight_layout(); fig.savefig(cfg.FIGURES_DIR / "cross_subset_comparison.png", dpi=150); plt.close(fig)
