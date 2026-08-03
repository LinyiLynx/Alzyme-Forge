#!/usr/bin/env python3
"""Nature-style multi-panel performance figure for trained EPP-GT models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

C_BLUE = "#5B8DBE"
C_BLUE_LIGHT = "#A7C7E7"
C_GREEN = "#7CB87C"
C_GREEN_LIGHT = "#C1E1C1"
C_ORANGE = "#E8A838"
C_CORAL = "#D4786A"
C_CORAL_LIGHT = "#E8A0A0"
C_NAVY = "#2E5A88"
C_GRAY = "#B8B8B8"
C_GRAY_LIGHT = "#E8E8E8"


def apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 350,
            "savefig.dpi": 350,
        }
    )


def style_axes(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333", length=3, width=0.6)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis="both", color=C_GRAY_LIGHT, linestyle="--", linewidth=0.5, alpha=0.9)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left", color="#222222")


def ensure_predictions(split: str, checkpoint: Path, artifact_dir: Path, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    csv_path = DATA_DIR / f"{split}.csv"
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "predict",
        "--checkpoint",
        str(checkpoint),
        "--csv",
        str(csv_path),
        "--config",
        str(artifact_dir / "resolved_config.yaml"),
        "--output",
        str(out_path),
        "--device",
        "cuda",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out_path


def bootstrap_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    rng = np.random.default_rng(seed)
    base_fpr = np.linspace(0, 1, 101)
    tprs: list[np.ndarray] = []
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        y = labels[idx]
        s = scores[idx]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        tprs.append(np.interp(base_fpr, fpr, tpr))
        aucs.append(roc_auc_score(y, s))
    tprs_arr = np.vstack(tprs)
    return (
        base_fpr,
        tprs_arr.mean(axis=0),
        np.percentile(tprs_arr, 2.5, axis=0),
        np.percentile(tprs_arr, 97.5, axis=0),
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def plot_roc_panel(
    ax: plt.Axes,
    labels: np.ndarray,
    scores: np.ndarray,
    title: str,
    line_color: str,
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    base_fpr, mean_tpr, lo, hi, boot_auc, auc_lo, auc_hi = bootstrap_roc(labels, scores)
    ax.fill_between(base_fpr, lo, hi, color=C_GRAY, alpha=0.35, linewidth=0)
    ax.plot(base_fpr, mean_tpr, color=line_color, linewidth=1.8, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=C_GRAY, linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, pad=6)
    ax.legend(
        frameon=False,
        loc="lower right",
        title=f"95% CI [{auc_lo:.3f}, {auc_hi:.3f}]",
        title_fontsize=6,
    )
    style_axes(ax, grid=True)


def plot_learning_curves(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    epochs = metrics["epoch"]
    ax.plot(epochs, metrics["auroc"], color=C_ORANGE, marker="o", markersize=3.5, linewidth=1.5, label="Val AUROC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUROC", color=C_ORANGE)
    ax.tick_params(axis="y", labelcolor=C_ORANGE)
    ax.set_ylim(max(0.5, float(metrics["auroc"].min()) - 0.05), 1.0)
    ax2 = ax.twinx()
    ax2.plot(epochs, metrics["mrr"], color=C_GREEN, marker="s", markersize=3.5, linewidth=1.5, label="Val MRR")
    ax2.set_ylabel("Val MRR", color=C_GREEN)
    ax2.tick_params(axis="y", labelcolor=C_GREEN)
    ax2.set_ylim(max(0.4, float(metrics["mrr"].min()) - 0.05), 1.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#333333")
    ax.set_title("Training dynamics", pad=6)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right")
    style_axes(ax, grid=True)


def plot_metric_bars(ax: plt.Axes, val_metrics: dict, test_metrics: dict) -> None:
    names = ["AUROC", "AUPRC", "MRR", "Mean group AUROC"]
    val_vals = [val_metrics["auroc"], val_metrics["auprc"], val_metrics["mrr"], val_metrics["mean_group_auroc"]]
    test_vals = [test_metrics["auroc"], test_metrics["auprc"], test_metrics["mrr"], test_metrics["mean_group_auroc"]]
    x = np.arange(len(names))
    width = 0.34
    bars1 = ax.bar(x - width / 2, val_vals, width, label="Validation", color=C_BLUE_LIGHT, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, test_vals, width, label="Test", color=C_ORANGE, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Generalization performance", pad=6)
    ymax = 1.0
    for bar, val in list(zip(bars1, val_vals)) + list(zip(bars2, test_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    style_axes(ax, grid=True)


def plot_score_density(ax: plt.Axes, labels: np.ndarray, scores: np.ndarray) -> None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    xs = np.linspace(0, 1, 200)
    if len(pos) > 1:
        kde_pos = stats.gaussian_kde(pos)
        ax.fill_between(xs, kde_pos(xs), color=C_GREEN_LIGHT, alpha=0.75, linewidth=0)
        ax.plot(xs, kde_pos(xs), color=C_GREEN, linewidth=1.2, label="Positive")
    if len(neg) > 1:
        kde_neg = stats.gaussian_kde(neg)
        ax.fill_between(xs, kde_neg(xs), color=C_CORAL_LIGHT, alpha=0.75, linewidth=0)
        ax.plot(xs, kde_neg(xs), color=C_CORAL, linewidth=1.2, label="Negative")
    threshold = 0.5
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=0.9)
    ax.text(threshold + 0.02, ax.get_ylim()[1] * 0.92, f"threshold = {threshold:.2f}", fontsize=6, va="top")
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Density")
    ax.set_title("Class separation (test set)", pad=6)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax, grid=True)


def plot_ranking_metrics(ax: plt.Axes, test_metrics: dict) -> None:
    keys = ["top1_hit", "top5_hit", "top10_hit", "mrr", "ndcg", "mean_group_auroc"]
    labels = ["Top-1 hit", "Top-5 hit", "Top-10 hit", "MRR", "NDCG", "Mean group AUROC"]
    values = [test_metrics[k] for k in keys]
    y = np.arange(len(labels))
    colors = [C_BLUE if i < 3 else C_NAVY for i in range(len(labels))]
    ax.barh(y, values, color=colors, height=0.62, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score")
    ax.set_title("Ranking metrics (test set)", pad=6)
    for yi, val in zip(y, values):
        ax.text(val + 0.015, yi, f"{val:.3f}", va="center", ha="left", fontsize=6)
    style_axes(ax, grid=True)


def compute_split_metrics(df: pd.DataFrame) -> dict[str, float]:
    labels = df["Label"].to_numpy()
    scores = df["score"].to_numpy()
    from eppgt_repro.metrics import compute_binary_metrics, compute_grouped_ranking_metrics

    metrics = compute_binary_metrics(labels, scores)
    groups = df["Substrate_SMILES"].astype(str).tolist()
    metrics.update(compute_grouped_ranking_metrics(labels, scores, groups))
    return metrics


def load_test_eval(artifact_dir: Path, test_df: pd.DataFrame) -> dict:
    eval_path = artifact_dir / "eval_test.json"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            return json.load(f)
    sys.path.insert(0, str(ROOT / "code" / "EPPGT_repro" / "src"))
    return compute_split_metrics(test_df)


def build_figure(
    run_name: str,
    artifact_dir: Path,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    test_eval: dict,
    val_metrics: dict,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.5, 9.8))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 0.95, 1.0], hspace=0.55, wspace=0.38, top=0.93, bottom=0.07, left=0.10, right=0.97)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_roc_panel(ax_a, val_df["Label"].to_numpy(), val_df["score"].to_numpy(), "Validation ROC", C_ORANGE)
    add_panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    plot_roc_panel(ax_b, test_df["Label"].to_numpy(), test_df["score"].to_numpy(), "Test ROC", C_CORAL)
    add_panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    plot_learning_curves(ax_c, metrics_df)
    add_panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    plot_metric_bars(ax_d, val_metrics, test_eval)
    add_panel_label(ax_d, "D")

    ax_e = fig.add_subplot(gs[2, 0])
    plot_score_density(ax_e, test_df["Label"].to_numpy(), test_df["score"].to_numpy())
    add_panel_label(ax_e, "E")

    ax_f = fig.add_subplot(gs[2, 1])
    plot_ranking_metrics(ax_f, test_eval)
    add_panel_label(ax_f, "F")

    fig.suptitle(f"{run_name} model performance (ESMC + EPP-GT)", fontsize=9.5, fontweight="normal", color="#222222", y=0.985)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model AUC overview figure")
    parser.add_argument("--run", default="neg_fix_v1", help="Run name under artifacts/ (e.g. 0601, neg_fix_v1)")
    args = parser.parse_args()

    run_name = args.run
    artifact_dir = ROOT / "artifacts" / run_name
    apply_nature_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = artifact_dir / "best.pt"
    val_pred_path = ensure_predictions("val", checkpoint, artifact_dir, FIG_DIR / f"predictions_val_{run_name}.csv")
    test_pred_path = ensure_predictions("test", checkpoint, artifact_dir, FIG_DIR / f"predictions_test_{run_name}.csv")

    val_df = pd.read_csv(val_pred_path)
    test_df = pd.read_csv(test_pred_path)
    metrics_df = pd.read_csv(artifact_dir / "metrics.csv")
    test_eval = load_test_eval(artifact_dir, test_df)

    sys.path.insert(0, str(ROOT / "code" / "EPPGT_repro" / "src"))
    val_metrics = compute_split_metrics(val_df)

    fig = build_figure(run_name, artifact_dir, val_df, test_df, metrics_df, test_eval, val_metrics)

    out_png = FIG_DIR / f"model_auc_overview_{run_name}.png"
    out_pdf = FIG_DIR / f"model_auc_overview_{run_name}.pdf"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "run": run_name,
        "val_auroc": val_metrics["auroc"],
        "test_auroc": test_eval["auroc"],
        "val_mrr": val_metrics["mrr"],
        "test_mrr": test_eval["mrr"],
    }
    (FIG_DIR / f"model_auc_summary_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved figure to {out_png}")
    print(f"Saved figure to {out_pdf}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
