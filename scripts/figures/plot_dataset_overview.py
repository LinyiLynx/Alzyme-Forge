#!/usr/bin/env python3
"""Generate Nature-style dataset overview figure (panels A, B, C)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
EMBED_DIR = DATA_DIR / "esmc_embeddings" / "embeddings"
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from clustering_utils import (  # noqa: E402
    cluster_compounds,
    cluster_proteins,
    l2_normalize,
    load_protein_embeddings_from_cache,
    pca_project,
)

# Palette aligned with reference Figure 2 (muted blue / green / orange / coral)
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

PALETTE_MAIN = [C_BLUE, C_ORANGE, C_GREEN, C_CORAL, C_NAVY, C_BLUE_LIGHT]
PALETTE_CLUSTER = [
    C_BLUE,
    C_GREEN,
    C_ORANGE,
    C_CORAL,
    C_NAVY,
    C_BLUE_LIGHT,
    C_GREEN_LIGHT,
    C_CORAL_LIGHT,
    "#9B8EC4",
    "#C9A66B",
    "#6BA3A0",
    "#B07AA1",
]


def load_protein_embeddings(
    proteins: pd.DataFrame,
    cache_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    return load_protein_embeddings_from_cache(cache_path, proteins)


def cluster_compounds_for_plot(smiles_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    smiles_list = smiles_df["Substrate_SMILES"].tolist()
    compound_clusters, _ = cluster_compounds(smiles_list)

    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in compound_clusters["Substrate_SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    fp_array = np.asarray([np.array(fp) for fp in fps], dtype=np.float32)
    normalized = l2_normalize(fp_array)
    coords, var_ratio = pca_project(normalized, n_components=3)

    pair_counts = smiles_df.set_index("Substrate_SMILES")["pair_count"]
    compound_clusters = compound_clusters.copy()
    compound_clusters["pc1"] = coords[:, 0]
    compound_clusters["pc2"] = coords[:, 1]
    compound_clusters["pc3"] = coords[:, 2]
    compound_clusters["pair_count"] = [pair_counts.loc[smi] for smi in compound_clusters["Substrate_SMILES"]]
    return compound_clusters, var_ratio


def cluster_proteins_for_plot(matrix: np.ndarray, meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    protein_clusters, normalized, _ = cluster_proteins(matrix, meta)
    coords, var_ratio = pca_project(normalized, n_components=3)
    protein_clusters = protein_clusters.copy()
    protein_clusters["pc1"] = coords[:, 0]
    protein_clusters["pc2"] = coords[:, 1]
    protein_clusters["pc3"] = coords[:, 2]
    return protein_clusters, var_ratio


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
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": False,
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
        ax.grid(axis="y", color=C_GRAY_LIGHT, linestyle="--", linewidth=0.5, alpha=0.9)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.06) -> None:
    if not label:
        return
    text_kwargs = dict(
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color="#222222",
    )
    if hasattr(ax, "get_zlim"):
        ax.text2D(x, y, label, transform=ax.transAxes, **text_kwargs)
    else:
        ax.text(x, y, label, transform=ax.transAxes, **text_kwargs)


def annotate_bars_h(ax: plt.Axes, bars, values: list, fmt: str = "{:,}", pad: float = 0.01) -> None:
    xmax = max(values) if values else 1
    ax.set_xlim(0, xmax * 1.18)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + xmax * pad,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va="center",
            ha="left",
            fontsize=6.5,
            color="#333333",
        )


def annotate_bars_v(ax: plt.Axes, bars, labels: list[str], ypad: float = 0.04) -> None:
    ymax = max(bar.get_height() for bar in bars)
    ax.set_ylim(0, ymax * (1 + ypad))
    for bar, text in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.01,
            text,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#333333",
            linespacing=1.1,
        )


def load_splits() -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = pd.read_csv(DATA_DIR / f"{name}.csv")
    return splits


def load_all_pairs(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for split_name, df in splits.items():
        part = df.copy()
        part["split"] = split_name
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def add_panel_label_fig(fig: plt.Figure, ax: plt.Axes, label: str, pad_x: float = 0.012, pad_y: float = 0.006) -> None:
    """Place A/B/C labels at a consistent figure-x using the subplot bbox."""
    if not label:
        return
    pos = ax.get_position()
    fig.text(
        pos.x0 - pad_x,
        pos.y1 + pad_y,
        label,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="right",
        color="#222222",
    )


def plot_panel_a(
    fig: plt.Figure,
    gs: GridSpec,
    cazy: pd.DataFrame,
    backbone: pd.DataFrame,
    all_pairs: pd.DataFrame,
) -> plt.Axes:
    ax1 = fig.add_subplot(gs[0, 0:3])
    labels = ["CAZy characterized", "GT backbone", "Curated pairs", "Unique proteins", "Unique compounds"]
    values = [len(cazy), len(backbone), len(all_pairs), all_pairs["Protein_ID"].nunique(), all_pairs["Substrate_SMILES"].nunique()]
    colors = PALETTE_MAIN[: len(labels)]
    y = np.arange(len(labels))
    bars = ax1.barh(y, values, color=colors, height=0.62, edgecolor="white", linewidth=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=6.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Count")
    ax1.set_title("Data provenance", pad=8)
    annotate_bars_h(ax1, bars, values)
    style_axes(ax1, grid=True)

    ax2 = fig.add_subplot(gs[0, 3:6])
    kingdom_counts = cazy["kingdom_group"].value_counts()
    yk = np.arange(len(kingdom_counts))
    bars2 = ax2.barh(
        yk,
        kingdom_counts.values,
        color=[C_BLUE, C_ORANGE, C_GREEN, C_CORAL][: len(kingdom_counts)],
        height=0.62,
        edgecolor="white",
        linewidth=0.6,
    )
    ax2.set_yticks(yk)
    ax2.set_yticklabels(kingdom_counts.index, fontsize=6.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Entries")
    ax2.set_title("Kingdom (CAZy)", pad=8)
    pct_labels = [f"{100 * v / kingdom_counts.sum():.0f}%" for v in kingdom_counts.values]
    for bar, val, pct in zip(bars2, kingdom_counts.values, pct_labels):
        ax2.text(
            bar.get_width() + kingdom_counts.max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}  {pct}",
            va="center",
            ha="left",
            fontsize=6.5,
            color="#333333",
        )
    ax2.set_xlim(0, kingdom_counts.max() * 1.35)
    style_axes(ax2, grid=True)

    ax3 = fig.add_subplot(gs[0, 6:9])
    top_fam = cazy["cazy_family"].value_counts().head(10)
    yf = np.arange(len(top_fam))
    ax3.barh(yf, top_fam.values, color=C_BLUE, height=0.62, edgecolor="white", linewidth=0.4)
    ax3.set_yticks(yf)
    ax3.set_yticklabels(top_fam.index, fontsize=6.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("Entries")
    ax3.set_title("Top CAZy families", pad=8)
    style_axes(ax3, grid=True)

    ax4 = fig.add_subplot(gs[0, 9:12])
    seqlen = backbone["seq"].str.len()
    counts_hist, _, _ = ax4.hist(
        seqlen,
        bins=40,
        color=C_BLUE_LIGHT,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.95,
    )
    median_val = seqlen.median()
    ymax = max(counts_hist) if len(counts_hist) else 1
    ax4.axvline(median_val, color=C_CORAL, linestyle="--", linewidth=1.0)
    ax4.text(
        median_val + 40,
        ymax * 0.88,
        f"median = {median_val:.0f}",
        fontsize=6.5,
        color=C_CORAL,
        va="top",
        ha="left",
    )
    ax4.set_xlabel("Sequence length (aa)")
    ax4.set_ylabel("Proteins")
    ax4.set_title("Backbone length distribution", pad=8)
    style_axes(ax4, grid=True)
    return ax1


def style_axes_3d(ax) -> None:
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#DDDDDD")
    ax.yaxis.pane.set_edgecolor("#DDDDDD")
    ax.zaxis.pane.set_edgecolor("#DDDDDD")
    ax.tick_params(labelsize=6, pad=0)
    ax.grid(True, color=C_GRAY_LIGHT, linewidth=0.4, alpha=0.7)


def plot_cluster_scatter_3d(
    ax,
    df: pd.DataFrame,
    title: str,
    n_items: int,
    n_clusters: int,
    var_ratio: np.ndarray,
    size_col: str | None = None,
    alpha: float = 0.75,
    point_size: float = 8,
    draw_hulls: bool = True,
    elev: float = 22,
    azim: float = -58,
) -> None:
    cluster_ids = sorted(df["cluster_id"].unique())
    cols = ["pc1", "pc2", "pc3"]
    cum_var = 100.0 * var_ratio.sum()

    for idx, cid in enumerate(cluster_ids):
        mask = df["cluster_id"] == cid
        color = PALETTE_CLUSTER[idx % len(PALETTE_CLUSTER)]
        pts = df.loc[mask, cols].to_numpy()

        if draw_hulls and len(pts) >= 4:
            hull = ConvexHull(pts)
            triangles = [pts[simplex] for simplex in hull.simplices]
            ax.add_collection3d(
                Poly3DCollection(
                    triangles,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.10,
                    linewidths=0.4,
                )
            )

        sizes = point_size
        if size_col is not None:
            sizes = 14 + 2.0 * np.sqrt(df.loc[mask, size_col].to_numpy())
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=sizes,
            c=[color],
            alpha=alpha,
            depthshade=True,
            edgecolors="white" if size_col is not None else "none",
            linewidths=0.15,
        )

        if len(pts) > 0:
            centroid = pts.mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                centroid[2],
                s=48,
                marker="X",
                c=[color],
                edgecolors="#222222",
                linewidths=0.5,
                depthshade=False,
            )

    ax.set_xlabel(f"PC1 ({100 * var_ratio[0]:.1f}%)", labelpad=2, fontsize=6.5)
    ax.set_ylabel(f"PC2 ({100 * var_ratio[1]:.1f}%)", labelpad=2, fontsize=6.5)
    ax.set_zlabel(f"PC3 ({100 * var_ratio[2]:.1f}%)", labelpad=2, fontsize=6.5)
    ax.set_title(
        f"{title}\nKMeans + PCA 3D  |  n={n_items:,}, k={n_clusters}  |  cum. var. {cum_var:.1f}%",
        fontsize=7,
        pad=10,
        linespacing=1.2,
    )
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    ax.view_init(elev=elev, azim=azim)
    style_axes_3d(ax)


def plot_panel_b(
    fig: plt.Figure,
    gs: GridSpec,
    compound_clusters: pd.DataFrame,
    protein_clusters: pd.DataFrame,
    compound_var: np.ndarray,
    protein_var: np.ndarray,
) -> plt.Axes:
    ax1 = fig.add_subplot(gs[1, 0:6], projection="3d")
    plot_cluster_scatter_3d(
        ax1,
        compound_clusters,
        "Compound clustering",
        n_items=len(compound_clusters),
        n_clusters=int(compound_clusters["cluster_id"].nunique()),
        var_ratio=compound_var,
        size_col="pair_count",
        alpha=0.90,
        point_size=18,
        elev=24,
        azim=-52,
    )

    ax2 = fig.add_subplot(gs[1, 6:12], projection="3d")
    plot_cluster_scatter_3d(
        ax2,
        protein_clusters,
        "Protein clustering",
        n_items=len(protein_clusters),
        n_clusters=int(protein_clusters["cluster_id"].nunique()),
        var_ratio=protein_var,
        alpha=0.45,
        point_size=2.5,
        elev=20,
        azim=-62,
    )
    return ax1


def plot_panel_c(fig: plt.Figure, gs: GridSpec, splits: dict[str, pd.DataFrame]) -> plt.Axes:
    split_names = ["train", "val", "test"]
    split_labels = ["Train", "Val", "Test"]
    colors = [C_BLUE, C_ORANGE, C_GREEN]

    ax1 = fig.add_subplot(gs[2, 0:4])
    counts = [len(splits[s]) for s in split_names]
    total = sum(counts)
    bars = ax1.bar(split_labels, counts, color=colors, width=0.58, edgecolor="white", linewidth=0.6)
    ax1.set_ylabel("Pairs")
    ax1.set_title("8:1:1 split sizes", pad=8)
    annotate_bars_v(
        ax1,
        bars,
        [f"{val:,}\n({100.0 * val / total:.1f}%)" for val in counts],
        ypad=0.12,
    )
    style_axes(ax1, grid=True)

    ax2 = fig.add_subplot(gs[2, 4:8])
    x = np.arange(len(split_names))
    width = 0.32
    pos_counts = [int((splits[s]["Label"] == 1).sum()) for s in split_names]
    neg_counts = [int((splits[s]["Label"] == 0).sum()) for s in split_names]
    ax2.bar(x - width / 2, pos_counts, width, label="Positive", color=C_CORAL_LIGHT, edgecolor="white", linewidth=0.5)
    ax2.bar(x + width / 2, neg_counts, width, label="Negative", color=C_BLUE_LIGHT, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_labels)
    ax2.set_ylabel("Pairs")
    ax2.set_title("Label balance per split", pad=8)
    ymax2 = max(max(pos_counts), max(neg_counts))
    ax2.set_ylim(0, ymax2 * 1.10)
    ax2.legend(frameon=False, loc="upper right", fontsize=6, borderaxespad=0.4)
    style_axes(ax2, grid=True)

    ax3 = fig.add_subplot(gs[2, 8:12])
    prot_counts = [splits[s]["Protein_ID"].nunique() for s in split_names]
    smi_counts = [splits[s]["Substrate_SMILES"].nunique() for s in split_names]
    ax3.bar(x - width / 2, prot_counts, width, label="Proteins", color=C_BLUE, edgecolor="white", linewidth=0.5)
    ax3.bar(x + width / 2, smi_counts, width, label="Compounds", color=C_ORANGE, edgecolor="white", linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(split_labels)
    ax3.set_ylabel("Unique entities")
    ax3.set_title("Diversity per split", pad=8)
    ymax3 = max(max(prot_counts), max(smi_counts))
    ax3.set_ylim(0, ymax3 * 1.10)
    ax3.legend(frameon=False, loc="upper right", fontsize=6, borderaxespad=0.4)
    style_axes(ax3, grid=True)
    return ax1


def main() -> None:
    apply_nature_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cazy = pd.read_csv(DATA_DIR / "raw" / "cazy_gt_characterized_raw.csv")
    backbone = pd.read_csv(DATA_DIR / "raw" / "GT_backbone_metadata.csv")
    splits = load_splits()
    all_pairs = load_all_pairs(splits)

    smiles_freq = (
        all_pairs.groupby("Substrate_SMILES", as_index=False)
        .size()
        .rename(columns={"size": "pair_count"})
    )
    compound_clusters, compound_var = cluster_compounds_for_plot(smiles_freq)
    compound_clusters.to_csv(FIG_DIR / "compound_clusters.csv", index=False)

    unique_proteins = (
        all_pairs[["Protein_ID", "Protein_Sequence"]]
        .drop_duplicates("Protein_ID")
        .rename(columns={"Protein_ID": "protein_id", "Protein_Sequence": "sequence"})
        .reset_index(drop=True)
    )
    protein_matrix, protein_meta = load_protein_embeddings(unique_proteins, FIG_DIR / "protein_meanpool.npz")
    protein_clusters, protein_var = cluster_proteins_for_plot(protein_matrix, protein_meta)
    protein_clusters.to_csv(FIG_DIR / "protein_clusters.csv", index=False)

    fig = plt.figure(figsize=(9.2, 14.0))
    gs = fig.add_gridspec(
        3,
        12,
        figure=fig,
        height_ratios=[1.0, 1.55, 1.05],
        hspace=0.42,
        wspace=0.38,
        top=0.91,
        bottom=0.07,
        left=0.09,
        right=0.98,
    )

    ax_a = plot_panel_a(fig, gs, cazy, backbone, all_pairs)
    ax_b = plot_panel_b(fig, gs, compound_clusters, protein_clusters, compound_var, protein_var)
    ax_c = plot_panel_c(fig, gs, splits)

    fig.canvas.draw()
    add_panel_label_fig(fig, ax_a, "A")
    add_panel_label_fig(fig, ax_b, "B")
    add_panel_label_fig(fig, ax_c, "C")

    fig.suptitle(
        "Glycosyltransferase dataset: raw sources, clustering, and 8:1:1 split",
        fontsize=10,
        fontweight="normal",
        color="#222222",
        y=0.975,
    )
    fig.savefig(FIG_DIR / "dataset_overview.png", bbox_inches="tight", facecolor="white", pad_inches=0.08)
    fig.savefig(FIG_DIR / "dataset_overview.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)

    print(f"Saved figure to {FIG_DIR / 'dataset_overview.png'}")
    print(f"Saved figure to {FIG_DIR / 'dataset_overview.pdf'}")
    print(f"Saved compound clusters to {FIG_DIR / 'compound_clusters.csv'}")
    print(f"Saved protein clusters to {FIG_DIR / 'protein_clusters.csv'}")


if __name__ == "__main__":
    main()
