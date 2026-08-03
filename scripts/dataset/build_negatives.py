#!/usr/bin/env python3
"""Build cluster-based negative samples and regenerate train/val/test splits."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clustering_utils import (  # noqa: E402
    cluster_compounds,
    cluster_distance_matrix,
    cluster_proteins,
    load_protein_embeddings_from_cache,
)

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUTPUT_COLUMNS = ["Protein_ID", "Protein_Sequence", "Substrate_SMILES", "Label"]


def load_positive_pairs(split_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        pos = df[df["Label"] == 1].copy()
        frames.append(pos)
    if not frames:
        raise FileNotFoundError(f"No positive pairs found under {split_dir}")
    positives = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Protein_ID", "Substrate_SMILES"]
    )
    return positives.reset_index(drop=True)


def build_protein_sequence_map(positives: pd.DataFrame, backbone_csv: Path | None) -> dict[str, str]:
    seq_map = (
        positives[["Protein_ID", "Protein_Sequence"]]
        .drop_duplicates("Protein_ID")
        .set_index("Protein_ID")["Protein_Sequence"]
        .to_dict()
    )
    if backbone_csv and backbone_csv.exists():
        backbone = pd.read_csv(backbone_csv)
        for row in backbone.itertuples(index=False):
            if row.uid not in seq_map:
                seq_map[row.uid] = row.seq
    return seq_map


def build_cluster_member_index(cluster_df: pd.DataFrame, id_col: str) -> dict[int, list[str]]:
    members: dict[int, list[str]] = defaultdict(list)
    for row in cluster_df.itertuples(index=False):
        members[int(getattr(row, "cluster_id"))].append(getattr(row, id_col))
    return members


def rank_distant_clusters(
    anchor_cluster: int,
    cluster_ids: list[int],
    dist_matrix: np.ndarray,
) -> list[int]:
    idx = {cid: i for i, cid in enumerate(cluster_ids)}
    anchor_idx = idx[anchor_cluster]
    distances = [(cid, dist_matrix[anchor_idx, idx[cid]]) for cid in cluster_ids if cid != anchor_cluster]
    distances.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in distances]


def _eligible_candidates(
    anchor_protein: str,
    anchor_cluster: int,
    substrate: str,
    positive_pairs: set[tuple[str, str]],
    used_negative_pairs: set[tuple[str, str]],
    protein_to_cluster: dict[str, int],
    cluster_members: dict[int, list[str]],
    cluster_ids: list[int],
    cluster_dist: np.ndarray,
    all_proteins: list[str],
    strategy: str,
) -> list[str]:
    blocked = positive_pairs | used_negative_pairs

    if strategy == "distant":
        distant_clusters = rank_distant_clusters(anchor_cluster, cluster_ids, cluster_dist)
        for cluster_id in distant_clusters:
            candidates = [
                pid
                for pid in cluster_members[cluster_id]
                if pid != anchor_protein and (pid, substrate) not in blocked
            ]
            if candidates:
                return candidates
        return []

    if strategy == "different_cluster":
        return [
            pid
            for pid in all_proteins
            if pid != anchor_protein
            and protein_to_cluster.get(pid) != anchor_cluster
            and (pid, substrate) not in blocked
        ]

    return [
        pid
        for pid in all_proteins
        if pid != anchor_protein and (pid, substrate) not in blocked
    ]


def sample_negative_protein(
    rng: np.random.Generator,
    anchor_protein: str,
    anchor_cluster: int,
    substrate: str,
    positive_pairs: set[tuple[str, str]],
    used_negative_pairs: set[tuple[str, str]],
    protein_to_cluster: dict[str, int],
    cluster_members: dict[int, list[str]],
    cluster_ids: list[int],
    cluster_dist: np.ndarray,
    all_proteins: list[str],
) -> str | None:
    for strategy in ("distant", "different_cluster", "any"):
        candidates = _eligible_candidates(
            anchor_protein=anchor_protein,
            anchor_cluster=anchor_cluster,
            substrate=substrate,
            positive_pairs=positive_pairs,
            used_negative_pairs=used_negative_pairs,
            protein_to_cluster=protein_to_cluster,
            cluster_members=cluster_members,
            cluster_ids=cluster_ids,
            cluster_dist=cluster_dist,
            all_proteins=all_proteins,
            strategy=strategy,
        )
        if candidates:
            return str(rng.choice(candidates))
    return None


def generate_negative_pairs(
    positives: pd.DataFrame,
    protein_clusters: pd.DataFrame,
    protein_seq_map: dict[str, str],
    cluster_members: dict[int, list[str]],
    cluster_ids: list[int],
    cluster_dist: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positive_pairs = set(zip(positives["Protein_ID"], positives["Substrate_SMILES"]))
    protein_to_cluster = protein_clusters.set_index("protein_id")["cluster_id"].astype(int).to_dict()
    all_proteins = protein_clusters["protein_id"].tolist()

    negatives: list[dict[str, object]] = []
    used_negative_pairs: set[tuple[str, str]] = set()
    failed = 0
    for row in positives.itertuples(index=False):
        anchor_cluster = int(protein_to_cluster.get(row.Protein_ID, -1))
        if anchor_cluster < 0:
            failed += 1
            continue
        neg_protein = sample_negative_protein(
            rng=rng,
            anchor_protein=row.Protein_ID,
            anchor_cluster=anchor_cluster,
            substrate=row.Substrate_SMILES,
            positive_pairs=positive_pairs,
            used_negative_pairs=used_negative_pairs,
            protein_to_cluster=protein_to_cluster,
            cluster_members=cluster_members,
            cluster_ids=cluster_ids,
            cluster_dist=cluster_dist,
            all_proteins=all_proteins,
        )
        if neg_protein is None or neg_protein not in protein_seq_map:
            failed += 1
            continue
        pair = (neg_protein, row.Substrate_SMILES)
        used_negative_pairs.add(pair)
        negatives.append(
            {
                "Protein_ID": neg_protein,
                "Protein_Sequence": protein_seq_map[neg_protein],
                "Substrate_SMILES": row.Substrate_SMILES,
                "Label": 0,
            }
        )

    if failed:
        print(f"[negative sampling] failed to sample {failed} negatives", file=sys.stderr)
    neg_df = pd.DataFrame(negatives)
    print(f"[negative sampling] generated {len(neg_df)} negative pairs", file=sys.stderr)
    return neg_df


def split_pairs(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["Label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["Label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def backup_existing_splits(data_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir / f"backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "val.csv", "test.csv"):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    print(f"[backup] saved previous splits to {backup_dir}", file=sys.stderr)
    return backup_dir


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir: Path,
) -> None:
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        out = frame[OUTPUT_COLUMNS].copy()
        out.to_csv(data_dir / f"{name}.csv", index=False)
        pos = int((out["Label"] == 1).sum())
        neg = int((out["Label"] == 0).sum())
        print(f"[save] {name}.csv: {len(out)} rows (pos={pos}, neg={neg})", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate cluster-based negative samples and splits.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    positives = load_positive_pairs(args.data_dir)
    print(f"[load] {len(positives)} unique positive pairs", file=sys.stderr)

    unique_proteins = (
        positives[["Protein_ID", "Protein_Sequence"]]
        .drop_duplicates("Protein_ID")
        .rename(columns={"Protein_ID": "protein_id", "Protein_Sequence": "sequence"})
        .reset_index(drop=True)
    )
    protein_matrix, protein_meta = load_protein_embeddings_from_cache(
        args.figures_dir / "protein_meanpool.npz",
        unique_proteins,
    )
    protein_clusters, _, protein_centroids = cluster_proteins(protein_matrix, protein_meta, random_state=args.seed)
    protein_clusters.to_csv(args.figures_dir / "protein_clusters.csv", index=False)

    unique_substrates = sorted(positives["Substrate_SMILES"].unique().tolist())
    compound_clusters, _ = cluster_compounds(unique_substrates)
    compound_clusters.to_csv(args.figures_dir / "compound_clusters.csv", index=False)

    protein_seq_map = build_protein_sequence_map(
        positives,
        args.data_dir / "raw" / "GT_backbone_metadata.csv",
    )
    cluster_members = build_cluster_member_index(protein_clusters, "protein_id")
    cluster_ids, cluster_dist = cluster_distance_matrix(protein_centroids, use_cosine=True)

    negatives = generate_negative_pairs(
        positives=positives,
        protein_clusters=protein_clusters,
        protein_seq_map=protein_seq_map,
        cluster_members=cluster_members,
        cluster_ids=cluster_ids,
        cluster_dist=cluster_dist,
        seed=args.seed,
    )

    pos_out = positives.rename(columns={"Protein_ID": "Protein_ID", "Protein_Sequence": "Protein_Sequence"})
    pos_out = pos_out[OUTPUT_COLUMNS].copy()
    pos_out["Label"] = 1

    if len(negatives) != len(positives):
        raise RuntimeError(
            f"Negative count ({len(negatives)}) does not match positive count ({len(positives)})"
        )

    full_df = pd.concat([pos_out, negatives], ignore_index=True)

    overlap = set(zip(pos_out["Protein_ID"], pos_out["Substrate_SMILES"])) & set(
        zip(negatives["Protein_ID"], negatives["Substrate_SMILES"])
    )
    if overlap:
        raise RuntimeError(f"Positive/negative overlap detected: {len(overlap)} pairs")

    dup_neg = negatives.duplicated(subset=["Protein_ID", "Substrate_SMILES"]).sum()
    if dup_neg:
        raise RuntimeError(f"Duplicate negative pairs detected: {dup_neg}")

    train_df, val_df, test_df = split_pairs(full_df, seed=args.seed)

    if not args.no_backup:
        backup_existing_splits(args.data_dir)

    save_splits(train_df, val_df, test_df, args.data_dir)
    print("[done] regenerated train/val/test with cluster-based negatives", file=sys.stderr)


if __name__ == "__main__":
    main()
