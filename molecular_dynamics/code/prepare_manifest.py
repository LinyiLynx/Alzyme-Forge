#!/usr/bin/env python3
"""Sync MD manifest from Alphafold summary.csv (all structure-ready pairs)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from config import CAMPAIGN, MANIFEST, MANIFEST_INPUTS, SUMMARY_CSV


def sync_manifest(
    summary_csv: Path = SUMMARY_CSV,
    inputs_manifest: Path = MANIFEST_INPUTS,
    out_dir: Path = CAMPAIGN,
    top_n: int = 0,
    require_structure: bool = True,
) -> Path:
    summary = pd.read_csv(summary_csv)
    if require_structure:
        rows_df = summary[summary["structure_status"] == "done"].copy()
    else:
        rows_df = summary.copy()
    rows_df = rows_df.sort_values("rank")
    if top_n > 0:
        rows_df = rows_df.head(top_n)

    smiles_map: dict[str, str] = {}
    if inputs_manifest.exists():
        man = pd.read_csv(inputs_manifest)
        smiles_map = dict(zip(man["protein_id"].astype(str), man["smiles"].astype(str)))

    rows: list[dict] = []
    for _, row in rows_df.iterrows():
        protein_id = str(row["protein_id"])
        structure_path = str(row.get("structure_path", "")).strip()
        if require_structure and not structure_path:
            continue
        if require_structure and not Path(structure_path).exists():
            continue
        rows.append(
            {
                "rank": int(row["rank"]),
                "job_id": str(row["job_id"]),
                "compound_id": str(row["compound_id"]),
                "protein_id": protein_id,
                "structure_path": structure_path,
                "screen_score": float(row["screen_score"]),
                "confidence_score": float(row.get("confidence_score", 0) or 0),
                "iptm": float(row.get("iptm", 0) or 0),
                "smiles": smiles_map.get(protein_id, ""),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    if rows:
        with open(manifest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--inputs-manifest", type=Path, default=MANIFEST_INPUTS)
    parser.add_argument("--top-n", type=int, default=0, help="0 = all ready structures")
    parser.add_argument("--out-dir", type=Path, default=CAMPAIGN)
    args = parser.parse_args()

    manifest = sync_manifest(
        summary_csv=args.summary_csv,
        inputs_manifest=args.inputs_manifest,
        out_dir=args.out_dir,
        top_n=args.top_n,
    )
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"jobs: {len(rows)}")
    print(f"manifest: {manifest}")
    if rows:
        print(f"rank range: {rows[0]['rank']} - {rows[-1]['rank']}")


if __name__ == "__main__":
    main()
