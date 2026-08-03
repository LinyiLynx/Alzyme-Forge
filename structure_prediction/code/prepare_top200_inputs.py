#!/usr/bin/env python3
"""Extract top-N pairs and generate Boltz YAML inputs (clean naming)."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import yaml

from config import COMPOUND, MANIFEST, YAML_DIR


def safe_name(text: str, max_len: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:max_len]


def make_yaml(protein_seq: str, smiles: str) -> dict:
    return {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": protein_seq}},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
    }


def job_id(rank: int, protein_id: str) -> str:
    return f"rank{rank:03d}_{safe_name(protein_id)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "predict" / "proteome_predictions_neg_fix_v1.csv",
    )
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    top = df[df["prediction"] == 1].sort_values("score", ascending=False).head(args.top_n)
    top = top.reset_index(drop=True)

    YAML_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, row in top.iterrows():
        rank = i + 1
        compound_id = str(row["compound_id"])
        protein_id = str(row["Protein_ID"])
        jid = job_id(rank, protein_id)
        yaml_path = YAML_DIR / f"{jid}.yaml"

        with open(yaml_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                make_yaml(str(row["Protein_Sequence"]).strip(), str(row["Substrate_SMILES"])),
                fh,
                sort_keys=False,
                allow_unicode=True,
            )

        rows.append(
            {
                "rank": rank,
                "job_id": jid,
                "compound_id": compound_id,
                "protein_id": protein_id,
                "yaml_path": str(yaml_path),
                "screen_score": float(row["score"]),
                "protein_length": int(row["protein_length"]),
                "smiles": str(row["Substrate_SMILES"]),
                "gt_tier": row.get("gt_tier", ""),
            }
        )

    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"compound: {compound_id} | jobs: {len(rows)}")
    print(f"manifest: {MANIFEST}")
    print(f"score range: {rows[-1]['screen_score']:.4f} - {rows[0]['screen_score']:.4f}")


if __name__ == "__main__":
    main()
