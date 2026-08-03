#!/usr/bin/env python3
"""Migrate legacy results/top200 into neg_fix_v1_top200 layout."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from config import COMPOUND, MANIFEST, STRUCTURES, WORK

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "results" / "top200"
LEGACY_MANIFEST = ROOT / "inputs" / "top200" / "manifest.csv"


def legacy_to_job_id(name: str) -> str | None:
    m = re.match(r"rank(\d{3})_compound_1_(.+)", name)
    if m:
        return f"rank{m.group(1)}_{m.group(2)}"
    m = re.match(r"rank(\d{3})_(.+)", name)
    if m:
        return f"rank{m.group(1)}_{m.group(2)}"
    return None


def migrate_structures() -> int:
    count = 0
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    for legacy_dir in LEGACY.glob("boltz_results_rank*"):
        stem = legacy_dir.name.replace("boltz_results_", "")
        job_id = legacy_to_job_id(stem)
        if not job_id:
            continue
        hits = list(legacy_dir.glob(f"predictions/*/{stem}_model_0.cif"))
        if not hits:
            continue
        dest = STRUCTURES / f"{job_id}.cif"
        if not dest.exists():
            shutil.copy2(hits[0], dest)
            count += 1
        wd = WORK / job_id
        wd.mkdir(parents=True, exist_ok=True)
        legacy_work = WORK / job_id / legacy_dir.name
        if not legacy_work.exists() and not any(wd.iterdir()):
            shutil.move(str(legacy_dir), str(wd / legacy_dir.name))
    return count


def rebuild_manifest_from_legacy() -> None:
    if MANIFEST.exists():
        return
    if not LEGACY_MANIFEST.exists():
        return
    rows = []
    with open(LEGACY_MANIFEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            jid = legacy_to_job_id(row["job_name"])
            if not jid:
                continue
            rows.append(
                {
                    "rank": row["rank"],
                    "job_id": jid,
                    "compound_id": row["compound_id"],
                    "protein_id": row["protein_id"],
                    "yaml_path": str(COMPOUND / "inputs" / "yaml" / f"{jid}.yaml"),
                    "screen_score": row["score"],
                    "protein_length": row["protein_length"],
                    "smiles": row["smiles"],
                    "gt_tier": row.get("gt_tier", ""),
                }
            )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rebuild_manifest_from_legacy()
    n = migrate_structures()
    print(f"migrated {n} structures -> {STRUCTURES}")
    print(f"existing structures: {len(list(STRUCTURES.glob('*.cif')))}")


if __name__ == "__main__":
    main()
