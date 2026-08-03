#!/usr/bin/env python3
"""Build master summary table for Boltz predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from config import COMPOUND, MANIFEST, STRUCTURES, SUMMARY, WORK


def find_confidence(job_id: str) -> dict | None:
    wd = WORK / job_id
    if not wd.exists():
        return None
    hits = sorted(wd.rglob("confidence_*model_0.json"))
    if not hits:
        return None
    with open(hits[-1], encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")

    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))

    rows = []
    for job in manifest:
        job_id = job["job_id"]
        cif = STRUCTURES / f"{job_id}.cif"
        conf = find_confidence(job_id)
        row = {
            "rank": job["rank"],
            "job_id": job_id,
            "compound_id": job["compound_id"],
            "protein_id": job["protein_id"],
            "screen_score": job["screen_score"],
            "protein_length": job["protein_length"],
            "gt_tier": job.get("gt_tier", ""),
            "structure_status": "done" if cif.exists() else "pending",
            "structure_path": str(cif) if cif.exists() else "",
            "confidence_score": conf.get("confidence_score") if conf else "",
            "ptm": conf.get("ptm") if conf else "",
            "iptm": conf.get("iptm") if conf else "",
            "ligand_iptm": conf.get("ligand_iptm") if conf else "",
            "complex_plddt": conf.get("complex_plddt") if conf else "",
            "complex_iplddt": conf.get("complex_iplddt") if conf else "",
            "complex_pde": conf.get("complex_pde") if conf else "",
            "complex_ipde": conf.get("complex_ipde") if conf else "",
        }
        rows.append(row)

    done = sum(1 for r in rows if r["structure_status"] == "done")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(SUMMARY, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"summary: {SUMMARY}")
    print(f"structures: {done}/{len(rows)} done")


if __name__ == "__main__":
    main()
