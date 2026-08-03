#!/usr/bin/env python3
"""Aggregate MD job status into a summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import CAMPAIGN, MANIFEST, SUMMARY_CSV, job_dir, status_path
from prepare_manifest import sync_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--out", type=Path, default=CAMPAIGN / "md_summary.csv")
    args = parser.parse_args()

    sync_manifest(summary_csv=args.summary_csv, out_dir=args.manifest.parent)
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        jobs = list(csv.DictReader(fh))

    rows: list[dict] = []
    for job in jobs:
        row = dict(job)
        sp = status_path(job["job_id"])
        row["md_status"] = "pending"
        row["job_dir"] = str(job_dir(job["job_id"]))
        if sp.exists():
            st = json.loads(sp.read_text(encoding="utf-8"))
            row["md_status"] = st.get("status", "unknown")
            row["elapsed_min"] = st.get("elapsed_min", "")
            row["trajectory"] = st.get("trajectory", "")
            row["final_pdb"] = st.get("final_pdb", "")
            if st.get("status") == "failed":
                row["error"] = st.get("error", "")
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["md_status"]] = counts.get(r["md_status"], 0) + 1
    print(f"Wrote {args.out} ({len(rows)} jobs) {counts}")


if __name__ == "__main__":
    main()
