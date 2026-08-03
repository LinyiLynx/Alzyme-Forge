#!/usr/bin/env python3
"""Run GROMACS setup + OpenMM MD for one manifest job."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from config import DEFAULT_PROD_NS, job_dir
from run_openmm_md import run_openmm_md
from setup_md_job import setup_job


def run_one(job: dict, gpu_id: int, prod_ns: float) -> dict:
    job_id = job["job_id"]
    jdir = job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    work = jdir / "work"
    out = jdir / "outputs"
    result_path = jdir / "md_status.json"

    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "done":
            return existing

    started = time.time()
    status = {"job_id": job_id, "gpu_id": gpu_id, "status": "running"}
    result_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    try:
        setup_job(jdir, Path(job["structure_path"]), smiles=job.get("smiles", ""))
        md = run_openmm_md(work, out, gpu_id=gpu_id, prod_ns=prod_ns)
        status = {
            "job_id": job_id,
            "rank": job.get("rank"),
            "protein_id": job.get("protein_id"),
            "gpu_id": gpu_id,
            "prod_ns": prod_ns,
            "elapsed_min": (time.time() - started) / 60,
            "status": "done",
            **md,
        }
    except Exception as exc:
        status = {
            "job_id": job_id,
            "gpu_id": gpu_id,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_min": (time.time() - started) / 60,
        }

    result_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prod-ns", type=float, default=DEFAULT_PROD_NS)
    args = parser.parse_args()

    import csv

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    job = next(r for r in rows if r["job_id"] == args.job_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result = run_one(job, gpu_id=0, prod_ns=args.prod_ns)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("status") == "done" else 1)


if __name__ == "__main__":
    main()
