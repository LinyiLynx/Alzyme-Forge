#!/usr/bin/env python3
"""Launch Boltz-2 predictions across multiple GPUs."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "inputs" / "top200"
DEFAULT_OUT_DIR = ROOT / "results" / "top200"
CONDA_ENV = "colabfold_gpu"


def load_manifest(input_dir: Path) -> list[dict]:
    manifest = input_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    with open(manifest, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def is_done(out_dir: Path, job_name: str) -> bool:
    candidates = [
        out_dir / f"boltz_results_{job_name}",
        out_dir / "boltz_results" / job_name,
    ]
    for job_out in candidates:
        if job_out.exists() and (any(job_out.rglob("*.cif")) or any(job_out.rglob("*.pdb"))):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU Boltz launcher")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--model", choices=["boltz1", "boltz2"], default="boltz2")
    parser.add_argument("--use-msa-server", action="store_true", default=True)
    parser.add_argument("--skip-done", action="store_true", default=True)
    parser.add_argument("--max-jobs", type=int, default=0, help="Limit jobs per GPU (0=all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(args.input_dir)
    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    pending = []
    for row in rows:
        job_name = row["job_name"]
        if args.skip_done and is_done(args.out_dir, job_name):
            continue
        pending.append(row)

    print(f"Total jobs in manifest: {len(rows)}")
    print(f"Pending jobs: {len(pending)}")
    print(f"GPUs: {gpu_list}")

    if not pending:
        print("Nothing to run.")
        return

    chunks: list[list[dict]] = [[] for _ in gpu_list]
    for i, row in enumerate(pending):
        chunks[i % len(gpu_list)].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[int, subprocess.Popen, object]] = []
    for gpu_id, chunk in zip(gpu_list, chunks):
        if not chunk:
            continue
        if args.max_jobs > 0:
            chunk = chunk[: args.max_jobs]

        queue_csv = log_dir / f"queue_gpu{gpu_id}.csv"
        with open(queue_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(chunk[0].keys()))
            writer.writeheader()
            writer.writerows(chunk)

        worker = ROOT / "run_boltz_worker.py"
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            CONDA_ENV,
            "python",
            "-u",
            str(worker),
            "--queue-csv",
            str(queue_csv),
            "--out-dir",
            str(args.out_dir),
            "--model",
            args.model,
        ]
        if args.use_msa_server:
            cmd.append("--use-msa-server")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        env["BOLTZ_CACHE"] = str(ROOT / "cache")
        env["JAX_COMPILATION_CACHE_DIR"] = str(ROOT / "jax_cache")

        log_path = log_dir / f"gpu_{gpu_id}.log"
        print(f"GPU {gpu_id}: {len(chunk)} jobs -> {log_path}")

        if args.dry_run:
            print(" ".join(cmd))
            continue

        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((gpu_id, proc, log_fh))

    if args.dry_run:
        return

    print(f"Launched {len(procs)} workers. Waiting...")
    for gpu_id, proc, log_fh in procs:
        code = proc.wait()
        log_fh.close()
        status = "OK" if code == 0 else f"FAILED ({code})"
        print(f"GPU {gpu_id}: {status}")

    print("All workers finished.")


if __name__ == "__main__":
    main()
