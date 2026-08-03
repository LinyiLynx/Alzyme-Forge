#!/usr/bin/env python3
"""Run Boltz predict sequentially for jobs listed in a queue CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="boltz2")
    parser.add_argument("--use-msa-server", action="store_true")
    args = parser.parse_args()

    with open(args.queue_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        yaml_path = Path(row["yaml_path"])
        job_name = row["job_name"]
        job_out = args.out_dir / "boltz_results" / job_name
        print(f"[{idx}/{total}] {job_name}", flush=True)

        cmd = [
            "boltz",
            "predict",
            str(yaml_path),
            "--out_dir",
            str(args.out_dir),
            "--model",
            args.model,
            "--output_format",
            "mmcif",
            "--devices",
            "1",
            "--accelerator",
            "gpu",
            "--recycling_steps",
            "3",
            "--sampling_steps",
            "200",
            "--diffusion_samples",
            "1",
            "--no_kernels",
            "--override",
        ]
        if args.use_msa_server:
            cmd.append("--use_msa_server")

        started = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - started
        status = "ok" if result.returncode == 0 else f"fail({result.returncode})"
        print(f"  -> {status} in {elapsed/60:.1f} min", flush=True)

        if result.returncode != 0:
            print(f"  ERROR running {yaml_path}", file=sys.stderr, flush=True)

    print("Worker finished.", flush=True)


if __name__ == "__main__":
    main()
