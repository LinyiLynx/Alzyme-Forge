#!/usr/bin/env python3
"""Rolling 4-GPU MD scheduler: keeps GPUs busy until summary.csv is fully processed."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from config import CAMPAIGN, DEFAULT_POLL_SEC, DEFAULT_PROD_NS, LOGS, MANIFEST, SUMMARY_CSV
from prepare_manifest import sync_manifest


def load_jobs(manifest: Path) -> list[dict]:
    if not manifest.exists():
        return []
    with open(manifest, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_status(job_id: str) -> str | None:
    from config import status_path

    path = status_path(job_id)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if '"status": "done"' in text:
        return "done"
    if '"status": "running"' in text:
        return "running"
    if '"status": "failed"' in text:
        return "failed"
    return None


def busy_gpus_from_ps() -> set[int]:
    busy: set[int] = set()
    try:
        out = subprocess.check_output(["pgrep", "-af", "Gromacs/md_worker.py"], text=True)
    except subprocess.CalledProcessError:
        return busy
    for line in out.splitlines():
        match = re.search(r"--gpu (\d+)", line)
        if match:
            busy.add(int(match.group(1)))
    return busy


def pending_jobs(manifest: Path, retry_failed: bool) -> list[dict]:
    jobs: list[dict] = []
    for job in load_jobs(manifest):
        job_id = job["job_id"]
        status = read_status(job_id)
        if status == "done":
            continue
        if status == "running":
            continue
        if status == "failed" and not retry_failed:
            continue
        jobs.append(job)
    jobs.sort(key=lambda j: int(j["rank"]))
    return jobs


def worker(job: dict, gpu_id: int, prod_ns: float, root: Path, manifest: Path) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = "/home/jnz/miniconda3/envs/colabfold_gpu/bin:" + env.get("PATH", "")

    cmd = [
        sys.executable,
        "-u",
        str(root / "md_worker.py"),
        "--job-id",
        job["job_id"],
        "--manifest",
        str(manifest),
        "--gpu",
        "0",
        "--prod-ns",
        str(prod_ns),
    ]
    log_path = LOGS / f"{job['job_id']}_gpu{gpu_id}.log"
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n=== {job['job_id']} GPU {gpu_id} {time.strftime('%F %T')} ===\n")
        log_fh.flush()
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=log_fh, stderr=subprocess.STDOUT, check=False)
    return {
        "job_id": job["job_id"],
        "rank": job.get("rank"),
        "gpu_id": gpu_id,
        "returncode": proc.returncode,
        "log": str(log_path),
    }


def refresh_manifest(manifest: Path, summary_csv: Path) -> int:
    sync_manifest(summary_csv=summary_csv, out_dir=manifest.parent)
    return len(load_jobs(manifest))


def run_rolling(
    manifest: Path,
    summary_csv: Path,
    gpu_list: list[int],
    prod_ns: float,
    poll_sec: int,
    retry_failed: bool,
    root: Path,
) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    n_gpus = len(gpu_list)
    total_done = total_fail = 0
    idle_rounds = 0

    print(
        f"Rolling scheduler: gpus={gpu_list} prod_ns={prod_ns} poll={poll_sec}s",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=n_gpus) as pool:
        active: dict[Future, int] = {}

        while True:
            n_ready = refresh_manifest(manifest, summary_csv)
            queue = pending_jobs(manifest, retry_failed=retry_failed)

            busy = busy_gpus_from_ps() | set(active.values())
            for gpu in gpu_list:
                if gpu in busy:
                    continue
                if gpu in active.values():
                    continue
                if not queue:
                    break
                job = queue.pop(0)
                fut = pool.submit(worker, job, gpu, prod_ns, root, manifest)
                active[fut] = gpu
                print(f"[start] rank={job['rank']} {job['job_id']} -> GPU {gpu}", flush=True)

            if not active:
                external_busy = busy_gpus_from_ps()
                if external_busy:
                    time.sleep(15)
                    continue

                pending_n = len(pending_jobs(manifest, retry_failed=retry_failed))
                running_n = sum(1 for j in load_jobs(manifest) if read_status(j["job_id"]) == "running")
                if pending_n == 0 and running_n == 0:
                    idle_rounds += 1
                    print(
                        f"Idle round {idle_rounds}: waiting for new Alphafold structures "
                        f"(ready={n_ready} pending={pending_n} running={running_n})",
                        flush=True,
                    )
                    time.sleep(poll_sec)
                    continue

                time.sleep(5)
                continue

            idle_rounds = 0
            done_set, _ = wait(active.keys(), return_when=FIRST_COMPLETED, timeout=30)
            if not done_set:
                continue
            for fut in done_set:
                gpu = active.pop(fut)
                try:
                    res = fut.result()
                    ok = res["returncode"] == 0
                    if ok:
                        total_done += 1
                        print(f"[OK] rank={res.get('rank')} {res['job_id']} GPU{gpu}", flush=True)
                    else:
                        total_fail += 1
                        print(f"[FAIL] rank={res.get('rank')} {res['job_id']} GPU{gpu} -> {res['log']}", flush=True)
                except Exception as exc:
                    total_fail += 1
                    print(f"[ERR] GPU{gpu}: {exc}", file=sys.stderr, flush=True)

    print(f"Session finished: ok={total_done} fail={total_fail}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--prod-ns", type=float, default=DEFAULT_PROD_NS)
    parser.add_argument("--poll-sec", type=int, default=DEFAULT_POLL_SEC)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    refresh_manifest(args.manifest, args.summary_csv)

    run_rolling(
        manifest=args.manifest,
        summary_csv=args.summary_csv,
        gpu_list=gpu_list,
        prod_ns=args.prod_ns,
        poll_sec=args.poll_sec,
        retry_failed=args.retry_failed,
        root=root,
    )


if __name__ == "__main__":
    main()
