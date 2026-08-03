#!/usr/bin/env python3
"""High-utilization Boltz scheduler: multiple workers per GPU, pipelined MSA + inference."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import BOLTZ_CACHE, COMPOUND, LOGS, MANIFEST, STRUCTURES, WORK


def structure_path(job_id: str) -> Path:
    return STRUCTURES / f"{job_id}.cif"


def work_dir(job_id: str) -> Path:
    return WORK / job_id


def is_done(job_id: str) -> bool:
    if structure_path(job_id).exists():
        return True
    wd = work_dir(job_id)
    return any(wd.rglob(f"predictions/*/{job_id}_model_0.cif"))


def collect_structure(job_id: str) -> bool:
    dest = structure_path(job_id)
    if dest.exists():
        return True
    wd = work_dir(job_id)
    hits = sorted(wd.rglob(f"predictions/*/{job_id}_model_0.cif"))
    if not hits:
        return False
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(hits[-1].read_bytes())
    return True


def needs_msa(job_id: str, yaml_path: Path) -> bool:
    wd = work_dir(job_id)
    stem = yaml_path.stem
    processed = wd / f"boltz_results_{stem}" / "processed" / "manifest.json"
    if processed.exists():
        return False
    legacy = list(wd.glob(f"boltz_results_{stem}/processed/manifest.json"))
    return not legacy


def run_job(job: dict, gpu_id: int, model: str, use_msa_server: bool) -> dict:
    job_id = job["job_id"]
    yaml_path = Path(job["yaml_path"])
    out = work_dir(job_id)
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["BOLTZ_CACHE"] = str(BOLTZ_CACHE)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["PATH"] = "/home/jnz/miniconda3/envs/colabfold_gpu/bin:" + env.get("PATH", "")

    cmd = [
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out),
        "--model",
        model,
        "--output_format",
        "mmcif",
        "--devices",
        "1",
        "--accelerator",
        "gpu",
        "--recycling_steps",
        "3",
        "--sampling_steps",
        "100",
        "--diffusion_samples",
        "1",
        "--no_kernels",
        "--preprocessing-threads",
        "4",
    ]

    msa_needed = needs_msa(job_id, yaml_path)
    if use_msa_server and msa_needed:
        cmd.append("--use_msa_server")
        cmd.append("--override")
    elif not msa_needed:
        pass  # reuse cached MSA/processed
    else:
        cmd.append("--override")

    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and collect_structure(job_id)

    return {
        "job_id": job_id,
        "gpu_id": gpu_id,
        "status": "done" if ok else "fail",
        "elapsed_min": round(elapsed / 60, 2),
        "msa_phase": msa_needed,
        "stderr_tail": (proc.stderr or "")[-500:] if not ok else "",
    }


def _worker(args: tuple) -> dict:
    job, gpu_id, model, use_msa_server = args
    return run_job(job, gpu_id, model, use_msa_server)


def load_manifest() -> list[dict]:
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--model", default="boltz2")
    parser.add_argument("--use-msa-server", action="store_true", default=True)
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()

    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    workers = [g for g in gpu_list for _ in range(args.workers_per_gpu)]
    jobs = [j for j in load_manifest() if not is_done(j["job_id"])]
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    LOGS.mkdir(parents=True, exist_ok=True)
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print(f"pending={len(jobs)} workers={len(workers)} gpus={gpu_list} ({args.workers_per_gpu}/gpu)")

    if not jobs:
        print("All jobs done.")
        return

    task_args = [(job, workers[i % len(workers)], args.model, args.use_msa_server) for i, job in enumerate(jobs)]

    done = fail = 0
    with ProcessPoolExecutor(max_workers=len(workers)) as pool:
        futures = {pool.submit(_worker, ta): ta[0]["job_id"] for ta in task_args}
        for fut in as_completed(futures):
            jid = futures[fut]
            try:
                res = fut.result()
                if res["status"] == "done":
                    done += 1
                    print(f"[OK] {jid} gpu={res['gpu_id']} {res['elapsed_min']}min msa={res['msa_phase']}")
                else:
                    fail += 1
                    print(f"[FAIL] {jid} gpu={res['gpu_id']} {res['stderr_tail']}", file=sys.stderr)
            except Exception as exc:
                fail += 1
                print(f"[ERR] {jid}: {exc}", file=sys.stderr)

    print(f"finished: done={done} fail={fail}")


if __name__ == "__main__":
    main()
