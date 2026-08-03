#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /home/jnz/miniconda3/etc/profile.d/conda.sh

conda activate esmc
python prepare_top200_inputs.py --top-n 200
python migrate_legacy_results.py

conda activate colabfold_gpu
export BOLTZ_CACHE="$ROOT/cache"
mkdir -p neg_fix_v1_top200/logs

nohup python -u pipeline_scheduler.py \
  --gpus 0,1,2,3 \
  --workers-per-gpu 2 \
  --model boltz2 \
  --use-msa-server \
  > neg_fix_v1_top200/logs/scheduler.log 2>&1 &

echo "scheduler pid=$!"
sleep 3
python build_summary.py
tail -5 neg_fix_v1_top200/logs/scheduler.log || true
