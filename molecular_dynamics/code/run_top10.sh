#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /home/jnz/miniconda3/etc/profile.d/conda.sh
conda activate colabfold_gpu

export PROD_NS="${PROD_NS:-2.0}"
export GPUS="${GPUS:-0,1,2,3}"

python prepare_top10.py --top-n 10
mkdir -p neg_fix_v1_top10/logs

nohup python -u pipeline_scheduler.py \
  --gpus "${GPUS}" \
  --prod-ns "${PROD_NS}" \
  --max-workers 4 \
  > neg_fix_v1_top10/logs/scheduler.log 2>&1 &

echo "scheduler pid=$!"
echo "log: neg_fix_v1_top10/logs/scheduler.log"
echo "PROD_NS=${PROD_NS} GPUS=${GPUS}"
