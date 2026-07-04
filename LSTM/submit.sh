#!/bin/bash -l
#SBATCH --job-name=candidate_train
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --output=logs_candidate_%j.out

set -euo pipefail

module load enseignement/GLO-4030

cd /project/ens-h26-glo4030-projet-eq02/LSTM
# source env_eevdf/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# echo "===== DEBUG ENV ====="
# which python
# python -V
# echo "PYTHONPATH=$PYTHONPATH"
# echo "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"


# python - <<'PY'
# import sys
# print("executable:", sys.executable)
# print("sys.path:")
# for p in sys.path:
#    print("  ", p)
# import torch
# print("torch:", torch.__version__)
# print("torch file:", torch.__file__)
# PY

python scripts/train_baseline.py \
  --dataset-dir artifacts/candidate_dataset \
  --output-dir artifacts/training_runs/candidate_baseline \
  --batch-size 512 \
  --epochs 10 \
  --lr 3e-3 \
  --max-history 256 \
  --num-workers 4 \
  --task-hidden-dim 128 \
  --history-hidden-dim 128 \
  --num-layers 2 \
  --dropout 0.1 \
  --weight-decay 1e-5 \
  --early-stopping-patience 3 \
  --wandb-mode offline
