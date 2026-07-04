#!/bin/bash -l
#SBATCH --job-name=distill_lstm
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --output=logs_distill_%j.out

set -euo pipefail

module load enseignement/GLO-4030

cd /project/ens-h26-glo4030-projet-eq02/LSTM
# source env_eevdf/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/train_distillation.py \
  --dataset-dir artifacts/candidate_dataset \
  --output-dir artifacts/training_runs/lstm_distilled \
  --batch-size 512 \
  --epochs 5 \
  --lr 1e-3 \
  --max-history 64 \
  --num-workers 4 \
  --teacher-temperature 1.0 \
  --ce-weight 0.1 \
  --listwise-weight 1.0 \
  --pairwise-weight 0.2 \
  --task-hidden-dim 64 \
  --history-hidden-dim 64 \
  --num-layers 2 \
  --dropout 0.1 \
  --weight-decay 1e-5 \
  --early-stopping-patience 3 \
  --wandb-mode offline