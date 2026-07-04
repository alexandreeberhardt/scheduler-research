#!/bin/bash -l
#SBATCH --job-name=eval_distill
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --output=logs_eval_distill_%j.out

set -euo pipefail

module load enseignement/GLO-4030

cd /project/ens-h26-glo4030-projet-eq02/LSTM
# source env_eevdf/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/evaluate_distilled.py \
  --dataset-dir artifacts/candidate_dataset \
  --checkpoint artifacts/training_runs/lstm_distilled/best_model.pt \
  --artifacts-json artifacts/training_runs/lstm_distilled/dataset_artifacts.json \
  --run-config artifacts/training_runs/lstm_distilled/run_config.json \
  --output-dir artifacts/evaluation/lstm_distilled \
  --batch-size 1024 \
  --num-workers 4 \
  --max-history 256