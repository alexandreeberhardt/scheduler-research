#!/bin/bash -l
#SBATCH --job-name=eval_candidate
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --output=logs_eval_%j.out

set -euo pipefail

module load enseignement/GLO-4030

cd /project/ens-h26-glo4030-projet-eq02/model
# source env_eevdf/bin/activate

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/evaluate_baseline.py \
  --dataset-dir artifacts/candidate_dataset \
  --checkpoint artifacts/training_runs/candidate_baseline/best_model.pt \
  --artifacts-json artifacts/training_runs/candidate_baseline/dataset_artifacts.json \
  --run-config artifacts/training_runs/candidate_baseline/run_config.json \
  --output-dir artifacts/evaluation/candidate_baseline \
  --batch-size 256 \
  --num-workers 4 \
  --max-history 256
