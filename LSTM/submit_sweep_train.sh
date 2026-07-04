#!/bin/bash -l
#SBATCH --job-name=candidate_sweep
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch_gpu
#SBATCH --gres=shard:4
#SBATCH --output=logs_candidate_sweep_%j.out

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

python scripts/sweep_train.py --dataset-dir artifacts/candidate_dataset
