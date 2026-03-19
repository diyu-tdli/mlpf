#!/bin/bash
#SBATCH --job-name=mlpf_eval_clust_500k_arc
#SBATCH --output=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc399

set -euo pipefail

ml miniforge/24.3.0-0
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF

cd /gpfs/scratch/ehpc399/vincent/code/mlpf
nvidia-smi
pwd

export WANDB_MODE=offline
export WANDB_PROJECT=mlpf_debug_eval
export WANDB_ENTITY=ml4hep
export WANDB_DIR="/gpfs/scratch/ehpc399/vincent/wandb/${SLURM_JOB_ID}"
mkdir -p "${WANDB_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export BLIS_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=8

# Adjust DATA_TEST to your preferred evaluation sample if needed.
DATA_TEST="/gpfs/scratch/ehpc399/vincent/data/500k_mix/evaluation/evaluation_eCH/arc/*.parquet"
CFG_DATA="config_files/config_hits_track_v4.yaml"
CFG_NET="src/models/wrapper/example_mode_gatr_noise.py"
MODEL_DIR="/gpfs/scratch/ehpc399/vincent/models/500k_arc"

CKPT="$(ls -1t "${MODEL_DIR}"/*.ckpt 2>/dev/null | head -n1 || true)"
if [[ -z "${CKPT}" ]]; then
  echo "No checkpoint found in ${MODEL_DIR}"
  exit 1
fi
echo "Using checkpoint: ${CKPT}"

python -m src.train_lightning1 \
  --predict \
  --data-test "${DATA_TEST}" \
  --name-output "eval_clustering_500k_arc.pkl" \
  --data-config "${CFG_DATA}" \
  -clust \
  -clust_dim 3 \
  --network-config "${CFG_NET}" \
  --model-prefix "${MODEL_DIR}" \
  --load-model-weights "${CKPT}" \
  --wandb-displayname "eval_clustering_500k_arc" \
  --num-workers 16 \
  --gpus 0 \
  --batch-size 1 \
  --start-lr 1e-3 \
  --num-epochs 100 \
  --optimizer ranger \
  --fetch-step 0.1 \
  --condensation \
  --log-wandb \
  --wandb-projectname "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --frac_cluster_loss 0 \
  --qmin 1 \
  --use-average-cc-pos 0.99 \
  --lr-scheduler reduceplateau \
  --tracks \
  --add-track-chis
