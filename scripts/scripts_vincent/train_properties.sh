#!/bin/bash
#SBATCH --job-name=mlpf_props_05
#SBATCH --output=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc399/vincent/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Lightning spawns per-GPU workers (if strategy uses DDP etc.)
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1                 # DEBUG: 1 GPU. For full run later set to 4 and --gpus 0,1,2,3
#SBATCH --time=01:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc399

set -euo pipefail

# --- Environment ---
ml miniforge/24.3.0-0
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF

nvidia-smi

cd "/gpfs/scratch/ehpc399/vincent/code/mlpf"
pwd

# --- Runtime env ---
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
export TOKENIZERS_PARALLELISM=false

# Optional speed on H100 tensor cores (harmless):
# export TORCH_LOGS=""  # keep empty unless debugging
# (Better set inside python, but env is ok if your code reads it; otherwise ignore)

# --- Weights & Biases (offline at BSC) ---
export WANDB_MODE=offline
export WANDB_PROJECT=mlpf_debug
export WANDB_ENTITY=ml4hep
export WANDB_DIR=/gpfs/scratch/ehpc399/vincent/wandb/${SLURM_JOB_ID}
mkdir -p "${WANDB_DIR}"

# --- Paths / configs (EDIT THESE) ---
# Properties-training dataset (05):
DATA_DIR="/gpfs/scratch/ehpc399/vincent/data/5k_mix/05/"   # <- anpassen: dein 05 property dataset
CFG_DATA="config_files/config_hits_track_v4.yaml"
CFG_NET="src/models/wrapper/example_mode_gatr_noise.py"

# Output folder for this run:
MODEL_PREFIX="/gpfs/scratch/ehpc399/vincent/models/PROPS_05_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${MODEL_PREFIX}"

# --- Threading ---
CPUS=${SLURM_CPUS_PER_TASK:-80}
NGPU=1
THREADS_PER_RANK=$(( (CPUS + NGPU - 1) / NGPU ))
if [[ $THREADS_PER_RANK -lt 1 ]]; then THREADS_PER_RANK=1; fi

export OMP_NUM_THREADS=${THREADS_PER_RANK}
export MKL_NUM_THREADS=${THREADS_PER_RANK}
export OPENBLAS_NUM_THREADS=${THREADS_PER_RANK}
export BLIS_NUM_THREADS=${THREADS_PER_RANK}
export NUMEXPR_MAX_THREADS=${THREADS_PER_RANK}
export NUMEXPR_NUM_THREADS=${THREADS_PER_RANK}

mkdir -p logs checkpoints

# --- Debug-friendly limits ---
NUM_EPOCHS=1
TRAIN_BATCHES=50          # debug: small
BATCH_SIZE=8              # debug: small-ish

# --- Launch ---
python -m src.train_lightning1 \
  --data-train "${DATA_DIR}" \
  --data-config "${CFG_DATA}" \
  --network-config "${CFG_NET}" \
  --model-prefix "${MODEL_PREFIX}/" \
  --num-workers 16 \
  --gpus 0 \
  --batch-size "${BATCH_SIZE}" \
  --start-lr 1e-3 \
  --num-epochs "${NUM_EPOCHS}" \
  --optimizer ranger \
  --fetch-step 1 \
  --condensation \
  --log-wandb \
  --wandb-displayname "E_PID_05_props_debug" \
  --wandb-projectname "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --frac_cluster_loss 0 \
  --qmin 1 \
  --use-average-cc-pos 0.99 \
  --lr-scheduler reduceplateau \
  --tracks \
  --correction \
  --ec-model gatr-neutrals \
  --regress-pos \
  --add-track-chis \
  --freeze-clustering \
  --regress-unit-p \
  --separate-PID-GATr \
  --n-layers-PID-head 3 \
  --fetch-by-files \
  --train-val-split 0.98 \
  --restrict_PID_charge \
  --PID-4-class \
  --balance-pid-classes \
  --train-batches "${TRAIN_BATCHES}" \
  --use-gt-clusters
