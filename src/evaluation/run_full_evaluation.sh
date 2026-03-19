#!/usr/bin/env bash

set -euo pipefail

ml miniforge/24.3.0-0
CONDA_BASE="$(dirname "$(dirname "$(which conda)")")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/ehpc399/vincent/envs/HitPF

# Full-evaluation outputs for the properties regression models.

ARC_MLPF="/gpfs/scratch/ehpc399/vincent/models/PROPS_500k_arc_DoloresLike_20260309_163158/showers_df_evaluation/eval_full_500k_arc.pkl0_0_None.pt"
O5_MLPF="/gpfs/scratch/ehpc399/vincent/models/PROPS_500k_05_DoloresLike_20260309_130956/showers_df_evaluation/eval_full_500k_05.pkl0_0_None.pt"

# Leave these empty for a pure HitPF comparison.
# If you set both and put "pandora" into DATATYPE, the script will plot ARC/05 HitPF plus ARC/05 Pandora.
ARC_PANDORA=""
O5_PANDORA=""

DATATYPE="hitpf"
OUTPUT_DIR="/gpfs/scratch/ehpc399/vincent/models/full_evaluation_compare_500k_arc_05"

CMD=(
  python -m src.evaluation.full_evaluation
  --arc-mlpf "${ARC_MLPF}"
  --o5-mlpf "${O5_MLPF}"
  --datatype "${DATATYPE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${ARC_PANDORA}" && -n "${O5_PANDORA}" ]]; then
  CMD+=(--arc-pandora "${ARC_PANDORA}" --o5-pandora "${O5_PANDORA}")
fi

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
