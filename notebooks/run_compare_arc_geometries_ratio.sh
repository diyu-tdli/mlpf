#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/eos/user/v/vriecher/mlpf_arc/mlpf/notebooks/compare_arc_geometries_ratio.py"
DATASET_ROOT="/eos/user/v/vriecher/mlpf_events_new/CLD_o2_v05_ARC_ARCmod_test_5k"
OUTPUT_DIR="${DATASET_ROOT}/comparison_plots_ratio"

# Versuche zuerst direkten Python-Pfad aus dem env.
PYTHON_BIN="/tmp/vriecher/envs/pytorch_cpuOnly/bin/python"
if [[ -x "${PYTHON_BIN}" ]]; then
  echo "Using python: ${PYTHON_BIN}"
else
  # Fallback: conda aktivieren
  if [[ -f "/tmp/vriecher/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source /tmp/vriecher/miniforge3/etc/profile.d/conda.sh
    conda activate /tmp/vriecher/envs/pytorch_cpuOnly
    PYTHON_BIN="python"
  else
    echo "ERROR: Konnte env nicht aktivieren. /tmp/vriecher/miniforge3 fehlt."
    exit 1
  fi
fi

WORKERS="${WORKERS:-3}"
LOG_EVERY="${LOG_EVERY:-1000}"
MAX_FILES="${MAX_FILES:-}"
MAX_EVENTS_PER_FILE="${MAX_EVENTS_PER_FILE:-5000}"
DPI="${DPI:-140}"

CMD=("${PYTHON_BIN}" "${SCRIPT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-events-per-file "${MAX_EVENTS_PER_FILE}" \
  --dpi "${DPI}" \
  --workers "${WORKERS}" \
  --log-every "${LOG_EVERY}")

if [[ -n "${MAX_FILES}" ]]; then
  CMD+=(--max-files "${MAX_FILES}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Output files:"
ls -1 "${OUTPUT_DIR}" | head -n 200
