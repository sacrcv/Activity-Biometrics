#!/usr/bin/env bash
# Stage 0: train the bias-less GaitGL teacher on silhouettes.
#
#   bash scripts/train_teacher.sh configs/teacher_ntu_rgb_ab.yaml
#   bash scripts/train_teacher.sh configs/teacher_ntu_rgb_ab.yaml run.max_epoch=30
#
# Must be run before scripts/train.sh: ABNet distils this model's identity
# distribution (Eq. 1), and each dataset needs its own teacher because the
# label space differs.
#
# GPU count comes from CUDA_VISIBLE_DEVICES, or NPROC_PER_NODE if you set it.
set -euo pipefail

CFG="${1:?usage: bash scripts/train_teacher.sh <teacher_config.yaml> [key=value ...]}"
shift || true

cd "$(dirname "$0")/.."

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NPROC_PER_NODE=$(python -c "import torch; print(max(torch.cuda.device_count(), 1))")
  fi
fi

MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p outputs
LOG_FILE="outputs/teacher_$(basename "${CFG}" .yaml)_$(date +%Y%m%d_%H%M%S).log"
echo "config: ${CFG}"
echo "gpus:   ${NPROC_PER_NODE}"
echo "log:    ${LOG_FILE}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
    train_teacher.py --cfg "${CFG}" "$@" 2>&1 | tee "${LOG_FILE}"
else
  python train_teacher.py --cfg "${CFG}" "$@" 2>&1 | tee "${LOG_FILE}"
fi
