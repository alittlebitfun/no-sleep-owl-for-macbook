#!/usr/bin/env bash
set -Eeuo pipefail

# Durable 8xH20 launcher for the Unified57 smoke/formal run.  This file holds
# paths and hyperparameters only; credentials are intentionally unsupported.

PROJECT_ROOT="${PROJECT_ROOT:-/maas_data/workspaces/bosideng-model-lab}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/maas_data/datasets/bosideng/bosideng_unified57_v1_20260717_r4/train.jsonl}"
SCHEMA_PATH="${SCHEMA_PATH:-${PROJECT_ROOT}/configs/bosideng_unified57_schema.json}"
MODEL_PATH="${MODEL_PATH:-/maas_data/tagvlm/Qwen3-VL-8B-Instruct}"
AGGREGATE18_CHECKPOINT="${AGGREGATE18_CHECKPOINT:-/maas_data/artifacts/bosideng-model-lab/jd_aggregate18/runs/aggregate18_h2_20260715_0315/checkpoints/latest.pt}"
AGGREGATE18_CONFIG="${AGGREGATE18_CONFIG:-/maas_data/artifacts/bosideng-model-lab/unified57_20260717/jd_multilabel_31/delivery/bosideng_jd_aggregate18_lora_step751/model_config.json}"
RUN_ROOT="${RUN_ROOT:-/maas_data/artifacts/bosideng-model-lab/unified57/runs}"
RUN_ID="${RUN_ID:-unified57_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
UNIT_NAME="${UNIT_NAME:-bosideng-unified57-${RUN_ID//[^a-zA-Z0-9_.-]/-}}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

# Safety-first defaults: the first invocation is exactly a 20-step smoke.
# Set MAX_STEPS=0 only after smoke_report.json approves the formal projection.
MAX_STEPS="${MAX_STEPS:-20}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-112896}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
UNIFORM_PER_RANK="${UNIFORM_PER_RANK:-6}"
BALANCED_PER_RANK="${BALANCED_PER_RANK:-2}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
PU_MARGIN="${PU_MARGIN:-1.0}"
PU_LOSS_WEIGHT="${PU_LOSS_WEIGHT:-0.2}"
SAVE_EVERY="${SAVE_EVERY:-20}"
LOG_EVERY="${LOG_EVERY:-1}"
SEED="${SEED:-20260717}"
WALL_CLOCK_HOURS="${WALL_CLOCK_HOURS:-5}"
RESERVE_MINUTES="${RESERVE_MINUTES:-10}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

write_status() {
  local state="$1"
  local exit_code="$2"
  local destination="${RUN_DIR}/exit_status.json"
  local temporary="${destination}.tmp"
  printf '{"state":"%s","exit_code":%s,"updated_at":"%s","unit":"%s"}\n' \
    "${state}" "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${UNIT_NAME}" \
    > "${temporary}"
  mv -f "${temporary}" "${destination}"
}

run_worker() {
  mkdir -p "${RUN_DIR}/logs"
  printf '%s\n' "$$" > "${RUN_DIR}/worker.pid"
  write_status "running" -1

  local checkpoint="${RESUME_CHECKPOINT}"
  if [[ -z "${checkpoint}" && -f "${RUN_DIR}/checkpoints/latest.pt" ]]; then
    checkpoint="${RUN_DIR}/checkpoints/latest.pt"
  fi

  local state_args=()
  if [[ -n "${checkpoint}" ]]; then
    state_args=(--resume "${checkpoint}")
  else
    state_args=(
      --init-from-aggregate18 "${AGGREGATE18_CHECKPOINT}"
      --aggregate18-config "${AGGREGATE18_CONFIG}"
    )
  fi

  local command=(
    "${TORCHRUN_BIN}"
    --standalone
    --nnodes=1
    --nproc-per-node=8
    "${PROJECT_ROOT}/scripts/train_unified57_qwen3vl_multilabel.py"
    --manifest "${TRAIN_MANIFEST}"
    --schema "${SCHEMA_PATH}"
    --model "${MODEL_PATH}"
    --output-dir "${RUN_DIR}"
    "${state_args[@]}"
    --expected-world-size 8
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --uniform-per-rank "${UNIFORM_PER_RANK}"
    --balanced-per-rank "${BALANCED_PER_RANK}"
    --gradient-accumulation "${GRADIENT_ACCUMULATION}"
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --image-max-pixels "${IMAGE_MAX_PIXELS}"
    --learning-rate "${LEARNING_RATE}"
    --pu-margin "${PU_MARGIN}"
    --pu-loss-weight "${PU_LOSS_WEIGHT}"
    --max-steps "${MAX_STEPS}"
    --save-every "${SAVE_EVERY}"
    --log-every "${LOG_EVERY}"
    --seed "${SEED}"
    --wall-clock-hours "${WALL_CLOCK_HOURS}"
    --reserve-minutes "${RESERVE_MINUTES}"
  )

  printf '%q ' "${command[@]}" > "${RUN_DIR}/resolved_command.txt"
  printf '\n' >> "${RUN_DIR}/resolved_command.txt"
  if [[ -f "${RUN_DIR}/smoke_report.json" ]]; then
    mv -f "${RUN_DIR}/smoke_report.json" "${RUN_DIR}/smoke_report.previous.json"
  fi
  set +e
  "${command[@]}" 2>&1 | tee -a "${RUN_DIR}/logs/train.log"
  local training_exit="${PIPESTATUS[0]}"
  local worker_exit="${training_exit}"
  set -e
  if [[ "${training_exit}" -eq 0 ]] && \
    grep -q '"status": "complete"' "${RUN_DIR}/smoke_report.json" 2>/dev/null; then
    write_status "exited" 0
  elif grep -q '"status": "partial"' "${RUN_DIR}/smoke_report.json" 2>/dev/null; then
    write_status "partial" 3
    worker_exit=0
  else
    if [[ "${worker_exit}" -eq 0 ]]; then
      worker_exit=4
    fi
    write_status "failed" "${worker_exit}"
  fi
  exit "${worker_exit}"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
fi

for required_path in \
  "${PROJECT_ROOT}/scripts/train_unified57_qwen3vl_multilabel.py" \
  "${TRAIN_MANIFEST}" \
  "${SCHEMA_PATH}" \
  "${MODEL_PATH}"; do
  if [[ ! -e "${required_path}" ]]; then
    printf 'missing required path: %s\n' "${required_path}" >&2
    exit 2
  fi
done
if [[ -z "${RESUME_CHECKPOINT}" && ! -f "${RUN_DIR}/checkpoints/latest.pt" ]]; then
  for init_path in "${AGGREGATE18_CHECKPOINT}" "${AGGREGATE18_CONFIG}"; do
    if [[ ! -f "${init_path}" ]]; then
      printf 'missing Aggregate18 initialization asset: %s\n' "${init_path}" >&2
      exit 2
    fi
  done
fi
if [[ "$(nvidia-smi -L | wc -l | tr -d ' ')" -ne 8 ]]; then
  printf 'node1 must expose exactly 8 GPUs for this launcher\n' >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs"
printf '%s\n' \
  "RUN_DIR=${RUN_DIR}" \
  "UNIT_NAME=${UNIT_NAME}" \
  "TRAIN_MANIFEST=${TRAIN_MANIFEST}" \
  "SCHEMA_PATH=${SCHEMA_PATH}" \
  "MODEL_PATH=${MODEL_PATH}" \
  "MAX_STEPS=${MAX_STEPS}" \
  "IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS}" \
  "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}" \
  "UNIFORM_PER_RANK=${UNIFORM_PER_RANK}" \
  "BALANCED_PER_RANK=${BALANCED_PER_RANK}" \
  "GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION}" \
  "PU_MARGIN=${PU_MARGIN}" \
  "PU_LOSS_WEIGHT=${PU_LOSS_WEIGHT}" \
  > "${RUN_DIR}/launch_config.env"

systemd-run \
  --unit "${UNIT_NAME}" \
  --collect \
  --working-directory "${PROJECT_ROOT}" \
  --property "Restart=no" \
  --property "TimeoutStopSec=600" \
  --setenv "PROJECT_ROOT=${PROJECT_ROOT}" \
  --setenv "TRAIN_MANIFEST=${TRAIN_MANIFEST}" \
  --setenv "SCHEMA_PATH=${SCHEMA_PATH}" \
  --setenv "MODEL_PATH=${MODEL_PATH}" \
  --setenv "AGGREGATE18_CHECKPOINT=${AGGREGATE18_CHECKPOINT}" \
  --setenv "AGGREGATE18_CONFIG=${AGGREGATE18_CONFIG}" \
  --setenv "RUN_ROOT=${RUN_ROOT}" \
  --setenv "RUN_ID=${RUN_ID}" \
  --setenv "RUN_DIR=${RUN_DIR}" \
  --setenv "UNIT_NAME=${UNIT_NAME}" \
  --setenv "TORCHRUN_BIN=${TORCHRUN_BIN}" \
  --setenv "MAX_STEPS=${MAX_STEPS}" \
  --setenv "IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS}" \
  --setenv "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}" \
  --setenv "UNIFORM_PER_RANK=${UNIFORM_PER_RANK}" \
  --setenv "BALANCED_PER_RANK=${BALANCED_PER_RANK}" \
  --setenv "GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION}" \
  --setenv "NUM_WORKERS=${NUM_WORKERS}" \
  --setenv "PREFETCH_FACTOR=${PREFETCH_FACTOR}" \
  --setenv "LEARNING_RATE=${LEARNING_RATE}" \
  --setenv "PU_MARGIN=${PU_MARGIN}" \
  --setenv "PU_LOSS_WEIGHT=${PU_LOSS_WEIGHT}" \
  --setenv "SAVE_EVERY=${SAVE_EVERY}" \
  --setenv "LOG_EVERY=${LOG_EVERY}" \
  --setenv "SEED=${SEED}" \
  --setenv "WALL_CLOCK_HOURS=${WALL_CLOCK_HOURS}" \
  --setenv "RESERVE_MINUTES=${RESERVE_MINUTES}" \
  --setenv "RESUME_CHECKPOINT=${RESUME_CHECKPOINT}" \
  /usr/bin/env bash "${PROJECT_ROOT}/scripts/launch_unified57_node1.sh" --worker

printf 'unit=%s\nrun_dir=%s\nlog=%s\nprogress=%s\n' \
  "${UNIT_NAME}" \
  "${RUN_DIR}" \
  "${RUN_DIR}/logs/train.log" \
  "${RUN_DIR}/progress.json"
