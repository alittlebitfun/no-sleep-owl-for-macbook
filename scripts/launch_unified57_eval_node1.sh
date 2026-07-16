#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/maas_data/artifacts/bosideng-model-lab/unified57_20260717}"
DATASET_ROOT="${DATASET_ROOT:-/maas_data/datasets/bosideng/bosideng_unified57_v1_20260717_r4}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/unified57_formal_e1_b8_20260717_0136}"
MODEL="${MODEL:-/maas_data/tagvlm/Qwen3-VL-8B-Instruct}"
SCHEMA="${SCHEMA:-${PROJECT_ROOT}/configs/bosideng_unified57_schema.json}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/latest.pt}"
VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-${DATASET_ROOT}/val.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-${DATASET_ROOT}/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/evaluation}"
PYTHON="${PYTHON:-/maas_data/tagvlm/venv4train/bin/python}"

MODEL_CONFIG_SHA256="${MODEL_CONFIG_SHA256:-5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661}"
SCHEMA_FILE_SHA256="${SCHEMA_FILE_SHA256:-43620d06b5db44f667803038b5039732bd70140c8522e70cc04158b51aed3a9a}"
VALIDATION_MANIFEST_SHA256="${VALIDATION_MANIFEST_SHA256:-1cbb2aca6c98c32cb1e7666185a5fa5d5a780836cd8c0bcfca712d19d5a42891}"
TEST_MANIFEST_SHA256="${TEST_MANIFEST_SHA256:-cd81ba30c1266afebbef333a69fb4e8ddcb7b12f0ab94c6edfbc34db0039aef8}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-$(sha256sum "${CHECKPOINT}" | awk '{print $1}')}"

mkdir -p "${OUTPUT_DIR}/logs"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

exec "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/evaluate_unified57_multilabel.py" \
  --model "${MODEL}" \
  --model-config-sha256 "${MODEL_CONFIG_SHA256}" \
  --schema "${SCHEMA}" \
  --schema-file-sha256 "${SCHEMA_FILE_SHA256}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --validation-manifest-sha256 "${VALIDATION_MANIFEST_SHA256}" \
  --test-manifest "${TEST_MANIFEST}" \
  --test-manifest-sha256 "${TEST_MANIFEST_SHA256}" \
  --output-dir "${OUTPUT_DIR}" \
  --wall-clock-seconds "${WALL_CLOCK_SECONDS:-2700}" \
  --expected-world-size 8 \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --image-max-pixels "${IMAGE_MAX_PIXELS:-112896}" \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --head-dropout 0.1
