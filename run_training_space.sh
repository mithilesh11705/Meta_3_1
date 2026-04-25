#!/bin/sh
set -eu

echo "[INFO] Starting local API on :7860 for Space health checks..."
uvicorn server.app:app --host 0.0.0.0 --port 7860 &
API_PID=$!

# Allow model download auth from either HF_TOKEN or HUGGINGFACE_HUB_TOKEN secret.
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

ENV_BASE_URL="${TRAIN_ENV_BASE_URL:-https://hitanshjain1812-mete-final.hf.space}"
MODEL_NAME="${TRAIN_MODEL_NAME:-google/gemma-2-9b-it}"
OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-artifacts/grpo_hf_space_run}"

echo "[INFO] Starting GRPO training job..."
python train_grpo.py \
  --env-base-url "${ENV_BASE_URL}" \
  --model-name "${MODEL_NAME}" \
  --num-samples "${TRAIN_NUM_SAMPLES:-120}" \
  --num-train-epochs "${TRAIN_NUM_TRAIN_EPOCHS:-1}" \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${TRAIN_GRAD_ACCUM:-4}" \
  --num-generations "${TRAIN_NUM_GENERATIONS:-2}" \
  --learning-rate "${TRAIN_LEARNING_RATE:-1e-5}" \
  --episodes-per-task "${TRAIN_EPISODES_PER_TASK:-4}" \
  --max-episode-steps "${TRAIN_MAX_EPISODE_STEPS:-3}" \
  --eval-tasks-per-difficulty "${TRAIN_EVAL_TASKS_PER_DIFFICULTY:-1}" \
  --skip-initial-eval \
  --strict-json-reward \
  --parse-failure-reward "${TRAIN_PARSE_FAILURE_REWARD:-0.01}" \
  --max-completion-length "${TRAIN_MAX_COMPLETION_LENGTH:-160}" \
  --max-new-tokens "${TRAIN_MAX_NEW_TOKENS:-160}" \
  --output-dir "${OUTPUT_DIR}" || true

echo "[INFO] Training finished. Artifacts directory: ${OUTPUT_DIR}"
echo "[INFO] Keeping Space app alive on :7860"
wait "${API_PID}"
