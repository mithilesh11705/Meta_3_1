#!/bin/sh
set -eux

echo "[INFO] Starting local API on :7860 for Space health checks..."
uvicorn server.app:app --host 0.0.0.0 --port 7860 &
API_PID=$!
echo "[INFO] API PID: ${API_PID}"
echo "[INFO] Working directory: $(pwd)"
echo "[INFO] User: $(id)"

# Allow model download auth from either HF_TOKEN or HUGGINGFACE_HUB_TOKEN secret.
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
echo "[INFO] HUGGINGFACE_HUB_TOKEN present: $( [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] && echo yes || echo no )"

ENV_BASE_URL="${TRAIN_ENV_BASE_URL:-https://hitanshjain1812-mete-final.hf.space}"
MODEL_NAME="${TRAIN_MODEL_NAME:-google/gemma-2-9b-it}"
OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/home/appuser/artifacts/grpo_hf_space_run}"
mkdir -p "${OUTPUT_DIR}"

echo "[INFO] Starting GRPO training job..."
EXTRA_ARGS=""
if [ "${TRAIN_SKIP_INITIAL_EVAL:-0}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --skip-initial-eval"
fi
if [ "${TRAIN_STRICT_JSON_REWARD:-0}" = "1" ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --strict-json-reward"
  if [ -n "${TRAIN_STRICT_JSON_WARMUP_STEPS:-}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --strict-json-warmup-steps ${TRAIN_STRICT_JSON_WARMUP_STEPS}"
  fi
else
  EXTRA_ARGS="${EXTRA_ARGS} --no-strict-json-reward"
fi

python -u train_grpo.py \
  --env-base-url "${ENV_BASE_URL}" \
  --model-name "${MODEL_NAME}" \
  --num-samples "${TRAIN_NUM_SAMPLES:-120}" \
  --num-train-epochs "${TRAIN_NUM_TRAIN_EPOCHS:-1}" \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${TRAIN_GRAD_ACCUM:-4}" \
  --num-generations "${TRAIN_NUM_GENERATIONS:-4}" \
  --learning-rate "${TRAIN_LEARNING_RATE:-1e-5}" \
  --episodes-per-task "${TRAIN_EPISODES_PER_TASK:-4}" \
  --max-episode-steps "${TRAIN_MAX_EPISODE_STEPS:-3}" \
  --eval-tasks-per-difficulty "${TRAIN_EVAL_TASKS_PER_DIFFICULTY:-1}" \
  --parse-failure-reward "${TRAIN_PARSE_FAILURE_REWARD:-0.001}" \
  --max-completion-length "${TRAIN_MAX_COMPLETION_LENGTH:-96}" \
  --max-new-tokens "${TRAIN_MAX_NEW_TOKENS:-96}" \
  --suppress-train-log-keys "${TRAIN_SUPPRESS_LOG_KEYS:-loss,completions/clipped_ratio,completions/mean_terminated_length}" \
  --output-dir "${OUTPUT_DIR}" \
  ${EXTRA_ARGS} 2>&1 | tee "${OUTPUT_DIR}/training.log"

echo "[INFO] Training finished. Artifacts directory: ${OUTPUT_DIR}"
echo "[INFO] Keeping Space app alive on :7860"
wait "${API_PID}"
