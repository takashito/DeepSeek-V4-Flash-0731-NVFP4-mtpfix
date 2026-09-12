#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731-NVFP4-mtpfix with vLLM on 2x RTX PRO 6000 Blackwell (SM120).
#
# Defaults reflect what is deployed and benchmarked (see ../README.md):
#   * -mtpfix checkpoint  : drafter experts cast MXFP4 -> NVFP4, so DSpark actually drafts well
#   * DSpark k=7, greedy  : NVIDIA's recommended setting for DeepSeek-V4-Flash-DSpark
#   * max-model-len       : 987136 — the drafter's memory is what caps it below 1M
#   * autotune disabled   : startup FlashInfer MoE autotune deadlocks the TP ranks on SM120
#   * DeepGEMM stays on   : VLLM_USE_DEEP_GEMM=0 fails, CUTLASS scaled_mm rejects SM120
#
# Configuration (all optional):
#   MODEL_DIR     checkpoint path on the host, bind-mounted read-only (default below)
#   MODELS_ROOT   directory to mount; defaults to MODEL_DIR's parent
#   DSV4_API_KEY  bearer token for the OpenAI-compatible API; unset = no authentication
#   ENV_FILE      file to source first, e.g. one holding DSV4_API_KEY=...
#   GPUS HOST PORT IMAGE MAX_LEN GPU_UTIL MAX_SEQS MNBT
#   SPEC_ARGS     speculative decoding flags; SPEC_ARGS= (empty) turns MTP off
#   DOCKER_ENV    extra docker flags, e.g. DOCKER_ENV="-e VLLM_LOGGING_LEVEL=DEBUG"
#   EXTRA_ARGS    extra vLLM flags
#
# Examples:
#   DSV4_API_KEY=secret ./serve_dsv4.sh
#   SPEC_ARGS= MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4 MAX_LEN=1048576 ./serve_dsv4.sh
set -euo pipefail

[ -n "${ENV_FILE:-}" ] && . "$ENV_FILE"

MODEL_DIR="${MODEL_DIR:-/data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix}"
MODELS_ROOT="${MODELS_ROOT:-$(dirname "$MODEL_DIR")}"
SPEC_DEFAULT='--speculative-config {"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'

auth=()
[ -n "${DSV4_API_KEY:-}" ] && auth=(--api-key "$DSV4_API_KEY")

docker rm -f dsv4-vllm 2>/dev/null || true
docker run -d --name dsv4-vllm --restart unless-stopped \
  --gpus "\"device=${GPUS:-0,1}\"" --ipc host --network host \
  ${DOCKER_ENV:-} \
  -v "$MODELS_ROOT":"$MODELS_ROOT":ro \
  "${IMAGE:-vllm/vllm-openai:v0.29.0}" \
  "$MODEL_DIR" \
  --served-model-name deepseek-v4-flash \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8002}" ${auth[@]+"${auth[@]}"} \
  --tensor-parallel-size 2 \
  --max-model-len ${MAX_LEN:-987136} \
  --gpu-memory-utilization ${GPU_UTIL:-0.95} \
  --max-num-seqs ${MAX_SEQS:-4} \
  --max-num-batched-tokens ${MNBT:-512} \
  --kv-cache-dtype fp8 --block-size 256 \
  --kernel-config "{\"enable_flashinfer_autotune\": false}" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --trust-remote-code ${SPEC_ARGS-$SPEC_DEFAULT} ${EXTRA_ARGS:-}
