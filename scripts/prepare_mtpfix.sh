#!/usr/bin/env bash
# Build DeepSeek-V4-Flash-0731-NVFP4-mtpfix from the stock nvidia/DeepSeek-V4-Flash-0731-NVFP4.
#
#   SRC=/data/models/DeepSeek-V4-Flash-0731-NVFP4 \
#   DST=/data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix ./prepare_mtpfix.sh
#
# The destination is a hardlink farm of the source: only the drafter shards and the three
# metadata files are rewritten (~11 GB), everything else shares inodes with the source.
# The cast runs on CPU inside a throwaway container because modelopt's shard_cast_utils
# lives on Model-Optimizer main, not in any pip release.
set -euo pipefail

SRC="${SRC:-/data/models/DeepSeek-V4-Flash-0731-NVFP4}"
DST="${DST:-/data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.29.0}"          # any image with torch + safetensors works
MODELOPT_VERSION="${MODELOPT_VERSION:-0.46.0}"
SHARD_CAST_URL="${SHARD_CAST_URL:-https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/modelopt/torch/export/shard_cast_utils.py}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -d "$SRC" ] || { echo "source checkpoint not found: $SRC" >&2; exit 1; }
[ -e "$DST" ] && { echo "destination already exists: $DST" >&2; exit 1; }

echo "==> hardlink copy $SRC -> $DST"
cp -al "$SRC" "$DST"

# Shards that hold mtp.* tensors, read from the index rather than hardcoded.
mapfile -t SHARDS < <(python3 - "$SRC" <<'PY'
import json, sys
index = json.load(open(sys.argv[1] + "/model.safetensors.index.json"))["weight_map"]
print("\n".join(sorted({v for k, v in index.items() if k.startswith("mtp.")})))
PY
)
[ "${#SHARDS[@]}" -gt 0 ] || { echo "no mtp.* tensors in the index" >&2; exit 1; }
echo "==> drafter shards: ${SHARDS[*]}"

# Unlink before rewriting: writing through a hardlink would corrupt the source checkpoint.
for f in "${SHARDS[@]}" model.safetensors.index.json config.json hf_quant_config.json; do
    rm "$DST/$f"
done

echo "==> rewriting metadata and casting drafter experts MXFP4 -> NVFP4 (CPU, a few minutes)"
CONTAINER="mtpcast-$$"
cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT
SRC_ROOT="$(cd "$(dirname "$SRC")" && pwd)"
DST_ROOT="$(cd "$(dirname "$DST")" && pwd)"
mounts=(-v "$SRC_ROOT":"$SRC_ROOT")
[ "$DST_ROOT" != "$SRC_ROOT" ] && mounts+=(-v "$DST_ROOT":"$DST_ROOT")
docker run -d --name "$CONTAINER" "${mounts[@]}" --entrypoint sleep "$IMAGE" infinity >/dev/null
docker exec "$CONTAINER" pip install -q "nvidia-modelopt==${MODELOPT_VERSION}" 2>&1 | grep -vE "WARNING|notice" || true
docker exec "$CONTAINER" curl -fsSL "$SHARD_CAST_URL" -o /tmp/shard_cast_utils.py
docker cp "$HERE/cast_mtp_to_nvfp4.py" "$CONTAINER:/tmp/cast_mtp_to_nvfp4.py"
docker exec -e SRC="$SRC" -e DST="$DST" "$CONTAINER" python3 /tmp/cast_mtp_to_nvfp4.py

echo "==> done: $DST"
du -sh --apparent-size "$DST" 2>/dev/null || true
