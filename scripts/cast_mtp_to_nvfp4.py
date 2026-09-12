#!/usr/bin/env python3
"""Turn a hardlink copy of DeepSeek-V4-Flash-0731-NVFP4 into the -mtpfix checkpoint.

Two steps, in order:

1. Metadata — drop ``mtp.*`` from the ignore / exclude lists, register the drafter layers as
   NVFP4, and point the index at the scale tensors the cast produces.
2. Weights — cast the DSpark drafter experts from MXFP4 to NVFP4. This is the same lossless
   cast NVIDIA applied when packaging ``nvidia/DeepSeek-V4-Flash-nvfp4-DSpark``
   (Model-Optimizer ``examples/deepseek/deepseek_v4/quantize_to_nvfp4.py
   --cast_mxfp4_to_nvfp4``): the E2M1 nibbles are carried over bit-for-bit and only the block
   scales are re-encoded, so ``weight_scale * weight_scale_2`` reproduces the source E8M0
   power-of-two exactly.

``input_scale`` follows NVIDIA's fallback for experts that have no calibrated activation
amax: the maximum input amax of the target model's routed experts, per projection, with
w1/w3 fused. It is read from the source checkpoint unless overridden.

Only the shards holding ``mtp.*`` tensors are rewritten; every other tensor in those shards
is copied through unchanged. DST must be a hardlink copy of SRC whose three metadata files
and drafter shards have been unlinked — prepare_mtpfix.sh does that and is the supported
entry point. Requires torch, safetensors, nvidia-modelopt, and Model-Optimizer main's
``shard_cast_utils.py`` on sys.path.

Env: SRC, DST (checkpoint directories), SHARD_CAST_DIR (default /tmp, holds
shard_cast_utils.py), INPUT_AMAX_W13 / INPUT_AMAX_W2 (override the computed constants).
"""

import json
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.environ.get("SHARD_CAST_DIR", "/tmp"))

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from shard_cast_utils import (  # Model-Optimizer main, not in any pip release
    build_w13_kmax_overrides,
    mxfp4_kmax,
    quantize_mxfp4_to_nvfp4_lossless,
)

SRC = os.environ.get("SRC", "/data/models/DeepSeek-V4-Flash-0731-NVFP4").rstrip("/") + "/"
DST = os.environ.get("DST", "/data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix").rstrip("/") + "/"
E2M1_MAX, E4M3_MAX = 6.0, 448.0  # NVFP4: scale_2 = amax / (E2M1_MAX * E4M3_MAX)
WEIGHT_RE = re.compile(r"^mtp\.\d+\.ffn\.experts\.\d+\.(w[123])\.weight$")
SCALE_RE = re.compile(r"^(mtp\.\d+\.ffn\.experts\.\d+\.w[123])\.scale$")
TARGET_INPUT_SCALE_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.(w[123])\.input_scale$")
NEW_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


def read_header(path):
    """(header dict, offset of the tensor data) of a safetensors file."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def write_json(name, obj):
    path = DST + name
    # The destination is a hardlink farm: writing in place would edit the source file too.
    if os.path.exists(path):
        raise SystemExit(f"{path} still exists — unlink it first (see prepare_mtpfix.sh)")
    json.dump(obj, open(path, "w"), indent=2)


def rewrite_metadata():
    """Register the drafter as NVFP4 and index the tensors the cast is about to write."""
    index = json.load(open(SRC + "model.safetensors.index.json"))
    layers = sorted({k.split(".ffn.experts.")[0] + ".ffn.experts"
                     for k in index["weight_map"] if SCALE_RE.match(k)})
    if not layers:
        raise SystemExit("no mtp.*.ffn.experts.*.scale tensors found — wrong checkpoint?")

    new_map, replaced = {}, 0
    for key, shard in index["weight_map"].items():
        m = SCALE_RE.match(key)
        if m:
            for suffix in NEW_SUFFIXES:
                new_map[m.group(1) + suffix] = shard
            replaced += 1
        else:
            new_map[key] = shard
    index["weight_map"] = new_map
    write_json("model.safetensors.index.json", index)

    config = json.load(open(SRC + "config.json"))
    quant = config["quantization_config"]
    quant["ignore"] = [x for x in quant["ignore"] if x != "mtp.*"]
    for layer in layers:
        quant["quantized_layers"][layer] = {"group_size": 16, "quant_algo": "NVFP4"}
    write_json("config.json", config)

    hf_quant = json.load(open(SRC + "hf_quant_config.json"))
    section = hf_quant["quantization"]
    section["exclude_modules"] = [x for x in section["exclude_modules"] if x != "mtp.*"]
    for layer in layers:
        section["quantized_layers"][layer] = {"quant_algo": "NVFP4", "group_size": 16}
    write_json("hf_quant_config.json", hf_quant)

    print(f"metadata: {replaced} drafter .scale entries -> {len(new_map)} tensors, "
          f"quantized layers {layers}", flush=True)
    return new_map


def target_input_amax():
    """Max calibrated input amax over the target's routed experts, per projection."""
    index = json.load(open(SRC + "model.safetensors.index.json"))["weight_map"]
    shards = sorted({v for k, v in index.items() if TARGET_INPUT_SCALE_RE.match(k)})
    peak = {}
    for shard in shards:
        header, base = read_header(SRC + shard)
        with open(SRC + shard, "rb") as fh:
            for key, meta in header.items():
                m = TARGET_INPUT_SCALE_RE.match(key)
                if not m:
                    continue
                start, _ = meta["data_offsets"]
                fh.seek(base + start)
                value = struct.unpack("<f", fh.read(4))[0]
                peak[m.group(1)] = max(peak.get(m.group(1), 0.0), value)
    scale = E2M1_MAX * E4M3_MAX
    w13 = max(peak["w1"], peak["w3"]) * scale  # w1/w3 share one fused GEMM1 scale
    return {"w1": w13, "w3": w13, "w2": peak["w2"] * scale}


def cast_drafter(index):
    amax = target_input_amax()
    if os.environ.get("INPUT_AMAX_W13"):
        amax["w1"] = amax["w3"] = float(os.environ["INPUT_AMAX_W13"])
    if os.environ.get("INPUT_AMAX_W2"):
        amax["w2"] = float(os.environ["INPUT_AMAX_W2"])
    print(f"input amax: w1/w3={amax['w1']} w2={amax['w2']}", flush=True)

    shards = sorted({v for k, v in index.items() if k.startswith("mtp.")})
    total_blocks = lossless_blocks = 0

    for shard in shards:
        started, out = time.time(), {}
        with safe_open(SRC + shard, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            weight_keys = [k for k in keys if WEIGHT_RE.match(k)]
            bases = [k[: -len(".weight")] for k in weight_keys]
            # w1 and w3 must share k_max: vLLM consumes a single fused GEMM1 weight scale.
            kmax = build_w13_kmax_overrides(bases, lambda b: f.get_tensor(b + ".scale"), "cpu")
            scale_keys = {b + ".scale" for b in bases}

            for key in keys:
                if key in scale_keys:
                    continue  # replaced by weight_scale / weight_scale_2 / input_scale
                if key not in weight_keys:
                    out[key] = f.get_tensor(key)
                    continue
                base, proj = key[: -len(".weight")], WEIGHT_RE.match(key).group(1)
                weight, scale = f.get_tensor(key), f.get_tensor(base + ".scale")
                k_max = kmax.get(base)
                k_max = mxfp4_kmax(scale, "cpu") if k_max is None else k_max
                packed, w_scale, w_scale_2, n_blocks, n_lossless = quantize_mxfp4_to_nvfp4_lossless(
                    weight, scale, k_max, "cpu"
                )
                assert packed.dtype == torch.uint8 and tuple(packed.shape) == tuple(weight.shape)
                assert w_scale.dtype == torch.float8_e4m3fn
                assert w_scale.shape[-1] == weight.shape[-1] * 2 // 16  # one e4m3 scale per 16 values
                out[key] = packed.contiguous()
                out[base + ".weight_scale"] = w_scale.contiguous()
                out[base + ".weight_scale_2"] = w_scale_2.float().reshape(())
                out[base + ".input_scale"] = (
                    torch.tensor(amax[proj], dtype=torch.float32) / (E2M1_MAX * E4M3_MAX)
                ).reshape(())
                total_blocks += n_blocks
                lossless_blocks += n_lossless

        expected = {k for k, v in index.items() if v == shard}
        assert set(out) == expected, sorted(set(out) ^ expected)[:5]
        save_file(out, DST + shard, metadata={"format": "pt"})
        print(f"{shard}: {len(weight_keys)} experts cast, {len(out)} tensors, "
              f"{time.time() - started:.0f}s", flush=True)

    ratio = 100.0 * lossless_blocks / max(total_blocks, 1)
    print(f"lossless blocks: {lossless_blocks}/{total_blocks} ({ratio:.4f}%)")
    assert lossless_blocks == total_blocks, "cast was not lossless"


if __name__ == "__main__":
    cast_drafter(rewrite_metadata())
    print("CAST_DONE")
