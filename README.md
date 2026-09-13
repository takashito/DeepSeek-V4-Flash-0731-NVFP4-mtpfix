# DeepSeek-V4-Flash-0731-NVFP4-mtpfix

Make speculative decoding work for
[`nvidia/DeepSeek-V4-Flash-0731-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)
on an unmodified vLLM, by repairing the model files instead of patching vLLM.

## 📖 Terms

| term | meaning |
|---|---|
| **original model** | `nvidia/DeepSeek-V4-Flash-0731-NVFP4`, exactly as downloaded from Hugging Face |
| **repaired model** | the `DeepSeek-V4-Flash-0731-NVFP4-mtpfix` folder this repo builds from the original model |
| **speculative decoding** | the technique: a small draft model guesses the next few tokens, the main model checks them in one pass and keeps the correct ones. Faster, same answer quality. |
| **main model** | the 43-layer model that produces the answer |
| **DSpark** | the draft model DeepSeek ships inside the checkpoint, next to the main model. vLLM enables it with `"method":"dspark"`. |
| **`mtp.*`** | the tensor names DSpark's weights are stored under (from Multi-Token Prediction). This repo's name comes from them. |
| **SM120** | the chip generation of the RTX PRO 6000 Blackwell (CUDA compute capability 12.0) |

In the original model, vLLM loads DSpark in the wrong format and DSpark guesses badly. In the
repaired model, single-request speed roughly doubles.

> **Measured on 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each), tensor parallel 2,
> vLLM `v0.29.0`, no vLLM patches.** Other GPU counts work, but context length and batch size
> settings change.

| | original model, DSpark off | original model, DSpark on | **repaired model, DSpark on** |
|---|---|---|---|
| speed, 1 request | 98.5 tokens/s | 99.1 tokens/s | **212 tokens/s** |
| speed, 4 requests (combined) | 264 tokens/s | 226 tokens/s | **415 tokens/s** |
| tokens accepted per step | — | 1.4–1.5 | **3.1–3.7** |
| IFBench accuracy | — | — | **79.0** (the original model's Hugging Face page reports 75.5) |
| max context | 1,048,576 | 1,048,576 | 1,038,592 |

The repair is lossless: DSpark's weights decode to exactly the same values as in the original
model, and the main model's weights are not touched.

---

## 🔍 Why DSpark does not work in the original model

The original model mixes two formats:

* the **main model**'s experts are **NVFP4** (one scale per 16 values, plus one per-tensor scale);
* **DSpark**'s experts (`mtp.0/1/2`, 3 layers x 256 experts) are still **MXFP4**, as DeepSeek
  published them (one power-of-two scale per 32 values). `config.json` and
  `hf_quant_config.json` list `mtp.*` as excluded from NVFP4.

vLLM renames DSpark's tensors on load (`mtp.*` becomes `layers.43` and later), so the `mtp.*`
exclusion stops matching. vLLM then reads DSpark's MXFP4 data as NVFP4. Nothing crashes; DSpark
just guesses badly.

Measured on the original model with 3 draft tokens: 1.4–1.5 tokens accepted per step, +1% speed
for 1 request, −14% for 4 requests, compared with DSpark off.

Upstream status (September 2026):

* vLLM PR [#49133](https://github.com/vllm-project/vllm/pull/49133) fixed this inside vLLM and was
  closed: *"low priority — we have nvfp4 drafter for dsv4-nvfp4 now"*.
* That comment refers to [`nvidia/DeepSeek-V4-Flash-nvfp4-DSpark`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark),
  where NVIDIA converted DSpark to NVFP4 — but for the pre-0731 DeepSeek-V4-Flash.
* NVIDIA has not published the same for 0731. This repo applies NVIDIA's conversion
  (Model-Optimizer `--cast_mxfp4_to_nvfp4`) to the original model.

Ways to solve it:

| approach | what changes | survives a vLLM upgrade | base checkpoint |
|---|---|---|---|
| **this repo** | model files (+11 GB) | yes | `nvidia/DeepSeek-V4-Flash-0731-NVFP4` |
| download [`Rarri/DeepSeek-V4-Flash-0731-NVFP4`](https://huggingface.co/Rarri/DeepSeek-V4-Flash-0731-NVFP4) | nothing (176 GB download) | yes | 0731, quantized by Rarri instead of NVIDIA; same DSpark conversion, published 2026-08-03 |
| patch vLLM to read DSpark as MXFP4 (e.g. [hikarioyama](https://github.com/hikarioyama/dsv4-flash-nvfp4-sm120)) | vLLM source | no | `nvidia/DeepSeek-V4-Flash-0731-NVFP4` |
| serve `nvidia/DeepSeek-V4-Flash-nvfp4-DSpark` | nothing | yes | pre-0731 DeepSeek-V4-Flash |

## 🧬 What the repair changes

For each DSpark expert weight (`mtp.{0,1,2}.ffn.experts.{0..255}.w{1,2,3}`):

| stored value | original model (MXFP4) | repaired model (NVFP4) |
|---|---|---|
| 4-bit weights | `I8` | `U8`, same bytes |
| scale per 32 values | `F8_E8M0` | removed |
| scale per 16 values | — | `F8_E4M3` |
| per-tensor scale | — | `F32` |
| input scale | — | `F32` |

Per-16 scale x per-tensor scale equals the original per-32 scale exactly. All 603,979,776 groups
were verified identical. `w1` and `w3` share one per-tensor scale because vLLM fuses them.

The input scale is the only new value. DSpark was never calibrated, so it gets the same fallback
Model-Optimizer uses for uncalibrated experts: the largest input scale among the main model's
experts, **8.4375** for `w1`/`w3` and **150.0** for `w2`. `nvidia/DeepSeek-V4-Flash-nvfp4-DSpark`
uses 9.0 and 150.0.

Metadata changes: `mtp.*` is removed from the exclusion lists in `config.json` and
`hf_quant_config.json`, the three DSpark layers are registered as NVFP4, and
`model.safetensors.index.json` lists the new scale tensors.

## 🖥️ Requirements

Everything runs on the GPU server that holds the original model. Your laptop only copies this
repo over with `scp`.

| what | notes |
|---|---|
| the original model on local disk | `hf download nvidia/DeepSeek-V4-Flash-0731-NVFP4 --local-dir …` |
| ~11 GB free on the **same filesystem** as the original model | unchanged files are hardlinked, not copied |
| Docker | the build runs in a temporary container, CPU only (GPUs can keep serving) |
| `bash`, `python3` (stdlib only) | the build script reads `model.safetensors.index.json` |
| internet | installs `nvidia-modelopt` from PyPI and downloads `shard_cast_utils.py` from GitHub |
| serving: 2x RTX PRO 6000 Blackwell, NVIDIA driver 580+ | tested on 595.71.05 |

Build time: a few minutes of CPU.

## 🔧 Build

`SRC` is the original model's folder, `DST` is where the repaired model is created.

```bash
SRC=/data/models/DeepSeek-V4-Flash-0731-NVFP4 \
DST=/data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix \
./scripts/prepare_mtpfix.sh
```

What it does:

1. `cp -al SRC DST` — a complete model folder made of hardlinks, costing almost no space.
2. Deletes, in `DST` only, the 6 files it will rewrite: `model-00046…48-of-00048.safetensors`
   (the shards holding DSpark), `model.safetensors.index.json`, `config.json`,
   `hf_quant_config.json`.
3. In a temporary container, installs `nvidia-modelopt==0.46.0`, downloads
   `shard_cast_utils.py` from Model-Optimizer `main` (not in any release), and runs
   `scripts/cast_mtp_to_nvfp4.py`, which rewrites the 3 JSON files and converts the 3 shards.

```
SRC/model-00001…45  ─┐
                     ├─ shared data, 165 GB
DST/model-00001…45  ─┘
DST/model-00046…48   ── new, +11 GB
DST/*.json           ── new, a few MB
```

Expected last lines:

```
lossless blocks: 603979776/603979776 (100.0000%)
CAST_DONE
```

The script stops if any value would change, if the tensors written to a shard do not match
`model.safetensors.index.json`, or if `DST` already exists.

**Why delete before writing:** a hardlink is the same data under two names. Writing into
`DST/config.json` while it is linked would also change `SRC/config.json`. Deleting the name first
leaves the original model intact. The same applies to any tool that later edits a file in `DST`.

To roll back, serve `SRC` again. Deleting `DST` frees only its 11 GB.

## 🚀 Serve

```bash
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix \
DSV4_API_KEY=your-token ./scripts/serve_dsv4.sh      # omit DSV4_API_KEY to serve without a token
```

Key options (full command in `scripts/serve_dsv4.sh`):

```
vllm serve /data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix \
    --tensor-parallel-size 2 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}' \
    --max-model-len 1038592 --gpu-memory-utilization 0.95 \
    --max-num-seqs 4 --max-num-batched-tokens 512 \
    --kv-cache-dtype fp8 --block-size 256 \
    --kernel-config '{"enable_flashinfer_autotune": false}' \
    --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice --trust-remote-code
```

Why, on 2x RTX PRO 6000 Blackwell (SM120):

* `--tensor-parallel-size 2` — 176 GB of weights do not fit on one 96 GB card.
* `enable_flashinfer_autotune: false` — FlashInfer's MoE autotuning at startup deadlocks the two
  GPUs (one at 100%, the other waits forever).
* `--max-num-batched-tokens 512` — vLLM's DeepSeek-V4 code reserves GPU memory in proportion to
  this value; the default 8192 leaves about 4x less KV cache.
* `--max-model-len 1038592` — DSpark and its CUDA graphs take GPU memory, so 1,048,576 does not
  fit. vLLM prints the largest value that fits (987,136 with 7 draft tokens).
* `num_speculative_tokens: 5` — about 3.3 tokens are accepted per step, so draft tokens 6–7 are
  mostly rejected. 5 beats 7 on speed (212 vs 207 tokens/s for 1 request, 415 vs 390 for 4) and
  leaves room for 51K more context.
* `--kv-cache-dtype fp8` — required by DeepSeek-V4.
* Do **not** set `VLLM_USE_DEEP_GEMM=0`: vLLM then uses CUTLASS `scaled_mm`, which does not
  support SM120.
* `VLLM_USE_BREAKABLE_CUDAGRAPH=0` is not used: it gave +28.6% on DGX Spark, but −33% at
  4 requests on 2x RTX PRO 6000.

Check it worked:

```bash
docker logs dsv4-vllm 2>&1 | grep -E "DSpark draft model loaded|GPU KV cache size"
# DSpark draft model loaded: 109 params
# GPU KV cache size: 1,041,593 tokens, Maximum concurrency for 1,038,592 tokens per request: 1.00x

docker logs dsv4-vllm 2>&1 | grep "SpecDecoding metrics" | tail -1
# Mean acceptance length: 3.35, ... Per-position acceptance rate: 0.797, 0.593, 0.427, 0.306, 0.224
```

Mean acceptance length 3.1–3.7 means the repair worked. 1.0–1.6 means vLLM still reads DSpark as
the wrong format: check that `mtp.*` is gone from the exclusion list in `DST/config.json` and
that `MODEL_DIR` points at `DST`, not `SRC`.

## 📊 Speed

Raw output in `results/`. `benchmarks/realbench.py` runs 8 realistic prompts (English and
Japanese; code, math, prose, explanation), 512 tokens each, streamed, measuring tokens/s after
the first token. Random-token prompts distort acceptance (vLLM issue
[#51009](https://github.com/vllm-project/vllm/issues/51009)).

```bash
BASE_URL=http://127.0.0.1:8002/v1 DSV4_API_KEY=your-token \
    python3 benchmarks/realbench.py mtpfix   # "mtpfix" is the label for the results file; run twice, the first run after a restart is slow
```

| setup | 1 request | 4 requests (combined) | tokens accepted per step |
|---|---|---|---|
| original model, DSpark off (1M context) | 98.5 tokens/s | 264 tokens/s | — |
| original model, DSpark on, 3 draft tokens | 99.1 tokens/s | 226 tokens/s | 1.4–1.5 |
| repaired model, DSpark on, 7 draft tokens | 207 tokens/s | 390 tokens/s | 3.3–3.8 |
| **repaired model, DSpark on, 5 draft tokens** | **212 tokens/s** | **415 tokens/s** | **3.1–3.7** |

* Per prompt, 1-request speed on the repaired model (5 draft tokens) ranges from 163 tokens/s
  (open-ended prose) to 300 tokens/s (listing primes): DSpark guesses predictable text better.
* Prefill (original model, DSpark off, no prefix cache): ~6,100 tokens/s at 8K and 32K input
  tokens, ~5,700 at 128K.
* GPU memory bandwidth alone caps 1 request at ~279 tokens/s (5.72 GB read per GPU per token at
  1,597 GB/s). The rest is GPU-to-GPU communication and small-kernel overhead, which accepting
  several tokens per step spreads out.
* The first 4-request run after a vLLM restart is 40–60% slower with multi-second first-token
  latency (5 draft tokens: 226 → 415 tokens/s from the first to the second run). All numbers above are from the second run.

## 🎯 Accuracy

IFBench, 300 prompts, official [allenai/IFBench](https://github.com/allenai/IFBench) scorer.
Main metric is prompt-level loose (the one the IFBench paper and Artificial Analysis report).
Settings match the original model's Hugging Face page: `temperature=1.0`, `top_p=1.0`,
`reasoning_effort=max`.

| | prompt-level loose | prompt-level strict | instruction-level loose |
|---|---|---|---|
| **repaired model, DSpark on, 7 draft tokens** (2x RTX PRO 6000) | **79.0** | 74.0 | 81.4 |
| original model, reported by NVIDIA (B200) | 75.5 | — | — |
| `deepseek-ai/DeepSeek-V4-Flash-0731`, reported by NVIDIA (B200) | 75.8 | — | — |

The 95% margin is about ±4.6 points, so read this as "no regression", not "better than NVIDIA's
result". 2 answers hit this run's 131,072-token output limit and counted as failures (NVIDIA's
run allowed 384,000); without them the score is 79.5. The run took 2h23m and 3.2M generated
tokens. The number of draft tokens does not change the output distribution, so this result also
applies to 5 draft tokens.

```bash
git clone https://github.com/allenai/IFBench.git && cd IFBench
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd -

BASE_URL=http://127.0.0.1:8002/v1 DSV4_API_KEY=your-token \
IFBENCH_DATA=IFBench/data/IFBench_test.jsonl \
    python3 benchmarks/ifbench_generate.py     # writes responses.jsonl, 2-3 h

IFBench/.venv/bin/python IFBench/run_eval.py \
    --input_data=IFBench/data/IFBench_test.jsonl \
    --input_response_data=responses.jsonl --output_dir=eval
```

* Send `reasoning_effort` as a top-level field of the chat request. Passed through
  `chat_template_kwargs` instead, it silently gives shorter thinking.
* Run `ifbench_generate.py` with `nohup`. It appends each answer to `responses.jsonl`, so an
  interrupted run resumes.

## ⚠️ Limitations

* **Less context and KV cache** than the original model with DSpark off: max context 1,038,592
  (−1%), KV cache 2.47M → ~1.04M tokens.
* **Prefix caching may lower acceptance** — vLLM issue
  [#47930](https://github.com/vllm-project/vllm/issues/47930) (open). Agents that resend a long
  system prompt may see less speedup.
* **Not validated by NVIDIA.** The original model's Hugging Face page says speculative decoding
  was not tested with it. This repo's accuracy check is one IFBench run; long-document recall is untested.
* **Output is not token-identical** to DSpark off, even at temperature 0. Speculative decoding
  keeps the output distribution, not the exact tokens, and DeepSeek-V4 in vLLM is already
  non-deterministic at temperature 0 with concurrent requests (vLLM issue
  [#53257](https://github.com/vllm-project/vllm/issues/53257)).
* **Open vLLM issues on SM120, unrelated to DSpark**:
  [#53635](https://github.com/vllm-project/vllm/issues/53635),
  [#41063](https://github.com/vllm-project/vllm/issues/41063).

## 📁 Files

```
scripts/prepare_mtpfix.sh      build: hardlink copy -> JSON update + DSpark conversion
scripts/cast_mtp_to_nvfp4.py   rewrites the 3 JSON files and converts DSpark's experts to NVFP4
scripts/serve_dsv4.sh          the vLLM command all numbers were measured with
benchmarks/realbench.py        speed on realistic prompts (1 and 4 requests)
benchmarks/ifbench_generate.py IFBench answer generation
results/                       raw speed and IFBench output
```

## 🔗 References

* [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) — the original model
* [nvidia/DeepSeek-V4-Flash-nvfp4-DSpark](https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark) — NVIDIA's NVFP4 DSpark for the pre-0731 model, the example this repo follows
* [Model-Optimizer `quantize_to_nvfp4.py`](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/deepseek/deepseek_v4/quantize_to_nvfp4.py) — `--cast_mxfp4_to_nvfp4`, the reference implementation
* [Rarri/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/Rarri/DeepSeek-V4-Flash-0731-NVFP4) — the same DSpark conversion, on a 0731 checkpoint quantized by Rarri
* [vLLM PR #49133](https://github.com/vllm-project/vllm/pull/49133) — the fix inside vLLM, closed
* [Infatoshi/dsv4-flash-2x-rtxpro6000s](https://github.com/Infatoshi/dsv4-flash-2x-rtxpro6000s), [hikarioyama/dsv4-flash-nvfp4-sm120](https://github.com/hikarioyama/dsv4-flash-nvfp4-sm120) — recipes for RTX PRO 6000 that patch vLLM instead
* [allenai/IFBench](https://github.com/allenai/IFBench) — accuracy benchmark and scorer
