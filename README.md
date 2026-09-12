# DeepSeek-V4-Flash-0731-NVFP4-mtpfix

Make **speculative decoding** work for
[`nvidia/DeepSeek-V4-Flash-0731-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)
on an unmodified vLLM, by repairing the **downloaded model files** instead of patching the
inference engine.

Speculative decoding (this model calls it MTP, or DSpark) is a small helper model that guesses
the next few tokens; the real model checks them all in one pass and keeps the correct ones. It
does not change what the model answers — only how fast the answer arrives. With the files as
downloaded, the helper guesses badly and the feature is worthless; after the repair it roughly
doubles single-request speed.

> **Hardware this assumes: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each),**
> **running the model split across both cards (tensor parallel = 2).**
> Every number below was measured on that setup with vLLM `v0.29.0` and no engine patches.
> Other GPU counts work, but the memory-dependent settings (context length, batch size) change.

| | stock files, MTP off | stock files, MTP on | **repaired files, MTP on** |
|---|---|---|---|
| speed, 1 request at a time | 98.5 tokens/s | 99.5 tokens/s | **207 tokens/s** |
| speed, 4 requests at once (combined) | 264 tokens/s | 150 tokens/s | **390 tokens/s** |
| tokens confirmed per step (higher = better guessing) | — | 1.4–1.6 | **3.3–3.8** |
| IFBench accuracy | — | — | **79.0** (NVIDIA publishes 75.5 for this model) |
| longest prompt+answer the server accepts | 1,048,576 tokens | 1,048,576 | 987,136 |

The model's actual numbers are **unchanged, bit for bit**. Only the way a few scaling values
are stored is rewritten, so the model answers exactly as before — just faster.

---

## 🖥️ Where this runs

**Everything runs on the GPU server that stores the model files** — the 176 GB never leaves it.
Your laptop only needs `ssh`/`scp` to copy this folder over.

```
laptop  ──ssh──►  GPU server (2x RTX PRO 6000 Blackwell)
                    ├── /data/models/DeepSeek-V4-Flash-0731-NVFP4        (downloaded, 176 GB)
                    ├── /data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix (built here, +11 GB)
                    └── this folder (scripts)
```

What the server needs:

| what | why | notes |
|---|---|---|
| the downloaded model files on a local disk | input to the repair | `hf download nvidia/DeepSeek-V4-Flash-0731-NVFP4 --local-dir …` |
| **~11 GB free space on the same disk/filesystem as those files** | the build shares the unchanged weight files instead of copying them, and that sharing cannot cross filesystems | not 176 GB — see [Why share files instead of copying](#why-share-files-instead-of-copying-) |
| Docker | the repair runs in a temporary container; vLLM also runs in one | the repair uses **CPU only** — it can run while the GPUs keep serving |
| `bash` and `python3` (standard library only) | the build script reads the model's file index | nothing to pip install on the host |
| internet access | fetches NVIDIA's conversion helper from PyPI and GitHub | download both in advance if the server is offline |
| for serving: **2x RTX PRO 6000 Blackwell (96 GB), driver 580 or newer** | the setup all numbers come from | tested on driver 595.71.05 |

Build cost: a few minutes of CPU (3 weight files, ~2,300 tensors).

## 🔍 Why MTP does not work with the stock `nvidia/DeepSeek-V4-Flash-0731-NVFP4`

Turning MTP on with the files exactly as downloaded gives you almost nothing: the helper model
is loaded wrong, so its guesses are mostly rejected. Here is why.

The stock files mix two storage formats:

* the **main model**'s expert weights use **NVFP4** (one scale per 16 values, plus a per-file
  scale);
* the bundled **helper model** (`mtp.0/1/2`, 3 layers x 256 experts) is still in **MXFP4**, the
  format DeepSeek published it in (one power-of-two scale per 32 values). The config file even
  lists `mtp.*` as "not quantized in the NVFP4 scheme — skip it".

When vLLM loads the model it renames the helper's tensors (`mtp.*` becomes `layers.43` and
later). The "skip `mtp.*`" rule no longer matches those new names, so the helper's MXFP4 data
is read as if it were NVFP4: the scaling bytes are interpreted in the wrong format and two
values NVFP4 expects are missing. ⚠️ Nothing crashes — the helper simply produces bad guesses.

Measured with the stock files and 3 guessed tokens: 1.4–1.6 tokens confirmed per step,
+1% for a single request, −43% when serving 4 at once. Not worth turning on.

Where upstream stands (September 2026):

* vLLM PR [#49133](https://github.com/vllm-project/vllm/pull/49133) fixed exactly this and was
  closed with *"low priority — we have nvfp4 drafter for dsv4-nvfp4 now"*.
* That refers to [`nvidia/DeepSeek-V4-Flash-nvfp4-DSpark`](https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark),
  where NVIDIA re-encoded the helper so both parts use one format — but it is built on the
  **older, non-0731** version of DeepSeek-V4-Flash.
* No 0731 equivalent exists, so this repo applies NVIDIA's own conversion to the 0731 files.

Other ways to solve it:

| approach | what you modify | survives a vLLM upgrade | keeps the newer 0731 model |
|---|---|---|---|
| **this repo** (re-encode the helper) | the model files | ✅ | ✅ |
| patch vLLM's `quant_config.py` so the helper is read as MXFP4 (e.g. [hikarioyama](https://github.com/hikarioyama/dsv4-flash-nvfp4-sm120)) | the inference engine | ❌ redo after every upgrade | ✅ |
| serve `nvidia/DeepSeek-V4-Flash-nvfp4-DSpark` instead | nothing | ✅ | ❌ |

## 🧬 What the repair changes

For each of the helper model's expert weights
(`mtp.{0,1,2}.ffn.experts.{0..255}.w{1,2,3}`):

| stored value | before (MXFP4) | after (NVFP4) |
|---|---|---|
| the weights themselves | `I8`, packed 4-bit values | `U8`, **exactly the same bytes** |
| scale, one per 32 values | `F8_E8M0` | removed |
| scale, one per 16 values | — | `F8_E4M3` |
| one scale for the whole tensor | — | `F32` |
| one scale for the inputs | — | `F32` |

The new per-group scale multiplied by the new whole-tensor scale reproduces the original scale
exactly, which is why nothing is lost — ✅ all 603,979,776 groups in these files were verified
identical. The `w1` and `w3` parts share one scale because vLLM merges them into a single
operation.

The input scale is the one value that is not simply copied. The helper model has no measured
input range of its own, so — exactly as NVIDIA's documentation prescribes — we reuse the
largest input range measured for the main model's experts: **8.4375** for `w1`/`w3` and
**150.0** for `w2`. NVIDIA's own converted helper (for the older model) shipped 9.0 and 150.0,
which is the main reason to trust this rule.

Config changes: `mtp.*` is removed from the "skip" lists in `config.json` and
`hf_quant_config.json`, the three helper layers are registered as NVFP4, and the file index is
updated to point at the new scale entries.

## 🔧 Build the repaired model

```bash
# on the GPU server
SRC=/data/models/DeepSeek-V4-Flash-0731-NVFP4 \
DST=/data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix \
./scripts/prepare_mtpfix.sh
```

### Why share files instead of copying 🔗

vLLM needs one **complete** folder: all 48 weight files, the index, and the config files. Only
3 of the 48 contain helper-model data, so a normal copy would duplicate 165 GB of bytes that
never change.

`cp -al` creates a folder whose entries point at the *same data on disk* as the originals
(hardlinks). It costs almost nothing until a file is replaced:

```
SRC/model-00001…45  ─┐
                     ├─ one copy of the data, shared      (165 GB)
DST/model-00001…45  ─┘
DST/model-00046…48   ── written fresh by the repair       (+11 GB)
DST/{index,config,hf_quant_config}.json ── written fresh  (a few MB)
```

So you get a self-contained model folder for ~11 GB instead of 176 GB, and the original stays
untouched — 🛟 rolling back is just pointing the server at the original folder, and deleting the
repaired folder frees only the 11 GB it really owns.

⚠️ **This is also why the build deletes those files before writing them.** A shared file is not
a copy: both names refer to the same data. Writing into `DST/config.json` while it is still
shared would change `SRC/config.json` as well, damaging the original download. The script
therefore deletes the 6 files it is about to replace (removing a name, not the data the other
name still uses), and refuses to run if the destination folder already exists.

Two practical consequences: source and destination must sit on the **same filesystem** (file
sharing cannot cross one), and any tool that later rewrites a shared file in place hits the
same trap — always delete first.

### What the script does

1. `cp -al SRC DST` — the shared-file copy described above.
2. Delete, inside `DST` only, the 3 helper weight files plus
   `model.safetensors.index.json`, `config.json` and `hf_quant_config.json`.
3. Start a temporary container, install `nvidia-modelopt==0.46.0`, download NVIDIA's conversion
   helper `shard_cast_utils.py` from Model-Optimizer `main` (it is not in any released package),
   and run `scripts/cast_mtp_to_nvfp4.py`, which updates the config files and then re-encodes
   the 3 helper weight files.

Expected last lines:

```
lossless blocks: 603979776/603979776 (100.0000%)
CAST_DONE
```

The run stops if any group of values would change, if the tensors written into a file disagree
with the index, or if the destination already exists. These scripts are the tidied-up form of
the run that produced the numbers below; the conversion and config logic is identical.

## 🚀 Serve it

```bash
MODEL_DIR=/data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix \
DSV4_API_KEY=your-token ./scripts/serve_dsv4.sh      # leave DSV4_API_KEY out to serve without a token
```

The important options (full command in `scripts/serve_dsv4.sh`):

```
vllm serve /data/models/DeepSeek-V4-Flash-0731-NVFP4-mtpfix \
    --tensor-parallel-size 2 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --max-model-len 987136 --gpu-memory-utilization 0.95 \
    --max-num-seqs 4 --max-num-batched-tokens 512 \
    --kv-cache-dtype fp8 --block-size 256 \
    --kernel-config '{"enable_flashinfer_autotune": false}' \
    --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice --trust-remote-code
```

Why the non-obvious ones are there — all specific to these GPUs (compute capability 12.0,
"SM120"), which are not the data-center Blackwell chips the model was tuned on:

* `--tensor-parallel-size 2` — the model is split across the two cards; its ~176 GB of weights
  do not fit on one 96 GB card.
* `enable_flashinfer_autotune: false` — vLLM's startup kernel tuning hangs here: one GPU spins
  at 100% and ~97 W while the other waits forever.
* `--max-num-batched-tokens 512` — DeepSeek-V4 sizes an internal cache from this value, and the
  default of 8192 wastes most of the GPU memory that would otherwise hold requests. At 512 the
  request pool is about 4x larger.
* `--max-model-len 987136` — with 7 guessed tokens the helper model and its CUDA graphs need
  ~4.2 GiB, so the full 1,048,576 no longer fits. vLLM refuses at startup and prints the largest
  value that does fit. Guess fewer tokens if you want the full 1M back.
* `--kv-cache-dtype fp8` — required; this model stores attention state in 8-bit.
* ❌ Do **not** set `VLLM_USE_DEEP_GEMM=0`: the fallback math library rejects these GPUs and
  startup fails.
* ❌ `VLLM_USE_BREAKABLE_CUDAGRAPH=0` is deliberately not used: it gave +28.6% on DGX Spark, but
  here it cost 33% of the 4-request throughput and gained nothing for a single request.

✅ Check it worked:

```bash
docker logs dsv4-vllm 2>&1 | grep -E "DSpark draft model loaded|GPU KV cache size"
# DSpark draft model loaded: 109 params
# GPU KV cache size: 1,037,893 tokens, Maximum concurrency for 987,136 tokens per request: 1.05x

docker logs dsv4-vllm 2>&1 | grep "SpecDecoding metrics" | tail -1
# Mean acceptance length: 3.54, ... Per-position acceptance rate: 0.778, 0.601, 0.434, ...
```

"Mean acceptance length" of 3.3–3.8 on normal text means the repair worked. If it sits at
1.0–1.6, the helper model is still being read in the wrong format — check that `mtp.*` is gone
from the "skip" list in `config.json` and that you are serving the repaired folder.

## 📊 Speed

Raw output is in `results/`. Numbers come from `benchmarks/realbench.py`: 8 realistic prompts,
512 tokens generated each, streamed, measuring tokens per second after the first token.
Random-token test data is useless here — it makes the text unpredictable in an artificial way
and distorts how often guesses are accepted (vLLM issue
[#51009](https://github.com/vllm-project/vllm/issues/51009)).

```bash
BASE_URL=http://127.0.0.1:8002/v1 DSV4_API_KEY=your-token \
    python3 benchmarks/realbench.py mtpfix   # run it twice; see the warm-up note
```

The prompt list mixes English and Japanese across code, math, prose and explanation, because
how often guesses succeed depends heavily on the kind of text. Those prompts are test input,
not documentation, and are kept as-is so the numbers stay reproducible.

| setup | 1 request | 4 requests (combined) | tokens confirmed per step |
|---|---|---|---|
| MTP off (1M context) | 98.5 tokens/s | 264 tokens/s | — |
| MTP on, stock files, 3 guesses | 99.5 tokens/s | 150 tokens/s | 1.4–1.6 |
| **MTP on, repaired files, 7 guesses** | **207 tokens/s** | **390 tokens/s** | **3.3–3.8** |

Per prompt, single-request speed ranges from 160 tokens/s (open-ended prose) to 298 tokens/s
(listing prime numbers) — predictable text is easier to guess.

Reading the prompt (MTP off, no cache reuse): ~6,100 tokens/s for 8K and 32K inputs, ~5,700
tokens/s for 128K.

For context: memory bandwidth alone caps a single request at ~279 tokens/s here (each token
reads 5.72 GB per GPU at 1,597 GB/s). The rest of the time goes to the two GPUs exchanging
results and to many small operations — which is exactly what guessing several tokens at once
amortizes.

⚠️ **Measurement warning:** the first 4-request run after a server restart reads ~180 tokens/s
with multi-second first-token latency, and recovers to ~260 on the second run. Always warm up
before recording.

## 🎯 Accuracy

IFBench (300 prompts), scored with the official
[allenai/IFBench](https://github.com/allenai/IFBench) tool, using "prompt-level loose" accuracy —
the same figure Artificial Analysis and the IFBench paper report: the share of prompts where
*every* instruction was followed, ignoring cosmetic wrapping. Generation settings match
NVIDIA's model page: `temperature=1.0`, `top_p=1.0`, `reasoning_effort=max`.

| | prompt-level loose | prompt-level strict | per-instruction loose |
|---|---|---|---|
| **repaired files, MTP on** | **79.0** | 74.0 | 81.4 |
| NVIDIA's published NVFP4 result (B200) | 75.5 | — | — |
| NVIDIA's published original-format result (B200) | 75.8 | — | — |

With 300 prompts at temperature 1.0 the 95% margin is about ±4.6 points, so the honest reading
is "no regression", not "better than NVIDIA". 2 of 300 answers hit the 131,072-token output
limit while thinking and were counted as failures (NVIDIA allowed 384,000); without them the
score is 79.5. The run took 2h23m and 3.2M generated tokens, median 6.2k per prompt.

```bash
git clone https://github.com/allenai/IFBench.git && cd IFBench
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd -

BASE_URL=http://127.0.0.1:8002/v1 DSV4_API_KEY=your-token \
IFBENCH_DATA=IFBench/data/IFBench_test.jsonl \
    python3 benchmarks/ifbench_generate.py     # writes responses.jsonl (2-3 h at max effort)

IFBench/.venv/bin/python IFBench/run_eval.py \
    --input_data=IFBench/data/IFBench_test.jsonl \
    --input_response_data=responses.jsonl --output_dir=eval
```

💡 Send `reasoning_effort` as a normal request field. Routing it through
`chat_template_kwargs` quietly produces *shorter* thinking instead.

💡 For long runs, start the generator on the server with `nohup`. It appends each answer as it
finishes, so an interrupted run continues instead of losing hours of work.

## ⚠️ Known limitations

* **Maximum context drops to 987,136 tokens** (−6%), and the pool of GPU memory for in-flight
  requests drops from 2.47M to ~1.04M tokens, because the helper model and its CUDA graphs take
  space. Serving several very long requests at once suffers accordingly.
* **Reusing a cached prompt prefix may reduce guessing accuracy** — vLLM issue
  [#47930](https://github.com/vllm-project/vllm/issues/47930) (still open) reports exactly that.
  Agent workloads that resend one long system prompt may see less speedup than the table above.
* **NVIDIA has not validated this.** The 0731 model page says speculative decoding was not
  exercised for these NVFP4 files. Accuracy here is one IFBench run; long-document recall is
  untested.
* **Answers are not byte-identical** to a run with guessing turned off, even at temperature 0.
  Speculative decoding preserves the output *distribution*, not the exact token stream, and this
  model is already non-deterministic at temperature 0 when several requests run together (vLLM
  issue [#53257](https://github.com/vllm-project/vllm/issues/53257)).
* **Unrelated GPU-specific issues remain**: vLLM
  [#53635](https://github.com/vllm-project/vllm/issues/53635) and
  [#41063](https://github.com/vllm-project/vllm/issues/41063) track gaps in kernel support for
  these cards.

## 📁 Files

```
scripts/prepare_mtpfix.sh      one-command build: shared-file copy -> config update + re-encode
scripts/cast_mtp_to_nvfp4.py   updates the config files and re-encodes the helper model
scripts/serve_dsv4.sh          the vLLM launch every number here was measured with
benchmarks/realbench.py        speed on realistic prompts (1 and 4 requests)
benchmarks/ifbench_generate.py IFBench answer generation (temperature 1.0, max thinking)
results/                       raw speed and IFBench output
```

## 🔗 References

* [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) — the model files this repairs
* [nvidia/DeepSeek-V4-Flash-nvfp4-DSpark](https://huggingface.co/nvidia/DeepSeek-V4-Flash-nvfp4-DSpark) — NVIDIA's re-encoded helper (older base model), the example this follows
* [Model-Optimizer `quantize_to_nvfp4.py`](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/deepseek/deepseek_v4/quantize_to_nvfp4.py) — `--cast_mxfp4_to_nvfp4`, the reference implementation
* [vLLM PR #49133](https://github.com/vllm-project/vllm/pull/49133) — the engine-side fix, closed in favour of re-encoded helpers
* [Infatoshi/dsv4-flash-2x-rtxpro6000s](https://github.com/Infatoshi/dsv4-flash-2x-rtxpro6000s), [hikarioyama/dsv4-flash-nvfp4-sm120](https://github.com/hikarioyama/dsv4-flash-nvfp4-sm120) — recipes for these GPUs that patch vLLM instead
* [allenai/IFBench](https://github.com/allenai/IFBench) — the accuracy benchmark and its scorer
