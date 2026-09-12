#!/usr/bin/env python3
"""Generate IFBench responses for scoring with allenai/IFBench's run_eval.

Settings match NVIDIA's model card for DeepSeek-V4-Flash-0731-NVFP4: temperature 1.0,
top_p 1.0, reasoning_effort=max. Pass reasoning_effort as a request *field*; routing it
through chat_template_kwargs silently yields a shorter reasoning chain.

Writes responses_full.jsonl (one record per prompt, appended as each finishes, so an
interrupted run resumes where it stopped) and responses.jsonl (prompt/response only, the
input run_eval expects). Only the answer text is scored — the reasoning chain is dropped,
as the IFBench authors do for thinking models.

Usage:
    BASE_URL=http://127.0.0.1:8002/v1 DSV4_API_KEY=your-token \\
    IFBENCH_DATA=IFBench/data/IFBench_test.jsonl python3 ifbench_generate.py

Env: BASE_URL, DSV4_API_KEY (or K), IFBENCH_DATA, OUT_DIR (default "."),
MODEL (default deepseek-v4-flash), MAX_TOKENS (default 131072), CONCURRENCY (default 4).

A full run takes hours (median ~6k reasoning tokens per prompt). Start it under nohup on the
serving host — a driver that dies takes an unsaved run with it.
"""

import concurrent.futures as cf
import json
import os
import threading
import time
import urllib.request

URL = os.environ.get("BASE_URL", "http://127.0.0.1:8002/v1").rstrip("/") + "/chat/completions"
API_KEY = os.environ.get("DSV4_API_KEY") or os.environ.get("K", "")
DATA = os.environ.get("IFBENCH_DATA", "IFBench/data/IFBench_test.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", ".")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "131072"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))

FULL = os.path.join(OUT_DIR, "responses_full.jsonl")
SCORED = os.path.join(OUT_DIR, "responses.jsonl")
lock = threading.Lock()


def generate(row):
    body = {"model": MODEL, "temperature": 1.0, "top_p": 1.0, "max_tokens": MAX_TOKENS,
            "reasoning_effort": "max", "messages": [{"role": "user", "content": row["prompt"]}]}
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    record = None
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), headers)
            data = json.load(urllib.request.urlopen(req, timeout=3600))
            choice, usage = data["choices"][0], data["usage"]
            record = {"prompt": row["prompt"], "response": choice["message"].get("content") or "",
                      "finish": choice["finish_reason"], "completion_tokens": usage["completion_tokens"],
                      "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
            break
        except Exception as exc:  # transient 5xx, timeouts, dropped connections
            record = {"prompt": row["prompt"], "response": "", "finish": "error",
                      "error": repr(exc), "completion_tokens": 0}
            time.sleep(5)
    with lock, open(FULL, "a") as f:  # append per result: an interrupted run stays resumable
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    rows = [json.loads(line) for line in open(DATA) if line.strip()]
    done = {json.loads(line)["prompt"] for line in open(FULL)} if os.path.exists(FULL) else set()
    todo = [r for r in rows if r["prompt"] not in done]
    print(f"{len(done)} already generated, {len(todo)} to go", flush=True)

    started = time.time()
    with cf.ThreadPoolExecutor(CONCURRENCY) as pool:
        for i, _ in enumerate(pool.map(generate, todo), 1):
            if i % 25 == 0:
                print(f"{i}/{len(todo)} done, {time.time() - started:.0f}s", flush=True)

    records = [json.loads(line) for line in open(FULL)]
    with open(SCORED, "w") as f:
        for r in records:
            f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}, ensure_ascii=False) + "\n")

    finishes = {}
    for r in records:
        finishes[r["finish"]] = finishes.get(r["finish"], 0) + 1
    tokens = sorted(r["completion_tokens"] for r in records)
    print(f"GEN_DONE n={len(records)} wall={time.time() - started:.0f}s finish={finishes} "
          f"tokens total={sum(tokens)} median={tokens[len(tokens) // 2]} max={tokens[-1]}")


if __name__ == "__main__":
    main()
