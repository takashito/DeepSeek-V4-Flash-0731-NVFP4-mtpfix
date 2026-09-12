#!/usr/bin/env python3
"""Decode-throughput benchmark on realistic prompts, at concurrency 1 and 4.

Speculative decoding must be measured on real text: a random-token dataset makes the
continuation degenerate, which distorts acceptance in both directions (vLLM issue #51009).
The prompt set below is deliberately mixed English/Japanese across code, math, prose and
explanation, because acceptance depends strongly on content; it is benchmark input and is
kept verbatim so the numbers in ../README.md stay reproducible.

Usage:
    BASE_URL=http://127.0.0.1:8002/v1 DSV4_API_KEY=your-token python3 realbench.py <label>

Env: BASE_URL (default http://127.0.0.1:8002/v1), DSV4_API_KEY (or K), MODEL
(default deepseek-v4-flash), MAX_TOKENS (default 512), CONCURRENCY (default "1,4").

Run it twice and report the second pass: the first concurrent pass after a server restart
reads far below steady state.
"""

import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("BASE_URL", "http://127.0.0.1:8002/v1").rstrip("/") + "/chat/completions"
API_KEY = os.environ.get("DSV4_API_KEY") or os.environ.get("K", "")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

PROMPTS = [
    "Pythonでクイックソートを実装して、各行にコメントを付けて",
    "Explain how TCP congestion control works, step by step.",
    "1から100までの素数をすべて列挙し、その合計を計算過程とともに示して",
    "東京の観光プランを3日分、時間ごとに詳しく作って",
    "Write a bash script that backs up /etc to a timestamped tar.gz and keeps only the last 7 backups.",
    "日本の歴史を縄文時代から現代まで簡潔にまとめて",
    "Implement a thread-safe LRU cache in Go with unit tests.",
    "Summarize the key ideas of the transformer architecture for a beginner.",
]


def one(prompt):
    """Stream one completion; decode rate excludes the prefill (time to first token)."""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    started, first, tokens = time.time(), None, 0
    with urllib.request.urlopen(urllib.request.Request(URL, body, headers), timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("choices") and first is None:
                first = time.time()
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]  # server-side count, spec-decode safe
    ended = time.time()
    return {"ttft": first - started, "tokens": tokens,
            "decode_tps": (tokens - 1) / (ended - first), "wall": ended - started}


def run(label, concurrency):
    started = time.time()
    with cf.ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(one, PROMPTS))
    wall = time.time() - started
    total = sum(r["tokens"] for r in results)
    per_stream = sum(r["decode_tps"] for r in results) / len(results)
    ttft = sum(r["ttft"] for r in results) / len(results) * 1000
    print(f"{label} c={concurrency}: per-stream decode {per_stream:.1f} tok/s | "
          f"aggregate {total / wall:.1f} tok/s | mean TTFT {ttft:.0f} ms | tokens {total}")
    for prompt, r in zip(PROMPTS, results):
        print(f"   {r['decode_tps']:6.1f} tok/s  {r['tokens']:4d} tok  {prompt[:40]}")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    one(PROMPTS[0])  # warm up the shapes this run will use
    for c in [int(x) for x in os.environ.get("CONCURRENCY", "1,4").split(",")]:
        run(label, c)
