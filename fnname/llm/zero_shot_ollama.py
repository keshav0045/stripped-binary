"""Zero-shot baseline with a local model served by Ollama — free, no API key.

Same instructions and input as the API baseline (zero_shot.py), with Ollama's
JSON-schema output so every answer parses. Resumable: rows already in --out are
skipped, so Ctrl-C and rerun whenever.

    ollama pull qwen2.5-coder:7b
    python -m fnname.llm.zero_shot_ollama --model qwen2.5-coder:7b --sample 1000 --out preds/qwen7b_zeroshot.jsonl
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from .prompt import SCHEMA, SYSTEM, build_prompt


def suggest(host, model, code):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(code)}],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = json.load(resp)["message"]["content"]

    seen, cands = set(), []
    for c in json.loads(text)["candidates"]:
        name = str(c.get("name", "")).strip()
        if name and name not in seen:
            seen.add(name)
            cands.append({"name": name, "score": max(float(c.get("score", 0)), 0.0)})
    cands = cands[:5]
    total = sum(c["score"] for c in cands) or 1.0  # small models rarely sum to exactly 1
    for c in cands:
        c["score"] /= total
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", default="preds/ollama_zeroshot.jsonl", type=Path)
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--sample", type=int, default=1000, help="random test subset (seed 0, same as the other runs); 0 = all")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test)]
    if args.sample and args.sample < len(rows):
        rows = random.Random(0).sample(rows, args.sample)
    # Failed rows are retried on the next run; the evaluator keeps the last line per id.
    prior = [json.loads(l) for l in open(args.out)] if args.out.exists() else []
    done = {r["id"] for r in prior if "error" not in r}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go with {args.model}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start, errors = time.time(), 0
    with open(args.out, "a") as f:
        for i, row in enumerate(todo, 1):
            try:
                rec = {"id": row["id"], "preds": suggest(args.host, args.model, row["code"])}
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as e:
                errors += 1
                rec = {"id": row["id"], "preds": [], "error": type(e).__name__}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                per = (time.time() - start) / i
                print(f"{len(done) + i}/{len(rows)}  {per:.1f}s each, ~{per * (len(todo) - i) / 60:.0f} min left, {errors} errors", flush=True)


if __name__ == "__main__":
    main()
