"""Zero-shot baseline: ask a large API model (Claude) for five ranked names.

No examples in the prompt; same normalized input the fine-tuned model sees.
Run it on a fixed random sample of the test set to bound cost.

    python -m fnname.llm.zero_shot --test data/test.jsonl --out preds/zeroshot.jsonl --sample 2000

Uses server-side refusal fallbacks: decompiled code can trip safety classifiers
now and then, and a fallback model answering beats a silently empty row.
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from .prompt import SCHEMA, SYSTEM, build_prompt


def suggest(client, model, code):
    resp = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(code)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        return [], "refusal"
    text = next((b.text for b in resp.content if b.type == "text"), "")
    cands = json.loads(text)["candidates"][:5]
    return cands, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", default="preds/zeroshot.jsonl", type=Path)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--sample", type=int, default=2000, help="0 = whole test set")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test)]
    if args.sample and args.sample < len(rows):
        rows = random.Random(0).sample(rows, args.sample)
    client = anthropic.Anthropic()

    def run(row):
        try:
            cands, err = suggest(client, args.model, row["code"])
        except anthropic.BadRequestError as e:
            cands, err = [], f"bad_request: {e.message}"
        except anthropic.APIStatusError as e:  # retries already exhausted by the SDK
            cands, err = [], f"api_{e.status_code}"
        except anthropic.APIConnectionError:
            cands, err = [], "connection"
        except (json.JSONDecodeError, KeyError):
            cands, err = [], "unparseable"
        return {"id": row["id"], "preds": cands, **({"error": err} if err else {})}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with open(args.out, "w") as f, ThreadPoolExecutor(args.workers) as pool:
        for i, rec in enumerate(pool.map(run, rows), 1):
            errors += "error" in rec
            f.write(json.dumps(rec) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(rows)} ({errors} errors)", flush=True)
    print(f"wrote {args.out}: {len(rows)} rows, {errors} errors")


if __name__ == "__main__":
    main()
