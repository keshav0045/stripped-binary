"""Top-k name suggestions from the fine-tuned model via beam search.

Confidence is the softmax over the k beams' length-normalized log-probs, so the
five scores sum to 1 — comparable in spirit to the kNN vote share.

    python -m fnname.llm.predict_lora --adapter models/lora --test data/test.jsonl --out preds/lora.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from .prompt import build_prompt


class LoraNamer:
    def __init__(self, adapter, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(adapter)
        self.tok.padding_side = "left"
        self.model = AutoPeftModelForCausalLM.from_pretrained(adapter, torch_dtype=torch.bfloat16).to(self.device).eval()

    @torch.no_grad()
    def predict(self, codes, top=5):
        enc = self.tok([build_prompt(c) for c in codes], return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=16, num_beams=top, num_return_sequences=top,
            output_scores=True, return_dict_in_generate=True, early_stopping=True,
            pad_token_id=self.tok.pad_token_id,
        )
        texts = self.tok.batch_decode(out.sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        probs = out.sequences_scores.view(len(codes), top).softmax(-1).tolist()
        results = []
        for i in range(len(codes)):
            seen, cands = set(), []
            for t, p in zip(texts[i * top:(i + 1) * top], probs[i]):
                name = t.strip().split()[0] if t.strip() else ""
                if name and name not in seen:
                    seen.add(name)
                    cands.append({"name": name, "score": p})
            results.append(cands)
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/lora")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out", default="preds/lora.jsonl", type=Path)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    namer = LoraNamer(args.adapter)
    rows = [json.loads(l) for l in open(args.test)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            for r, p in zip(chunk, namer.predict([r["code"] for r in chunk])):
                f.write(json.dumps({"id": r["id"], "preds": p}) + "\n")
            print(f"{i + len(chunk)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
