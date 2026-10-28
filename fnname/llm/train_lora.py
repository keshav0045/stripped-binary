"""Stage 3: LoRA fine-tune a small code model to emit the function name.

Loss is computed on the name tokens only; the prompt is masked out.

    python -m fnname.llm.train_lora --base Qwen/Qwen2.5-Coder-1.5B \
        --train data/train.jsonl --val data/val.jsonl --out models/lora-qwen1.5b

Needs a CUDA GPU with ~16 GB (bf16, batch 8 x grad-accum 4 at 2K tokens).
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

from .prompt import build_prompt


def encode(tok, max_len):
    def fn(row):
        prompt = tok(build_prompt(row["code"]), add_special_tokens=False)["input_ids"]
        target = tok(" " + row["name"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        prompt = prompt[: max_len - len(target)]
        return {
            "input_ids": prompt + target,
            "attention_mask": [1] * (len(prompt) + len(target)),
            "labels": [-100] * len(prompt) + target,
        }
    return fn


def load(path, tok, max_len, limit=None):
    rows = [json.loads(l) for l in open(path)][:limit]
    ds = Dataset.from_list([{"code": r["code"], "name": r["name"]} for r in rows])
    return ds.map(encode(tok, max_len), remove_columns=["code", "name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="models/lora")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=8, per_device_eval_batch_size=8, gradient_accumulation_steps=4,
            lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=20,
            eval_strategy="steps", eval_steps=500, save_steps=500, save_total_limit=2,
            load_best_model_at_end=True, group_by_length=True, report_to="none",
        ),
        train_dataset=load(args.train, tok, args.max_len),
        eval_dataset=load(args.val, tok, args.max_len, limit=2000),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
