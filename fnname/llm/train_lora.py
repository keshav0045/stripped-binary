"""Stage 3: LoRA fine-tune a small code model to emit the function name.

Loss is computed on the name tokens only; the prompt is masked out.

    python -m fnname.llm.train_lora --base Qwen/Qwen2.5-Coder-1.5B \
        --train data/train.jsonl --val data/val.jsonl --out models/lora-qwen1.5b

Runs on a free Kaggle/Colab T4 (16 GB): fp16 is picked automatically where bf16
isn't supported, gradient checkpointing keeps memory down, and --limit trains on
a random subset so a run fits in one session. Checkpoints are saved every 100
steps (--save-steps); rerun with --resume after a disconnect.

    # free T4, ~2-3 h
    python -m fnname.llm.train_lora --limit 30000 --max-len 1024 --resume
"""

import argparse
import inspect
import json
import random
from pathlib import Path

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
    rows = [json.loads(l) for l in open(path)]
    if limit and limit < len(rows):
        rows = random.Random(0).sample(rows, limit)  # files are ordered by project
    ds = Dataset.from_list([{"code": r["code"], "name": r["name"]} for r in rows])
    return ds.map(encode(tok, max_len), remove_columns=["code", "name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="models/lora")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--limit", type=int, help="random subset of train to use")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=100, help="checkpoint interval; a resume loses at most this many steps")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    bf16 = torch.cuda.is_bf16_supported()  # False on T4 / P100: fall back to fp16
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16 if bf16 else torch.float16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    if not bf16:
        # fp16 mixed precision needs fp32 trainable weights, or the grad scaler refuses to unscale
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model.print_trainable_parameters()

    train_ds = load(args.train, tok, args.max_len, limit=args.limit)
    # transformers 5 dropped warmup_ratio and group_by_length; warmup_steps works in 4 and 5
    steps = len(train_ds) * args.epochs / (args.batch * args.grad_accum * max(torch.cuda.device_count(), 1))
    extra = {}
    accepted = inspect.signature(TrainingArguments).parameters
    if "group_by_length" in accepted:
        extra["group_by_length"] = True
    elif "train_sampling_strategy" in accepted:
        extra["train_sampling_strategy"] = "group_by_length"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type="cosine", warmup_steps=max(int(0.03 * steps), 1), bf16=bf16, fp16=not bf16,
            logging_steps=20, eval_strategy="steps", eval_steps=500, save_steps=args.save_steps, save_total_limit=2, report_to="none", **extra,
        ),
        train_dataset=train_ds,
        eval_dataset=load(args.val, tok, args.max_len, limit=500),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    resume = args.resume and any(Path(args.out).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
