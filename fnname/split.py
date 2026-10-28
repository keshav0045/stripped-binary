"""Stage 1d: split by project, then remove cross-project leakage.

Splitting by function would put zlib's inflate() at -O0 in train and the same
inflate() at -O2 in test. Splitting by project fixes that, but projects still
vendor each other's code (gnulib, getopt, md5...). So after the project split,
any val/test function whose normalized body also appears in train is dropped.

    python -m fnname.split --data data/functions.jsonl --out data/
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def assign(project, val_frac, test_frac, seed):
    h = int(hashlib.sha1(f"{seed}:{project}".encode()).hexdigest(), 16) / 16**40
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/functions.jsonl", type=Path)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--test-projects", nargs="*", default=[], help="force these into test")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    split_of = {}
    for p in sorted({r["project"] for r in rows}):
        split_of[p] = "test" if p in args.test_projects else assign(p, args.val, args.test, args.seed)

    train_hashes = {r["code_hash"] for r in rows if split_of[r["project"]] == "train"}
    train_names = {r["name"] for r in rows if split_of[r["project"]] == "train"}
    stats = Counter()
    files = {s: open(args.out / f"{s}.jsonl", "w") for s in ("train", "val", "test")}
    for i, r in enumerate(rows):
        s = split_of[r["project"]]
        if s != "train" and r["code_hash"] in train_hashes:
            stats[f"{s}_dropped_body_seen_in_train"] += 1
            continue
        stats[s] += 1
        if s != "train" and r["name"] in train_names:
            stats[f"{s}_name_seen_in_train"] += 1
        r["id"] = i
        files[s].write(json.dumps(r) + "\n")
    for f in files.values():
        f.close()

    json.dump(split_of, open(args.out / "split_projects.json", "w"), indent=2)
    for k, v in sorted(stats.items()):
        print(f"{k:<36} {v}")


if __name__ == "__main__":
    main()
