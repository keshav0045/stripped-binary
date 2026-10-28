"""Score prediction files and build the results table with breakdowns.

    python -m fnname.evaluate --test data/test.jsonl preds/*.jsonl [--failures 20]

Prediction files are JSONL: {"id": ..., "preds": [{"name": ..., "score": ...}, ...]}
"""

import argparse
import json
from pathlib import Path

from .metrics import canonical, score

SIZE_BUCKETS = [(0, 10, "tiny <10"), (10, 50, "small 10-49"), (50, 200, "medium 50-199"), (200, 10**9, "large 200+")]


def size_bucket(n):
    return next(label for lo, hi, label in SIZE_BUCKETS if lo <= n < hi)


def opt_level(config):
    return config.split("-")[-1]


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows])


def fmt(m):
    return [m["n"], f"{100 * m['exact@1']:.1f}", f"{100 * m['exact@5']:.1f}", f"{100 * m['f1']:.1f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test.jsonl", type=Path)
    ap.add_argument("--failures", type=int, default=0, help="print the first N misses per model")
    ap.add_argument("--intersect", action="store_true",
                    help="score every model on the ids all prediction files cover (e.g. a zero-shot sample)")
    ap.add_argument("preds", nargs="+", type=Path)
    args = ap.parse_args()

    test = {r["id"]: r for r in map(json.loads, open(args.test))}
    if args.intersect:
        for path in args.preds:
            covered = {p["id"] for p in map(json.loads, open(path))}
            test = {i: r for i, r in test.items() if i in covered}
    header = ["model", "n", "exact@1", "exact@5", "subtoken F1"]
    overall, by_opt, by_size, by_cc = [], [], [], []

    for path in args.preds:
        model = path.stem
        preds = {p["id"]: [c["name"] for c in p["preds"]] for p in map(json.loads, open(path))}
        pairs = [(test[i], preds.get(i, [])) for i in test]
        overall.append([model] + fmt(score((r["name"], p) for r, p in pairs)))

        for group, key, sink in [
            ("opt", lambda r: opt_level(r["config"]), by_opt),
            ("size", lambda r: size_bucket(r["n_instructions"]), by_size),
            ("config", lambda r: r["config"], by_cc),
        ]:
            for g in sorted({key(r) for r, _ in pairs}):
                sub = [(r["name"], p) for r, p in pairs if key(r) == g]
                sink.append([model, g] + fmt(score(sub)))

        if args.failures:
            print(f"\n### Failures: {model}\n")
            misses = [(r, p) for r, p in pairs if not p or canonical(p[0]) != canonical(r["name"])]
            for r, p in misses[: args.failures]:
                print(f"- `{r['name']}` ({r['project']}, {r['config']}) -> {p[:3]}")

    print("\n## Overall (unseen projects)\n")
    print(table(overall, header))
    for title, rows, col in [("optimization level", by_opt, "opt"), ("function size (instructions)", by_size, "size"), ("compiler config", by_cc, "config")]:
        print(f"\n## By {title}\n")
        print(table(rows, ["model", col] + header[1:]))


if __name__ == "__main__":
    main()
