"""Baselines: frequency prior and TF-IDF nearest-neighbour name lookup.

kNN retrieves the k most similar training functions by cosine similarity over
TF-IDF of code tokens (identifiers, callees, string-literal words, constants)
and ranks candidate names by summed similarity. Confidence is a candidate's
share of the total neighbour similarity.

    python -m fnname.baselines.knn --train data/train.jsonl --test data/test.jsonl \
        --out preds/knn.jsonl --save models/knn.pkl
"""

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from ..normalize import tokens

# Placeholders and Ghidra's generated locals carry no meaning across functions.
_NOISE = re.compile(r"^(FUN|DAT|LAB|PTR|UNK|switchD|caseD)_\d+$|^[a-z]*Var\d+$|^(local|param|in|stack)_\w+$")


def features(code):
    out = []
    for t in tokens(code):
        if t.startswith('"'):
            out += ["S:" + w.lower() for w in re.findall(r"[A-Za-z]{3,}", t)]
        elif not _NOISE.match(t):
            out.append(t)
    return out


class KNNNamer:
    def __init__(self, k=20, max_df=0.5):
        self.k = k
        self.vec = TfidfVectorizer(analyzer=features, sublinear_tf=True, min_df=2, max_df=max_df)

    def fit(self, codes, names):
        self.X = self.vec.fit_transform(codes)
        self.names = list(names)
        return self

    def predict(self, codes, top=5):
        sims = self.vec.transform(codes) @ self.X.T
        results = []
        for row in sims:
            row = row.tocoo()
            order = row.data.argsort()[::-1][: self.k]
            votes = defaultdict(float)
            for j in order:
                votes[self.names[row.col[j]]] += row.data[j]
            total = sum(votes.values()) or 1.0
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])[:top]
            results.append([{"name": n, "score": s / total} for n, s in ranked])
        return results


def load(path):
    return [json.loads(l) for l in open(path)]


def write_preds(path, rows, preds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"id": r["id"], "preds": p}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl", type=Path)
    ap.add_argument("--test", default="data/test.jsonl", type=Path)
    ap.add_argument("--out", default="preds/knn.jsonl", type=Path)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    # Import by package path so the pickle references fnname.baselines.knn, not __main__.
    from fnname.baselines.knn import KNNNamer as Namer

    train, test = load(args.train), load(args.test)
    model = Namer(k=args.k).fit([r["code"] for r in train], [r["name"] for r in train])
    write_preds(args.out, test, model.predict([r["code"] for r in test]))

    prior = Counter(r["name"] for r in train).most_common(5)
    total = len(train)
    freq = [{"name": n, "score": c / total} for n, c in prior]
    write_preds(args.out.with_name("frequency.jsonl"), test, [freq] * len(test))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(model, open(args.save, "wb"))
    print(f"wrote {args.out} and {args.out.with_name('frequency.jsonl')}")


if __name__ == "__main__":
    main()
