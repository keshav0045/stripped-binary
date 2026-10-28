"""Evaluation metrics, following the code2seq / DIRE convention.

- exact@1 / exact@5: canonical name of a top-k prediction equals the truth
- subtoken precision / recall / F1: multiset overlap of name subtokens on the
  top-1 prediction, micro-averaged over the test set
"""

from collections import Counter

from .names import canonical, subtokens


def subtoken_counts(pred, true):
    p, t = Counter(subtokens(pred)), Counter(subtokens(true))
    tp = sum((p & t).values())
    return tp, sum(p.values()), sum(t.values())


def score(examples, k=5):
    """examples: iterable of (true_name, [ranked predicted names])."""
    n = ex1 = exk = tp = np_ = nt = 0
    for true, preds in examples:
        n += 1
        truth = canonical(true)
        canon = [canonical(p) for p in preds[:k]]
        ex1 += bool(canon) and canon[0] == truth
        exk += truth in canon
        a, b, c = subtoken_counts(preds[0] if preds else "", true)
        tp, np_, nt = tp + a, np_ + b, nt + c
    prec = tp / np_ if np_ else 0.0
    rec = tp / nt if nt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n,
        "exact@1": ex1 / n if n else 0.0,
        f"exact@{k}": exk / n if n else 0.0,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
