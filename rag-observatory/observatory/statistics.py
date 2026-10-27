"""Metric definitions and reproducible, paired bootstrap comparisons."""
import math
import random
from collections import Counter
from statistics import mean
from .retrieval import tokens


def retrieval_metrics(retrieved, relevant):
    if not relevant:
        return {"recall_at_k": None, "mrr": None, "ndcg": None}
    rel = set(relevant)
    hits = [int(i in rel) for i in retrieved]
    dcg = sum(h / math.log2(rank + 2) for rank, h in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(rel), len(retrieved))))
    return {"recall_at_k": len(set(retrieved) & rel) / len(rel),
            "mrr": next((1 / i for i, h in enumerate(hits, 1) if h), 0),
            "ndcg": dcg / ideal if ideal else 0}


def lexical_f1(answer, reference):
    a, b = Counter(tokens(answer)), Counter(tokens(reference))
    shared = sum((a & b).values())
    p, r = shared / max(sum(a.values()), 1), shared / max(sum(b.values()), 1)
    return 2*p*r/(p+r) if p+r else 0


def quantile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered)-1)*p
    low = int(index)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low])*(index-low)


def interval(values, seed=42, draws=2000):
    if not values:
        return None
    rng = random.Random(seed)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    return {"mean": mean(values), "low": quantile(samples, .025), "high": quantile(samples, .975), "n": len(values)}


def paired_delta(baseline, candidate, metric):
    a = {r["case_id"]: r["metrics"].get(metric) for r in baseline}
    differences = [r["metrics"][metric] - a[r["case_id"]] for r in candidate
                   if r["case_id"] in a and a[r["case_id"]] is not None and r["metrics"].get(metric) is not None]
    return interval(differences)


def psi(reference, current, edges=(0, 5, 10, 15, 25, float("inf"))):
    """Population stability index on fixed query-length bins, with pseudocounts."""
    def dist(values):
        counts = [.5] * (len(edges)-1)
        for v in values:
            for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
                if lo <= v < hi:
                    counts[i] += 1
                    break
        total = sum(counts)
        return [x / total for x in counts]
    if not reference or not current:
        return None
    a, b = dist(reference), dist(current)
    return {"value": sum((q-p)*math.log(q/p) for p, q in zip(a, b)),
            "reference": a, "current": b, "bins": ["0–4", "5–9", "10–14", "15–24", "25+"],
            "reference_n": len(reference), "current_n": len(current)}
