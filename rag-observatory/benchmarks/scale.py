"""Synthetic local retrieval benchmark; no claims about provider/production scale.

Run: python -m benchmarks.scale --sizes 100 1000 5000 --queries 100
"""
import argparse
import json
import platform
import random
import time
import tracemalloc
from pathlib import Path
from observatory.retrieval import Retriever
from observatory.statistics import quantile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000])
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    results = []
    for size in args.sizes:
        rng = random.Random(42)
        corpus = [{"id": f"doc-{i:06d}", "text": " ".join([f"topic{i%100}"]+[f"term{rng.randrange(1000)}" for _ in range(80)])} for i in range(size)]
        tracemalloc.start()
        start = time.perf_counter()
        retriever = Retriever(corpus)
        build_seconds = time.perf_counter()-start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        for strategy in ("tfidf", "bm25", "rrf"):
            timings = []
            for i in range(args.queries):
                question = f"topic{i%100} term{i%1000} term{(i+17)%1000}"
                start = time.perf_counter()
                retriever.search(question, strategy, min(3, size))
                timings.append((time.perf_counter()-start)*1000)
            results.append({"documents": size, "strategy": strategy, "queries": args.queries,
                            "index_build_seconds_with_tracemalloc": build_seconds,
                            "index_peak_python_mib": peak/(1024**2), "query_p50_ms": quantile(timings,.5),
                            "query_p95_ms": quantile(timings,.95), "serial_queries_per_second": args.queries/(sum(timings)/1000)})
    report = {"python": platform.python_version(), "platform": platform.platform(), "seed": 42,
              "scope": "Synthetic in-memory retrieval only. Excludes ingestion, HTTP, generation, RAGAS, storage, bootstrap and concurrency. Python allocation peaks exclude corpus creation/native/process memory.", "results": results}
    folder = Path(__file__).resolve().parents[1]/"artifacts"
    folder.mkdir(exist_ok=True)
    (folder/"scale-benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
