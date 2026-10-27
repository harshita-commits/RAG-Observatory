import argparse
import json
import os
import sys
from .store import Store


def main():
    # MLflow prints Unicode run links when ending a run. Windows pipe defaults
    # such as cp1252 cannot encode those symbols and can otherwise fail a job
    # after its metrics/artifacts have already been saved.
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="RAG Observatory: offline experiments, live RAGAS, MLflow tracing")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Run three credential-free retrieval experiments on dev only")
    serve = sub.add_parser("serve", help="Open the local observability dashboard")
    serve.add_argument("--port", type=int, default=8000)
    sweep = sub.add_parser("sweep", help="Run a reproducible retrieval ablation")
    sweep.add_argument("--split", choices=["dev", "test"], default="dev")
    sweep.add_argument("--mode", choices=["offline", "live"], default="offline")
    sweep.add_argument("--mlflow", action="store_true")
    sweep.add_argument("--ragas", action="store_true")
    sweep.add_argument("--dataset", help="Path to a labeled custom benchmark JSON")
    run = sub.add_parser("run", help="Run a single configuration, including final held-out evaluation")
    run.add_argument("--strategy", choices=["tfidf", "bm25", "rrf"], default="bm25")
    run.add_argument("--top-k", type=int, default=3)
    run.add_argument("--split", choices=["dev", "test"], default="dev")
    run.add_argument("--mode", choices=["offline", "live"], default="offline")
    run.add_argument("--mlflow", action="store_true")
    run.add_argument("--ragas", action="store_true")
    run.add_argument("--dataset", help="Path to a labeled custom benchmark JSON")
    ingest = sub.add_parser("ingest", help="Parse supported document formats into an unlabeled benchmark JSON")
    ingest.add_argument("--documents", required=True, help="Folder to scan recursively")
    ingest.add_argument("--output", required=True, help="New JSON filename; never overwrites existing files")
    ingest.add_argument("--chunk-words", type=int, default=300)
    ingest.add_argument("--overlap-words", type=int, default=50)
    comparison = sub.add_parser("compare")
    comparison.add_argument("baseline")
    comparison.add_argument("candidate")
    args = parser.parse_args()
    if args.command == "ingest":
        from .benchmark import import_documents
        data = import_documents(args.documents, args.output, args.chunk_words, args.overlap_words)
        print(f"Imported {len(data['documents'])} chunks into {args.output}. Add labeled cases before running experiments.")
        return
    store = Store()
    if args.command == "serve":
        from .server import serve
        serve(store, args.port)
    elif args.command == "compare":
        from .experiment import compare
        print(json.dumps(compare(store, args.baseline, args.candidate), indent=2))
    else:
        from .experiment import run_experiment
        from .benchmark import load
        size = len(load(getattr(args, "dataset", None))["documents"])
        configs = [("tfidf", 1), ("bm25", 3), ("rrf", 3)] if args.command == "demo" else ([(s, k) for s in ("tfidf", "bm25", "rrf") for k in (1, 3, 5) if k <= size] if args.command == "sweep" else [(args.strategy, args.top_k)])
        failed = False
        for strategy, k in configs:
            run = run_experiment(store, strategy, k, getattr(args, "split", "dev"), getattr(args, "mode", "offline"), getattr(args, "mlflow", False), getattr(args, "ragas", False), getattr(args, "dataset", None))
            print(json.dumps({"id": run["id"], "config": run["config"], "status": run["status"], "summary": run["summary"], "gates": run["gates"]}, indent=2))
            failed |= run["status"] == "FAILED" or not all(g["passed"] for g in run["gates"].values())
        # Demo intentionally includes weak baselines. Other commands are useful as CI gates.
        if failed and args.command != "demo":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
