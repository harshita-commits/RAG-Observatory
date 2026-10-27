import hashlib
import json
import os
import platform
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .dataset import cases
from .benchmark import load as load_benchmark, fingerprint as benchmark_fingerprint
from .retrieval import Retriever, extract_answer, tokens
from .statistics import interval, lexical_f1, paired_delta, psi, quantile, retrieval_metrics
from .telemetry import Trace
from .integrations import PROMPT_VERSION, SYSTEM_PROMPT
from .providers import require_provider, settings


def code_fingerprint():
    files = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


def summarize(records):
    valid = [r for r in records if r["status"] == "OK"]
    keys = sorted({k for r in valid for k in r["metrics"]})
    summary = {k: interval([r["metrics"][k] for r in valid if r["metrics"].get(k) is not None]) for k in keys}
    elapsed = [r["latency_ms"] for r in records]
    summary.update({"latency_p50_ms": quantile(elapsed, .5), "latency_p95_ms": quantile(elapsed, .95),
                    "error_rate": 1 - len(valid)/max(len(records), 1), "requests": len(records),
                    "input_tokens": sum(r["usage"]["input_tokens"] for r in records) if all(r["usage"]["input_tokens"] is not None for r in records) else None,
                    "output_tokens": sum(r["usage"]["output_tokens"] for r in records) if all(r["usage"]["output_tokens"] is not None for r in records) else None,
                    "generator_cost_usd": sum(r["usage"]["generator_cost_usd"] for r in records)
                    if all(r["usage"]["generator_cost_usd"] is not None for r in records) else None})
    return summary


def execute_case(case, retriever, config, mlflow=None, judge=None):
    trace = Trace(mlflow)
    record = {"trace_id": trace.id, "case_id": case["id"], "question": case["question"], "slice": case["slice"],
              "reference": case["reference"], "relevant_ids": case["relevant_ids"], "status": "OK",
              "metrics": {}, "answer": "", "retrieved": [], "citations": [], "flags": [],
              "usage": {"input_tokens": 0, "output_tokens": 0, "model": "extractive", "token_source": "estimated",
                        "generator_cost_usd": 0 if config["mode"] == "offline" else None}}
    started = time.perf_counter()
    try:
        with trace.span("rag.request", inputs={"question": case["question"], "case_id": case["id"], "config": config}) as root:
            with trace.span("guard.input", "GUARDRAIL") as span:
                detected = bool(re.search(r"ignore previous|disregard the rules|reveal secrets|disclose passwords", case["question"], re.I))
                if detected:
                    record["flags"].append("instruction_override_pattern")
                span["outputs"] = {"instruction_override_pattern": detected, "action": "isolate_question_from_system_prompt"}
            with trace.span("retrieve." + config["strategy"], "RETRIEVER", {"top_k": config["top_k"]}) as span:
                docs = retriever.search(case["question"], config["strategy"], config["top_k"])
                record["retrieved"] = docs
                span["outputs"] = {"documents": docs}
            with trace.span("generate." + config["mode"], "LLM", {"prompt_version": PROMPT_VERSION}) as span:
                if config["mode"] == "live":
                    from .integrations import generate
                    answer, citations, usage = generate(case["question"], docs)
                    record["usage"].update(usage, token_source="provider")
                    if "INPUT_USD_PER_MILLION" in os.environ and "OUTPUT_USD_PER_MILLION" in os.environ:
                        record["usage"]["generator_cost_usd"] = (
                            usage["input_tokens"]*float(os.environ["INPUT_USD_PER_MILLION"]) +
                            usage["output_tokens"]*float(os.environ["OUTPUT_USD_PER_MILLION"])) / 1e6
                else:
                    answer, citations = extract_answer(case["question"], docs)
                    record["usage"].update(input_tokens=len(tokens(case["question"] + " ".join(d["text"] for d in docs))),
                                           output_tokens=len(tokens(answer)))
                record.update(answer=answer, citations=citations)
                span["outputs"] = {"answer": answer, "citations": citations, "usage": record["usage"]}
            with trace.span("evaluate.retrieval_and_proxies", "EVALUATOR") as span:
                metrics = retrieval_metrics([d["id"] for d in docs], case["relevant_ids"])
                # Surface validity of citation IDs is not a semantic faithfulness score.
                metrics["citation_validity_proxy"] = (sum(c in {d["id"] for d in docs} for c in citations)/len(citations)) if citations else None
                metrics["lexical_f1_proxy"] = lexical_f1(answer, case["reference"])
                abstained = "i do not know" in answer.lower()
                metrics["abstention_accuracy"] = float(abstained) if not case["relevant_ids"] else None
                if abstained:
                    record["flags"].append("abstained")
                if case["relevant_ids"] and metrics["recall_at_k"] < 1:
                    record["flags"].append("retrieval_miss")
                if case["relevant_ids"] and abstained:
                    record["flags"].append("false_abstention")
                if not case["relevant_ids"] and not abstained:
                    record["flags"].append("unsupported_answer")
                record["metrics"] = metrics
                span["outputs"] = metrics.copy()
            if judge:
                with trace.span("evaluate.ragas", "EVALUATOR", {"judge_model": config["judge_model"]}) as span:
                    # Context recall has no meaningful reference evidence for unanswerables.
                    if case["relevant_ids"]:
                        scores = judge.score(case["question"], answer, [d["text"] for d in docs], case["reference"])
                        record["metrics"].update(scores)
                        span["outputs"] = scores
                    else:
                        span["outputs"] = {"skipped": "unanswerable; evaluated separately for abstention"}
            root["outputs"] = {"answer": record["answer"], "metrics": record["metrics"], "flags": record["flags"]}
    except Exception as exc:
        record.update(status="ERROR", error_type=type(exc).__name__)
        record["flags"].append("pipeline_error")
    record["latency_ms"] = (time.perf_counter()-started)*1000
    record["spans"] = trace.spans
    return record


def run_experiment(store, strategy="bm25", top_k=3, split="dev", mode="offline", use_mlflow=False, use_ragas=False, dataset_path=None, benchmark_data=None, progress=None):
    if split not in {"dev", "test"} or mode not in {"offline", "live"}:
        raise ValueError("Invalid split or mode")
    from .benchmark import validate
    benchmark = validate(benchmark_data) if benchmark_data is not None else load_benchmark(dataset_path)
    documents = benchmark["documents"]
    selected_cases = [c for c in benchmark["cases"] if c["split"] == split]
    if not selected_cases:
        raise ValueError(f"No labeled cases in the {split} split")
    dataset_hash = benchmark_fingerprint(benchmark)
    if strategy not in {"tfidf", "bm25", "rrf"} or not 1 <= top_k <= len(documents):
        raise ValueError("Invalid retrieval configuration")
    if use_ragas and mode != "live":
        raise ValueError("RAGAS is enabled only in live mode; offline proxies are explicitly separate")
    if mode == "live":
        require_provider(generation=True, judge=use_ragas)
    from .integrations import RagasJudge, tracking
    mlflow = tracking() if use_mlflow else None
    judge = RagasJudge() if use_ragas else None
    config = dict(strategy=strategy, top_k=top_k, split=split, mode=mode,
                  ragas=use_ragas, prompt_version=PROMPT_VERSION,
                  provider=settings()["provider"] if mode == "live" else "offline",
                  generator_model=settings()["generator_model"] if mode == "live" else "extractive",
                  judge_model=settings()["judge_model"] if use_ragas else None,
                  embedding_model=settings()["embedding_model"] if use_ragas else None)
    run = {"id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
           "dataset_sha256": dataset_hash, "code_sha256": code_fingerprint(), "python": platform.python_version(),
           "dataset": {"name": benchmark["name"], "documents": len(documents), "cases": len(benchmark["cases"]),
                       "source": "custom" if dataset_path or benchmark_data else "bundled", "ingestion": benchmark.get("ingestion")},
           "mlflow_run_id": None, "sample_size_warning": "Question-level bootstrap intervals do not establish production performance. Validate on a sufficiently large, independent test set."}
    remote = mlflow.start_run(run_name=f"{strategy}-k{top_k}-{split}-{mode}") if mlflow else None
    try:
        if mlflow:
            run["mlflow_run_id"] = remote.info.run_id
            mlflow.log_params({k: str(v) for k, v in config.items()})
            mlflow.set_tags({"dataset_sha256": dataset_hash, "code_sha256": run["code_sha256"], "observatory.run_id": run["id"]})
        retriever = Retriever(documents)
        records = []
        for index, case in enumerate(selected_cases, 1):
            records.append(execute_case(case, retriever, config, mlflow, judge))
            if progress: progress(index, len(selected_cases))
        run["summary"] = summarize(records)
        run["slices"] = {s: summarize([r for r in records if r["slice"] == s]) for s in sorted({r["slice"] for r in records})}
        run["gates"] = {
            "recall_at_k": {"passed": bool(run["summary"].get("recall_at_k") and run["summary"]["recall_at_k"]["mean"] >= .9), "threshold": .9},
            "pipeline_errors": {"passed": run["summary"]["error_rate"] == 0, "threshold": 0},
            "unanswerable_abstention": {"passed": bool(run["summary"].get("abstention_accuracy") and run["summary"]["abstention_accuracy"]["mean"] == 1), "threshold": 1}}
        run["status"] = "FAILED" if any(r["status"] != "OK" for r in records) else "COMPLETE"
        if mlflow:
            for key, value in run["summary"].items():
                if isinstance(value, dict):
                    mlflow.log_metric(key, value["mean"])
                elif isinstance(value, (float, int)):
                    mlflow.log_metric(key, value)
            mlflow.log_dict(run, "report.json")
            mlflow.log_dict(benchmark, "benchmark.json")
            mlflow.log_text(SYSTEM_PROMPT, "system_prompt.txt")
        store.save(run, records)
        artifacts = store.path.parent / "reports"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / f"{run['id']}.json").write_text(json.dumps({"run": run, "records": records}, indent=2, allow_nan=False), encoding="utf-8")
        return run
    finally:
        if mlflow:
            mlflow.end_run(status="FAILED" if "records" not in locals() or any(r["status"] != "OK" for r in records) else "FINISHED")


def compare(store, baseline_id, candidate_id):
    a, b = store.run(baseline_id), store.run(candidate_id)
    if a["dataset_sha256"] != b["dataset_sha256"] or a["config"].get("provider") != b["config"].get("provider") or any(a["config"][k] != b["config"][k] for k in ("split", "mode", "ragas", "generator_model", "judge_model", "embedding_model", "prompt_version")):
        raise ValueError("Compare runs with the same benchmark, split, generation mode and evaluator configuration")
    left, right = store.traces(baseline_id), store.traces(candidate_id)
    if any(r["status"] != "OK" for r in left + right):
        raise ValueError("Fix failed requests before comparing experiments; complete-case selection would bias the result")
    if {r["case_id"] for r in left} != {r["case_id"] for r in right}:
        raise ValueError("Paired comparisons require identical case IDs")
    delta = paired_delta(left, right, "recall_at_k")
    if delta is None:
        raise ValueError("Paired retrieval comparison needs at least one answerable question")
    return {"baseline": baseline_id, "candidate": candidate_id, "metric": "recall_at_k",
            "paired_delta": delta,
            "interpretation": "Candidate minus baseline; 95% paired percentile bootstrap CI, 2000 draws, seed 42. Small benchmark; exploratory, no multiple-testing correction."}


def drift_report():
    return {"feature": "query length in content tokens", "baseline": "authored dev split", "current": "authored test split",
            "psi": psi([len(tokens(c["question"])) for c in cases("dev")], [len(tokens(c["question"])) for c in cases("test")]),
            "warning": "Cross-split descriptive comparison, not evidence of production drift. Only 12 queries per cohort."}
