"""Evaluate outputs from an existing pipeline without invoking its retriever/model."""
import json
import math
import os
import platform
import re
import uuid
from datetime import datetime, timezone
from .benchmark import required_text
from .experiment import summarize, code_fingerprint
from .statistics import retrieval_metrics, lexical_f1, quantile
from .telemetry import Trace
from .providers import require_provider, settings


def validate_records(records):
    if not isinstance(records, list) or not 1 <= len(records) <= 500:
        raise ValueError("Provide between 1 and 500 evaluation records per batch")
    ids = set()
    for row in records:
        if not isinstance(row, dict): raise ValueError("Each record must be an object")
        identifier = required_text(row, "case_id")
        if identifier in ids: raise ValueError("Duplicate case_id")
        ids.add(identifier)
        for field in ("question", "response", "reference", "slice"): required_text(row, field)
        relevant = row.get("relevant_ids")
        if not isinstance(relevant, list) or any(not isinstance(i, str) for i in relevant) or len(set(relevant)) != len(relevant):
            raise ValueError("relevant_ids must be a list of unique gold IDs; [] explicitly means unanswerable")
        if (not relevant) != (row["slice"] == "unanswerable"):
            raise ValueError("Unanswerable records require slice=unanswerable and relevant_ids=[]")
        contexts = row.get("contexts")
        if not isinstance(contexts, list): raise ValueError("contexts must be a list of {id, text} objects")
        context_ids = set()
        for context in contexts:
            for field in ("id", "text"): required_text(context, field)
            if context["id"] in context_ids: raise ValueError("Duplicate context ID in one record")
            context_ids.add(context["id"])
        if "latency_ms" in row and (isinstance(row["latency_ms"], bool) or not isinstance(row["latency_ms"], (int, float)) or not math.isfinite(row["latency_ms"]) or row["latency_ms"] < 0):
            raise ValueError("latency_ms must be a finite, nonnegative number")
    return records


def evaluate_external(store, project, records, pipeline_config=None, use_ragas=False, use_mlflow=False, progress=None):
    import hashlib
    validate_records(records)
    if not isinstance(project, str) or not project.strip(): raise ValueError("project is required")
    if use_ragas: require_provider(judge=True)
    from .integrations import RagasJudge, tracking
    judge = RagasJudge() if use_ragas else None
    mlflow = tracking() if use_mlflow else None
    config = {"strategy": "external", "top_k": max(len(r["contexts"]) for r in records), "split": "external",
              "mode": "external", "ragas": use_ragas, "prompt_version": "caller-managed",
              "provider": settings()["provider"] if use_ragas else "caller-managed",
              "generator_model": "caller-managed", "judge_model": settings()["judge_model"] if use_ragas else None,
              "embedding_model": settings()["embedding_model"] if use_ragas else None,
              "project": project, "pipeline_config": pipeline_config or {}}
    # Pair evaluations by gold labels, not predictions. The caller must use stable
    # evidence IDs tied to an unchanged corpus version across configurations.
    gold = sorted([{k: row[k] for k in ("case_id", "question", "reference", "relevant_ids", "slice")} for row in records], key=lambda r: r["case_id"])
    dataset_hash = hashlib.sha256(json.dumps({"project": project, "gold": gold}, sort_keys=True).encode()).hexdigest()
    run = {"id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
           "dataset_sha256": dataset_hash, "code_sha256": code_fingerprint(), "python": platform.python_version(),
           "dataset": {"name": project, "documents": len({c["id"] for r in records for c in r["contexts"]}),
                       "cases": len(records), "source": "external"}, "mlflow_run_id": None,
           "sample_size_warning": "Caller-provided labels and predictions; validate stable corpus IDs and independent test design."}
    remote = mlflow.start_run(run_name=f"external-{project}") if mlflow else None
    results = []
    try:
        if mlflow:
            run["mlflow_run_id"] = remote.info.run_id
            mlflow.log_params({"project": project, "mode": "external", "ragas": use_ragas})
            mlflow.log_dict(config, "pipeline_config.json")
        for index, row in enumerate(records, 1):
            trace = Trace(mlflow)
            citations = re.findall(r"\[([A-Za-z0-9_.-]+)\]", row["response"])
            result = {"trace_id": trace.id, "case_id": row["case_id"], "question": row["question"], "slice": row["slice"],
                      "reference": row["reference"], "relevant_ids": row["relevant_ids"], "answer": row["response"],
                      "retrieved": row["contexts"], "citations": citations, "status": "OK", "flags": [], "metrics": {},
                      "source_latency_ms": row.get("latency_ms"),
                      "usage": {"input_tokens": None, "output_tokens": None, "model": "caller-managed",
                                "token_source": "not_reported", "generator_cost_usd": None}}
            try:
                with trace.span("evaluate.external", "EVALUATOR", {"case_id": row["case_id"], "question": row["question"], "caller_reported_pipeline_latency_ms": row.get("latency_ms")}) as root:
                    with trace.span("evaluate.retrieval_and_proxies", "EVALUATOR") as span:
                        metrics = retrieval_metrics([c["id"] for c in row["contexts"]], row["relevant_ids"])
                        metrics["lexical_f1_proxy"] = lexical_f1(row["response"], row["reference"])
                        metrics["citation_validity_proxy"] = sum(c in {d["id"] for d in row["contexts"]} for c in citations)/len(citations) if citations else None
                        abstained = "i do not know" in row["response"].lower()
                        metrics["abstention_accuracy"] = float(abstained) if not row["relevant_ids"] else None
                        if row["relevant_ids"] and metrics["recall_at_k"] < 1: result["flags"].append("retrieval_miss")
                        if abstained: result["flags"].append("abstained")
                        if not row["relevant_ids"] and not abstained: result["flags"].append("unsupported_answer")
                        result["metrics"] = metrics
                        span["outputs"] = metrics.copy()
                    if judge and row["relevant_ids"]:
                        with trace.span("evaluate.ragas", "EVALUATOR") as span:
                            scores = judge.score(row["question"], row["response"], [d["text"] for d in row["contexts"]], row["reference"])
                            result["metrics"].update(scores)
                            span["outputs"] = scores
                    root["outputs"] = {"answer": row["response"], "metrics": result["metrics"]}
            except Exception as exc:
                result.update(status="ERROR", error_type=type(exc).__name__)
                result["flags"].append("pipeline_error")
            result["spans"] = trace.spans
            result["latency_ms"] = trace.spans[0]["duration_ms"]
            results.append(result)
            if progress: progress(index, len(records))
        run["summary"] = summarize(results)
        reported = [r["source_latency_ms"] for r in results if r["source_latency_ms"] is not None]
        run["summary"]["source_pipeline_p95_ms"] = quantile(reported, .95)
        run["summary"]["source_latency_n"] = len(reported)
        run["slices"] = {s: summarize([r for r in results if r["slice"] == s]) for s in {r["slice"] for r in results}}
        run["status"] = "FAILED" if any(r["status"] != "OK" for r in results) else "COMPLETE"
        run["gates"] = {"pipeline_errors": {"passed": run["summary"]["error_rate"] == 0, "threshold": 0}}
        if mlflow:
            mlflow.log_dict(run, "report.json")
            for key, value in run["summary"].items():
                if isinstance(value, dict): mlflow.log_metric(key, value["mean"])
                elif isinstance(value, (int, float)): mlflow.log_metric(key, value)
        store.save(run, results)
        reports = store.path.parent/"reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports/f"{run['id']}.json").write_text(json.dumps({"run": run, "records": results}, indent=2, allow_nan=False), encoding="utf-8")
        return run
    finally:
        if mlflow: mlflow.end_run(status="FAILED" if any(r["status"] != "OK" for r in results) else "FINISHED")
