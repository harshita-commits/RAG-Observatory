"""One synthetic question through the dashboard worker, local RAGAS and MLflow.

docker compose exec dashboard python examples/smoke_local.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observatory.client import ObservatoryClient


def main():
    client = ObservatoryClient()
    benchmark = json.loads(Path(__file__).with_name("benchmark.json").read_text(encoding="utf-8"))
    benchmark["name"] = "Local Llama smoke test (one synthetic question)"
    benchmark["cases"] = benchmark["cases"][:1]
    identifier = client.register_benchmark(benchmark)
    job = client.run_experiment(identifier, top_k=1, mode="live", ragas=True, mlflow=True)
    print("Submitted local smoke job:", job["id"], flush=True)
    result = client.wait(job["id"], timeout=1800, interval=3)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "COMPLETE":
        raise RuntimeError("Local generation/evaluation failed; inspect the run traces")
    runs = client._request("GET", "/api/runs")
    run = next(r for r in runs if r["id"] == result["run_id"])
    scores = {k: v for k, v in run["summary"].items() if k.startswith("ragas_")}
    if len(scores) != 3 or not run["mlflow_run_id"] or run["config"].get("provider") != "ollama":
        raise RuntimeError("Expected three real local RAGAS metrics and an MLflow run")
    print(json.dumps({"config": run["config"], "ragas": scores, "mlflow_run_id": run["mlflow_run_id"]}, indent=2))


if __name__ == "__main__":
    main()
