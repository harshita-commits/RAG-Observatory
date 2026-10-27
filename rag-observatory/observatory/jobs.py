"""One bounded, durable local job queue. It is not a distributed task broker."""
import importlib.util
import os
import threading
from .benchmark import validate
from .external import validate_records, evaluate_external
from .experiment import run_experiment
from .providers import require_provider, settings


def check_integrations(mode, ragas, mlflow):
    if mode == "live" or ragas:
        require_provider(generation=mode == "live", judge=ragas)
    for needed, module in ((mode == "live" and settings()["provider"] == "openai", "openai"),
                           (ragas and settings()["provider"] == "ollama", "langchain_ollama"),
                           (ragas, "ragas"), (mlflow, "mlflow")):
        if needed and importlib.util.find_spec(module) is None:
            raise ValueError(f"Missing {module}; install requirements-live.txt in the server environment")


class Jobs:
    def __init__(self, store):
        self.store = store
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        self.store.recover_jobs()
        self.thread = threading.Thread(target=self.work, daemon=True, name="observatory-worker")
        self.thread.start()

    def submit_experiment(self, request):
        benchmark = self.store.benchmark(request.get("dataset_id", ""))
        validate(benchmark)
        strategy, mode, split = request.get("strategy", "bm25"), request.get("mode", "offline"), request.get("split", "dev")
        top_k = request.get("top_k", 3)
        if strategy not in {"bm25", "tfidf", "rrf"} or mode not in {"offline", "live"} or split not in {"dev", "test"}:
            raise ValueError("Invalid strategy, mode or split")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(benchmark["documents"]):
            raise ValueError("top_k must be an integer between 1 and the corpus chunk count")
        selected = [c for c in benchmark["cases"] if c["split"] == split]
        if not selected or len(selected) > 500: raise ValueError("Selected split must contain between 1 and 500 labeled questions")
        ragas, mlflow = request.get("ragas", False), request.get("mlflow", False)
        if not isinstance(ragas, bool) or not isinstance(mlflow, bool): raise ValueError("ragas and mlflow must be booleans")
        if ragas and mode != "live": raise ValueError("Built-in RAGAS runs require live generation mode")
        check_integrations(mode, ragas, mlflow)
        return self.store.enqueue({"kind": "experiment", "dataset_id": request["dataset_id"], "strategy": strategy,
            "top_k": top_k, "split": split, "mode": mode, "ragas": ragas, "mlflow": mlflow, "total": len(selected)})

    def submit_external(self, request):
        validate_records(request.get("records"))
        if not isinstance(request.get("project"), str) or not request["project"].strip(): raise ValueError("project is required")
        if not isinstance(request.get("pipeline_config", {}), dict): raise ValueError("pipeline_config must be an object")
        ragas, mlflow = request.get("ragas", False), request.get("mlflow", False)
        if not isinstance(ragas, bool) or not isinstance(mlflow, bool): raise ValueError("ragas and mlflow must be booleans")
        check_integrations("external", ragas, mlflow)
        return self.store.enqueue({"kind": "external", "project": request["project"], "records": request["records"],
                                  "pipeline_config": request.get("pipeline_config", {}), "ragas": ragas, "mlflow": mlflow,
                                  "total": len(request["records"])})

    def process_one(self):
        job = self.store.claim_job()
        if not job: return False
        request = job["request"]
        progress = lambda done, total: self.store.update_job(job["id"], progress=done, total=total)
        try:
            if request["kind"] == "experiment":
                run = run_experiment(self.store, strategy=request["strategy"], top_k=request["top_k"], split=request["split"],
                    mode=request["mode"], use_mlflow=request["mlflow"], use_ragas=request["ragas"],
                    benchmark_data=self.store.benchmark(request["dataset_id"]), progress=progress)
            else:
                run = evaluate_external(self.store, request["project"], request["records"], request["pipeline_config"],
                                        request["ragas"], request["mlflow"], progress)
            self.store.update_job(job["id"], status=run["status"], run_id=run["id"], gates=run["gates"])
        except Exception as exc:
            # Type only: provider exception messages may contain sensitive request text.
            self.store.update_job(job["id"], status="FAILED", error=f"{type(exc).__name__}: worker failed; check server configuration. No secret-bearing error text is exported.")
        return True

    def work(self):
        while not self.stop.is_set():
            if not self.process_one(): self.stop.wait(.5)

    def close(self):
        self.stop.set()
        if self.thread: self.thread.join(timeout=2)
