"""Framework-independent local API client; standard library only."""
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ObservatoryClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.token = self._request("GET", "/api/session")["token"]

    def _request(self, method, path, payload=None):
        body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if hasattr(self, "token"): headers["X-Observatory-Token"] = self.token
        try:
            with urlopen(Request(self.base_url+path, body, headers, method=method), timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            try: message = json.load(exc).get("error", str(exc))
            except ValueError: message = str(exc)
            raise ValueError(message) from exc

    def register_benchmark(self, benchmark):
        return self._request("POST", "/api/datasets", benchmark)["id"]

    def run_experiment(self, dataset_id, strategy="bm25", top_k=3, split="dev", mode="offline", ragas=False, mlflow=False):
        return self._request("POST", "/api/experiments", dict(dataset_id=dataset_id, strategy=strategy,
            top_k=top_k, split=split, mode=mode, ragas=ragas, mlflow=mlflow))

    def evaluate(self, project, records, pipeline_config=None, ragas=False, mlflow=False):
        return self._request("POST", "/api/evaluations", dict(project=project, records=records,
            pipeline_config=pipeline_config or {}, ragas=ragas, mlflow=mlflow))

    def job(self, job_id): return self._request("GET", "/api/jobs/"+job_id)

    def wait(self, job_id, timeout=300, interval=1):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            job = self.job(job_id)
            if job["status"] not in {"QUEUED", "RUNNING"}: return job
            time.sleep(interval)
        raise TimeoutError("Job is still pending; query it by ID later")

    def evaluate_pipeline(self, project, benchmark, pipeline, split="test", pipeline_config=None, ragas=False, mlflow=False):
        """callback(question) -> {response: str, contexts: [{id, text, ...}]}.

        Only the question enters your callback. Gold labels stay with evaluation.
        This does not reconstruct the pipeline's original internal spans.
        """
        records = []
        for case in benchmark["cases"]:
            if case["split"] != split: continue
            start = time.perf_counter()
            prediction = pipeline(case["question"])
            elapsed = (time.perf_counter()-start)*1000
            records.append({"case_id": case["id"], "question": case["question"], "slice": case["slice"],
                "reference": case["reference"], "relevant_ids": case["relevant_ids"],
                "response": prediction["response"], "contexts": prediction["contexts"], "latency_ms": elapsed})
        return self.evaluate(project, records, pipeline_config, ragas, mlflow)
