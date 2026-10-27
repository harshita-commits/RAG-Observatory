import base64
import copy
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from observatory.benchmark import load
from observatory.client import ObservatoryClient
from observatory.external import evaluate_external
from observatory.jobs import Jobs
from observatory.server import create_server
from observatory.store import Store

EXAMPLE = Path(__file__).resolve().parents[1]/"examples"/"benchmark.json"


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name)/"db.sqlite")
        self.server = create_server(self.store, port=0, start_worker=False)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.05), daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.client = ObservatoryClient(self.base)

    def tearDown(self):
        self.server.shutdown()
        self.server.jobs.close()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def test_dashboard_job_flow_and_durable_progress(self):
        dataset = self.client.register_benchmark(load(EXAMPLE))
        job = self.client.run_experiment(dataset, top_k=1)
        self.assertEqual(job["status"], "QUEUED")
        self.assertTrue(self.server.jobs.process_one())
        done = self.client.job(job["id"])
        self.assertEqual(done["status"], "COMPLETE")
        self.assertEqual(done["progress"], 2)
        self.assertTrue(done["run_id"])
        self.assertEqual(len(self.store.traces(done["run_id"])), 2)
        self.assertNotIn("request", done)

    def test_background_worker_completes_without_blocking_api(self):
        self.server.jobs.start()
        dataset = self.client.register_benchmark(load(EXAMPLE))
        job = self.client.run_experiment(dataset, top_k=1)
        done = self.client.wait(job["id"], timeout=10, interval=.05)
        self.assertEqual(done["status"], "COMPLETE")

    def test_upload_retains_location_and_requires_labels(self):
        result = self.client._request("POST", "/api/uploads", {"name": "CSV import", "files": [{"name": "policy.csv", "content_base64": base64.b64encode(b"policy,days\nannual leave,24\n").decode()}]})
        benchmark = result["benchmark"]
        self.assertEqual(benchmark["documents"][0]["location"]["row"], 2)
        self.assertIn("days: 24", benchmark["documents"][0]["text"])
        with self.assertRaisesRegex(ValueError, "labeled"):
            self.client.run_experiment(result["id"], top_k=1)
        benchmark["cases"] = [{"id": "c1", "question": "How many days of annual leave?", "reference": "24 days", "split": "dev", "slice": "direct", "relevant_ids": [benchmark["documents"][0]["id"]]}]
        new_id = self.client.register_benchmark(benchmark)
        self.assertNotEqual(new_id, result["id"])
        self.assertEqual(self.store.benchmark(result["id"])["cases"], [])

    def test_external_pipeline_evaluates_existing_predictions(self):
        questions = []
        def pipeline(question):
            questions.append(question)
            if "carryover" in question:
                return {"response": "Five days. [leave-policy-1]", "contexts": [{"id": "leave-policy-1", "text": "Carryover is limited to five days."}]}
            return {"response": "I do not know based on the available evidence.", "contexts": []}
        job = self.client.evaluate_pipeline("existing-rag-v1", load(EXAMPLE), pipeline, split="test")
        self.server.jobs.process_one()
        done = self.client.wait(job["id"])
        run = self.store.run(done["run_id"])
        self.assertEqual(run["config"]["mode"], "external")
        self.assertEqual(run["summary"]["recall_at_k"]["mean"], 1)
        self.assertIsNone(run["summary"]["generator_cost_usd"])
        self.assertIsNone(run["summary"]["input_tokens"])
        self.assertEqual(len(questions), 2)
        for trace in self.store.traces(run["id"]):
            self.assertEqual(trace["spans"][0]["name"], "evaluate.external")
            self.assertFalse(any(s["kind"] == "LLM" for s in trace["spans"]))
            self.assertIsNotNone(trace["source_latency_ms"])

    def test_security_rejects_missing_token_and_foreign_origin(self):
        for headers in ({"Content-Type": "application/json"},
                        {"Content-Type": "application/json", "Origin": "https://evil.example", "X-Observatory-Token": self.client.token}):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.base+"/api/datasets", b"{}", headers), timeout=5)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()

    def test_invalid_config_and_pagination(self):
        identifier = self.client.register_benchmark(load(EXAMPLE))
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.client.run_experiment(identifier, top_k=999)
        with self.assertRaises(ValueError):
            self.client._request("GET", "/api/runs?limit=10000")
        self.assertEqual(self.client._request("GET", "/api/runs?limit=1&offset=0"), [])

    def test_comparison_requires_an_answerable_case(self):
        from observatory.experiment import compare
        records = [{"case_id": "unknown", "question": "What is unavailable?", "response": "I do not know.",
                    "reference": "I do not know.", "relevant_ids": [], "contexts": [], "slice": "unanswerable"}]
        run = evaluate_external(self.store, "unknown-corpus", records)
        with self.assertRaisesRegex(ValueError, "answerable question"):
            compare(self.store, run["id"], run["id"])

    def test_queue_backpressure_and_restart_recovery(self):
        for i in range(8): self.store.enqueue({"total": i})
        with self.assertRaises(OverflowError): self.store.enqueue({})
        claimed = self.store.claim_job()
        self.assertEqual(claimed["status"], "RUNNING")
        self.store.recover_jobs()
        self.assertEqual(self.store.job(claimed["id"])["status"], "INTERRUPTED")
        self.assertEqual(sum(j["status"] == "QUEUED" for j in self.store.jobs()), 7)


if __name__ == "__main__": unittest.main()
