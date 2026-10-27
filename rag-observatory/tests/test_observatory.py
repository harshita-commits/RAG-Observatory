import json
import math
import tempfile
import unittest
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from observatory.dataset import cases, corpus, fingerprint
from observatory.experiment import compare, execute_case, run_experiment
from observatory.retrieval import Retriever, extract_answer
from observatory.statistics import interval, paired_delta, psi, retrieval_metrics
from observatory.store import Store
from observatory.telemetry import Trace, redact
from observatory.integrations import RagasJudge, generate


class NumericalTests(unittest.TestCase):
    def test_ranking_metrics_with_two_relevant_documents(self):
        result = retrieval_metrics(["x", "b", "a"], ["a", "b"])
        self.assertEqual(result["recall_at_k"], 1)
        self.assertEqual(result["mrr"], .5)
        self.assertAlmostEqual(result["ndcg"], (1/math.log2(3)+.5)/(1+1/math.log2(3)))
        self.assertEqual(retrieval_metrics([], ["a"])["ndcg"], 0)
        self.assertIsNone(retrieval_metrics(["x"], [])["recall_at_k"])

    def test_bootstrap_and_paired_alignment(self):
        self.assertEqual(interval([0, 1, .5]), interval([0, 1, .5]))
        self.assertEqual(interval([.5]*3)["low"], .5)
        a = [{"case_id": "a", "metrics": {"x": .3}}, {"case_id": "b", "metrics": {"x": .8}}]
        b = [{"case_id": "b", "metrics": {"x": .9}}, {"case_id": "a", "metrics": {"x": .4}}]
        self.assertAlmostEqual(paired_delta(a, b, "x")["mean"], .1)
        self.assertEqual(psi([1, 7, 14], [1, 7, 14])["value"], 0)
        self.assertGreater(psi([1]*10, [30]*10)["value"], 0)

    def test_corpus_and_split_invariants(self):
        dev, test = cases("dev"), cases("test")
        self.assertEqual(len(dev), 12)
        self.assertEqual(len(test), 12)
        self.assertFalse({c["id"] for c in dev} & {c["id"] for c in test})
        ids = {d["id"] for d in corpus()}
        for case in dev+test:
            self.assertTrue(set(case["relevant_ids"]) <= ids)
        self.assertEqual(len(fingerprint()), 64)

    def test_retrievers_rank_specific_evidence_and_ties_deterministically(self):
        r = Retriever(corpus())
        for strategy in ("tfidf", "bm25", "rrf"):
            results = r.search("HMAC-SHA256 webhook payloads", strategy, 3)
            self.assertEqual(results[0]["id"], "webhooks")
            self.assertEqual(results, r.search("HMAC-SHA256 webhook payloads", strategy, 3))
            self.assertEqual(r.search("zzzzz", strategy, 3), [])
        with self.assertRaises(ValueError):
            r.search("test", "bm25", 0)

    def test_extractive_answer_uses_only_document_sentences(self):
        docs = [{"id": "r", "text": "Database backups run every six hours."}]
        answer, citations = extract_answer("How often do database backups run?", docs)
        self.assertIn(docs[0]["text"], answer)
        self.assertEqual(citations, ["r"])
        answer, citations = extract_answer("zzzzz", docs)
        self.assertIn("I do not know", answer)
        self.assertEqual(citations, [])


class TelemetryTests(unittest.TestCase):
    def test_tree_records_exception_and_restores_parent_stack(self):
        trace = Trace()
        with self.assertRaises(RuntimeError):
            with trace.span("root"):
                with trace.span("child"):
                    raise RuntimeError("sensitive request details")
        root, child = trace.spans
        self.assertEqual(child["parent_id"], root["id"])
        self.assertEqual(root["status"], "ERROR")
        self.assertGreaterEqual(child["duration_ms"], 0)
        self.assertNotIn("sensitive", json.dumps(trace.spans))
        self.assertFalse(trace.stack)

    def test_redaction_of_emails_and_nested_secrets(self):
        value = redact({"api_key": "private", "list": [{"text": "Contact me@example.com"}]})
        self.assertEqual(value["api_key"], "[REDACTED]")
        self.assertNotIn("me@example.com", json.dumps(value))

    def test_mlflow_span_contract_has_one_connected_root(self):
        class Remote:
            def set_inputs(self, v): pass
            def set_outputs(self, v): pass
            def set_attribute(self, k, v): pass
        class MLflow:
            def __init__(self): self.depth=0; self.depths=[]
            @contextmanager
            def start_span(self, name, span_type):
                self.depths.append(self.depth)
                self.depth += 1
                try: yield Remote()
                finally: self.depth -= 1
        remote = MLflow()
        trace = Trace(remote)
        with trace.span("root"):
            with trace.span("retrieval", "RETRIEVER"):
                pass
        self.assertEqual(remote.depths, [0, 1])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name)/"experiments.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def test_offline_run_persists_reports_metrics_and_complete_trace_trees(self):
        run = run_experiment(self.store, "bm25", 3)
        self.assertEqual(run["status"], "COMPLETE")
        records = self.store.traces(run["id"])
        self.assertEqual(len(records), 12)
        self.assertEqual(run["summary"]["recall_at_k"]["n"], 11)
        self.assertEqual(run["summary"]["abstention_accuracy"]["n"], 1)
        self.assertNotIn("ragas_faithfulness", run["summary"])
        self.assertEqual(run["summary"]["generator_cost_usd"], 0)
        for record in records:
            self.assertEqual(len(record["spans"]), 5)
            self.assertEqual(record["spans"][0]["parent_id"], None)
            self.assertEqual(self.store.trace(record["trace_id"]), record)
        report = Path(self.tmp.name)/"reports"/f"{run['id']}.json"
        self.assertEqual(json.loads(report.read_text())["run"]["id"], run["id"])
        delta = compare(self.store, run["id"], run["id"])["paired_delta"]
        self.assertEqual(delta["mean"], 0)
        self.assertEqual(delta["n"], 11)

    def test_comparison_rejects_mismatched_splits(self):
        a = run_experiment(self.store, split="dev")
        b = run_experiment(self.store, split="test")
        with self.assertRaisesRegex(ValueError, "same benchmark"):
            compare(self.store, a["id"], b["id"])

    def test_failed_judge_is_visible_and_not_a_successful_proxy_result(self):
        class BrokenJudge:
            def score(self, *args): raise TimeoutError("provider timeout")
        config = dict(strategy="bm25", top_k=3, mode="offline", judge_model="test")
        record = execute_case(cases()[0], Retriever(corpus()), config, judge=BrokenJudge())
        self.assertEqual(record["status"], "ERROR")
        self.assertEqual(record["error_type"], "TimeoutError")
        self.assertEqual(record["spans"][-1]["status"], "ERROR")
        self.assertEqual(record["spans"][0]["status"], "ERROR")

    def test_invalid_modes_are_rejected_before_any_provider_call(self):
        with self.assertRaises(ValueError):
            run_experiment(self.store, use_ragas=True)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "LLM_PROVIDER": "openai"}):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                run_experiment(self.store, mode="live")


class IntegrationContractTests(unittest.TestCase):
    def test_ragas_adapter_passes_single_turn_schema_and_rejects_nan(self):
        class Sample:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class Metric:
            name = "faithfulness"
            value = .75
            async def single_turn_ascore(self, sample):
                self.sample = sample
                return self.value
        metric = Metric()
        judge = object.__new__(RagasJudge)
        judge.metrics = [metric]
        with patch.dict("sys.modules", {"ragas": types.SimpleNamespace(SingleTurnSample=Sample)}):
            result = judge.score("question", "answer", ["evidence"], "reference")
            self.assertEqual(result, {"ragas_faithfulness": .75})
            self.assertEqual(metric.sample.retrieved_contexts, ["evidence"])
            self.assertEqual(metric.sample.reference, "reference")
            metric.value = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                judge.score("q", "a", ["c"], "r")

    def test_generator_uses_provider_tokens_and_separates_system_instructions(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="One hour. [sla]"))], usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=8))
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
        with patch.dict("os.environ", {"LLM_PROVIDER": "openai"}), patch.dict("sys.modules", {"openai": types.SimpleNamespace(OpenAI=lambda **kwargs: client)}):
            answer, citations, usage = generate("Response time?", [{"id": "sla", "text": "One hour."}])
        self.assertEqual(citations, ["sla"])
        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 8)
        self.assertEqual(calls[0]["messages"][0]["role"], "system")
        self.assertNotIn("Response time?", calls[0]["messages"][0]["content"])
        self.assertEqual(answer, "One hour. [sla]")


if __name__ == "__main__":
    unittest.main()
