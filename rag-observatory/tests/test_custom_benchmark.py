import copy
import json
import tempfile
import unittest
from pathlib import Path
from observatory.benchmark import load, fingerprint, validate, import_documents
from observatory.experiment import run_experiment, compare
from observatory.store import Store

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "benchmark.json"


class CustomBenchmarkTests(unittest.TestCase):
    def test_example_runs_and_uses_custom_evidence_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "db.sqlite")
            run = run_experiment(store, top_k=1, dataset_path=EXAMPLE)
            self.assertEqual(run["dataset"]["name"], "Example leave policy")
            self.assertEqual(run["dataset"]["documents"], 1)
            records = store.traces(run["id"])
            self.assertEqual({r["case_id"] for r in records}, {"leave-dev-1", "leave-dev-2"})
            self.assertEqual(records[0]["retrieved"][0]["id"], "leave-policy-1")
            self.assertEqual(run["summary"]["recall_at_k"]["mean"], 1)
            self.assertEqual(run["summary"]["abstention_accuracy"]["mean"], 1)
            self.assertTrue(all(g["passed"] for g in run["gates"].values()))

    def test_invalid_labels_and_duplicate_ids_are_rejected(self):
        for mutation in (lambda d: d["cases"][0].update(relevant_ids=["missing"]),
                         lambda d: d["documents"].append(d["documents"][0].copy()),
                         lambda d: d["cases"][0].update(reference=""),
                         lambda d: d["cases"][1].update(slice="direct")):
            data = load(EXAMPLE)
            mutation(data)
            with self.assertRaises(ValueError):
                validate(data)

    def test_ingest_overlap_sources_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            docs = base / "docs"
            docs.mkdir()
            (docs / "one.md").write_text("one two three four five six seven", encoding="utf-8")
            output = base / "imported.json"
            data = import_documents(docs, output, chunk_words=4, overlap_words=1)
            self.assertEqual([d["text"] for d in data["documents"]], ["one two three four", "four five six seven"])
            self.assertEqual(data["documents"][0]["source"], "one.md")
            self.assertEqual(data["cases"], [])
            with self.assertRaisesRegex(ValueError, "labeled questions"):
                load(output)
            with self.assertRaisesRegex(ValueError, "already exists"):
                import_documents(docs, output)
            with self.assertRaises(ValueError):
                import_documents(docs, base/"invalid.json", chunk_words=4, overlap_words=4)

    def test_hash_changes_with_content_and_labels_but_not_row_order(self):
        original = load(EXAMPLE)
        changed = copy.deepcopy(original)
        changed["cases"].reverse()
        self.assertEqual(fingerprint(original), fingerprint(changed))
        changed["cases"][0]["reference"] += " changed"
        self.assertNotEqual(fingerprint(original), fingerprint(changed))

    def test_empty_selected_split_and_cross_dataset_comparison_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder)/"db.sqlite")
            data = load(EXAMPLE)
            data["cases"] = [c for c in data["cases"] if c["split"] == "dev"]
            custom = Path(folder)/"custom.json"
            custom.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No labeled cases"):
                run_experiment(store, top_k=1, split="test", dataset_path=custom)
            a = run_experiment(store, top_k=1, dataset_path=EXAMPLE)
            b = run_experiment(store, top_k=1, dataset_path=custom)
            with self.assertRaisesRegex(ValueError, "same benchmark"):
                compare(store, a["id"], b["id"])


if __name__ == "__main__":
    unittest.main()
