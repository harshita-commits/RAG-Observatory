"""Check installed adapters without making paid LLM calls.

Run: .venv/Scripts/python.exe scripts/verify_integrations.py
"""
import importlib.metadata
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from observatory.integrations import RagasJudge


def main():
    for package in ("ragas", "mlflow", "langchain", "langchain-core", "langchain-openai"):
        print(f"{package}=={importlib.metadata.version(package)}")
    import mlflow
    from ragas import SingleTurnSample
    sample = SingleTurnSample(user_input="How many leave days?", response="24 days.",
                              retrieved_contexts=["Annual leave is 24 days."], reference="24 days.")
    assert sample.reference == "24 days."
    # Construct real adapters using a dummy key solely for local SDK validation.
    # No scoring or generation method is called and no provider request is made.
    with patch.dict(os.environ, {"OPENAI_API_KEY": "local-adapter-construction-check"}):
        judge = RagasJudge()
    print("Real RAGAS metrics initialized:", ", ".join(metric.name for metric in judge.metrics))
    assert callable(mlflow.start_span)
    print("MLflow tracing API import: OK")
    print("No paid provider calls were made.")


if __name__ == "__main__": main()
