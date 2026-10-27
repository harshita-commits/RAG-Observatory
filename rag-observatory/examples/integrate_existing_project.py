"""Run from rag-observatory while its server is listening on port 8000.

Replace existing_pipeline with a wrapper around YOUR retriever/generator. The
included function is an extractive demonstration, not an external LLM call.
"""
import json
from pathlib import Path
from observatory.client import ObservatoryClient
from observatory.retrieval import Retriever, extract_answer


def main():
    benchmark = json.loads((Path(__file__).parent/"benchmark.json").read_text())
    retriever = Retriever(benchmark["documents"])

    def existing_pipeline(question):
        contexts = retriever.search(question, "bm25", 1)
        response, _ = extract_answer(question, contexts)
        return {"response": response, "contexts": contexts}

    client = ObservatoryClient()
    job = client.evaluate_pipeline("example-existing-project-corpus-v1", benchmark, existing_pipeline,
                                   split="test", pipeline_config={"retriever": "bm25", "k": 1, "generator": "extractive-demo"})
    print(json.dumps(client.wait(job["id"]), indent=2))


if __name__ == "__main__": main()
