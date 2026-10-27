# Integrate an existing RAG project

Observatory can evaluate outputs your project already generated. You keep your retriever, vector database, prompt, model provider and application. The adapter only requires a stable prediction/evidence schema and independent evaluation labels.

## Three integration paths

1. **Dashboard:** Integrations → paste a JSON array or select JSON/JSONL → Evaluate supplied outputs.
2. **Python SDK:** `ObservatoryClient.evaluate(...)` submits existing predictions; `evaluate_pipeline(...)` executes your callback in your application process and submits the resulting predictions.
3. **HTTP:** `POST /api/evaluations` accepts the same data contract from any language.

These are batch evaluation paths. They do not attach a live observer automatically to an existing deployment or reconstruct internal source spans.

## Prediction schema

```json
{
  "project": "support-rag-corpus-v1",
  "pipeline_config": {"retriever": "my-retriever", "model": "my-model", "k": 3},
  "ragas": false,
  "mlflow": false,
  "records": [{
    "case_id": "q1",
    "question": "What is the annual leave entitlement?",
    "response": "24 days. [policy-1]",
    "contexts": [{"id": "policy-1", "text": "Annual leave entitlement is 24 days.", "source": "policy.pdf", "score": 0.9}],
    "reference": "24 days.",
    "relevant_ids": ["policy-1"],
    "slice": "direct",
    "latency_ms": 180
  }]
}
```

`case_id`, `question`, `response`, `contexts`, `reference`, `relevant_ids` and `slice` are required. `latency_ms` and context `source`/`score` are optional. The contexts array is ordered by retrieval rank. Context IDs must be stable and match your relevance annotations; missing gold evidence should remain missing from contexts so a retrieval miss can be measured.

`relevant_ids: []` together with `slice: "unanswerable"` means the corpus is known not to contain an answer. It never means unknown/unannotated relevance. Unlabeled production log ingestion is outside this API's scope. One external batch contains 1–500 records with unique case IDs.

External dataset fingerprints cover project/corpus version and the question/reference/relevance labels, excluding predictions. This permits comparing different configurations on the same gold set. The service cannot independently prove that your underlying external corpus stayed unchanged: include the corpus version in `project` and use immutable context IDs. Otherwise a paired comparison could be misleading.

## Python adapter

Install this repository with `pip install -e PATH_TO_RAG_OBSERVATORY`, then:

```python
import json
from observatory.client import ObservatoryClient

client = ObservatoryClient()
benchmark = json.load(open("my-benchmark.json", encoding="utf-8"))

def predict(question):
    # Replace these calls with your application functions.
    documents = my_retriever.invoke(question)
    response = my_generator(question, documents)
    return {
        "response": response,
        "contexts": [
            {"id": doc.metadata["chunk_id"], "text": doc.page_content}
            for doc in documents
        ],
    }

job = client.evaluate_pipeline(
    project="my-project-corpus-v1",
    benchmark=benchmark,
    pipeline=predict,
    split="test",
    pipeline_config={"retriever": "my-retriever", "model": "my-model"},
    ragas=False,
    mlflow=False,
)
print(client.wait(job["id"], timeout=600))
```

`my_retriever`/`my_generator` are placeholders to adapt to your actual application. The example shape resembles a common document interface, but Observatory has no dependency on that framework. A LlamaIndex-style result can be converted using its own text and metadata fields; plain Python can return dictionaries directly.

Only the question is passed to the callback. The client adds gold answers and relevance labels after obtaining the prediction. Callback duration is reported as caller-side pipeline latency. Exceptions in your callback propagate to your application before submission; the SDK does not invent predictions for failed calls. To evaluate a saved failure cohort, prepare and review an explicit prediction dataset rather than silently dropping errors.

The existing-generation step executes where your application calls the SDK, not inside Observatory's background worker. The service background job handles the submitted evaluation. Large or expensive application generation should use your application's own execution controls.

## HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/session` | Local session token, parser/provider capabilities and limits |
| GET | `/api/datasets` | Registered immutable benchmark metadata |
| GET | `/api/datasets/{id}` | Full benchmark JSON |
| POST | `/api/datasets` | Register validated benchmark; returns content-based ID |
| POST | `/api/uploads` | Extract/chunk uploaded files; returns unlabeled benchmark |
| POST | `/api/experiments` | Queue a built-in RAG experiment on a registered benchmark |
| POST | `/api/evaluations` | Queue evaluation of existing predictions |
| GET | `/api/jobs` | Latest 30 public job statuses |
| GET | `/api/jobs/{id}` | Progress, terminal status and resulting run ID |
| GET | `/api/runs?limit=100&offset=0` | Paginated experiment reports |
| GET | `/api/traces?run={id}&limit=100&offset=0` | Paginated per-question records |
| GET | `/api/trace/{id}` | Full trace record |
| GET | `/api/compare?baseline={id}&candidate={id}` | Paired recall comparison |

Read-page `limit` is capped at 200. Invalid inputs receive 400/404, cross-origin or missing-session-token writes receive 403, and a full local queue receives 429. Submission returns 202 with a job ID. It does not wait for LLM/judge requests to complete.

For direct HTTP clients, first read `/api/session`, then include its `token` as `X-Observatory-Token` for POST requests with `Content-Type: application/json`. Do not include credentials in URLs. The Python client does this automatically. Tokens change when the server restarts; initialize a new client after restart.

Built-in experiment payload:

```json
{"dataset_id":"REGISTERED_ID","strategy":"bm25","top_k":3,"split":"dev","mode":"offline","ragas":false,"mlflow":false}
```

Upload payload:

```json
{"name":"My documents","chunk_words":300,"overlap_words":50,"files":[{"name":"policy.pdf","content_base64":"BASE64_FILE_BYTES"}]}
```

The raw JSON body has a 30 MiB limit, accounting for base64 expansion. The extracted input bytes have a separate 20 MiB batch limit. Uploaded files are parsed from bytes, not written to arbitrary server paths.

## Observability semantics

- `evaluate.external` is an Observatory evaluation trace root. It is not the original application's end-to-end trace.
- Nested evaluator spans show ID-based metrics, proxies and optional RAGAS calls.
- `source_latency_ms` is caller-reported; `latency_ms` measures local evaluation wall time.
- Source pipeline p95 is reported with the number of records that supplied latency. Missing source durations remain missing.
- External generation tokens and costs are not collected by this adapter. They remain unknown.
- To capture your original retrieval/generation/tool spans, instrument your application with its tracing framework independently.

## Credentials and data

For external RAGAS evaluation, configure the judge and embeddings in the Observatory server environment (Ollama with downloaded local models, or a hosted provider with credentials). The client does not send provider credentials. External RAGAS can evaluate saved answers from any generator provider; it only requires the configured evaluation provider to work.

The local session mechanism is cross-site request protection, not user authentication. Any trusted local process can access this loopback service. There are no remote deployments, tenant permissions or public CORS configuration here. Use only one server process per database.

Local reports preserve supplied data. RAGAS may transmit questions, references, answers and contexts to its configured judge/embedding providers. MLflow traces preserve selected evaluation data when enabled. Use approved documents/destinations and add comprehensive access control/redaction before processing sensitive workloads.
