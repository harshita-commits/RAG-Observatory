"""Persisted local span trees with optional connected MLflow spans."""
import re
import time
import uuid
from contextlib import contextmanager


def redact(value):
    if isinstance(value, str):
        return re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", value)
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if any(s in k.lower() for s in ("password", "secret", "api_key", "authorization")) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class Trace:
    def __init__(self, mlflow=None):
        self.id = uuid.uuid4().hex
        self.origin = time.perf_counter()
        self.spans = []
        self.stack = []
        self.mlflow = mlflow

    @contextmanager
    def span(self, name, kind="CHAIN", inputs=None):
        item = {"id": uuid.uuid4().hex[:12], "parent_id": self.stack[-1] if self.stack else None,
                "name": name, "kind": kind, "start_ms": (time.perf_counter()-self.origin)*1000,
                "inputs": redact(inputs or {}), "outputs": {}, "status": "OK"}
        self.spans.append(item)
        self.stack.append(item["id"])
        started = time.perf_counter()
        try:
            if self.mlflow:
                with self.mlflow.start_span(name=name, span_type=kind) as remote:
                    remote.set_inputs(item["inputs"])
                    remote.set_attribute("observatory.trace_id", self.id)
                    yield item
                    remote.set_outputs(redact(item["outputs"]))
            else:
                yield item
        except Exception as exc:
            item["status"] = "ERROR"
            # Never export exception text: provider errors may contain request data.
            item["error_type"] = type(exc).__name__
            raise
        finally:
            item["outputs"] = redact(item["outputs"])
            item["duration_ms"] = (time.perf_counter()-started)*1000
            self.stack.pop()
