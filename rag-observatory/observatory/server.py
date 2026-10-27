"""Loopback HTTP API and dashboard with a bounded asynchronous job worker."""
import base64
import binascii
import hmac
import importlib.util
import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from .benchmark import load, validate
from .documents import FORMATS, chunks_from_files
from .experiment import compare, drift_report
from .jobs import Jobs
from .providers import capabilities

STATIC = Path(__file__).parent / "static"
MAX_BODY = 30 * 1024 * 1024


def create_server(store, port=8000, start_worker=True):
    jobs = Jobs(store)
    token = secrets.token_urlsafe(32)
    store.register_benchmark(load())
    example = Path(__file__).resolve().parents[1]/"examples"/"benchmark.json"
    if example.exists(): store.register_benchmark(load(example))

    class Handler(BaseHTTPRequestHandler):
        def local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host", "") not in allowed: raise PermissionError("Invalid Host header; access the loopback address")
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed}: raise PermissionError("Cross-origin requests are disabled")
            if self.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}: raise PermissionError("Use the dashboard's own origin")

        def do_GET(self):
            try:
                self.local_request()
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                def number(key, default, maximum):
                    value = int(query.get(key, [default])[0])
                    if not 0 <= value <= maximum: raise ValueError(f"Invalid {key}")
                    return value
                if parsed.path == "/api/session":
                    self.respond({"token": token, "formats": sorted(FORMATS), "limits": {"file_mib": 10, "total_upload_mib": 20, "chunks": 10000, "cases_per_job": 500, "pending_jobs": 8, "workers": 1},
                                  "capabilities": {**capabilities(), "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
                                  **{module: importlib.util.find_spec(module) is not None for module in ("mlflow", "ragas", "openai", "pypdf", "docx", "pptx", "openpyxl")}}})
                elif parsed.path == "/api/runs": self.respond(store.runs(number("limit", 100, 200), number("offset", 0, 10000000)))
                elif parsed.path == "/api/traces": self.respond(store.traces(query.get("run", [""])[0], number("limit", 100, 200), number("offset", 0, 10000000)))
                elif parsed.path.startswith("/api/trace/"):
                    value = store.trace(parsed.path.rsplit("/", 1)[-1])
                    self.respond(value or {"error": "Trace not found"}, 200 if value else 404)
                elif parsed.path == "/api/datasets": self.respond(store.benchmarks())
                elif parsed.path.startswith("/api/datasets/"): self.respond(store.benchmark(parsed.path.rsplit("/", 1)[-1]))
                elif parsed.path == "/api/jobs": self.respond(store.jobs())
                elif parsed.path.startswith("/api/jobs/"): self.respond(store.job(parsed.path.rsplit("/", 1)[-1]))
                elif parsed.path == "/api/compare": self.respond(compare(store, query["baseline"][0], query["candidate"][0]))
                elif parsed.path == "/api/drift": self.respond(drift_report())
                elif parsed.path in ("/", "/app.js", "/style.css"):
                    name = "index.html" if parsed.path == "/" else parsed.path[1:]
                    mime = {"html": "text/html", "js": "text/javascript", "css": "text/css"}[name.rsplit(".", 1)[1]]
                    data = (STATIC/name).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime + "; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
                    self.end_headers()
                    self.wfile.write(data)
                else: self.respond({"error": "Not found"}, 404)
            except PermissionError as exc: self.respond({"error": str(exc)}, 403)
            except KeyError as exc: self.respond({"error": str(exc)}, 404)
            except (ValueError, TypeError) as exc: self.respond({"error": str(exc)}, 400)

        def do_POST(self):
            try:
                self.local_request()
                if not hmac.compare_digest(self.headers.get("X-Observatory-Token", ""), token):
                    raise PermissionError("Missing local session token; refresh the dashboard or initialize the SDK")
                if self.headers.get_content_type() != "application/json": raise ValueError("Use Content-Type: application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY: raise OverflowError("Request body must be between 1 byte and 30 MiB")
                def invalid_constant(value): raise ValueError(f"Non-finite JSON number: {value}")
                data = json.loads(self.rfile.read(length), parse_constant=invalid_constant)
                if not isinstance(data, dict): raise ValueError("JSON body must be an object")
                route = urlparse(self.path).path
                if route == "/api/datasets":
                    validate(data, require_cases=False)
                    if len(data["documents"]) > 10000 or len(data["cases"]) > 1000: raise ValueError("Local dataset limits: 10,000 chunks and 1,000 labeled cases")
                    self.respond({"id": store.register_benchmark(data)}, 201)
                elif route == "/api/uploads":
                    files = data.get("files")
                    if not isinstance(files, list) or not 1 <= len(files) <= 100: raise ValueError("Upload 1–100 files")
                    decoded = []
                    for file in files:
                        if not isinstance(file, dict) or not isinstance(file.get("name"), str): raise ValueError("Each file requires a name")
                        decoded.append((file["name"], base64.b64decode(file.get("content_base64", ""), validate=True)))
                    name = data.get("name", "Imported documents")
                    if not isinstance(name, str) or not name.strip(): raise ValueError("Dataset name is required")
                    benchmark = chunks_from_files(decoded, name, data.get("chunk_words", 300), data.get("overlap_words", 50))
                    self.respond({"id": store.register_benchmark(benchmark), "benchmark": benchmark}, 201)
                elif route == "/api/experiments": self.respond(jobs.submit_experiment(data), 202)
                elif route == "/api/evaluations": self.respond(jobs.submit_external(data), 202)
                else: self.respond({"error": "Not found"}, 404)
            except PermissionError as exc: self.respond({"error": str(exc)}, 403)
            except KeyError as exc: self.respond({"error": str(exc)}, 404)
            except OverflowError as exc: self.respond({"error": str(exc)}, 429)
            except (ValueError, TypeError, binascii.Error) as exc: self.respond({"error": str(exc)}, 400)
            except Exception as exc: self.respond({"error": f"{type(exc).__name__}: file or request could not be processed"}, 400)

        def respond(self, data, status=200):
            payload = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((os.environ.get("OBSERVATORY_BIND_HOST", "127.0.0.1"), port), Handler)
    server.daemon_threads = True
    server.jobs = jobs
    if start_worker: jobs.start()
    return server


def serve(store, port=8000):
    server = create_server(store, port)
    print(f"RAG Observatory: http://127.0.0.1:{server.server_port}", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.jobs.close()
        server.server_close()
