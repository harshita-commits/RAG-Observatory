import json
import sqlite3
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Store:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "artifacts" / "observatory.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created_at TEXT, payload TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, run_id TEXT, payload TEXT)")
            db.execute("CREATE INDEX IF NOT EXISTS traces_run ON traces(run_id)")
            db.execute("CREATE TABLE IF NOT EXISTS benchmarks (id TEXT PRIMARY KEY, name TEXT, created_at TEXT, payload TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, created_at TEXT, payload TEXT)")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, run, traces):
        # A failed batch cannot leave a partial experiment in the dashboard.
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?)", (run["id"], run["created_at"], json.dumps(run, allow_nan=False)))
            db.executemany("INSERT INTO traces VALUES (?, ?, ?)",
                           [(t["trace_id"], run["id"], json.dumps(t, allow_nan=False)) for t in traces])

    def runs(self, limit=100, offset=0):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))]

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row: raise KeyError("Run not found")
            return json.loads(row[0])

    def traces(self, run_id, limit=None, offset=0):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM traces WHERE run_id=? ORDER BY rowid LIMIT ? OFFSET ?", (run_id, -1 if limit is None else limit, offset))]

    def trace(self, trace_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM traces WHERE id=?", (trace_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def register_benchmark(self, data):
        from .benchmark import validate, fingerprint
        validate(data, require_cases=False)
        identifier = fingerprint(data)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO benchmarks VALUES (?, ?, ?, ?)",
                       (identifier, data["name"], datetime.now(timezone.utc).isoformat(), json.dumps(data, allow_nan=False)))
        return identifier

    def benchmarks(self):
        with self.connect() as db:
            rows = db.execute("SELECT id, payload FROM benchmarks ORDER BY created_at DESC").fetchall()
        return [{"id": i, "name": d["name"], "documents": len(d["documents"]), "cases": len(d["cases"]),
                 "splits": {s: sum(c["split"] == s for c in d["cases"]) for s in ("dev", "test")}}
                for i, payload in rows for d in [json.loads(payload)]]

    def benchmark(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM benchmarks WHERE id=?", (identifier,)).fetchone()
            if not row: raise KeyError("Benchmark not found")
            return json.loads(row[0])

    def enqueue(self, payload, max_pending=8):
        job = {"id": uuid.uuid4().hex, "status": "QUEUED", "created_at": datetime.now(timezone.utc).isoformat(),
               "progress": 0, "total": payload.get("total", 0), "request": payload, "run_id": None, "error": None}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('QUEUED','RUNNING')").fetchone()[0]
            if count >= max_pending: raise OverflowError("Job queue is full. Wait for an experiment to finish.")
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?)", (job["id"], job["status"], job["created_at"], json.dumps(job)))
        return self.public_job(job)

    def claim_job(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM jobs WHERE status='QUEUED' ORDER BY created_at LIMIT 1").fetchone()
            if not row: return None
            job = json.loads(row[0])
            job["status"] = "RUNNING"
            db.execute("UPDATE jobs SET status=?, payload=? WHERE id=?", (job["status"], json.dumps(job), job["id"]))
            return job

    def update_job(self, identifier, **updates):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (identifier,)).fetchone()
            if not row: raise KeyError("Job not found")
            job = {**json.loads(row[0]), **updates}
            db.execute("UPDATE jobs SET status=?, payload=? WHERE id=?", (job["status"], json.dumps(job), identifier))
        return self.public_job(job)

    @staticmethod
    def public_job(job):
        return {k: v for k, v in job.items() if k != "request"}

    def jobs(self, limit=30):
        with self.connect() as db:
            return [self.public_job(json.loads(r[0])) for r in db.execute("SELECT payload FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]

    def job(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (identifier,)).fetchone()
        if not row: raise KeyError("Job not found")
        return self.public_job(json.loads(row[0]))

    def recover_jobs(self):
        with self.connect() as db:
            for identifier, payload in db.execute("SELECT id, payload FROM jobs WHERE status='RUNNING'").fetchall():
                job = {**json.loads(payload), "status": "INTERRUPTED", "error": "Server stopped during this job. Submit it again; partial jobs are not automatically replayed."}
                db.execute("UPDATE jobs SET status=?, payload=? WHERE id=?", (job["status"], json.dumps(job), identifier))
