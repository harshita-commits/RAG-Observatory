"""Portable, validated benchmarks and deterministic UTF-8 text ingestion."""
import hashlib
import json
import re
from pathlib import Path
from .dataset import corpus, cases
from .documents import FORMATS, chunks_from_files


def required_text(item, key):
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def validate(data, require_cases=True):
    if not isinstance(data, dict):
        raise ValueError("Benchmark must be a JSON object")
    required_text(data, "name")
    documents, questions = data.get("documents"), data.get("cases")
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a non-empty list")
    ids = set()
    for doc in documents:
        if not isinstance(doc, dict):
            raise ValueError("Each document must be an object")
        identifier = required_text(doc, "id")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier):
            raise ValueError("Document IDs may contain only letters, digits, underscores, dots and hyphens")
        if identifier in ids:
            raise ValueError(f"Duplicate document id: {identifier}")
        ids.add(identifier)
        required_text(doc, "text")
    if not isinstance(questions, list) or (require_cases and not questions):
        raise ValueError("cases must contain labeled questions; add them to the imported JSON before running")
    case_ids = set()
    for case in questions:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        identifier = required_text(case, "id")
        if identifier in case_ids:
            raise ValueError(f"Duplicate case id: {identifier}")
        case_ids.add(identifier)
        for key in ("question", "reference", "slice"):
            required_text(case, key)
        if case.get("split") not in ("dev", "test"):
            raise ValueError(f"Case {identifier}: split must be dev or test")
        relevant = case.get("relevant_ids")
        if not isinstance(relevant, list) or any(not isinstance(i, str) for i in relevant):
            raise ValueError(f"Case {identifier}: relevant_ids must be a list of document IDs; [] means unanswerable")
        if len(relevant) != len(set(relevant)) or not set(relevant) <= ids:
            raise ValueError(f"Case {identifier}: duplicate or unknown relevant document IDs")
        if (not relevant) != (case["slice"] == "unanswerable"):
            raise ValueError(f"Case {identifier}: use slice=unanswerable exactly when relevant_ids is empty")
    return data


def load(path=None):
    if path is None:
        return {"name": "Bundled SaaS benchmark", "documents": corpus(), "cases": cases("dev") + cases("test")}
    return validate(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def fingerprint(data):
    # Canonicalize row ordering, while retaining the content, labels and metadata.
    canonical = {**data, "documents": sorted(data["documents"], key=lambda d: d["id"]),
                 "cases": sorted(data["cases"], key=lambda c: c["id"])}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def import_documents(folder, output, chunk_words=300, overlap_words=50):
    """Create evidence chunks from supported formats, preserving source locations."""
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if not folder.is_dir():
        raise ValueError("documents folder does not exist")
    if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
        raise ValueError("Require chunk_words > overlap_words >= 0")
    if output.exists():
        raise ValueError("Output already exists; choose a new filename to preserve your labels")
    files = sorted((p for p in folder.rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(folder).as_posix())
    imported = []
    for file in files:
        if file.is_symlink() or not file.resolve().is_relative_to(folder):
            raise ValueError("Document files must remain inside the input folder")
        source = file.relative_to(folder).as_posix()
        if file.suffix.lower() not in FORMATS:
            raise ValueError(f"Unsupported file in import folder: {source}. Move it out or convert it first.")
        imported.append((source, file.read_bytes()))
    data = chunks_from_files(imported, folder.name, chunk_words, overlap_words)
    validate(data, require_cases=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a second import overwriting a labeled benchmark.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return data
