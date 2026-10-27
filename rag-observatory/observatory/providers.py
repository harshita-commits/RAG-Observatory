"""Server-selected providers; Ollama keeps generation and evaluation local."""
import importlib.util
import json
import os
from urllib.request import Request, urlopen


def settings():
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider not in {"ollama", "openai"}:
        raise ValueError("LLM_PROVIDER must be ollama or openai")
    local = provider == "ollama"
    return {"provider": provider,
            "generator_model": os.environ.get("GENERATOR_MODEL", "llama3.2:3b" if local else "gpt-4o-mini"),
            "judge_model": os.environ.get("JUDGE_MODEL", "llama3.2:3b" if local else "gpt-4o-mini"),
            "embedding_model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text" if local else "text-embedding-3-small")}


def ollama_request(path, data=None, timeout=None):
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    request = Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout or float(os.environ.get("MODEL_TIMEOUT_SECONDS", "600"))) as response:
        return json.load(response)


def capabilities():
    config = settings()
    if config["provider"] == "openai":
        ready = bool(os.environ.get("OPENAI_API_KEY"))
        return {**config, "generation_ready": ready and importlib.util.find_spec("openai") is not None,
                "judge_ready": ready, "provider_status": "API key configured" if ready else "Set OPENAI_API_KEY"}
    try:
        models = {m["name"] for m in ollama_request("/api/tags", timeout=2).get("models", [])}
        def present(name):
            return name in models or (":" not in name and name + ":latest" in models)
        missing = sorted({config[key] for key in ("generator_model", "judge_model", "embedding_model") if not present(config[key])})
        return {**config, "generation_ready": present(config["generator_model"]),
                "judge_ready": present(config["judge_model"]) and present(config["embedding_model"]),
                "provider_status": "Pull missing models: " + ", ".join(missing) if missing else "All configured local models are ready",
                "available_models": sorted(models)}
    except (OSError, ValueError, KeyError):
        return {**config, "generation_ready": False, "judge_ready": False,
                "provider_status": "Ollama unavailable; start it and pull the configured models"}


def require_provider(generation=False, judge=False):
    status = capabilities()
    if (generation and not status["generation_ready"]) or (judge and not status["judge_ready"]):
        raise ValueError(status["provider_status"] + "; check generator, judge and embedding model configuration")
