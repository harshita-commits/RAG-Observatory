"""Extract the output contract from the pinned RAGAS 0.2.15 prompt format."""
import json


def ragas_output_schema(prompt):
    marker = "following schema as specified in JSON Schema:"
    text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    if marker not in text:
        raise ValueError("Unsupported RAGAS prompt format; expected the pinned JSON schema contract")
    schema, _ = json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())
    if not isinstance(schema, dict) or schema.get("type") != "object" or "properties" not in schema:
        raise ValueError("RAGAS output contract must be an object JSON schema")
    return schema
