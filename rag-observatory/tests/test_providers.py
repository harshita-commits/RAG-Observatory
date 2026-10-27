import unittest
from unittest.mock import patch
from observatory.providers import settings, capabilities, require_provider
from observatory.integrations import generate
from observatory.structured import ragas_output_schema


class LocalProviderTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_missing_embedding_disables_judge_but_not_generation(self):
        with patch("observatory.providers.ollama_request", return_value={"models": [{"name": "llama3.2:3b"}]}):
            status = capabilities()
            self.assertTrue(status["generation_ready"])
            self.assertFalse(status["judge_ready"])
            require_provider(generation=True)
            with self.assertRaises(ValueError): require_provider(judge=True)

    def test_no_key_needed_with_all_local_models(self):
        with patch("observatory.providers.ollama_request", return_value={"models": [
                {"name": "llama3.2:3b"}, {"name": "nomic-embed-text:latest"}]}):
            require_provider(generation=True, judge=True)
            self.assertEqual(settings()["embedding_model"], "nomic-embed-text")

    def test_native_token_usage_and_no_silent_cloud_fallback(self):
        response = {"message": {"content": "24 days [leave]"}, "prompt_eval_count": 42, "eval_count": 7}
        with patch("observatory.integrations.ollama_request", return_value=response) as request:
            answer, citations, usage = generate("Leave?", [{"id": "leave", "text": "24 days"}])
            self.assertEqual(citations, ["leave"])
            self.assertEqual(usage["input_tokens"], 42)
            self.assertEqual(usage["output_tokens"], 7)
            self.assertFalse(request.call_args.args[1]["stream"])
        with patch("observatory.integrations.ollama_request", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError): generate("Leave?", [])

    def test_unavailable_service_disables_models(self):
        with patch("observatory.providers.ollama_request", side_effect=OSError):
            self.assertFalse(capabilities()["generation_ready"])

    def test_unknown_provider_rejected(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "typo"}):
            with self.assertRaises(ValueError): settings()

    def test_judge_uses_output_schema_instead_of_input_json(self):
        schema = '{"type":"object","properties":{"statements":{"type":"array","items":{"type":"string"}}}}'
        prompt = 'Return following schema as specified in JSON Schema:\n' + schema + 'Do not use single quotes. Input: {"question":"q"}'
        self.assertEqual(ragas_output_schema(prompt)["properties"]["statements"]["type"], "array")
        with self.assertRaisesRegex(ValueError, "Unsupported RAGAS"):
            ragas_output_schema('Unrecognized format {"question":"q"}')
