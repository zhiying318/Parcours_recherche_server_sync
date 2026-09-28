import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from spatial_internalization.api_qwen import OpenAICompatibleQwenGenerator


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.completions = _FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


class APIQwenTests(unittest.TestCase):
    def _generator(self, client):
        previous = os.environ.get("TEST_XI_API_KEY")
        os.environ["TEST_XI_API_KEY"] = "  'Bearer secret-value'  "
        self.addCleanup(self._restore_key, previous)
        return OpenAICompatibleQwenGenerator(
            api_key_env="TEST_XI_API_KEY",
            base_url="https://example.test/v1/chat/completions",
            model="Qwen3.5-397B-A17B",
            request_retries=0,
            client=client,
        )

    @staticmethod
    def _restore_key(previous):
        if previous is None:
            os.environ.pop("TEST_XI_API_KEY", None)
        else:
            os.environ["TEST_XI_API_KEY"] = previous

    def test_generate_uses_sdk_chat_shape_and_final_content_only(self):
        response = SimpleNamespace(
            model="Qwen3.5-397B-A17B",
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content='{"ok":true}',
                    reasoning_content="must not enter target",
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4, total_tokens=16),
        )
        client = _FakeClient(response)
        generator = self._generator(client)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.png"
            image.write_bytes(b"not-a-real-png-for-request-shape-test")
            text, metadata = generator.generate(
                image,
                "Return JSON only.",
                enable_thinking=False,
                max_new_tokens=32,
            )
        self.assertEqual(text, '{"ok":true}')
        self.assertEqual(metadata["input_tokens"], 12)
        self.assertEqual(metadata["output_tokens"], 4)
        self.assertEqual(client.completions.payload["model"], "Qwen3.5-397B-A17B")
        self.assertEqual(generator.base_url, "https://example.test/v1")
        content = client.completions.payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[1], {"type": "text", "text": "/no_think\nReturn JSON only."})
        self.assertEqual(client.completions.payload["max_completion_tokens"], 32)

    def test_text_only_smoke_request_uses_same_sdk_path(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
            usage=None,
        )
        client = _FakeClient(response)
        generator = self._generator(client)
        generator.generate(None, "Return JSON only.", enable_thinking=True, max_new_tokens=8)
        self.assertEqual(client.completions.payload["messages"][0]["content"], [
            {"type": "text", "text": "/think\nReturn JSON only."},
        ])


if __name__ == "__main__":
    unittest.main()
