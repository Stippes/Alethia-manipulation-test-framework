import importlib, sys

import pytest

import api.chatgpt_api as cga


def test_call_chatgpt_converts_object(monkeypatch):
    class DummyResp:
        def __init__(self, content):
            self.choices = [type('C', (), {'message': type('M', (), {'content': content})()})()]

        def model_dump(self):
            return {"choices": [{"message": {"content": self.choices[0].message.content}}]}

    class DummyClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return DummyResp('{"ok": true}')

    class FakeOpenAI:
        __version__ = "1.2.0"
        OpenAI = DummyClient
        RateLimitError = Exception
        APIConnectionError = Exception
        OpenAIError = Exception

    monkeypatch.setitem(sys.modules, "openai", FakeOpenAI)
    importlib.reload(cga)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    result = cga.call_chatgpt("hi")
    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"] == '{"ok": true}'
