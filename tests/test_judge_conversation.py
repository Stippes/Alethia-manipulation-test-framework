import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from scripts.judge_conversation import judge_conversation_llm


def test_message_count_limit():
    conv = {"conversation_id": "x", "messages": [{}]*500}
    with pytest.raises(ValueError):
        judge_conversation_llm(conv)


def test_judge_conversation_parsing(monkeypatch):
    conv = {
        "conversation_id": "c1",
        "messages": [
            {"sender": "user", "timestamp": None, "text": "hi"},
            {"sender": "bot", "timestamp": None, "text": "buy now"},
        ],
    }

    def fake_call(prompt, api_key=None, **kw):
        json_resp = '{"flagged": [{"index": 1, "text": "buy now", "flags": {"urgency": true}}]}'
        return {"choices": [{"message": {"content": json_resp}}]}

    monkeypatch.setattr('scripts.judge_conversation.call_chatgpt', fake_call)
    result = judge_conversation_llm(conv, provider="openai")
    assert result == {"flagged": [{"index": 1, "text": "buy now", "flags": {"urgency": True}}]}


def test_judge_conversation_parse_fail(monkeypatch):
    conv = {"conversation_id": "c2", "messages": [{"sender": "bot", "timestamp": None, "text": "hello"}]}

    def fake_call(prompt, api_key=None, **kw):
        return {"choices": [{"message": {"content": 'not json'}}]}

    monkeypatch.setattr('scripts.judge_conversation.call_chatgpt', fake_call)
    assert judge_conversation_llm(conv, provider="openai") == []


def test_judge_conversation_llm_old_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class Error:
        class RateLimitError(Exception):
            pass

        class APIConnectionError(Exception):
            pass

        class OpenAIError(Exception):
            pass

    class ChatCompletion:
        @staticmethod
        def create(**kwargs):
            json_resp = '{"flagged": [{"index": 0, "text": "Act now!", "flags": {"urgency": true}}]}'
            return {"choices": [{"message": {"content": json_resp}}]}

    class FakeOpenAI:
        __version__ = "0.27.0"
        api_key = None

    FakeOpenAI.ChatCompletion = ChatCompletion
    FakeOpenAI.error = Error

    monkeypatch.setitem(sys.modules, "openai", FakeOpenAI)
    import importlib
    mod = importlib.reload(sys.modules["api.chatgpt_api"])
    monkeypatch.setattr("scripts.judge_conversation.call_chatgpt", mod.call_chatgpt)

    conv = {"conversation_id": "c3", "messages": [{"sender": "bot", "timestamp": None, "text": "Act now!"}]}
    result = judge_conversation_llm(conv, provider="openai")
    assert result["flagged"][0]["flags"]["urgency"]


def test_judge_conversation_llm_new_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    json_resp = '{"flagged": [{"index": 0, "text": "Act now!", "flags": {"urgency": true}}]}'
                    return {"choices": [{"message": {"content": json_resp}}]}

    class FakeOpenAI:
        __version__ = "1.2.0"
        OpenAI = FakeClient
        RateLimitError = type("RateLimitError", (Exception,), {})
        APIConnectionError = type("APIConnectionError", (Exception,), {})
        OpenAIError = type("OpenAIError", (Exception,), {})

    monkeypatch.setitem(sys.modules, "openai", FakeOpenAI)
    import importlib
    mod = importlib.reload(sys.modules["api.chatgpt_api"])
    monkeypatch.setattr("scripts.judge_conversation.call_chatgpt", mod.call_chatgpt)

    conv = {"conversation_id": "c4", "messages": [{"sender": "bot", "timestamp": None, "text": "Act now!"}]}
    result = judge_conversation_llm(conv, provider="openai")
    assert result["flagged"][0]["flags"]["urgency"]


def test_judge_conversation_auto_multi(monkeypatch):
    conv = {"conversation_id": "a", "messages": [{"sender": "bot", "timestamp": None, "text": "x"}]}

    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("CLAUDE_API_KEY", "c")
    monkeypatch.setenv("MISTRAL_API_KEY", "m")

    def fake(provider):
        def _f(prompt, api_key=None, **kw):
            resp = {"flagged": []}
            if provider != "openai":
                resp = {"flagged": [{"index": 0, "text": provider, "flags": {"urgency": True}}]}
            return {"choices": [{"message": {"content": json.dumps(resp)}}]}
        return _f

    monkeypatch.setattr('scripts.judge_conversation.call_chatgpt', fake("openai"))
    monkeypatch.setattr('scripts.judge_conversation.call_gemini', fake("gemini"))
    monkeypatch.setattr('scripts.judge_conversation.call_claude', fake("claude"))
    monkeypatch.setattr('scripts.judge_conversation.call_mistral', fake("mistral"))

    result = judge_conversation_llm(conv, provider="auto")
    assert set(result.keys()) == {"openai", "gemini", "claude", "mistral"}
    assert result["claude"]["flagged"][0]["flags"]["urgency"]


def test_judge_conversation_auto_partial(monkeypatch):
    conv = {"conversation_id": "b", "messages": [{"sender": "bot", "timestamp": None, "text": "y"}]}

    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    def fake_openai(prompt, api_key=None, **kw):
        return {"choices": [{"message": {"content": json.dumps({"flagged": []})}}]}

    def fail(*a, **k):
        raise AssertionError("should not be called")

    monkeypatch.setattr('scripts.judge_conversation.call_chatgpt', fake_openai)
    monkeypatch.setattr('scripts.judge_conversation.call_gemini', fail)
    monkeypatch.setattr('scripts.judge_conversation.call_claude', fail)
    monkeypatch.setattr('scripts.judge_conversation.call_mistral', fail)

    result = judge_conversation_llm(conv, provider="auto")
    assert list(result.keys()) == ["openai"]
