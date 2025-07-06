import os
import sys
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
