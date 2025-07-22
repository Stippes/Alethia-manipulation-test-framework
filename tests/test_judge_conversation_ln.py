import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import json
import pytest

from scripts.judge_conversation import judgeConversationLN


def test_judge_conversation_ln_auto(monkeypatch, caplog):
    conv = {"conversation_id": "ln", "messages": [{"sender": "bot", "text": "Go"}]}

    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    for env in ["GEMINI_API_KEY", "CLAUDE_API_KEY", "MISTRAL_API_KEY"]:
        monkeypatch.delenv(env, raising=False)

    def fake_call(prompt, api_key=None, **kw):
        content = (
            '{"flagged": [{"index": 0, "text": "a", "flags": {"urgency": true}}]}'
            '{"flagged": [{"index": 1, "text": "b", "flags": {"guilt": true}}]}'
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr('scripts.judge_conversation.call_chatgpt', fake_call)

    with caplog.at_level(logging.DEBUG):
        result = judgeConversationLN(conv, provider="auto")

    assert result == {
        "openai": {
            "flagged": [
                {"index": 0, "text": "a", "flags": {"urgency": True}},
                {"index": 1, "text": "b", "flags": {"guilt": True}},
            ]
        }
    }
    assert any('Calling provider openai' in rec.message or 'Raw content from openai' in rec.message for rec in caplog.records)
