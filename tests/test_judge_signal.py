import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import json
import pytest

from scripts.judge_conversation import judgeSignal


def test_judge_signal_merges_multi_json(monkeypatch, caplog):
    conv = {"conversation_id": "sig", "messages": [{"sender": "bot", "text": "Act"}]}

    def fake_call(prompt, api_key=None, **kw):
        content = (
            '{"flagged": [{"index": 0, "text": "a", "flags": {"urgency": true}}]}'
            '{"flagged": [{"index": 1, "text": "b", "flags": {"flattery": true}}]}'
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr('scripts.judge_conversation.call_chatgpt', fake_call)

    with caplog.at_level(logging.DEBUG):
        result = judgeSignal(conv, 'openai')

    assert result == {
        "flagged": [
            {"index": 0, "text": "a", "flags": {"urgency": True}},
            {"index": 1, "text": "b", "flags": {"flattery": True}},
        ]
    }
    assert any('Raw content from openai' in rec.message for rec in caplog.records)
