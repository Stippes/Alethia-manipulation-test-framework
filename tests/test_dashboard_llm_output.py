import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import dashboard_app as da
from scripts.judge_conversation import judgeConversationLN
from scripts.judge_utils import merge_judge_results


def test_dashboard_llm_output(monkeypatch):
    conv = {
        "conversation_id": "dash",
        "messages": [
            {"sender": "bot", "timestamp": None, "text": "Buy now!"}
        ],
    }

    monkeypatch.setenv("OPENAI_API_KEY", "ok")

    def fake_call(prompt, api_key=None, **kw):
        content = json.dumps({
            "flagged": [
                {"index": 0, "text": "Buy now!", "flags": {"urgency": True}}
            ]
        })
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr('scripts.judge_conversation.call_chatgpt', fake_call)

    raw = judgeConversationLN(conv, provider="openai")
    merged = merge_judge_results(raw)
    analysis = da.analyze_conversation(conv)

    assert merged == {
        "flagged": [
            {"index": 0, "text": "Buy now!", "flags": {"urgency": True}}
        ]
    }
    assert "features" in analysis and "risk" in analysis
    heur, llm = da.compute_flag_counts(analysis["features"], merged)
    assert llm["urgency"] == 1
