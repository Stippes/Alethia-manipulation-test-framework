import os, sys
os.environ["DEBUG_MODE"] = "1"
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


def test_update_output_callback(monkeypatch):
    conv = {
        "conversation_id": "cb",
        "messages": [{"sender": "bot", "timestamp": None, "text": "Buy now!"}],
    }
    analysis = {
        "features": [
            {"index": 0, "sender": "bot", "timestamp": None, "text": "Buy now!", "flags": {}}
        ],
        "risk": 0,
        "summary": {
            "dark_patterns": 0,
            "emotional_framing": 0,
            "parasocial_pressure": 0,
            "reinforcement_loops": 0,
            "guilt": 0,
            "social_proof": 0,
            "authority": 0,
            "reciprocity": 0,
            "consistency": 0,
            "dependency": 0,
            "fear": 0,
            "gaslighting": 0,
            "deception": 0,
        },
        "manipulation_ratio": 0,
        "manipulation_timeline": [0],
        "most_manipulative": {},
        "dominance_metrics": {
            "avg_user_msg_length": 0.0,
            "avg_bot_msg_length": 2.0,
            "user_msg_count": 0,
            "bot_msg_count": 1,
            "user_word_share": 0.0,
            "bot_word_share": 1.0,
        },
    }
    fake_results = {
        "flagged": [
            {"index": 0, "text": "Buy now!", "flags": {"urgency": True}}
        ]
    }
    monkeypatch.setattr(da, "parse_uploaded_file", lambda c, f: conv)
    monkeypatch.setattr(da, "analyze_conversation", lambda c: analysis)
    monkeypatch.setattr(da, "judge_conversation_llm", lambda c, provider="auto": fake_results)

    outputs = da.update_output(
        "data:text/plain;base64,Zm9v",  # dummy content
        "plain",
        0,
        1,
        "openai",
        [],
        "conv.json",
        False,
        [],
        None,
    )
    assert outputs[25] == fake_results
    assert any("processed results" in entry for entry in outputs[24])

    table = outputs[20]
    header_labels = [cell.children for cell in table.children[0].children]
    assert header_labels == ["Index", "Text", "Flags"]
    first_row = [cell.children for cell in table.children[1].children]
    assert first_row == [0, "Buy now!", "Urgency"]
