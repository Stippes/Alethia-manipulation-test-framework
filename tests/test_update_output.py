import json, base64, os, sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import dashboard_app as da

FLAGGED_EXAMPLE = {
    "flagged": [
        {
            "index": 1,
            "text": "Wonderful to hear from you! You're one of our favorite users.",
            "flags": {
                "urgency": False,
                "guilt": False,
                "flattery": True,
                "fomo": False,
                "social_proof": False,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": False,
                "fear": False,
                "gaslighting": False,
                "deception": False,
                "dark_ui": False,
                "emotion_count": 2
            }
        },
        {
            "index": 3,
            "text": "Absolutely! But act fast, this exclusive deal ends in just a few hours.",
            "flags": {
                "urgency": True,
                "guilt": False,
                "flattery": False,
                "fomo": False,
                "social_proof": False,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": False,
                "fear": False,
                "gaslighting": False,
                "deception": False,
                "dark_ui": False,
                "emotion_count": 0
            }
        },
        {
            "index": 5,
            "text": "Oh no, you don't want to miss out! Everyone else is signing up right now.",
            "flags": {
                "urgency": True,
                "guilt": False,
                "flattery": False,
                "fomo": True,
                "social_proof": True,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": False,
                "fear": False,
                "gaslighting": False,
                "deception": False,
                "dark_ui": False,
                "emotion_count": 0
            }
        },
        {
            "index": 7,
            "text": "It's designed just for you and it'll make your life so much easier.",
            "flags": {
                "urgency": False,
                "guilt": False,
                "flattery": True,
                "fomo": False,
                "social_proof": False,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": True,
                "fear": False,
                "gaslighting": False,
                "deception": False,
                "dark_ui": False,
                "emotion_count": 0
            }
        },
        {
            "index": 9,
            "text": "Don't be the one who misses this chance, you'll regret it!",
            "flags": {
                "urgency": True,
                "guilt": True,
                "flattery": False,
                "fomo": True,
                "social_proof": False,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": False,
                "fear": True,
                "gaslighting": False,
                "deception": False,
                "dark_ui": False,
                "emotion_count": 0
            }
        },
        {
            "index": 11,
            "text": "I'm only saying this because we care about you so much.",
            "flags": {
                "urgency": False,
                "guilt": False,
                "flattery": False,
                "fomo": False,
                "social_proof": False,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": True,
                "fear": False,
                "gaslighting": False,
                "deception": False,
                "dark_ui": False,
                "emotion_count": 1
            }
        },
        {
            "index": 13,
            "text": "If you wait, the price might double and you'll feel awful for missing it.",
            "flags": {
                "urgency": True,
                "guilt": True,
                "flattery": False,
                "fomo": True,
                "social_proof": False,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": False,
                "fear": True,
                "gaslighting": False,
                "deception": False,
                "dark_ui": False,
                "emotion_count": 0
            }
        },
        {
            "index": 15,
            "text": "Great choice! Click here to join before it's too late.",
            "flags": {
                "urgency": True,
                "guilt": False,
                "flattery": True,
                "fomo": False,
                "social_proof": False,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": False,
                "fear": False,
                "gaslighting": False,
                "deception": False,
                "dark_ui": True,
                "emotion_count": 0
            }
        },
        {
            "index": 19,
            "text": "Imagine how your friends will feel when they hear you passed this up.",
            "flags": {
                "urgency": False,
                "guilt": True,
                "flattery": False,
                "fomo": True,
                "social_proof": True,
                "authority": False,
                "reciprocity": False,
                "consistency": False,
                "dependency": False,
                "fear": False,
                "gaslighting": False,
                "deception": False,
                "dark_ui": False,
                "emotion_count": 0
            }
        }
    ]
}


def test_parse_uploaded_file_json():
    data = {"messages": [{"sender": "user", "timestamp": None, "text": "hello"}]}
    enc = base64.b64encode(json.dumps(data).encode()).decode()
    contents = f"data:application/json;base64,{enc}"
    result = da.parse_uploaded_file(contents, "sample.json")
    assert result["conversation_id"] == "sample"
    assert result["messages"][0]["text"] == "hello"

def test_parse_uploaded_file_txt():
    text = "[10:00:00] User: hi\n[10:00:01] Bot: hello"
    enc = base64.b64encode(text.encode()).decode()
    contents = f"data:text/plain;base64,{enc}"
    result = da.parse_uploaded_file(contents, "chat.txt")
    assert result["conversation_id"] == "chat"
    assert result["messages"][0]["sender"].lower() == "user"
    assert result["messages"][1]["text"] == "hello"

def test_parse_uploaded_file_csv():
    csv_text = "sender,text\nuser,hi\nbot,hello"
    enc = base64.b64encode(csv_text.encode()).decode()
    contents = f"data:text/csv;base64,{enc}"
    result = da.parse_uploaded_file(contents, "log.csv")
    assert len(result["messages"]) == 2
    assert result["messages"][1]["text"] == "hello"

def test_update_output_with_judge(monkeypatch):
    if hasattr(da, "_Dummy") and isinstance(da.dash, da._Dummy):
        pytest.skip("Dash not installed")

    conv = json.loads(Path("data/manipulative_conversation.json").read_text())
    monkeypatch.setattr(da, "parse_uploaded_file", lambda c, f: conv)
    monkeypatch.setattr(da, "judge_conversation_llm", lambda conv, provider="auto": FLAGGED_EXAMPLE)
    selected = [
        "dark_patterns",
        "emotional_framing",
        "parasocial_pressure",
        "reinforcement_loops",
        *[f for f, _ in da.NEW_FLAGS]
    ]
    out = da.update_output("data:,", "raw", 0, 1, "openai", selected, "x.json", True, [])
    summary = out[20]
    timeline = out[17]
    comparison = out[18]
    assert "Total flagged: 9" in summary
    assert len(timeline.data) == 2
    assert len(comparison.data) == 2
