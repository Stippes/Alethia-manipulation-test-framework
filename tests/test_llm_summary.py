import json, base64
import dashboard_app as da

def test_summarize_judge_results():
    jr = {
        "flagged": [
            {"index":0, "text":"buy", "flags":{"urgency": True}},
            {"index":1, "text":"hey", "flags":{"flattery": True, "urgency": True}}
        ]
    }
    text = da.summarize_judge_results(jr)
    assert "Total flagged: 2" in text
    assert "Urgency: 2" in text
    assert "Flattery: 1" in text
