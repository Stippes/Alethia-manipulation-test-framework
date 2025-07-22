import json, base64
import dashboard_app as da
go = da.go
from scripts.judge_utils import merge_judge_results

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


def test_summarize_judge_results_empty():
    assert da.summarize_judge_results({}) == "Total flagged: 0"
    assert da.summarize_judge_results(None) == "Total flagged: 0"


def test_judge_timeline_trace():
    features = [
        {"index": 0, "sender": "bot", "timestamp": None, "text": "buy", "flags": {"urgency": True}},
        {"index": 1, "sender": "user", "timestamp": None, "text": "ok", "flags": {}},
    ]
    timeline = da.compute_manipulation_timeline(features)
    judge = {"flagged": [{"index": 0, "text": "buy", "flags": {"urgency": True}}]}
    jtimeline = da.compute_llm_flag_timeline(judge, len(features))
    assert timeline == [1, 0]
    assert jtimeline == [1, 0]


def test_flag_comparison_figure():
    features = [
        {"index": 0, "sender": "bot", "timestamp": None, "text": "buy", "flags": {"urgency": True, "emotion_count": 1}},
        {"index": 1, "sender": "bot", "timestamp": None, "text": "please", "flags": {"guilt": True}},
    ]
    judge = {
        "flagged": [
            {"index": 0, "text": "buy", "flags": {"urgency": True}},
            {"index": 1, "text": "please", "flags": {"guilt": True}},
        ]
    }
    heur, llm = da.compute_flag_counts(features, judge)
    assert heur["urgency"] == 1 and heur["guilt"] == 1
    assert llm["urgency"] == 1 and llm["guilt"] == 1
    fig = da.build_flag_comparison_figure(features, judge, "#000", "white")
    assert fig is not None


def test_flag_helpers_with_merged_results():
    features = [
        {"index": 0, "sender": "bot", "timestamp": None, "text": "buy", "flags": {"urgency": True}},
        {"index": 1, "sender": "bot", "timestamp": None, "text": "hi", "flags": {}},
    ]
    raw_results = {
        "openai": {"flagged": [{"index": 0, "text": "buy", "flags": {"urgency": True}}]},
        "claude": {"flagged": [{"index": 1, "text": "hi", "flags": {"flattery": True}}]},
    }
    merged = merge_judge_results(raw_results)
    heur, llm = da.compute_flag_counts(features, merged)
    assert heur["urgency"] == 1
    assert llm["urgency"] == 1 and llm["flattery"] == 1
    timeline = da.compute_llm_flag_timeline(merged, len(features))
    assert timeline == [1, 1]
