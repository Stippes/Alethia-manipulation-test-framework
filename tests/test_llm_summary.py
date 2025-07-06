import json, base64
import plotly.graph_objs as go
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


def test_judge_timeline_trace():
    features = [
        {"index": 0, "sender": "bot", "timestamp": None, "text": "buy", "flags": {"urgency": True}},
        {"index": 1, "sender": "user", "timestamp": None, "text": "ok", "flags": {}},
    ]
    timeline = da.compute_manipulation_timeline(features)
    judge = {"flagged": [{"index": 0, "text": "buy", "flags": {"urgency": True}}]}
    jtimeline = da.compute_llm_flag_timeline(judge, len(features))
    fig = go.Figure([go.Scatter(y=timeline, name="Heuristic")])
    if any(jtimeline):
        fig.add_trace(go.Scatter(y=jtimeline, mode="markers", name="LLM Judge"))
    assert len(fig.data) == 2
