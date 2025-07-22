import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import dashboard_app as da

conv = json.loads(Path("data/manipulative_conversation.json").read_text())
analysis = da.analyze_conversation(conv)

if hasattr(da, "_Dummy") and isinstance(da.dash, da._Dummy):
    pytest.skip("Dash not installed", allow_module_level=True)

def test_pattern_breakdown_basic():
    fig = da.build_pattern_breakdown_figure(analysis["summary"], list(analysis["summary"].keys()), "#fff", "black")
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [analysis["summary"][k] for k in analysis["summary"]]

def test_timeline_figure_basic():
    fig = da.build_timeline_figure(analysis["manipulation_timeline"], [0]*len(analysis["manipulation_timeline"]), "#fff", "black")
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == analysis["manipulation_timeline"]

    fig2 = da.build_timeline_figure(analysis["manipulation_timeline"], [1,0,0]+[0]*(len(analysis["manipulation_timeline"])-3), "#fff", "black")
    assert len(fig2.data) == 2


def test_flag_comparison_figure():
    merged = da.merge_judge_results({"flagged": []})
    fig = da.build_flag_comparison_figure(analysis["features"], merged, "#fff", "black")
    assert len(fig.data) == 2
