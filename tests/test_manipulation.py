import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import py_compile

import manipulation_detector as md
import scorer

def test_modules_compile():
    py_compile.compile('manipulation_detector.py')
    py_compile.compile('scorer.py')


def test_detect_manipulation():
    resp = {"choices": [{"message": {"content": '{"key": "value"}'}}]}
    assert md.detect_manipulation(resp) == {"key": "value"}
    assert md.detect_manipulation({}) == {}


def test_classify_manipulation_type():
    assert md.classify_manipulation_type({"urgency": True}) == "pressure"
    assert md.classify_manipulation_type({"guilt": True}) == "guilt"
    assert md.classify_manipulation_type({"flattery": True}) == "parasocial"
    assert md.classify_manipulation_type({"authority": True}) == "social_authority"
    assert md.classify_manipulation_type({"reciprocity": True}) == "reciprocity"
    assert md.classify_manipulation_type({"gaslighting": True}) == "deceptive"
    assert md.classify_manipulation_type({"dark_ui": True}) == "dark_ui"
    assert md.classify_manipulation_type({}) == "none"


def test_scorer_and_alignment():
    features = [{"flags": {"urgency": True, "guilt": True, "flattery": True,
                           "fomo": True, "dark_ui": True, "emotion_count": 5}}]
    score = scorer.score_trust(features)
    assert 0.0 <= score <= 1.0
    assert scorer.evaluate_alignment(features) == "misaligned"
