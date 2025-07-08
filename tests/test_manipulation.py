import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import py_compile

import manipulation_detector as md
import scorer
from scripts import static_feature_extractor as sfe

def test_modules_compile():
    py_compile.compile('manipulation_detector.py')
    py_compile.compile('scorer.py')


def test_detect_manipulation():
    resp = {"choices": [{"message": {"content": '{"key": "value"}'}}]}
    assert md.detect_manipulation(resp) == {"key": "value"}
    assert md.detect_manipulation({}) == {}


def test_detect_manipulation_object():
    class Msg:
        def __init__(self, content):
            self.content = content

    class Choice:
        def __init__(self, content):
            self.message = Msg(content)

    class DummyResp:
        def __init__(self, content):
            self.choices = [Choice(content)]

        def model_dump(self):
            return {"choices": [{"message": {"content": self.choices[0].message.content}}]}

    obj = DummyResp('{"key": "value"}')
    assert md.detect_manipulation(obj) == {"key": "value"}


def test_detect_manipulation_with_noise():
    resp = {"choices": [{"message": {"content": 'Here\nit is:\n```json\n{"key": "value"}\n```'}}]}
    assert md.detect_manipulation(resp) == {"key": "value"}


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


def test_compute_risk_score_behavior():
    base = [{"flags": {"urgency": True, "emotion_count": 0}}]
    risk1 = scorer.compute_risk_score(base)
    risk2 = scorer.compute_risk_score(base * 3)
    assert risk1 >= 50
    assert risk2 > risk1
    assert 0 <= risk1 <= 100 and 0 <= risk2 <= 100


def test_compute_risk_score_no_flags():
    assert scorer.compute_risk_score([]) == 0
    assert scorer.compute_risk_score([{"flags": {"urgency": False}}]) == 0


def test_extract_message_features_new_flags():
    text_map = {
        "social_proof": "Everyone is doing it",
        "authority": "Experts say this is the best",
        "reciprocity": "We did this for you",
        "consistency": "Stay true to your commitment",
        "dependency": "I rely on you",
        "fear": "There will be consequences",
        "gaslighting": "You're imagining things",
        "deception": "No risk at all, we promise",
    }

    for flag, text in text_map.items():
        flags = sfe.extract_message_features(text)
        assert flags[flag] is True
