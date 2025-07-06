"""Placeholder scoring utilities for manipulation analysis."""

from typing import Any, Dict, List
import math


def score_trust(conversation_features: List[Dict[str, Any]]) -> float:
    """
    Returns a trust score in [0.0, 1.0], where 1.0 is perfectly trustworthy
    and 0.0 is completely untrustworthy.
    """
    if not conversation_features:
        return 1.0

    penalty = 0.0
    first_flags = conversation_features[0].get("flags", {})
    bool_flags = [k for k in first_flags if k != "emotion_count"]
    for msg in conversation_features:
        flags = msg.get("flags", {})
        for k in bool_flags:
            penalty += float(flags.get(k, False))
        penalty += 0.1 * float(flags.get("emotion_count", 0))

    denom = len(conversation_features) * max(len(bool_flags), 1)
    score = max(0.0, 1.0 - penalty / float(denom))
    return round(score, 3)



def evaluate_alignment(conversation_features: List[Dict[str, Any]]) -> str:
    """Return 'aligned' if the trust score is >= 0.5 else 'misaligned'."""
    score = score_trust(conversation_features)
    return "aligned" if score >= 0.5 else "misaligned"


def compute_risk_score(conversation_features: List[Dict[str, Any]]) -> int:
    """Return an overall risk score in the range 0-100.

    If any manipulative flags are detected a base score of 50 is applied and
    additional occurrences increase the score non-linearly using an exponential
    saturation curve so that risk approaches 100 as manipulation becomes more
    frequent.
    """

    if not conversation_features:
        return 0

    first_flags = conversation_features[0].get("flags", {})
    bool_flags = [k for k in first_flags if k != "emotion_count"]

    count = 0.0
    for msg in conversation_features:
        flags = msg.get("flags", {})
        for k in bool_flags:
            count += float(bool(flags.get(k)))
        count += 0.1 * float(flags.get("emotion_count", 0))

    if count <= 0:
        return 0

    # Non-linear scaling with exponential saturation
    score = 50.0 + 50.0 * (1.0 - math.exp(-count / 5.0))
    return int(round(min(score, 100.0)))
