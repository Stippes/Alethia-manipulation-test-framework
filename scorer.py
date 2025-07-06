"""Placeholder scoring utilities for manipulation analysis."""

from typing import Any, Dict, List


def score_trust(conversation_features: List[Dict[str, Any]]) -> float:
    """Return a crude trust score from 0.0 (low) to 1.0 (high)."""
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
