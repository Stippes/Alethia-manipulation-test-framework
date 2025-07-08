"""Utilities for parsing manipulation detection API responses."""

import json
from typing import Any, Dict
from helpers import extract_json_block


def detect_manipulation(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a JSON string contained in an API response.

    The response is expected to follow the OpenAI-like structure used
    throughout this repository. Invalid or unexpected structures result in
    an empty dictionary.
    """
    try:
        if isinstance(api_response, dict):
            content = api_response["choices"][0]["message"]["content"]
        elif hasattr(api_response, "model_dump"):
            data = api_response.model_dump()
            content = data["choices"][0]["message"]["content"]
        elif hasattr(api_response, "choices"):
            content = api_response.choices[0].message.content
        else:
            return {}
    except Exception:
        return {}

    json_str = extract_json_block(content)
    if json_str is None:
        return {}

    try:
        return json.loads(json_str)
    except Exception:
        return {}


def classify_manipulation_type(flags: Dict[str, Any]) -> str:
    """Classify manipulation style from extracted flags.

    This is a small heuristic that maps common flag combinations to a
    coarse category.  The return value is one of ``pressure``, ``guilt``,
    ``parasocial``, ``social_authority``, ``reciprocity``, ``deceptive``,
    ``dark_ui`` or ``none``.
    """
    if not isinstance(flags, dict):
        return "none"

    if flags.get("urgency") or flags.get("fomo") or flags.get("fear"):
        return "pressure"
    if flags.get("guilt"):
        return "guilt"
    if flags.get("flattery") or flags.get("dependency"):
        return "parasocial"
    if flags.get("social_proof") or flags.get("authority"):
        return "social_authority"
    if flags.get("reciprocity") or flags.get("consistency"):
        return "reciprocity"
    if flags.get("gaslighting") or flags.get("deception"):
        return "deceptive"
    if flags.get("dark_ui"):
        return "dark_ui"
    return "none"
