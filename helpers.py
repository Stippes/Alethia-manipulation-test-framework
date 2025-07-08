"""Common helper utilities used across the framework."""

import json
import logging
from datetime import datetime
from typing import Any, Optional


def format_timestamp(ts: Optional[Any] = None) -> str:
    """Return an ISO formatted timestamp."""
    if ts is None:
        ts = datetime.utcnow()
    elif isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            return ""
    return ts.isoformat()


def validate_json(data: str) -> bool:
    """Return True if the input string is valid JSON."""
    try:
        json.loads(data)
        return True
    except Exception:
        return False


def safe_load_json(data: str) -> Optional[Any]:
    """Safely load JSON returning None on failure."""
    try:
        return json.loads(data)
    except Exception:
        return None


def log_error(message: str) -> None:
    """Log an error message using the logging module."""
    logging.error(message)


def extract_json_block(text: str) -> Optional[str]:
    """Return the first valid JSON object found in ``text``.

    The function strips common code fences (``` or ```json) and then
    attempts to locate a JSON object within the remaining string. If a
    valid JSON block is found, the JSON substring is returned. Otherwise
    ``None`` is returned.
    """

    if not isinstance(text, str):
        return None

    cleaned = text.strip()

    # Remove ``` fences which LLMs often include
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned.partition("\n")[2]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]

    # Try direct parse first
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    # Scan for the first valid JSON object in the text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch in "{[":
            try:
                obj, end = decoder.raw_decode(cleaned[i:])
                return cleaned[i : i + end]
            except Exception:
                continue
#     # Otherwise try to find the first {...} substring that parses
#     start = cleaned.find("{")
#     end = cleaned.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         candidate = cleaned[start : end + 1]
#         try:
#             json.loads(candidate)
#             return candidate
#         except Exception:
#             return None

    return None
