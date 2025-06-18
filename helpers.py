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
