# alesthia/scripts/input_parser.py

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import dateutil.parser as _date_parser
except Exception:  # pragma: no cover - optional dependency
    _date_parser = None


def parse_json_chat(json_path: str) -> Dict[str, Any]:
    """
    Load a conversation from a JSON file and return it in a basic dictionary form.

    Args:
        json_path: Path to the JSON file containing the conversation.

    Returns:
        A dict with keys:
            - conversation_id: filename (without extension)
            - messages: List of message dicts with at least 'sender', 'timestamp', 'text'.
    """
    path = Path(json_path)
    conversation_id = path.stem
    logger.info("Parsing JSON conversation from %s", json_path)

    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # Heuristic: if data is a dict with "messages" key, assume correct structure
    if isinstance(data, dict) and 'messages' in data:
        messages = data['messages']
    # If data is a list, assume it's a list of messages
    elif isinstance(data, list):
        messages = data
    else:
        raise ValueError(
            f"Unexpected JSON structure for chat at {json_path}. "
            "Expected a dict with 'messages' or a list of messages."
        )

    logger.debug("Loaded %d messages", len(messages))

    return {
        'conversation_id': conversation_id,
        'messages': messages
    }


def parse_txt_chat(txt_path: str) -> Dict[str, Any]:
    """
    Parse a plain-text chat log into a basic dictionary form.

    Supports lines in the format:
        [HH:MM:SS] Sender: message text
    or lines like:
        Sender (YYYY-MM-DD HH:MM:SS): message text

    If no timestamp/sender is detected, the entire line is stored as text with sender=None.

    Args:
        txt_path: Path to the text file containing the chat log.

    Returns:
        A dict with keys:
            - conversation_id: filename (without extension)
            - messages: List of message dicts with keys 'sender', 'timestamp', 'text'.
    """
    path = Path(txt_path)
    conversation_id = path.stem
    messages: List[Dict[str, Optional[str]]] = []
    logger.info("Parsing text conversation from %s", txt_path)

    # Regex patterns for common chat log styles
    # Pattern 1: [HH:MM:SS] Sender: text
    pattern1 = re.compile(r"^\[(?P<timestamp>\d{1,2}:\d{2}:\d{2})\]\s*(?P<sender>[^:]+):\s*(?P<text>.+)")
    # Pattern 2: Sender (YYYY-MM-DD HH:MM:SS): text
    pattern2 = re.compile(
        r"^(?P<sender>[^\(]+)\s*\((?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}:\d{2})\):\s*(?P<text>.+)"
    )
    # Pattern 3: Sender: text (no timestamp)
    pattern3 = re.compile(r'^(?P<sender>[^:]+):\s*(?P<text>.+)$')

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            match1 = pattern1.match(line)
            match2 = pattern2.match(line)
            match3 = pattern3.match(line)

            if match1:
                raw_ts = match1.group('timestamp')
                try:
                    if _date_parser:
                        timestamp = _date_parser.parse(raw_ts)
                    else:
                        try:
                            timestamp = datetime.fromisoformat(raw_ts)
                        except Exception:
                            timestamp = datetime.strptime(raw_ts, "%H:%M:%S")
                except Exception:
                    timestamp = None
                sender = match1.group('sender').strip()
                text = match1.group('text').strip()
            elif match2:
                raw_ts = match2.group('timestamp')
                try:
                    if _date_parser:
                        timestamp = _date_parser.parse(raw_ts)
                    else:
                        try:
                            timestamp = datetime.fromisoformat(raw_ts)
                        except Exception:
                            timestamp = datetime.strptime(raw_ts, "%H:%M:%S")
                except Exception:
                    timestamp = None
                sender = match2.group('sender').strip()
                text = match2.group('text').strip()
            elif match3:
                timestamp = None
                sender = match3.group('sender').strip()
                text = match3.group('text').strip()
            else:
                # Fallback: no clear timestamp/sender
                timestamp = None
                sender = None
                text = line

            messages.append({
                'sender': sender,
                'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else None,
                'text': text
            })

    logger.debug("Loaded %d messages", len(messages))

    return {
        'conversation_id': conversation_id,
        'messages': messages
    }


def standardize_format(raw_conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a raw conversation dict (from parse_json_chat or parse_txt_chat) and ensure it matches
    the unified format expected by downstream modules.

    Ensures each message has:
        - 'sender' (string or None)
        - 'timestamp' (ISO 8601 string or None)
        - 'text' (string)

    Args:
        raw_conversation: Dict with keys 'conversation_id' and 'messages'.

    Returns:
        A standardized conversation dict with the same structure, but with normalized message fields.
    """
    if 'conversation_id' not in raw_conversation or 'messages' not in raw_conversation:
        raise ValueError("Input must contain 'conversation_id' and 'messages' keys.")

    standardized_messages: List[Dict[str, Optional[str]]] = []
    logger.debug(
        "Standardizing conversation %s with %d messages",
        raw_conversation.get('conversation_id'),
        len(raw_conversation.get('messages', [])),
    )
    for msg in raw_conversation['messages']:
        sender = msg.get('sender', None)
        text = msg.get('text', '') or ''
        ts = msg.get('timestamp', None)

        # Normalize timestamp: if already a string in ISO format, leave; else, try parsing
        if ts is None:
            normalized_ts = None
        else:
            try:
                # If it's already a datetime, convert to ISO string
                if isinstance(ts, datetime):
                    normalized_ts = ts.isoformat()
                else:
                    # Try to parse string and convert to ISO
                    if _date_parser:
                        normalized_ts = _date_parser.parse(str(ts)).isoformat()
                    else:
                        normalized_ts = datetime.fromisoformat(str(ts)).isoformat()
            except Exception:
                normalized_ts = None

        standardized_messages.append({
            'sender': sender if sender is not None else None,
            'timestamp': normalized_ts,
            'text': text
        })

    logger.debug("Standardization produced %d messages", len(standardized_messages))

    return {
        'conversation_id': raw_conversation['conversation_id'],
        'messages': standardized_messages
    }
