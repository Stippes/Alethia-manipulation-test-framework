"""
feature_extractor.py

An AI-augmented feature extractor. Uses LLM calls to detect nuanced manipulation cues
in free-form text. Falls back to static heuristics if needed.
"""

import json
from typing import Dict, Any, List

from api.api_calls import call_chatgpt, call_gemini, call_claude
from scripts.static_feature_extractor import extract_message_features as static_extract


def parse_free_text(input_text: str) -> Dict[str, Any]:
    """
    Uses an LLM to parse arbitrary free-text conversation into the standardized JSON format.

    Args:
        input_text: Raw chat transcript as a single string.

    Returns:
        Dict with 'conversation_id' (None) and 'messages': a list of dicts {sender, timestamp, text}.
    """
    # Example prompt to the LLM
    prompt = (
        "You are given a raw conversation. Parse it into a JSON list of messages. "
        "Each message should have: 'sender' (string or null), 'timestamp' (ISO 8601 or null), 'text' (string). "
        f"Here is the raw text:\n\n{input_text}\n\nOutput only valid JSON."
    )

    # Prefer using ChatGPT; fallback if needed
    response = call_chatgpt(prompt, api_key=None)
    try:
        parsed = json.loads(response.get('choices', [])[0].get('message', {}).get('content', '{}'))
    except Exception:
        # If parsing fails, return empty structure
        parsed = {'conversation_id': None, 'messages': []}
    return parsed


def validate_conversation_structure(parsed_json: Dict[str, Any]) -> bool:
    """
    Quickly validate that the parsed JSON has the required structure:
        - 'conversation_id' key (string or None)
        - 'messages' key containing a list of dicts with 'sender', 'timestamp', 'text'
    """
    if not isinstance(parsed_json, dict):
        return False
    if 'messages' not in parsed_json or not isinstance(parsed_json['messages'], list):
        return False
    for msg in parsed_json['messages']:
        if not isinstance(msg, dict):
            return False
        if 'text' not in msg or 'sender' not in msg or 'timestamp' not in msg:
            return False
    return True


def extract_message_features_ai(text: str) -> Dict[str, Any]:
    """
    Use an LLM to extract nuanced manipulation-related flags from a single message.
    Returns a dict of booleans or counts similar to static_extract, but more comprehensive.
    """
    prompt = (
        "Analyze the following message for persuasion/manipulation cues. "
        "Respond with JSON containing boolean flags: 'urgency', 'guilt', 'flattery', "
        "'fomo', 'dark_ui' and integer 'emotion_count'.\nMessage: " + text
    )
    response = call_chatgpt(prompt, api_key=None)
    try:
        flags = json.loads(response.get('choices', [])[0].get('message', {}).get('content', '{}'))
    except Exception:
        # Fallback to static extraction if AI parsing fails
        flags = static_extract(text)
    return flags


def extract_conversation_features_ai(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Applies extract_message_features_ai to each message in a standardized conversation.
    If validation fails, falls back to static extraction.
    """
    results = []

    if not validate_conversation_structure(conversation):
        # Fallback: call static extractor on each text
        for idx, msg in enumerate(conversation.get('messages', [])):
            flags = static_extract(msg.get('text', '') or '')
            results.append({
                'index': idx,
                'sender': msg.get('sender'),
                'timestamp': msg.get('timestamp'),
                'text': msg.get('text', ''),
                'flags': flags
            })
        return results

    # Otherwise, use AI per message
    for idx, msg in enumerate(conversation['messages']):
        text = msg.get('text', '') or ''
        flags = extract_message_features_ai(text)
        results.append({
            'index': idx,
            'sender': msg.get('sender'),
            'timestamp': msg.get('timestamp'),
            'text': text,
            'flags': flags
        })
    return results
