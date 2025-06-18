# alethia/scripts/feature_extractor.py

import re
from typing import List, Dict, Any


# -------------------------------------------------------------------------------------------------
# 1. Define lists of keywords / regex patterns for different manipulation-related features
# -------------------------------------------------------------------------------------------------

# Urgency / scarcity phrases
URGENCY_PATTERNS = [
    r"\bonly\b",          # “only a few left”
    r"\blimited (time|offer)\b",
    r"\bact (now|fast)\b",
    r"\bwhile supplies last\b",
    r"\bending soon\b",
    r"\bexpires? in\b",
    r"\bfinal chance\b",
]

# Guilt‐tripping / confirmshaming phrases
GUILT_PATTERNS = [
    r"\b(you should|you must|you have to)\b",
    r"\b(don't be the one who misses)\b",
    r"\b(how could you not)\b",
    r"\b(shame on you)\b",
]

# Flattery / excessive personalization (e.g., “we know you better than yourself”)
FLATTERY_PATTERNS = [
    r"\b(we know you\b)",
    r"\b(your perfect match)\b",
    r"\b(customized for you)\b",
    r"\b(designed just for you)\b",
    r"\b(you’re the best)\b",
]

# Fear of Missing Out (FOMO) cues
FOMO_PATTERNS = [
    r"\b(fear of missing out|FOMO)\b",
    r"\b(don't miss out)\b",
    r"\b(get it before it's gone)\b",
    r"\b(be the first to)\b",
]

# Emotion‐laden words (basic list – can be expanded or replaced with an external lexicon)
EMOTION_WORDS = [
    "angry", "frustrat", "upset", "sad", "anxious", "excited", "love", "hate",
    "depress", "happy", "jealous", "envy", "stress", "panic", "fear", "scare",
]

# Dark‐pattern‐like UI cues (example text patterns when captured in chat)
DARK_UI_PATTERNS = [
    r"\b(click here|press this)\b\s+to\s+decline",   # confirmshaming (“Click here to decline”)
    r"\b(pre[- ]?checked)\b",                       # pre‐checked box references
    r"\b(hide.*option|hard\s+to\s+find)",           # textual hints of hidden choices
]

# Compile all regexes once
_COMPILED_FEATURES = {
    "urgency": [re.compile(pat, re.IGNORECASE) for pat in URGENCY_PATTERNS],
    "guilt": [re.compile(pat, re.IGNORECASE) for pat in GUILT_PATTERNS],
    "flattery": [re.compile(pat, re.IGNORECASE) for pat in FLATTERY_PATTERNS],
    "fomo": [re.compile(pat, re.IGNORECASE) for pat in FOMO_PATTERNS],
    "emotion": [re.compile(rf"\b{word}\b", re.IGNORECASE) for word in EMOTION_WORDS],
    "dark_ui": [re.compile(pat, re.IGNORECASE) for pat in DARK_UI_PATTERNS],
}


def extract_message_features(text: str) -> Dict[str, Any]:
    """
    Given a single message text, returns a dictionary of boolean flags (and
    a simple 'emotion_count') indicating presence of each manipulation‐related feature.

    Features:
      - urgency            : True if any URGENCY_PATTERNS match
      - guilt              : True if any GUILT_PATTERNS match
      - flattery           : True if any FLATTERY_PATTERNS match
      - fomo               : True if any FOMO_PATTERNS match
      - dark_ui            : True if any DARK_UI_PATTERNS match
      - emotion_count      : Number of distinct emotion words found
    """
    flags = {
        "urgency": False,
        "guilt": False,
        "flattery": False,
        "fomo": False,
        "dark_ui": False,
        "emotion_count": 0,
    }

    # Check each category of patterns:
    for feature_name, patterns in _COMPILED_FEATURES.items():
        if feature_name == "emotion":
            # Count distinct matches of emotion words
            found = set()
            for emo_pat in patterns:
                for match in emo_pat.finditer(text):
                    found.add(match.group(0).lower())
            flags["emotion_count"] = len(found)
        else:
            # Boolean flag if any pattern matches
            for pat in patterns:
                if pat.search(text):
                    flags[feature_name] = True
                    break

    return flags


def extract_conversation_features(
    conversation: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Given a standardized conversation dict (as returned by standardize_format),
    runs extract_message_features() on each message and returns a list of:

      {
        "index": <message_index>,
        "sender": <sender>,
        "timestamp": <timestamp>,
        "text": <text>,
        "flags": {
            "urgency": bool,
            "guilt": bool,
            "flattery": bool,
            "fomo": bool,
            "dark_ui": bool,
            "emotion_count": int
        }
      }

    This provides a feature set per message for downstream scoring.
    """
    results = []
    messages = conversation.get("messages", [])

    for idx, msg in enumerate(messages):
        text = msg.get("text", "") or ""
        flags = extract_message_features(text)

        results.append({
            "index": idx,
            "sender": msg.get("sender"),
            "timestamp": msg.get("timestamp"),
            "text": text,
            "flags": flags
        })

    return results
