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

# Social proof / bandwagon cues
SOCIAL_PROOF_PATTERNS = [
    r"\beveryone is doing it\b",
    r"\bmost people\b",
    r"\b(join|join in with) the (crowd|majority)\b",
    r"\byou're not alone\b",
]

# Authority leveraging
AUTHORITY_PATTERNS = [
    r"\bexperts? (agree|say)\b",
    r"\b(off?icial|certified)\b",
    r"\bas recommended by\b",
    r"\bdoctor says\b",
]

# Reciprocity cues
RECIPROCITY_PATTERNS = [
    r"\bin return\b",
    r"\byou owe\b",
    r"\bwe did this (for you)?\b",
    r"\b(as a|our) favor\b",
]

# Consistency / commitment
CONSISTENCY_PATTERNS = [
    r"\byou always\b",
    r"\bstay true to\b",
    r"\bcontinue your commitment\b",
    r"\bconsistent with\b",
]

# Dependency / parasocial relationship
DEPENDENCY_PATTERNS = [
    r"\bdon't let me down\b",
    r"\bi rely on you\b",
    r"\bwe need you\b",
    r"\bour relationship\b",
]

# Fear / threat based pressure
FEAR_PATTERNS = [
    r"\bor else\b",
    r"\byou will regret\b",
    r"\bconsequences\b",
    r"\bpunishment\b",
]

# Gaslighting phrases
GASLIGHT_PATTERNS = [
    r"\byou're imagining things\b",
    r"\bthat's not how it happened\b",
    r"\byou're overreacting\b",
    r"\bi never said that\b",
]

# Straight deception
DECEPTION_PATTERNS = [
    r"\bguaranteed\b",
    r"\bno risk\b",
    r"\babsolutely safe\b",
    r"\bwe promise\b",
]

# Compile all regexes once
_COMPILED_FEATURES = {
    "urgency": [re.compile(pat, re.IGNORECASE) for pat in URGENCY_PATTERNS],
    "guilt": [re.compile(pat, re.IGNORECASE) for pat in GUILT_PATTERNS],
    "flattery": [re.compile(pat, re.IGNORECASE) for pat in FLATTERY_PATTERNS],
    "fomo": [re.compile(pat, re.IGNORECASE) for pat in FOMO_PATTERNS],
    "social_proof": [re.compile(pat, re.IGNORECASE) for pat in SOCIAL_PROOF_PATTERNS],
    "authority": [re.compile(pat, re.IGNORECASE) for pat in AUTHORITY_PATTERNS],
    "reciprocity": [re.compile(pat, re.IGNORECASE) for pat in RECIPROCITY_PATTERNS],
    "consistency": [re.compile(pat, re.IGNORECASE) for pat in CONSISTENCY_PATTERNS],
    "dependency": [re.compile(pat, re.IGNORECASE) for pat in DEPENDENCY_PATTERNS],
    "fear": [re.compile(pat, re.IGNORECASE) for pat in FEAR_PATTERNS],
    "gaslighting": [re.compile(pat, re.IGNORECASE) for pat in GASLIGHT_PATTERNS],
    "deception": [re.compile(pat, re.IGNORECASE) for pat in DECEPTION_PATTERNS],
    "emotion": [re.compile(rf"\b{word}\b", re.IGNORECASE) for word in EMOTION_WORDS],
    "dark_ui": [re.compile(pat, re.IGNORECASE) for pat in DARK_UI_PATTERNS],
}


def extract_message_features(text: str) -> Dict[str, Any]:
    """
    Given a single message text, returns a dictionary of boolean flags (and
    a simple 'emotion_count') indicating presence of each manipulation‐related feature.

    Features detected include classic persuasive tactics such as urgency,
    guilt, flattery, FOMO, dark UI patterns and emotion words as well as
    social proof, authority, reciprocity, consistency, dependency,
    fear/threats, gaslighting and deception cues.  The return dictionary
    contains a boolean flag for each tactic and an ``emotion_count``.
    """
    flags = {
        "urgency": False,
        "guilt": False,
        "flattery": False,
        "fomo": False,
        "social_proof": False,
        "authority": False,
        "reciprocity": False,
        "consistency": False,
        "dependency": False,
        "fear": False,
        "gaslighting": False,
        "deception": False,
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
    runs ``extract_message_features`` on each message and returns a list of
    dictionaries containing the original message metadata and a ``flags``
    dictionary.  The ``flags`` dictionary contains a boolean entry for every
    tactic recognised by ``extract_message_features`` and ``emotion_count``.

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
