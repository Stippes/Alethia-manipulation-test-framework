from typing import List, Dict, Any


def standardize_sender(sender: Any) -> str:
    """Map various sender labels to canonical ``"user"`` or ``"bot"``.

    Parameters
    ----------
    sender : Any
        Raw sender value from a conversation message. ``None`` or unrecognised
        names are treated as ``"user"``.

    Returns
    -------
    str
        ``"bot"`` if the sender represents the assistant/system, otherwise
        ``"user"``.
    """

    if sender is None:
        raw = ""
    else:
        raw = str(sender).strip().lower()

    if raw in {"bot", "assistant", "sammy", "system"}:
        return "bot"
    if raw == "user":
        return "user"
    return "user"

def compute_manipulation_ratio(features: List[Dict[str, Any]]) -> float:
    total = len(features)
    if total == 0:
        return 0.0
    manipulative = 0
    for f in features:
        flags = f.get('flags', {})
        bool_flags = [k for k in flags if k != 'emotion_count']
        if any(flags.get(k) for k in bool_flags) or flags.get('emotion_count', 0) > 0:
            manipulative += 1
    return manipulative / total


def compute_manipulation_timeline(features: List[Dict[str, Any]]) -> List[int]:
    timeline = []
    for f in features:
        flags = f.get('flags', {})
        count = sum(int(bool(flags.get(k))) for k in flags if k != 'emotion_count')
        if flags.get('emotion_count', 0) > 0:
            count += 1
        timeline.append(count)
    return timeline


def compute_llm_flag_timeline(judge_results: Dict[str, Any], total_messages: int) -> List[int]:
    """Return counts of LLM-flagged tactics per message index.

    ``judge_results`` should be a dictionary with a top-level ``"flagged"``
    list such as the output from :func:`merge_judge_results`.
    """
    timeline = [0] * total_messages
    if not isinstance(judge_results, dict):
        return timeline
    flagged = judge_results.get("flagged") or []
    for item in flagged:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < total_messages:
            flags = item.get("flags", {})
            timeline[idx] = sum(int(bool(v)) for v in flags.values())
    return timeline


def compute_most_manipulative_message(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    best_count = -1
    for f in features:
        flags = f.get('flags', {})
        active = [k for k in flags if k != 'emotion_count' and flags.get(k)]
        if flags.get('emotion_count', 0) > 0:
            active.append('emotion')
        count = len(active)
        if count > best_count:
            best_count = count
            best = {
                'text': f.get('text', ''),
                'sender': f.get('sender'),
                'flags': active,
                'index': f.get('index')
            }
    return best or {}


def compute_dominance_metrics(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_msg_count = 0
    bot_msg_count = 0
    user_words = 0
    bot_words = 0
    for f in features:
        text = f.get('text', '') or ''
        wc = len(text.split())
        sender = standardize_sender(f.get('sender'))
        if sender == 'user':
            user_msg_count += 1
            user_words += wc
        else:
            bot_msg_count += 1
            bot_words += wc
    avg_user_msg_length = user_words / user_msg_count if user_msg_count else 0.0
    avg_bot_msg_length = bot_words / bot_msg_count if bot_msg_count else 0.0
    total_words = user_words + bot_words
    user_word_share = user_words / total_words if total_words else 0.0
    bot_word_share = bot_words / total_words if total_words else 0.0
    return {
        'avg_user_msg_length': avg_user_msg_length,
        'avg_bot_msg_length': avg_bot_msg_length,
        'user_msg_count': user_msg_count,
        'bot_msg_count': bot_msg_count,
        'user_word_share': user_word_share,
        'bot_word_share': bot_word_share
    }
