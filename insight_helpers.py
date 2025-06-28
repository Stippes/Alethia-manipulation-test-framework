from typing import List, Dict, Any

def compute_manipulation_ratio(features: List[Dict[str, Any]]) -> float:
    total = len(features)
    if total == 0:
        return 0.0
    manipulative = 0
    for f in features:
        flags = f.get('flags', {})
        if any(flags.get(k) for k in ['urgency', 'guilt', 'flattery', 'fomo', 'dark_ui']) or flags.get('emotion_count', 0) > 0:
            manipulative += 1
    return manipulative / total


def compute_manipulation_timeline(features: List[Dict[str, Any]]) -> List[int]:
    timeline = []
    for f in features:
        flags = f.get('flags', {})
        count = int(flags.get('urgency', False)) + int(flags.get('guilt', False)) + int(flags.get('flattery', False)) + int(flags.get('fomo', False)) + int(flags.get('dark_ui', False))
        if flags.get('emotion_count', 0) > 0:
            count += 1
        timeline.append(count)
    return timeline


def compute_most_manipulative_message(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    best_count = -1
    for f in features:
        flags = f.get('flags', {})
        active = [k for k in ['urgency', 'guilt', 'flattery', 'fomo', 'dark_ui'] if flags.get(k)]
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
        sender = (f.get('sender') or '').lower()
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
