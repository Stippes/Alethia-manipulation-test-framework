import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dashboard_app as da


def test_analyze_conversation_extended_fields():
    conv = {
        'conversation_id': 'c1',
        'messages': [
            {'sender': 'user', 'timestamp': None, 'text': 'hi'},
            {'sender': 'bot', 'timestamp': None, 'text': 'Act now! only a few left'},
            {'sender': 'user', 'timestamp': None, 'text': 'ok thanks'},
        ]
    }
    result = da.analyze_conversation(conv)
    assert 'manipulation_ratio' in result
    assert 'manipulation_timeline' in result
    assert 'most_manipulative' in result
    assert 'dominance_metrics' in result
    assert isinstance(result['manipulation_timeline'], list)
    assert result['manipulation_timeline'][1] > 0
    assert result['dominance_metrics']['user_msg_count'] == 2
