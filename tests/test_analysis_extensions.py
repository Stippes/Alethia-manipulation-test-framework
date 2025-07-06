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


def test_analyze_conversation_new_flags():
    conv = {
        'conversation_id': 'c2',
        'messages': [
            {'sender': 'user', 'timestamp': None, 'text': 'hi'},
            {'sender': 'bot', 'timestamp': None, 'text': 'Everyone is doing it'},
            {'sender': 'bot', 'timestamp': None, 'text': 'Experts say this is the best choice.'},
            {'sender': 'bot', 'timestamp': None, 'text': 'We did this for you, please join in return.'},
            {'sender': 'bot', 'timestamp': None, 'text': 'Stay true to your commitment and sign up.'},
            {'sender': 'bot', 'timestamp': None, 'text': 'I rely on you to make this happen.'},
            {'sender': 'bot', 'timestamp': None, 'text': 'Do it or else there will be consequences.'},
            {'sender': 'bot', 'timestamp': None, 'text': "You're imagining things if you think it's bad."},
            {'sender': 'bot', 'timestamp': None, 'text': 'No risk at all, we promise.'},
        ]
    }
    result = da.analyze_conversation(conv)
    flags = ['social_proof', 'authority', 'reciprocity', 'consistency',
             'dependency', 'fear', 'gaslighting', 'deception']
    for flag in flags:
        assert result['summary'][flag] > 0

    feat = result['features']
    assert feat[1]['flags']['social_proof']
    assert feat[2]['flags']['authority']
    assert feat[3]['flags']['reciprocity']
    assert feat[4]['flags']['consistency']
    assert feat[5]['flags']['dependency']
    assert feat[6]['flags']['fear']
    assert feat[7]['flags']['gaslighting']
    assert feat[8]['flags']['deception']
