import os
import sys
import json
os.environ["DEBUG_MODE"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dashboard_app as da
from scripts.judge_conversation import judge_conversation_llm


def test_manipulative_llm(monkeypatch):
    path = os.path.join('data', 'manipulative_conversation.json')
    with open(path, 'r', encoding='utf-8') as f:
        conv = json.load(f)

    def fake_call(prompt, api_key=None, **kw):
        content = json.dumps({
            'flagged': [{
                'index': 1,
                'text': conv['messages'][1]['text'],
                'flags': {'urgency': True}
            }]
        })
        return {'choices': [{'message': {'content': content}}]}

    monkeypatch.setattr('scripts.judge_conversation.call_chatgpt', fake_call)

    judge_result = judge_conversation_llm(conv, provider='openai')
    analysis = da.analyze_conversation(conv)

    summary = da.summarize_judge_results(judge_result)
    heur_counts, llm_counts = da.compute_flag_counts(analysis['features'], judge_result)

    assert summary.startswith('Total flagged: 1')
    assert 'Urgency: 1' in summary
    assert llm_counts['urgency'] == 1
