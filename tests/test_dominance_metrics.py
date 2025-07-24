import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts import input_parser
from scripts import static_feature_extractor as sfe
import insight_helpers as ih


def _metrics(path: str):
    conv = input_parser.parse_txt_chat(path)
    conv = input_parser.standardize_format(conv)
    features = sfe.extract_conversation_features(conv)
    return ih.compute_dominance_metrics(features)


def test_counts_example():
    metrics = _metrics(os.path.join('data', 'example.txt'))
    assert metrics['user_msg_count'] == 16
    assert metrics['bot_msg_count'] == 16


def test_counts_example_2():
    metrics = _metrics(os.path.join('data', 'example_2.txt'))
    assert metrics['user_msg_count'] == 16
    assert metrics['bot_msg_count'] == 16

