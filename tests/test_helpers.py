import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import helpers


def test_extract_json_block():
    text = "prefix ```json\n{\"a\": 1}\n``` suffix"
    assert helpers.extract_json_block(text) == '{"a": 1}'


def test_extract_json_block_multiple_jsons():
    text = 'first {"a": 1} second {"b": 2}'
    assert helpers.extract_json_block(text) == '{"a": 1}'
