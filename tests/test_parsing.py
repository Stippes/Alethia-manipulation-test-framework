import json
import base64
import os
import sys
from pathlib import Path
os.environ["DEBUG_MODE"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dashboard_app as da
from scripts import input_parser


def test_parse_uploaded_file_json():
    data = {"messages": [{"sender": "user", "timestamp": None, "text": "hello"}]}
    enc = base64.b64encode(json.dumps(data).encode()).decode()
    contents = f"data:application/json;base64,{enc}"
    result = da.parse_uploaded_file(contents, "sample.json")
    assert result["conversation_id"] == "sample"
    assert result["messages"][0]["text"] == "hello"


def test_parse_uploaded_file_txt():
    text = "[10:00:00] User: hi\n[10:00:01] Bot: hello"
    enc = base64.b64encode(text.encode()).decode()
    contents = f"data:text/plain;base64,{enc}"
    result = da.parse_uploaded_file(contents, "chat.txt")
    assert result["conversation_id"] == "chat"
    assert result["messages"][0]["sender"].lower() == "user"
    assert result["messages"][1]["text"] == "hello"


def test_parse_uploaded_file_txt_no_timestamp():
    text = "User: hi\nBot: hello"
    enc = base64.b64encode(text.encode()).decode()
    contents = f"data:text/plain;base64,{enc}"
    result = da.parse_uploaded_file(contents, "chat.txt")
    assert result["conversation_id"] == "chat"
    assert result["messages"][0]["sender"].lower() == "user"
    assert result["messages"][0]["timestamp"] is None
    assert result["messages"][1]["sender"].lower() == "bot"


def test_parse_uploaded_file_csv():
    csv_text = "sender,text\nuser,hi\nbot,hello"
    enc = base64.b64encode(csv_text.encode()).decode()
    contents = f"data:text/csv;base64,{enc}"
    result = da.parse_uploaded_file(contents, "log.csv")
    assert len(result["messages"]) == 2
    assert result["messages"][1]["text"] == "hello"


def test_parse_txt_chat_timestamp(tmp_path):
    path = tmp_path / "chat.txt"
    path.write_text("[10:00:00] User: hi")
    result = input_parser.parse_txt_chat(str(path))
    ts = result["messages"][0]["timestamp"]
    assert ts is not None and ts.endswith("10:00:00")


def test_parse_txt_chat_timestamp_no_dateutil(tmp_path, monkeypatch):
    path = tmp_path / "chat.txt"
    path.write_text("[10:00:00] User: hi")

    monkeypatch.setattr(input_parser, "_date_parser", None)
    result = input_parser.parse_txt_chat(str(path))
    ts = result["messages"][0]["timestamp"]
    assert ts is not None and ts.endswith("10:00:00")
