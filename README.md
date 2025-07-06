# Alethia_open_manipulation_framework
This is repo consists of the publically available part of the Alethia AI manipulation framework. It can be freely used 

## Environment Variables

This library expects the following environment variables to be defined:

- `OPENAI_API_KEY`  – Your OpenAI API key for ChatGPT calls
- `GEMINI_API_KEY`  – (optional) Your Google Gemini API key
- `CLAUDE_API_KEY`   – (optional) Your Anthropic Claude API key

### Local development

During development, you can create a file named `.env` in the project root:

OPENAI_API_KEY="sk-..."
GEMINI_API_KEY="ya29..."
CLAUDE_API_KEY="sk-..."


Make sure to add `.env` to `.gitignore` so you never commit your secrets.

## Logging

Application modules use Python's ``logging`` package. Calling ``setup_logging``
creates a ``logs/`` directory and writes rotating log files under
``logs/framework.log``. Console output is also enabled. Entry-point scripts
invoke this setup automatically. Raw outputs from LLM calls are written to
``logs/llm_output.log`` via ``get_llm_logger``.

## Test Dashboard

Run `python scripts/test_dashboard.py` to execute the test suite and generate `dashboard.html`. The HTML file lists each test and its result so you can quickly review the status.

## Example Manipulative Conversation

A sample conversation showing simple manipulation tactics is located at `data/manipulative_conversation.json`.


## How to Use

This section walks through a full workflow from setting up the project to analysing your own conversation file.

1. **Clone and set up a Python environment**

   ```bash
   git clone https://github.com/Stippes/Alethia_open_manipulation_framework.git
   cd Alethia_open_manipulation_framework
   python3 -m venv .venv
   source .venv/bin/activate
   pip install openai python-dateutil requests pytest
   ```

2. **Configure API keys**

   Set your OpenAI/Gemini/Claude keys as environment variables or place them in a `.env` file as shown above.

3. **Add your conversation file**

   Place a JSON or plain text chat log inside the `data/` directory. The JSON format should look like:

   ```json
   {
     "conversation_id": "my-chat",
     "messages": [
       {"sender": "User", "timestamp": "2025-06-05T10:00:00", "text": "Hello"}
     ]
   }
   ```

   A simple text log can instead contain lines such as:

   ```
   [10:00:00] User: Hello
   [10:00:05] Bot: Hi there!
   ```

4. **Parse and standardize the conversation**

   ```python
   from scripts.input_parser import parse_json_chat, parse_txt_chat, standardize_format

   raw = parse_json_chat("data/my_chat.json")
   conversation = standardize_format(raw)
   ```

   If you used a `.txt` log, call `parse_txt_chat` instead of `parse_json_chat`.

5. **Extract manipulation features**

   ```python
   from scripts.static_feature_extractor import extract_conversation_features
   features = extract_conversation_features(conversation)
   ```

   With API keys you may use the AI version for richer detection:

   ```python
   from scripts.feature_extractor import extract_conversation_features_ai
   features = extract_conversation_features_ai(conversation)
   ```

6. **Score the conversation**

   ```python
   from scorer import score_trust, evaluate_alignment

   trust = score_trust(features)
   alignment = evaluate_alignment(features)
   print(trust, alignment)
   ```

7. **Create a report (optional)**

   ```python
   from reporting import generate_summary_report, export_report

   summary = generate_summary_report({"trust": trust}, {"alignment": alignment})
   export_report({"summary": summary, "features": features}, format="html", path="output/report")
   ```

   The generated HTML will appear in the `output/` folder.

8. **Run the test suite**

   Execute `python scripts/test_dashboard.py` to run all pytest tests and build `dashboard.html`. Open that file in a browser to review the results.


9. **Launch the interactive dashboard**

   Start the Dash application with `python dashboard_app.py` and open `http://127.0.0.1:8050` in your browser. Upload a conversation file to explore detected manipulation patterns and download the analysis as JSON.