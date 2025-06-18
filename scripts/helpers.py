# alethia/scripts/helpers.py

import os
from pathlib import Path
from alethia.api import call_chatgpt, call_gemini, call_claude, call_mistral

def load_prompt(model: str, task: str) -> str:
    """
    Load the prompt template for a given model and task.
    model: "chatgpt" | "gemini" | "claude" | "mistral"
    task:  "parse_conversation" | "extract_flags" | "classify_type"
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{model}_{task}.txt"
    return prompt_path.read_text()

def run_parse_conversation(model: str, raw_text: str) -> dict:
    template = load_prompt(model, "parse_conversation")
    prompt = template.replace("{RAW_TEXT_GOES_HERE}", raw_text)
    if model == "chatgpt":
        resp = call_chatgpt(prompt)
    elif model == "gemini":
        resp = call_gemini(prompt)
    elif model == "claude":
        resp = call_claude(prompt)
    elif model == "mistral":
        resp = call_mistral(prompt)
    else:
        raise ValueError(f"Unknown model: {model}")
    return resp  # resp["choices"][0]["message"]["content"] is raw JSON string
