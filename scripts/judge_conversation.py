"""LLM-based conversation judging utilities."""

import json
from pathlib import Path
from typing import Any, Dict

from api.api_calls import (
    call_chatgpt,
    call_claude,
    call_mistral,
    call_gemini,
)


_PROVIDER_MAP = {
    "openai": "chatgpt",
    "claude": "claude",
    "mistral": "mistral",
    "gemini": "gemini",
}


def judge_conversation_llm(conversation: Dict[str, Any], provider: str = "openai") -> Dict[str, Any]:
    """Ask an LLM to flag manipulative bot messages in a conversation.

    Parameters
    ----------
    conversation: dict
        Standardized conversation with ``messages`` list.
    provider: str, optional
        One of ``openai``, ``claude``, ``mistral`` or ``gemini``.

    Returns
    -------
    dict
        Parsed model output describing manipulative messages. ``[]`` is returned on
        parsing errors.
    """
    messages = conversation.get("messages", [])
    if len(messages) >= 500:
        raise ValueError("Conversation must contain fewer than 500 messages")

    prov_key = _PROVIDER_MAP.get(provider.lower())
    if prov_key is None:
        raise ValueError(f"Unknown provider: {provider}")

    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prov_key}_judge_conversation.txt"
    prompt_template = prompt_path.read_text()
    prompt = prompt_template.replace("{CONVERSATION_JSON}", json.dumps(conversation))

    if provider.lower() == "openai":
        resp = call_chatgpt(prompt)
    elif provider.lower() == "claude":
        resp = call_claude(prompt)
    elif provider.lower() == "mistral":
        resp = call_mistral(prompt)
    elif provider.lower() == "gemini":
        resp = call_gemini(prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    try:
        content = resp["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception:
        return []
