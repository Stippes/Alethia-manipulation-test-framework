"""LLM-based conversation judging utilities."""

import json
from pathlib import Path
from typing import Any, Dict
import os
import logging
from logging_utils import setup_logging, get_llm_logger
from helpers import extract_json_block


from api.api_calls import (
    call_chatgpt,
    call_claude,
    call_mistral,
    call_gemini,
)

setup_logging()
logger = logging.getLogger(__name__)

llm_logger = get_llm_logger()



_PROVIDER_MAP = {
    "openai": "chatgpt",
    "claude": "claude",
    "mistral": "mistral",
    "gemini": "gemini",
}

_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "claude": "CLAUDE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _judge_single(conversation: Dict[str, Any], provider: str) -> Dict[str, Any]:
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

    logger.info("Judging conversation with %s", provider)

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
        if isinstance(resp, dict):
            content = resp["choices"][0]["message"]["content"]
        elif hasattr(resp, "model_dump"):
            data = resp.model_dump()
            content = data["choices"][0]["message"]["content"]
        elif hasattr(resp, "choices"):
            content = resp.choices[0].message.content
        else:
            raise TypeError("Unsupported response type")

        logger.debug("Received response from %s", provider)
        llm_logger.info("%s: %s", provider, content)

        json_str = extract_json_block(content)
        if json_str is None:
            raise ValueError("no JSON found")
        return json.loads(json_str)
    except Exception as exc:
        logger.warning("Failed to parse response from %s: %s", provider, exc)
        return []


def judge_conversation_llm(conversation: Dict[str, Any], provider: str = "auto") -> Dict[str, Any]:
    """Flag manipulative bot messages using one or more LLM providers.

    When ``provider`` is ``"auto"``, providers will be tried sequentially in the
    order OpenAI, Gemini, Claude and Mistral. Only providers for which an API key
    is available will be called. The return value will be a mapping from each
    successful provider name to its parsed result. If a specific provider name is
    given, only that provider will be called and its parsed result returned.
    """

    messages = conversation.get("messages", [])
    if len(messages) >= 500:
        raise ValueError("Conversation must contain fewer than 500 messages")

    if provider.lower() == "auto":
        results: Dict[str, Any] = {}
        for prov in ["openai", "gemini", "claude", "mistral"]:
            env_var = _API_KEY_ENV.get(prov, "")
            if env_var and os.getenv(env_var):
                logger.info("Calling provider %s", prov)
                try:
                    results[prov] = _judge_single(conversation, prov)
                except Exception:
                    # skip failing providers
                    logger.warning("Provider %s failed", prov)
                    continue
        return results

    logger.info("Calling single provider %s", provider)
    return _judge_single(conversation, provider)
