# alethia/api/mistral_api.py

import os
import time
from typing import Dict, Any, Optional
import requests

# Load from environment (dotenv is already configured elsewhere)
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_ENDPOINT = os.getenv("MISTRAL_API_ENDPOINT", "https://api.mistral.ai/v1/generate")

def call_mistral(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "mistral-v1",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
    retry_attempts: int = 3,
    retry_backoff: float = 2.0,
) -> Dict[str, Any]:
    """
    Call Mistral’s text-generation endpoint via HTTP. Returns a JSON-like dict
    with at least `choices[0].message.content` similar to OpenAI/Gemini/Claude.

    Args:
        prompt: The text prompt to send.
        api_key: Optional override; if not provided, will use MISTRAL_KEY from env.
        model: Which Mistral model to use (e.g., "mistral-v1").
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0–1.0+).
        top_p: Nucleus sampling parameter.
        retry_attempts: How many times to retry on transient errors.
        retry_backoff: Base backoff time in seconds (exponential).

    Returns:
        A dict that mimics:
          {
            "choices": [
              { "message": { "content": "<generated text>" } }
            ]
          }
        or raises on permanent failure.
    """
    key = api_key or MISTRAL_KEY
    if not key:
        raise RuntimeError("Missing MISTRAL_API_KEY in environment.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    attempt = 0
    while attempt < retry_attempts:
        try:
            resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data
        except requests.HTTPError as e:
            # 429 Rate limit or 5xx server errors
            if resp.status_code in (429, 500, 502, 503, 504):
                attempt += 1
                backoff = retry_backoff ** attempt
                print(f"[Mistral] HTTP {resp.status_code} – retrying in {backoff:.1f}s "
                      f"(attempt {attempt}/{retry_attempts})")
                time.sleep(backoff)
            else:
                raise
        except requests.RequestException as e:
            # Network-related or timeout
            attempt += 1
            backoff = retry_backoff ** attempt
            print(f"[Mistral] Network error – retrying in {backoff:.1f}s "
                  f"(attempt {attempt}/{retry_attempts})")
            time.sleep(backoff)

    raise RuntimeError(f"[Mistral] Failed after {retry_attempts} attempts.")
