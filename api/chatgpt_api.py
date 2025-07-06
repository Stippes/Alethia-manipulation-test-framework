# alethia/api/chatgpt_api.py

import os
import time
from typing import Dict, Any, Optional

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency for tests
    openai = None

# Optionally: set your API key here or rely on OPENAI_API_KEY environment variable
# openai.api_key = "YOUR_OPENAI_API_KEY"


def call_chatgpt(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
    n: int = 1,
    retry_attempts: int = 3,
    retry_backoff: float = 2.0,
) -> Dict[str, Any]:
    """
    Call OpenAI's ChatCompletion API (ChatGPT).

    Args:
      - prompt: The text prompt to send.
      - api_key: Optional. If provided, will override the environment's OPENAI_API_KEY.
      - model: The ChatGPT model to use (e.g., "gpt-4", "gpt-3.5-turbo").
      - max_tokens: Maximum tokens to generate.
      - temperature: Sampling temperature.
      - top_p: Nucleus sampling parameter.
      - n: Number of completions to generate.
      - retry_attempts: How many times to retry on rate limit or transient errors.
      - retry_backoff: Base number of seconds to wait before retrying (exponential backoff).

    Returns:
      - The raw JSON response from OpenAI.
    """
    if openai is None:
        raise RuntimeError("openai package not available")

    openai_key = os.getenv("OPENAI_API_KEY")
    key = api_key or openai_key
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    version = getattr(openai, "__version__", "0")
    try:
        major = int(str(version).split(".")[0])
    except Exception:
        major = 0
    use_client = major >= 1 and hasattr(openai, "OpenAI")

    if hasattr(openai, "error"):
        error_mod = openai.error
    else:
        error_mod = openai
    RateLimitError = getattr(error_mod, "RateLimitError", Exception)
    APIConnectionError = getattr(error_mod, "APIConnectionError", Exception)
    OpenAIError = getattr(error_mod, "OpenAIError", Exception)

    client = openai.OpenAI(api_key=key) if use_client else None
    if not use_client:
        openai.api_key = key

    attempt = 0
    last_err: Optional[Exception] = None
    while attempt < retry_attempts:
        try:
            if use_client:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                )
            return response
        except (RateLimitError, APIConnectionError) as e:
            last_err = e
            attempt += 1
            wait_time = retry_backoff ** attempt
            print(
                f"[ChatGPT] {e.__class__.__name__} – retrying in {wait_time:.1f}s "
                f"(attempt {attempt}/{retry_attempts})"
            )
            time.sleep(wait_time)
        except OpenAIError as e:
            raise RuntimeError(f"[ChatGPT] API call failed: {e}") from e

    raise RuntimeError(
        f"[ChatGPT] Failed after {retry_attempts} attempts.") from last_err
