# alethia/api/chatgpt_api.py

import time
import os
import openai
from typing import Dict, Any, Optional

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
    # If an api_key was passed explicitly, set it:

    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    api_key = OPENAI_KEY


    attempt = 0
    while attempt < retry_attempts:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
            )
            return response
        except openai.error.RateLimitError as e:
            attempt += 1
            wait_time = retry_backoff ** attempt
            print(f"[ChatGPT] Rate limit hit – retrying in {wait_time:.1f}s (attempt {attempt}/{retry_attempts})")
            time.sleep(wait_time)
        except openai.error.APIConnectionError as e:
            attempt += 1
            wait_time = retry_backoff ** attempt
            print(f"[ChatGPT] Connection error – retrying in {wait_time:.1f}s (attempt {attempt}/{retry_attempts})")
            time.sleep(wait_time)
        except Exception as e:
            # For other errors, re-raise
            raise

    raise RuntimeError(f"[ChatGPT] Failed after {retry_attempts} attempts due to rate limits or network errors.")
