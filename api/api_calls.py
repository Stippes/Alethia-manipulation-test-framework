from api import call_chatgpt, call_gemini, call_claude

# api/api_calls.py

from .chatgpt_api import call_chatgpt
from .gemini_api import call_gemini
from .claude_api import call_claude
from .mistral_api import call_mistral

__all__ = ["call_chatgpt", "call_gemini", "call_claude", "call_mistral"]
