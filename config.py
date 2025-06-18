"""Central configuration settings for the framework."""

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")

DATA_PATH = os.getenv("DATA_PATH", "data")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output")
