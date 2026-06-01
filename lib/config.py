"""
lib/config.py — load and validate required environment variables at startup.
The server will refuse to start if any required variable is missing.
"""

import os


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


class Config:
    SUPABASE_URL:              str = _require("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY: str = _require("SUPABASE_SERVICE_ROLE_KEY")
    OPENAI_API_KEY:            str = _require("OPENAI_API_KEY")
    PORT:                      int = int(os.environ.get("PORT", "8000"))


config = Config()
