"""
lib/supabase_client.py — singleton Supabase admin client (service role, bypasses RLS).
"""

from supabase import Client, create_client
from lib.config import config

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
    return _client
