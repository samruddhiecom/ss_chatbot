"""Shared Supabase client for the SS Advisor backend.

Env vars:
  SUPABASE_URL       -- Project URL (https://ywctafxwkbekuqyhngvo.supabase.co)
  SUPABASE_ANON_KEY  -- anon/public key from Project Settings > API
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("ss_advisor")

_CLIENT = None


def get_client():
    """Return a shared Supabase client. Returns None if env vars are missing."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        logger.warning("SUPABASE_URL or SUPABASE_ANON_KEY not set — Supabase writes disabled.")
        return None
    try:
        from supabase import create_client
        _CLIENT = create_client(url, key)
        return _CLIENT
    except Exception as e:
        logger.error("Supabase client init failed: %s", e)
        return None
