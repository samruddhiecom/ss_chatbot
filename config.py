"""Central configuration. Values come from environment / .env; sane defaults otherwise.

Nothing here is a secret except the Groq key, which is read from the environment
and never hard-coded.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
DRAFT_KB_DIR = BASE_DIR / "app" / "kb" / "draft"
CHROMA_DIR = BASE_DIR / ".chroma"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Groq) ---
    groq_api_key: str = ""
    # A capable model for generation + snapshot assembly, a fast one for classification.
    smart_model: str = "llama-3.3-70b-versatile"
    fast_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2

    # --- Embeddings (local, no API cost, no torch) ---
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # --- Vector store ---
    chroma_dir: str = str(CHROMA_DIR)
    collection_name: str = "ssbp_allowlist"
    retrieval_top_k: int = 4

    # --- Conversation control ---
    max_followups_per_stage: int = 0
    max_messages_per_session: int = 60
    global_daily_cap: int = 2000
    transcript_retention_days: int = 30  # documented; enforced by the production store

    # --- Capture ---
    tool_source: str = "a52"

    @property
    def has_llm(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()
