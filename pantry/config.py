# Runtime configuration for the pantry agent, read once from environment variables.
# Loads .env explicitly with load_dotenv(".env"); secrets are never logged or printed.
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import os

from dotenv import load_dotenv

DEFAULT_DEMO_START_DATE: str = "2026-11-15"
DEFAULT_TICK_SECONDS: float = 2.0


def _get_str(name: str, default: str = "") -> str:
    """Return a stripped environment variable, or the default when unset or blank."""
    value = os.getenv(name, "").strip()
    return value or default


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable (true/false, 1/0, yes/no)."""
    raw = _get_str(name)
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be true or false, got {raw!r}")


def _get_float(name: str, default: float) -> float:
    """Parse a float environment variable."""
    raw = _get_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _get_date(name: str, default: str) -> date:
    """Parse an ISO date (YYYY-MM-DD) environment variable."""
    raw = _get_str(name, default)
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an ISO date, got {raw!r}") from exc


def _get_choice(name: str, default: str, choices: set[str]) -> str:
    """Parse an environment variable restricted to a fixed set of lowercase values."""
    value = _get_str(name, default).lower()
    if value not in choices:
        raise ValueError(f"Environment variable {name} must be one of {sorted(choices)}, got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of every environment variable the agent uses."""

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    mongodb_uri: str
    mongodb_db: str
    linkup_api_key: str
    tavily_api_key: str
    embed_provider: str
    embed_model: str
    mlflow_tracking_uri: str
    databricks_host: str
    databricks_token: str
    cache_first: bool
    store_backend: str
    tick_seconds: float
    demo_start_date: date

    def missing_keys(self) -> list[str]:
        """Return the names of unset variables whose features must fall back to Fakes."""
        required = {
            "LLM_BASE_URL": self.llm_base_url,
            "LLM_API_KEY": self.llm_api_key,
            "LLM_MODEL": self.llm_model,
            "MONGODB_URI": self.mongodb_uri,
            "LINKUP_API_KEY": self.linkup_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
        }
        return [name for name, value in required.items() if not value]


def load_settings() -> Settings:
    """Load .env (without overriding already-set variables) and build a fresh Settings."""
    load_dotenv(".env")
    return Settings(
        llm_base_url=_get_str("LLM_BASE_URL"),
        llm_api_key=_get_str("LLM_API_KEY"),
        llm_model=_get_str("LLM_MODEL"),
        mongodb_uri=_get_str("MONGODB_URI"),
        mongodb_db=_get_str("MONGODB_DB", "pantry_a"),
        linkup_api_key=_get_str("LINKUP_API_KEY"),
        tavily_api_key=_get_str("TAVILY_API_KEY"),
        embed_provider=_get_choice("EMBED_PROVIDER", "openai", {"openai", "local"}),
        embed_model=_get_str("EMBED_MODEL", "text-embedding-3-small"),
        mlflow_tracking_uri=_get_str("MLFLOW_TRACKING_URI"),
        databricks_host=_get_str("DATABRICKS_HOST"),
        databricks_token=_get_str("DATABRICKS_TOKEN"),
        cache_first=_get_bool("CACHE_FIRST", False),
        store_backend=_get_choice("STORE_BACKEND", "mongo", {"mongo", "json"}),
        tick_seconds=_get_float("TICK_SECONDS", DEFAULT_TICK_SECONDS),
        demo_start_date=_get_date("DEMO_START_DATE", DEFAULT_DEMO_START_DATE),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings, loading them on first use."""
    return load_settings()
