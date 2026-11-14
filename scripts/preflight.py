# Preflight checks before a demo: env variable names, Mongo ping, embedding dimension, chat JSON mode, search keys.
# Prints PASS/FAIL per check and exits non-zero on any failure; never prints secret values, only variable names.
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel  # noqa: E402

EXPECTED_EMBED_DIM: int = 1536
REQUIRED_VARS: tuple[str, ...] = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "MONGODB_URI",
    "MONGODB_DB",
    "LINKUP_API_KEY",
    "TAVILY_API_KEY",
    "EMBED_PROVIDER",
    "EMBED_MODEL",
    "STORE_BACKEND",
)
OPTIONAL_VARS: tuple[str, ...] = (
    "MLFLOW_TRACKING_URI",
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "CACHE_FIRST",
    "TICK_SECONDS",
    "DEMO_START_DATE",
)
SECRET_VARS: tuple[str, ...] = (
    "LLM_API_KEY",
    "MONGODB_URI",
    "LINKUP_API_KEY",
    "TAVILY_API_KEY",
    "DATABRICKS_TOKEN",
    "LLM_BASE_URL",
)


class PreflightPing(BaseModel):
    """Tiny schema used to prove JSON mode plus pydantic validation works end to end."""

    status: str


def _is_set(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def scrub(text: str) -> str:
    """Replace any secret value that leaks into an error message with its variable name."""
    for name in SECRET_VARS:
        value = os.getenv(name, "").strip()
        if len(value) >= 6:
            text = text.replace(value, f"<{name}>")
    return text[:300]


def check_required_vars() -> str:
    missing = [name for name in REQUIRED_VARS if not _is_set(name)]
    if missing:
        raise RuntimeError(f"unset: {', '.join(missing)}")
    return f"all {len(REQUIRED_VARS)} required variables set"


def check_config() -> str:
    from pantry.config import load_settings

    settings = load_settings()
    return f"STORE_BACKEND={settings.store_backend} EMBED_PROVIDER={settings.embed_provider} DEMO_START_DATE={settings.demo_start_date}"


def check_mongo() -> str:
    from pantry.store.mongo_store import MongoStore

    store = MongoStore(server_selection_timeout_ms=8000)
    try:
        if not store.ping():
            raise RuntimeError("ping returned ok=0")
        return f"ping ok (database name from MONGODB_DB: {store.db_name})"
    finally:
        store.close()


def check_embedding() -> str:
    from pantry.llm import OpenNextLLM

    vectors = OpenNextLLM().embed(["preflight embedding check"])
    dim = len(vectors[0]) if vectors else 0
    if dim != EXPECTED_EMBED_DIM:
        raise RuntimeError(f"embedding dimension {dim}, expected {EXPECTED_EMBED_DIM}")
    return f"dimension {dim}"


def check_chat_json() -> str:
    from pantry.llm import OpenNextLLM

    result = OpenNextLLM().chat_json(
        "You are a preflight check.", 'Return {"status": "ok"} as JSON.', PreflightPing
    )
    return f"JSON mode ok (status={result['status']!r})"


def check_search_keys() -> str:
    missing = [name for name in ("LINKUP_API_KEY", "TAVILY_API_KEY") if not _is_set(name)]
    if missing:
        raise RuntimeError(f"unset: {', '.join(missing)} (no API call made)")
    return "LINKUP_API_KEY and TAVILY_API_KEY set (no API call made)"


CHECKS: tuple[tuple[str, Callable[[], str], str], ...] = (
    ("env vars", check_required_vars, "fill the listed variables in .env (see .env.example)"),
    ("config", check_config, "fix the variable named in the error"),
    ("mongo ping", check_mongo, "set STORE_BACKEND=json to run offline"),
    ("embedding dim", check_embedding, "FakeLLM embeddings (EMBED_PROVIDER=local is not implemented yet)"),
    ("chat json mode", check_chat_json, "FakeLLM / cached plans with CACHE_FIRST=true"),
    ("search keys", check_search_keys, "FakeLinkupClient / FakeTavilyClient or the search cache"),
)


def main() -> int:
    load_dotenv(".env")
    failures = 0
    for name, check, fallback in CHECKS:
        try:
            detail = check()
            print(f"PASS  {name:<15} {detail}")
        except Exception as exc:  # noqa: BLE001 - every failure is reported, never raised
            failures += 1
            print(f"FAIL  {name:<15} {type(exc).__name__}: {scrub(str(exc))}")
            print(f"      fallback: {fallback}")
    for name in OPTIONAL_VARS:
        print(f"INFO  optional {name}: {'set' if _is_set(name) else 'unset'}")
    print(f"{'FAIL' if failures else 'PASS'}  preflight: {failures} failed check(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
