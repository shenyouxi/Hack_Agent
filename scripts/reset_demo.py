# Restores the demo to day 1 (DEMO_START_DATE) on the STORE_BACKEND store; safe to run repeatedly.
# Keeps search_cache, plan_cache and seed traces; clears runs, bus events, commands, ledger, feedback, shopping list.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pantry.config import get_settings  # noqa: E402
from pantry.store.factory import make_store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset the pantry demo to day 1.")
    parser.add_argument("--backend", choices=["mongo", "json"], help="override STORE_BACKEND")
    args = parser.parse_args(argv)

    settings = get_settings()
    backend = args.backend or settings.store_backend
    store = make_store(backend, settings)
    counts = store.reset_demo(settings.demo_start_date)

    problems = store.check_invariants()
    seeds = len(store.list_traces(is_seed=True, limit=10_000))
    print(f"Reset backend={backend} to day 1 ({store.get_today().isoformat()})")
    print(f"  restored pantry items: {counts.get('pantry_items', 0)}")
    for name in sorted(k for k in counts if k != "pantry_items"):
        print(f"  cleared {name}: {counts[name]}")
    print(f"  kept seed traces: {seeds}; search_cache and plan_cache untouched")
    if counts.get("pantry_items", 0) == 0:
        print("  warning: no pantry baseline found; run python -m pantry.seed first")
    if problems:
        print("FAIL invariant violations after reset:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASS inventory invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
