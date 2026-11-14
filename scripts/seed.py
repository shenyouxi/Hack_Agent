# Thin wrapper so `python scripts/seed.py` (docs/conventions.md) and `python -m pantry.seed` behave the same.
# All seeding logic lives in pantry/seed.py.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pantry.seed import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
