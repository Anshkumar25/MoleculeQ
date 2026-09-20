"""Pytest bootstrap: make the src package importable from the repo root.

Ensures `from src... import ...` works no matter where pytest is invoked.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))