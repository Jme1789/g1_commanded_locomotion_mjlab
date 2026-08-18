"""Command-line entry point for MJLab training."""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.training import train as _implementation

if __name__ == "__main__":
  _implementation.main()
else:
  # Preserve the historical import surface used by project tooling and tests.
  sys.modules[__name__] = _implementation
