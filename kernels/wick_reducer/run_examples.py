#!/usr/bin/env python3
"""Run and time the eight built-in qct correlator examples."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernels.wick_reducer.reducer import main


if __name__ == "__main__":
    main()
