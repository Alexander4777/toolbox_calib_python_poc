"""Entry point of the graphical interface.

Run it with::

    python -m calib.gui
"""

from __future__ import annotations

import sys

from .main_window import run

if __name__ == "__main__":
    sys.exit(run())
