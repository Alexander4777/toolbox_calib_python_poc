#!/usr/bin/env python3
"""Launch the camera calibration graphical interface.

Usage::

    python run_gui.py [image1.png image2.png ...]

The images given on the command line are loaded at startup.
"""

from __future__ import annotations

import sys
from pathlib import Path

from calib.gui import CalibrationWindow


def main(argv: list[str] | None = None) -> int:
    """Open the interface, optionally loading some images.

    Parameters
    ----------
    argv : list of str, optional
        The command line arguments.

    Returns
    -------
    int
        The exit code.
    """
    argv = list(sys.argv if argv is None else argv)

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)

    window = CalibrationWindow()
    window.show()

    images = [Path(a) for a in argv[1:] if Path(a).exists()]
    if images:
        added = window.session.add_images(images)
        window._refresh_table()
        if added:
            window.image_table.setCurrentCell(0, 0)
            window._show_view(0)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
