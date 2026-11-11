"""The main window of the calibration interface.

The window is organised in tabs, one per step of the workflow:

``Images``
    The list of the calibration images, and the image viewer.
``Corners``
    The interactive picking of the pattern corners.
``Model``
    The geometry of the pattern and the options of the calibration.
``Results``
    The calibrated parameters, the reprojection error and the undistorted
    image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..calibration import CalibrationOptions, calibrate
from ..images import undistort_image
from .picking import CornerPicker, detect_grid_corners
from .session import CalibrationSession
from .widgets import ImageView

__all__ = ["CalibrationWindow", "run"]


class _Worker(QObject):
    """Runs the calibration in a background thread."""

    finished = Signal(object, object, str)
    failed = Signal(str)

    def __init__(self, data, options):
        super().__init__()
        self.data = data
        self.options = options

    def run(self) -> None:
        """Run the calibration and report the outcome."""
        try:
            result = calibrate(self.data, self.options)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        if result is None:
            self.failed.emit("the calibration did not produce any result")
            return

        self.finished.emit(result, self.data, "")


class CalibrationWindow(QMainWindow):
    """The main window of the calibration interface.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.session = CalibrationSession()
        self._thread: QThread | None = None
        self._worker: _Worker | None = None

        self.setWindowTitle("Camera Calibration Toolbox")
        self.resize(1180, 780)

        self._build_actions()
        self._build_ui()
        self._update_status()

    # ==================================================================
    # Construction
    # ==================================================================
    def _build_actions(self) -> None:
        """Create the menus and the keyboard shortcuts."""
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")

        self.action_add = QAction("&Add images...", self)
        self.action_add.setShortcut(QKeySequence.Open)
        self.action_add.triggered.connect(self.on_add_images)
        file_menu.addAction(self.action_add)

        self.action_remove = QAction("&Remove selected image", self)
        self.action_remove.setShortcut(QKeySequence.Delete)
        self.action_remove.triggered.connect(self.on_remove_image)
        file_menu.addAction(self.action_remove)

        file_menu.addSeparator()

        self.action_load = QAction("&Load calibration...", self)
        self.action_load.setShortcut(QKeySequence("Ctrl+O"))
        self.action_load.triggered.connect(self.on_load_result)
        file_menu.addAction(self.action_load)

        self.action_save = QAction("&Save calibration...", self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.triggered.connect(self.on_save_result)
        file_menu.addAction(self.action_save)

        file_menu.addSeparator()

        self.action_quit = QAction("&Quit", self)
        self.action_quit.setShortcut(QKeySequence.Quit)
        self.action_quit.triggered.connect(self.close)
        file_menu.addAction(self.action_quit)

        calib_menu = menu.addMenu("&Calibration")

        self.action_run = QAction("&Run calibration", self)
        self.action_run.setShortcut(QKeySequence("Ctrl+R"))
        self.action_run.triggered.connect(self.on_run_calibration)
        calib_menu.addAction(self.action_run)

        self.action_export = QAction("&Export results as JSON...", self)
        self.action_export.triggered.connect(self.on_export_json)
        calib_menu.addAction(self.action_export)

        help_menu = menu.addMenu("&Help")

        self.action_about = QAction("&About", self)
        self.action_about.triggered.connect(self.on_about)
        help_menu.addAction(self.action_about)

    def _build_ui(self) -> None:
        """Create the widgets of the window."""
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(self._build_images_tab(), "&Images")
        self.tabs.addTab(self._build_corners_tab(), "&Corners")
        self.tabs.addTab(self._build_model_tab(), "&Model")
        self.tabs.addTab(self._build_results_tab(), "&Results")

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

        self.status = QLabel("Ready")
        self.statusBar().addWidget(self.status)

    # ------------------------------------------------------------------
    # The Images tab
    # ------------------------------------------------------------------
    def _build_images_tab(self) -> QWidget:
        """Build the tab that lists the images."""
        page = QWidget()
        layout = QHBoxLayout(page)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add images...")
        add_button.clicked.connect(self.on_add_images)
        buttons.addWidget(add_button)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.on_remove_image)
        buttons.addWidget(remove_button)

        left_layout.addLayout(buttons)

        self.image_table = QTableWidget(0, 3)
        self.image_table.setHorizontalHeaderLabels(["Image", "Corners", "Use"])
        self.image_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.image_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_table.verticalHeader().setVisible(False)
        self.image_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.image_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.image_table.currentCellChanged.connect(self.on_select_image)
        self.image_table.itemChanged.connect(self.on_item_changed)
        left_layout.addWidget(self.image_table)

        layout.addWidget(left, 2)

        self.image_view = ImageView()
        self.image_view.setMinimumWidth(360)
        layout.addWidget(self.image_view, 3)

        return page

    # ------------------------------------------------------------------
    # The Corners tab
    # ------------------------------------------------------------------
    def _build_corners_tab(self) -> QWidget:
        """Build the tab where the corners are picked."""
        page = QWidget()
        layout = QHBoxLayout(page)

        left = QWidget()
        left.setMaximumWidth(280)
        left_layout = QVBoxLayout(left)

        group = QGroupBox("Pattern")
        form = QFormLayout(group)

        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(2, 200)
        self.nx_spin.setValue(self.session.grid_nx)
        self.nx_spin.valueChanged.connect(self.on_pattern_changed)
        form.addRow("Corners along x:", self.nx_spin)

        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(2, 200)
        self.ny_spin.setValue(self.session.grid_ny)
        self.ny_spin.valueChanged.connect(self.on_pattern_changed)
        form.addRow("Corners along y:", self.ny_spin)

        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.01, 10000.0)
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setValue(self.session.grid_spacing)
        self.spacing_spin.valueChanged.connect(self.on_pattern_changed)
        form.addRow("Spacing:", self.spacing_spin)

        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(-10000.0, 10000.0)
        self.z_spin.setDecimals(3)
        self.z_spin.setValue(self.session.grid_z)
        self.z_spin.valueChanged.connect(self.on_pattern_changed)
        form.addRow("Z of the plane:", self.z_spin)

        left_layout.addWidget(group)

        buttons = QVBoxLayout()

        self.auto_button = QPushButton("Auto-detect corners")
        self.auto_button.setToolTip(
            "Detect the corners of the pattern automatically. "
            "Works on high contrast patterns."
        )
        self.auto_button.clicked.connect(self.on_auto_detect)
        buttons.addWidget(self.auto_button)

        undo_button = QPushButton("Undo last corner")
        undo_button.clicked.connect(self.on_undo_corner)
        buttons.addWidget(undo_button)

        clear_button = QPushButton("Clear corners")
        clear_button.clicked.connect(self.on_clear_corners)
        buttons.addWidget(clear_button)

        left_layout.addLayout(buttons)

        hint = QLabel(
            "Click the corners of the pattern in the image, in order:\n"
            "top-left first, then along the first row, then the next row, "
            "and so on.\n\n"
            "Right or middle drag pans the image,\n"
            "the wheel zooms, F fits the image,\n"
            "Backspace removes the last corner."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        left_layout.addWidget(hint)
        left_layout.addStretch(1)

        layout.addWidget(left)

        self.corner_view = ImageView()
        self.corner_view.cornerPicked.connect(self.on_corner_picked)
        self.corner_view.cornerRemoved.connect(self.on_corner_removed)
        layout.addWidget(self.corner_view, 1)

        return page

    # ------------------------------------------------------------------
    # The Model tab
    # ------------------------------------------------------------------
    def _build_model_tab(self) -> QWidget:
        """Build the tab with the calibration options."""
        page = QWidget()
        layout = QVBoxLayout(page)

        options = QGroupBox("Parameters to estimate")
        form = QFormLayout(options)

        self.est_fx_check = QCheckBox("Focal length fx")
        self.est_fx_check.setChecked(True)
        form.addRow(self.est_fx_check)

        self.est_fy_check = QCheckBox("Focal length fy")
        self.est_fy_check.setChecked(True)
        form.addRow(self.est_fy_check)

        self.aspect_check = QCheckBox("Estimate the aspect ratio (fx = fy otherwise)")
        self.aspect_check.setChecked(True)
        form.addRow(self.aspect_check)

        self.center_check = QCheckBox("Principal point")
        self.center_check.setChecked(True)
        form.addRow(self.center_check)

        self.alpha_check = QCheckBox("Pixel skew")
        form.addRow(self.alpha_check)

        layout.addWidget(options)

        distortion = QGroupBox("Distortion coefficients")
        dform = QFormLayout(distortion)

        self.dist_checks = []
        names = ["k1 (radial, 2nd order)", "k2 (radial, 4th order)",
                 "p1 (tangential x)", "p2 (tangential y)", "k3 (radial, 6th order)"]
        states = [True, True, True, True, False]
        for name, state in zip(names, states):
            check = QCheckBox(name)
            check.setChecked(state)
            dform.addRow(check)
            self.dist_checks.append(check)

        layout.addWidget(distortion)

        advanced = QGroupBox("Advanced")
        aform = QFormLayout(advanced)

        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 1000)
        self.iter_spin.setValue(30)
        aform.addRow("Maximum iterations:", self.iter_spin)

        self.damping_spin = QDoubleSpinBox()
        self.damping_spin.setRange(0.0, 1.0)
        self.damping_spin.setDecimals(6)
        self.damping_spin.setSingleStep(1e-4)
        self.damping_spin.setValue(1e-4)
        self.damping_spin.setToolTip(
            "Levenberg-Marquardt damping.  Set it to 0 to use the plain "
            "gradient descent of the original toolbox."
        )
        aform.addRow("Damping:", self.damping_spin)

        self.check_cond_check = QCheckBox("Reject ill-conditioned views")
        self.check_cond_check.setChecked(True)
        aform.addRow(self.check_cond_check)

        layout.addWidget(advanced)
        layout.addStretch(1)

        run_button = QPushButton("Run the calibration")
        run_button.setDefault(True)
        run_button.clicked.connect(self.on_run_calibration)
        layout.addWidget(run_button)

        return page

    # ------------------------------------------------------------------
    # The Results tab
    # ------------------------------------------------------------------
    def _build_results_tab(self) -> QWidget:
        """Build the tab that displays the results."""
        page = QWidget()
        layout = QVBoxLayout(page)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFontFamily("monospace")
        self.results_text.setPlaceholderText(
            "No calibration yet.\n\n"
            "Pick the corners of the pattern in the Corners tab, then run "
            "the calibration from the Model tab."
        )
        layout.addWidget(self.results_text)

        splitter = QSplitter(Qt.Vertical)

        self.error_table = QTableWidget(0, 2)
        self.error_table.setHorizontalHeaderLabels(["Image", "RMS error (px)"])
        self.error_table.verticalHeader().setVisible(False)
        self.error_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.error_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        splitter.addWidget(self.error_table)

        buttons = QHBoxLayout()
        self.undistort_button = QPushButton("Undistort the current image")
        self.undistort_button.clicked.connect(self.on_undistort)
        buttons.addWidget(self.undistort_button)

        self.export_button = QPushButton("Export as JSON...")
        self.export_button.clicked.connect(self.on_export_json)
        buttons.addWidget(self.export_button)

        buttons.addStretch(1)
        container = QWidget()
        container.setLayout(buttons)
        splitter.addWidget(container)

        layout.addWidget(splitter)

        return page

    # ==================================================================
    # Slots
    # ==================================================================
    def on_add_images(self) -> None:
        """Ask for images and add them to the session."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add calibration images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pgm *.ppm);;"
            "All files (*)",
        )
        if not paths:
            return

        added = self.session.add_images(paths)
        if not added:
            QMessageBox.warning(
                self, "Add images", "None of the selected files could be read."
            )
            return

        self._refresh_table()
        if self.session.views:
            self.image_table.setCurrentCell(0, 0)
            self._show_view(0)

    def on_remove_image(self) -> None:
        """Remove the selected image."""
        row = self.image_table.currentRow()
        if row < 0:
            return

        self.session.remove_view(row)
        self._refresh_table()

        if self.session.views:
            row = min(row, len(self.session.views) - 1)
            self.image_table.setCurrentCell(row, 0)
            self._show_view(row)
        else:
            self.image_view.set_image(None)
            self.corner_view.set_image(None)

    def on_select_image(self, row: int, column: int, prev_row: int, prev_column: int) -> None:
        """Show the selected image."""
        if row >= 0:
            self._show_view(row)

    def on_item_changed(self, item: QTableWidgetItem) -> None:
        """Handle the change of the "Use" checkbox of a view."""
        if item.column() != 2:
            return

        row = item.row()
        if 0 <= row < len(self.session.views):
            self.session.views[row].active = item.checkState() == Qt.Checked
            self.session.dirty = True
            self._update_status()

    def _show_view(self, index: int) -> None:
        """Display a view in the image viewers."""
        if not (0 <= index < len(self.session.views)):
            return

        view = self.session.views[index]

        self.image_view.set_image(view.image)
        self.corner_view.set_image(view.image)
        self.corner_view.set_picker(CornerPicker(self.session.grid_nx, self.session.grid_ny))
        self.corner_view.set_points(view.corners)

    def on_pattern_changed(self) -> None:
        """Handle a change of the pattern geometry."""
        self.session.grid_nx = self.nx_spin.value()
        self.session.grid_ny = self.ny_spin.value()
        self.session.grid_spacing = self.spacing_spin.value()
        self.session.grid_z = self.z_spin.value()
        self.session.dirty = True

        # The picker must know the expected number of corners.
        row = self.corner_view and self.image_table.currentRow()
        if 0 <= row < len(self.session.views):
            self.corner_view.set_picker(
                CornerPicker(self.session.grid_nx, self.session.grid_ny)
            )
            self.corner_view.set_points(self.session.views[row].corners)

        self._update_status()

    def on_corner_picked(self, x: float, y: float) -> None:
        """Store a picked corner."""
        row = self.image_table.currentRow()
        if not (0 <= row < len(self.session.views)):
            return

        view = self.session.views[row]
        corners = list(view.corners.T) if view.has_corners else []
        corners.append((x, y))
        view.corners = np.array(corners, dtype=float).T

        self.session.dirty = True
        self._update_table_row(row)
        self._update_status()

    def on_corner_removed(self) -> None:
        """Handle the removal of the last corner."""
        row = self.image_table.currentRow()
        if not (0 <= row < len(self.session.views)):
            return

        view = self.session.views[row]
        if view.has_corners:
            view.corners = view.corners[:, :-1]

        self.session.dirty = True
        self._update_table_row(row)
        self._update_status()

    def on_undo_corner(self) -> None:
        """Remove the last picked corner."""
        row = self.image_table.currentRow()
        if not (0 <= row < len(self.session.views)):
            return

        view = self.session.views[row]
        if view.has_corners:
            view.corners = view.corners[:, :-1]

        self.corner_view.set_points(view.corners)
        self.session.dirty = True
        self._update_table_row(row)
        self._update_status()

    def on_clear_corners(self) -> None:
        """Remove every corner of the current view."""
        row = self.image_table.currentRow()
        if not (0 <= row < len(self.session.views)):
            return

        self.session.views[row].corners = np.zeros((2, 0))
        self.corner_view.clear_points()
        self.session.dirty = True
        self._update_table_row(row)
        self._update_status()

    def on_auto_detect(self) -> None:
        """Detect the corners of the current view automatically."""
        row = self.image_table.currentRow()
        if not (0 <= row < len(self.session.views)):
            return

        view = self.session.views[row]

        corners = detect_grid_corners(
            view.image, self.session.grid_nx, self.session.grid_ny
        )

        if corners is None:
            QMessageBox.information(
                self,
                "Auto-detect corners",
                "The pattern could not be detected automatically.\n\n"
                "This usually means that the pattern is not a set of "
                "well separated high contrast blobs.  Pick the corners "
                "by hand instead.",
            )
            return

        view.corners = corners
        self.corner_view.set_points(corners)
        self.session.dirty = True
        self._update_table_row(row)
        self._update_status()

    def on_run_calibration(self) -> None:
        """Run the calibration in a background thread."""
        if self._thread is not None:
            return

        try:
            data = self.session.build_data()
        except ValueError as exc:
            QMessageBox.warning(self, "Run the calibration", str(exc))
            return

        options = self._collect_options()

        self.progress.setVisible(True)
        self.status.setText("Calibrating...")
        self.action_run.setEnabled(False)

        self._thread = QThread(self)
        self._worker = _Worker(data, options)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_calibration_done)
        self._worker.failed.connect(self._on_calibration_failed)

        # The thread is stopped once the worker is done.  The connection is
        # queued so that the slot runs in the thread that owns it.
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()

    def _on_thread_finished(self) -> None:
        """Clean up the worker once its thread has stopped."""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _collect_options(self) -> CalibrationOptions:
        """Build the calibration options from the widgets."""
        est_fc = np.array(
            [
                1.0 if self.est_fx_check.isChecked() else 0.0,
                1.0 if self.est_fy_check.isChecked() else 0.0,
            ]
        )
        est_dist = np.array(
            [1.0 if c.isChecked() else 0.0 for c in self.dist_checks]
        )

        return CalibrationOptions(
            est_fc=est_fc,
            est_aspect_ratio=self.aspect_check.isChecked(),
            center_optim=self.center_check.isChecked(),
            est_alpha=self.alpha_check.isChecked(),
            est_dist=est_dist,
            max_iter=self.iter_spin.value(),
            check_cond=self.check_cond_check.isChecked(),
            damping=self.damping_spin.value(),
            verbose=False,
        )

    def _on_calibration_done(self, result, data, _message: str) -> None:
        """Handle the end of a successful calibration."""
        self.session.set_result(result, data)

        self.progress.setVisible(False)
        self.action_run.setEnabled(True)
        self.status.setText("Calibration finished")

        self._show_results(result)
        self.tabs.setCurrentIndex(3)

    def _on_calibration_failed(self, message: str) -> None:
        """Handle a failed calibration."""
        self.progress.setVisible(False)
        self.action_run.setEnabled(True)
        self.status.setText("The calibration failed")

        QMessageBox.critical(
            self,
            "Run the calibration",
            f"The calibration failed:\n\n{message}",
        )

    def _show_results(self, result) -> None:
        """Fill the results tab."""
        lines = []
        lines.append("Calibration results")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Focal length:      fc = [{result.fc[0]:10.5f} {result.fc[1]:10.5f}]"
                     f"  +/- [{result.fc_error[0]:.5f} {result.fc_error[1]:.5f}]")
        lines.append(f"Principal point:   cc = [{result.cc[0]:10.5f} {result.cc[1]:10.5f}]"
                     f"  +/- [{result.cc_error[0]:.5f} {result.cc_error[1]:.5f}]")
        lines.append(f"Skew:         alpha_c = {result.alpha_c:10.5f}"
                     f"  +/- {result.alpha_c_error:.5f}")
        lines.append("")
        lines.append("Distortion coefficients")
        for i, (name, value) in enumerate(
            zip(["k1", "k2", "p1", "p2", "k3"], result.kc)
        ):
            lines.append(f"    {name} = {value:12.6f}"
                         f"   +/- {result.kc_error[i]:.6f}")
        lines.append("")
        lines.append(f"Pixel error:        err = [{result.err_std[0]:.5f} {result.err_std[1]:.5f}]")
        lines.append(f"RMS error:              {result.rmse:.5f} px")
        lines.append(f"Views used:            {int(np.sum(result.active_images))}"
                     f" / {len(result.active_images)}")
        lines.append(f"Iterations:             {result.n_iter}")
        lines.append("")
        lines.append("Camera matrix K")
        for row in result.KK:
            lines.append("    " + "  ".join(f"{v:12.6f}" for v in row))

        if result.kc[3] != 0 or result.kc[2] != 0:
            lines.append("")
            lines.append("Note: the tangential distortion coefficients are not zero;")
            lines.append("      the skew is usually better set to zero.")

        self.results_text.setPlainText("\n".join(lines))

        # The error per view.
        errors = self.session.error_per_view()
        self.error_table.setRowCount(len(errors))
        for row, (name, rms) in enumerate(errors):
            self.error_table.setItem(row, 0, QTableWidgetItem(name))
            self.error_table.setItem(row, 1, QTableWidgetItem(f"{rms:.5f}"))

    def on_undistort(self) -> None:
        """Undistort the current image and show it."""
        if self.session.result is None:
            QMessageBox.information(
                self,
                "Undistort",
                "Run the calibration first.",
            )
            return

        row = self.image_table.currentRow()
        if not (0 <= row < len(self.session.views)):
            return

        view = self.session.views[row]
        result = self.session.result

        try:
            undistorted = undistort_image(view.image, result)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            QMessageBox.critical(self, "Undistort", str(exc))
            return

        self.image_view.set_image(undistorted)
        self.tabs.setCurrentIndex(0)
        self.status.setText("Showing the undistorted image of the current view")

    def on_save_result(self) -> None:
        """Save the calibration to a file."""
        if self.session.result is None:
            QMessageBox.information(
                self, "Save", "There is no calibration to save."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save the calibration", "Calib_Results.npz", "NumPy archive (*.npz)"
        )
        if not path:
            return

        from ..io import save_calibration

        try:
            save_calibration(self.session.result, path, self.session.data)
        except OSError as exc:
            QMessageBox.critical(self, "Save", str(exc))
            return

        self.status.setText(f"Calibration saved to {Path(path).name}")

    def on_load_result(self) -> None:
        """Load a calibration from a file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load a calibration", "", "NumPy archive (*.npz)"
        )
        if not path:
            return

        from ..io import load_calibration

        try:
            result = load_calibration(path)
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.critical(self, "Load", str(exc))
            return

        self.session.result = result
        self._show_results(result)
        self.tabs.setCurrentIndex(3)
        self.status.setText(f"Calibration loaded from {Path(path).name}")

    def on_export_json(self) -> None:
        """Export the calibration as JSON."""
        if self.session.result is None:
            QMessageBox.information(
                self, "Export", "There is no calibration to export."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export the calibration", "Calib_Results.json", "JSON (*.json)"
        )
        if not path:
            return

        from ..io import export_calibration_json

        try:
            export_calibration_json(self.session.result, path)
        except OSError as exc:
            QMessageBox.critical(self, "Export", str(exc))
            return

        self.status.setText(f"Calibration exported to {Path(path).name}")

    def on_about(self) -> None:
        """Show the about box."""
        QMessageBox.about(
            self,
            "About",
            "<h3>Camera Calibration Toolbox</h3>"
            "<p>A PySide6 interface for the Python port of Jean-Yves "
            "Bouguet's Camera Calibration Toolbox for Matlab.</p>"
            "<p>The calibration core is a faithful port of the original "
            "algorithms.</p>",
        )

    # ==================================================================
    # Helpers
    # ==================================================================
    def _refresh_table(self) -> None:
        """Rebuild the list of the images."""
        self.image_table.blockSignals(True)
        self.image_table.setRowCount(len(self.session.views))

        for row, view in enumerate(self.session.views):
            name_item = QTableWidgetItem(view.name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.image_table.setItem(row, 0, name_item)

            count_item = QTableWidgetItem(str(view.corners.shape[1]))
            count_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.image_table.setItem(row, 1, count_item)

            check_item = QTableWidgetItem("")
            check_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            check_item.setCheckState(Qt.Checked if view.active else Qt.Unchecked)
            self.image_table.setItem(row, 2, check_item)

        self.image_table.blockSignals(False)
        self._update_status()

    def _update_table_row(self, row: int) -> None:
        """Update the number of corners shown for a view."""
        if not (0 <= row < len(self.session.views)):
            return
        item = self.image_table.item(row, 1)
        if item is not None:
            item.setText(str(self.session.views[row].corners.shape[1]))

    def _update_status(self) -> None:
        """Refresh the status bar."""
        n_views = len(self.session.views)
        n_corners = sum(1 for v in self.session.views if v.has_corners)
        n_active = sum(1 for v in self.session.views if v.active and v.has_corners)

        message = (
            f"{n_views} images, {n_corners} with corners, {n_active} used"
        )
        if self.session.result is not None:
            message += f" - RMS error {self.session.result.rmse:.4f} px"
        self.status.setText(message)

    def closeEvent(self, event) -> None:
        """Wait for the worker thread before closing."""
        if self._thread is not None and self._thread.isRunning():
            # The worker cannot be interrupted, so the window waits for it
            # to finish.
            self._thread.quit()
            self._thread.wait(30000)
        super().closeEvent(event)


def run(argv=None) -> int:
    """Run the graphical interface.

    Parameters
    ----------
    argv : list of str, optional
        The command line arguments.  The application arguments are used
        by default.

    Returns
    -------
    int
        The exit code of the application.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)

    window = CalibrationWindow()
    window.show()

    return app.exec()
