"""
Foot Pressure Real-time Monitor — SLIM build
============================================
Receives **two MCU** streams over TCP (same IP: left foot+knee **5000**, right **6000**),
bilateral pair-wise sync by PC receive time, writes labeled + raw CSV, and runs the
hierarchical RF recognizer (``realtime_recognizer.OnlineRecognizer``).

This rewrite **removes** all heavy visualisations that were causing the UI to lag:
  • no foot heatmap (``pyqtgraph`` is no longer imported)
  • no per-sensor histogram dialog
  • no timer-driven redraw loop

What is still here (and all the user actually needs for live testing):
  • IP / port inputs, Start / Stop, connection dots
  • **Live raw ADC display** — 8 channel values updated on every synced frame
  • **Cascade status HUD** — Layer-1 (Active / Inactive knee gate) + Layer-2 (Motion /
    Static) + the four-branch RF's final label, probability and step counters
  • Data-capture label strip (writes the same 9-column / 10-column dual CSV format)
  • Personal calibration wizard (two-step: stand → knee bend)

Per-MCU pin mapping:   A1 → toe    A2 → forefoot    A5 → heel    A8 → knee
Arduino line format:   ``timestamp_ms,toe,forefoot,heel,knee``
"""

from __future__ import annotations

import csv
import datetime
import math
import os
import socket
import sys
import time
import traceback
from collections import deque
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import personal_calibration as personal_calib
from realtime_recognizer import OnlineRecognizer

# 12-bit ADC range
SENSOR_MAX = 4095.0
# L/R pairing uses PC receive time (monotonic); MCU millis differ per board.
SYNC_PC_MAX_S = 0.15


# ═══════════════════════════════════════════════════════════════════════════
#  DARK THEME
# ═══════════════════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow { background-color: #0a0a1a; }
QWidget {
    background-color: transparent;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Arial', sans-serif;
}
QFrame#card {
    background-color: #10102a;
    border: 1px solid #1e1e40;
    border-radius: 12px;
}
QPushButton {
    background-color: #ff6b6b;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton:hover  { background-color: #ff8787; }
QPushButton:pressed{ background-color: #e05252; }
QPushButton:disabled{ background-color: #2a2a48; color: #555; }
QPushButton#stopBtn          { background-color: #3a3a5a; }
QPushButton#stopBtn:hover    { background-color: #4a4a6a; }
QPushButton#stopBtn:pressed  { background-color: #2a2a4a; }
QPushButton#labelBtn {
    background:#1e1e40; color:#d0d0e0; font-size:11px; font-weight:normal;
    border:1px solid #28284e; border-radius:6px; padding:6px 8px;
}
QPushButton#labelBtn:hover { background:#2e2e58; }
QLineEdit {
    background-color: #14143a;
    border: 1.5px solid #28284e;
    border-radius: 7px;
    padding: 6px 10px;
    color: #d0d0e0;
    font-size: 13px;
    selection-background-color: #ff6b6b;
}
QLineEdit:focus { border-color: #ff6b6b; }
QLabel { color: #d0d0e0; background: transparent; }
QStatusBar {
    background-color: #08081a;
    color: #606080;
    font-size: 12px;
    border-top: 1px solid #18183a;
    padding: 4px 12px;
}
QTabWidget::pane { border: 1px solid #28284e; border-radius: 8px; }
QTabBar::tab { background:#14143a; padding:7px 16px; color:#b0b0d0; }
QTabBar::tab:selected { background:#1e1e40; color:#ff6b6b; font-weight:bold; }
"""


# ═══════════════════════════════════════════════════════════════════════════
#  TCP LINE PARSING
# ═══════════════════════════════════════════════════════════════════════════

FootFour = Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]


def parse_line_timestamped(line: str) -> Optional[Tuple[int, FootFour]]:
    """Arduino line → ``(ts_ms, (toe, forefoot, heel, knee))``; ``None`` on any parse error."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None
    try:
        ts = int(float(parts[0]))
    except ValueError:
        return None
    vals: list[float] = []
    for p in parts[1:5]:
        try:
            v = float(p)
            if math.isnan(v) or math.isinf(v):
                return None
            vals.append(v)
        except ValueError:
            return None
    return (ts, (vals[0], vals[1], vals[2], vals[3]))


def _deque_remove_one(d: deque, item: tuple) -> bool:
    """Remove exactly one occurrence of ``item`` from ``d`` (preserve ``maxlen``)."""
    before = len(d)
    ml = d.maxlen
    tmp = deque((x for x in d if x != item), maxlen=ml)
    if len(tmp) == before:
        return False
    d.clear()
    d.extend(tmp)
    return True


def foot_tuple_for_recognizer(tup: Optional[FootFour]) -> Optional[Tuple[float, float, float, float]]:
    """Replace any ``None`` channel with ``SENSOR_MAX`` (= no pressure / straight knee)."""
    if tup is None:
        return None
    return tuple(SENSOR_MAX if x is None else float(x) for x in tup)


# ═══════════════════════════════════════════════════════════════════════════
#  SOCKET THREAD — one per MCU
# ═══════════════════════════════════════════════════════════════════════════

class SocketThread(QThread):
    """TCP server per MCU; emits raw samples (no CSV — main window handles it)."""

    # side, timestamp_ms, toe, forefoot, heel, knee (ts as object avoids int-overflow)
    data_received = pyqtSignal(str, object, float, float, float, float)
    status_changed = pyqtSignal(str)
    connection_state = pyqtSignal(str)  # "connected" | "waiting" | "disconnected"

    def __init__(self, host: str, port: int, side: str, parent=None):
        super().__init__()
        self.host = host
        self.port = port
        self.side = side  # "L" or "R"
        self.parent_window = parent
        self._running = False
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None

    def run(self):
        self._running = True
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(1.0)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)

            self.connection_state.emit("waiting")
            self.status_changed.emit(
                f"[{self.side}] listening {self.host}:{self.port} — waiting for MCU..."
            )

            while self._running:
                try:
                    self.client_socket, addr = self.server_socket.accept()
                    self.connection_state.emit("connected")
                    self.status_changed.emit(
                        f"[{self.side}] connected: {addr[0]}:{addr[1]} → port {self.port}"
                    )
                    break
                except socket.timeout:
                    continue

            if not self._running:
                return

            buffer = ""
            header_skipped = False
            self.client_socket.settimeout(0.5)

            while self._running:
                try:
                    chunk = self.client_socket.recv(1024).decode("utf-8")
                    if not chunk:
                        self.status_changed.emit(f"[{self.side}] MCU disconnected")
                        break

                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip().lstrip("\ufeff")
                        if not line:
                            continue
                        if not header_skipped:
                            lo = line.lower()
                            if "toe" in lo and ("knee" in lo or "timestamp" in lo):
                                header_skipped = True
                                continue
                            header_skipped = True
                        parsed = parse_line_timestamped(line)
                        if parsed is not None:
                            ts_ms, (s_toe, s_forefoot, s_heel, s_knee) = parsed
                            self.data_received.emit(
                                self.side, ts_ms,
                                s_toe, s_forefoot, s_heel, s_knee,
                            )
                except socket.timeout:
                    continue
                except Exception as exc:  # noqa: BLE001
                    self.status_changed.emit(f"[{self.side}] receive error: {exc}")
                    break

        except OSError as exc:
            self.status_changed.emit(f"[{self.side}] server failed: {exc}")
        finally:
            self._cleanup()
            self.connection_state.emit("disconnected")

    def _cleanup(self):
        for s in (self.client_socket, self.server_socket):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

    def stop(self):
        self._running = False
        self._cleanup()


# ═══════════════════════════════════════════════════════════════════════════
#  SMALL CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════════

class StatusDot(QWidget):
    """Tri-state LED dot (disconnected / waiting / connected)."""

    _COLORS = {
        "disconnected": QColor("#ff4455"),
        "waiting":      QColor("#ffaa00"),
        "connected":    QColor("#44ff88"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._state = "disconnected"

    def set_state(self, state: str):
        self._state = state
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = self._COLORS.get(self._state, self._COLORS["disconnected"])
        glow = QColor(c); glow.setAlpha(60)
        p.setBrush(QBrush(glow)); p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, 14, 14)
        p.setBrush(QBrush(c))
        p.drawEllipse(3, 3, 8, 8)
        p.end()


class ChannelReadout(QLabel):
    """Single-channel live ADC readout — raw + pressure percent, updated via ``set_raw``."""

    def __init__(self, title: str, accent: str, is_knee: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        self._accent = accent
        self._is_knee = is_knee
        self.setAlignment(Qt.AlignCenter)
        self.setTextFormat(Qt.RichText)
        self.setMinimumWidth(110)
        self.set_raw(None)

    def set_raw(self, raw: Optional[float]):
        if raw is None:
            body = "<span style='color:#606080;font-size:24px;font-weight:bold;'>—</span>"
            pct = "<span style='color:#404060;font-size:10px;'>no data</span>"
        else:
            raw_i = int(round(float(raw)))
            body = (
                f"<span style='color:#ffffff;font-size:24px;font-weight:bold;'>"
                f"{raw_i}</span>"
            )
            if self._is_knee:
                # Knee: lower raw = more bent; 4095 = straight
                bend = max(0.0, (SENSOR_MAX - raw)) / SENSOR_MAX * 100.0
                tag_clr = "#69db7c" if bend < 3 else "#ffd43b" if bend < 25 else "#ff6b6b"
                pct = (
                    f"<span style='color:{tag_clr};font-size:10px;'>"
                    f"bend {bend:4.0f}%</span>"
                )
            else:
                load = max(0.0, (SENSOR_MAX - raw)) / SENSOR_MAX * 100.0
                tag_clr = "#44dd66" if load < 40 else "#ffaa00" if load < 70 else "#ff4444"
                pct = (
                    f"<span style='color:{tag_clr};font-size:10px;'>"
                    f"load {load:4.0f}%</span>"
                )
        self.setText(
            f"<div style='line-height:115%;'>"
            f"<span style='color:{self._accent};font-size:11px;font-weight:bold;'>"
            f"{self._title}</span><br>{body}<br>{pct}"
            f"</div>"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PERSONAL CALIBRATION WIZARD (two-step: stand → knee bend)
# ═══════════════════════════════════════════════════════════════════════════

class CalibrationPanel(QWidget):
    """Embedded two-step personal-calibration wizard.

    Layout fix
    ----------
    The panel's content lives inside a :class:`QScrollArea` so no button
    is ever clipped, regardless of the host window's height or Windows
    display scaling (100% / 125% / 150%).  The outer panel itself has
    an Expanding size policy so the parent tab widget can give it the
    remaining vertical space.

    Live-data guard
    ---------------
    ``Start Step 1 / Step 2`` are blocked unless :meth:`set_live_state`
    has been called with ``True`` (i.e. MainWindow confirmed at least
    one synced bilateral frame arrived).  Feeding the :class:`OnlineCalibrator`
    is also a no-op while ``self._live_ok`` is ``False`` — the progress bar
    can never fake-advance.

    On **Save** a ``personal_calibration.json`` is written to the project
    root and :class:`MainWindow` hot-reloads its :class:`OnlineRecognizer`
    so the new per-wearer range takes effect immediately.  The Layer-1 knee
    4095 rule is still applied to the *pre*-calibration raw ADC inside the
    recognizer; calibration only changes the ML feature scale.
    """

    STEP1_SECONDS = 5.0
    STEP2_SECONDS = 5.0
    JSON_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        personal_calib.DEFAULT_CALIBRATION_FILENAME,
    )

    calibration_saved = pyqtSignal(str)  # emits absolute path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calibrator: Optional[personal_calib.OnlineCalibrator] = None
        self._last_preview: Optional[personal_calib.PersonalCalibration] = None
        self._live_ok: bool = False        # set via set_live_state()
        self._step_ready_notified: set[str] = set()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._build_ui()
        self._apply_initial_button_state()
        self._refresh_existing_calibration_status()

    # ═══════════════════ UI CONSTRUCTION ════════════════════════════
    def _build_ui(self):
        # Outer layout holds ONE widget: a QScrollArea with the real
        # content inside.  This is what keeps every button visible even
        # on 125–150% Windows display scaling and short windows.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea(self)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setStyleSheet(
            "QScrollArea { background:transparent; border:none; }"
            "QScrollBar:vertical { background:#10102a; width:10px; }"
            "QScrollBar::handle:vertical { background:#3a3a5a; border-radius:5px; }"
            "QScrollBar::add-line, QScrollBar::sub-line { height:0; }"
        )
        outer.addWidget(self._scroll)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(14, 10, 14, 14)
        root.setSpacing(10)

        # ── usage-order banner (bright, so users can't miss it) ───────
        banner = QLabel(
            "Usage order:  connect both MCUs → click top-right <b>Start</b> → "
            "wait until live bilateral packets arrive → open Calibration → "
            "Step 1 → Step 2 → Save"
        )
        banner.setTextFormat(Qt.RichText)
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel{background:#241830;border:1px solid #ff9060;border-radius:6px;"
            "color:#ffd0a0;font-size:12px;padding:8px 10px;}"
        )
        root.addWidget(banner)

        # ── per-wearer intro (what the wizard does) ───────────────────
        header = QLabel(
            "Personal calibration wizard — aligns per-channel [min, max] ADC to this "
            "wearer.  Layer-1 knee 4095 rule is unaffected; only ML feature scale changes."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color:#d0d0e0; font-size:12px;")
        root.addWidget(header)

        # ── existing-calibration status line ──────────────────────────
        self.status_lbl = QLabel("—")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#9090b0; font-size:11px;")
        root.addWidget(self.status_lbl)

        # ── live-data indicator line (flips when MainWindow notifies) ──
        self.live_lbl = QLabel(
            "Live data: NO.  Press Start on the top bar and wait for packets."
        )
        self.live_lbl.setStyleSheet(
            "color:#ff8080; background:#1a0a14; border:1px solid #602030;"
            "border-radius:5px; padding:4px 8px; font-size:11px;"
        )
        self.live_lbl.setWordWrap(True)
        root.addWidget(self.live_lbl)

        # ── subject row ───────────────────────────────────────────────
        subj_row = QHBoxLayout()
        subj_row.addWidget(self._small("Subject:"))
        self.subject_in = QLineEdit("default")
        self.subject_in.setFixedWidth(200)
        subj_row.addWidget(self.subject_in)
        subj_row.addStretch()
        root.addLayout(subj_row)

        # ── Step 1 card ───────────────────────────────────────────────
        step1_box = self._card(
            title="Step 1 — Stand still (capture personal pressure range)",
            title_color="#ff9060",
            desc=(
                "Have the wearer stand naturally upright, weight on both feet. "
                f"Hold ~{self.STEP1_SECONDS:.0f} s.  The lowest ADC values seen "
                "here become each pressure pad's personal \"fully loaded\" reference."
            ),
        )
        s1 = step1_box.layout()
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        self.s1_start_btn = QPushButton("Start Step 1")
        self.s1_start_btn.setCursor(Qt.PointingHandCursor)
        self.s1_start_btn.clicked.connect(self._start_step1)
        self.s1_finish_btn = QPushButton("Finish Step 1")
        self.s1_finish_btn.setCursor(Qt.PointingHandCursor)
        self.s1_finish_btn.clicked.connect(self._finish_step1)
        row1.addWidget(self.s1_start_btn)
        row1.addWidget(self.s1_finish_btn)
        row1.addStretch()
        s1.addLayout(row1)

        self.s1_progress = QProgressBar()
        self.s1_progress.setRange(0, 100)
        self.s1_progress.setTextVisible(True)
        self.s1_progress.setFormat("%p%")
        self.s1_progress.setMinimumHeight(18)
        s1.addWidget(self.s1_progress)

        self.s1_count_lbl = QLabel("frames captured: 0 / 0")
        self.s1_count_lbl.setStyleSheet("color:#a0a0c0; font-size:11px;")
        s1.addWidget(self.s1_count_lbl)
        root.addWidget(step1_box)

        # ── Step 2 card ───────────────────────────────────────────────
        step2_box = self._card(
            title="Step 2 — Bend knee ~90° (capture personal stretch range)",
            title_color="#60c0ff",
            desc=(
                "Have the wearer flex BOTH knees to roughly 90°, or as far as comfortable. "
                f"Hold ~{self.STEP2_SECONDS:.0f} s.  The lowest ADC values here become each "
                "knee's personal \"maximum bend\"."
            ),
        )
        s2 = step2_box.layout()
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.s2_start_btn = QPushButton("Start Step 2")
        self.s2_start_btn.setCursor(Qt.PointingHandCursor)
        self.s2_start_btn.clicked.connect(self._start_step2)
        self.s2_finish_btn = QPushButton("Finish Step 2")
        self.s2_finish_btn.setCursor(Qt.PointingHandCursor)
        self.s2_finish_btn.clicked.connect(self._finish_step2)
        row2.addWidget(self.s2_start_btn)
        row2.addWidget(self.s2_finish_btn)
        row2.addStretch()
        s2.addLayout(row2)

        self.s2_progress = QProgressBar()
        self.s2_progress.setRange(0, 100)
        self.s2_progress.setTextVisible(True)
        self.s2_progress.setFormat("%p%")
        self.s2_progress.setMinimumHeight(18)
        s2.addWidget(self.s2_progress)

        self.s2_count_lbl = QLabel("frames captured: 0 / 0")
        self.s2_count_lbl.setStyleSheet("color:#a0a0c0; font-size:11px;")
        s2.addWidget(self.s2_count_lbl)
        root.addWidget(step2_box)

        # ── save + reset row ──────────────────────────────────────────
        save_row = QHBoxLayout()
        save_row.setSpacing(10)
        self.save_btn = QPushButton("Save personal_calibration.json + reload")
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.clicked.connect(self._save)
        self.reset_btn = QPushButton("Reset wizard")
        self.reset_btn.setObjectName("stopBtn")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self._reset)
        save_row.addWidget(self.save_btn)
        save_row.addWidget(self.reset_btn)
        save_row.addStretch()
        root.addLayout(save_row)

        # ── log text box (events / errors) — small fixed height ──────
        log_hdr = QLabel("Log")
        log_hdr.setStyleSheet(
            "color:#a0a0c0;font-size:11px;font-weight:bold;margin-top:4px;"
        )
        root.addWidget(log_hdr)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(90)
        self.log.setMaximumHeight(140)
        self.log.setStyleSheet(
            "QTextEdit{background:#0a0a1a;color:#c0d0ff;"
            "font-family:'Consolas','Courier New',monospace;font-size:11px;"
            "border:1px solid #1c1c3a;border-radius:6px;}"
        )
        root.addWidget(self.log)

        # ── calibration summary preview (bigger, monospace, scrollable) ─
        preview_hdr = QLabel("Calibration summary (populated after Finish Step 2)")
        preview_hdr.setStyleSheet(
            "color:#a0a0c0;font-size:11px;font-weight:bold;margin-top:4px;"
        )
        root.addWidget(preview_hdr)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(180)
        self.preview.setStyleSheet(
            "QTextEdit{background:#0a0a1a;color:#a0ffa0;"
            "font-family:'Consolas','Courier New',monospace;font-size:11px;"
            "border:1px solid #1c1c3a;border-radius:6px;}"
        )
        root.addWidget(self.preview)

        # Trailing stretch: lets the scroll area show the last widgets
        # fully without them being forced to fill all vertical space.
        root.addStretch(0)

    @staticmethod
    def _card(title: str, title_color: str, desc: str) -> QFrame:
        box = QFrame()
        box.setObjectName("card")
        box.setStyleSheet(
            "#card{background:#12122a;border:1px solid #28284e;border-radius:8px;}"
        )
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        t = QLabel(title)
        t.setStyleSheet(
            f"color:{title_color};font-size:13px;font-weight:bold;"
        )
        lay.addWidget(t)
        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet("color:#9090b0;font-size:11px;")
        lay.addWidget(d)
        return box

    @staticmethod
    def _small(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#707090; font-size:11px;")
        return lbl

    # ═══════════════════ PUBLIC API (MainWindow) ═══════════════════════
    def feed(self, raw_l: tuple, raw_r: tuple) -> None:
        """Push one synced bilateral frame to the active phase, if any."""
        if self._calibrator is None or not self._live_ok:
            return
        phase = self._calibrator.phase
        if phase not in ("STEP1_STANDING", "STEP2_KNEE_BEND"):
            return
        if raw_l is None or raw_r is None:
            return
        raw8 = [
            float(raw_l[0]), float(raw_l[1]), float(raw_l[2]), float(raw_l[3]),
            float(raw_r[0]), float(raw_r[1]), float(raw_r[2]), float(raw_r[3]),
        ]
        self._calibrator.feed(raw8)
        self._refresh_progress_labels()
        # Auto-enable Finish when the target is hit, and log once.
        if phase == "STEP1_STANDING" and self._calibrator.step1_ready:
            self.s1_finish_btn.setEnabled(True)
            if "step1" not in self._step_ready_notified:
                self._step_ready_notified.add("step1")
                self._log("Step 1 ready — target frames reached.  "
                          "You can press Finish Step 1 now.")
        if phase == "STEP2_KNEE_BEND" and self._calibrator.step2_ready:
            self.s2_finish_btn.setEnabled(True)
            if "step2" not in self._step_ready_notified:
                self._step_ready_notified.add("step2")
                self._log("Step 2 ready — target frames reached.  "
                          "You can press Finish Step 2 now.")

    def set_live_state(self, is_live: bool) -> None:
        """Called by MainWindow when bilateral live packets start / stop."""
        self._live_ok = bool(is_live)
        if self._live_ok:
            self.live_lbl.setText(
                "Live data: YES — bilateral packets arriving."
            )
            self.live_lbl.setStyleSheet(
                "color:#a0ffa0; background:#0a1a14; border:1px solid #306040;"
                "border-radius:5px; padding:4px 8px; font-size:11px;"
            )
        else:
            self.live_lbl.setText(
                "Live data: NO.  Press Start on the top bar and wait for packets."
            )
            self.live_lbl.setStyleSheet(
                "color:#ff8080; background:#1a0a14; border:1px solid #602030;"
                "border-radius:5px; padding:4px 8px; font-size:11px;"
            )

    # ═══════════════════ WIZARD ACTIONS ════════════════════════════════
    def _start_step1(self):
        if not self._live_ok:
            self._warn_no_live()
            return
        self._step_ready_notified.clear()
        self._calibrator = personal_calib.OnlineCalibrator(
            sample_hz=10,
            step1_seconds=self.STEP1_SECONDS,
            step2_seconds=self.STEP2_SECONDS,
        )
        try:
            self._calibrator.start_step1()
        except RuntimeError as exc:
            self._error("Step 1", str(exc))
            return
        self.s1_progress.setValue(0)
        self.s2_progress.setValue(0)
        self.s1_start_btn.setEnabled(False)
        self.s1_finish_btn.setEnabled(False)   # enabled once min_samples reached
        self.s2_start_btn.setEnabled(False)
        self.s2_finish_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_lbl.setText(
            "Capturing Step 1 — stand still.  Progress updates on every synced frame."
        )
        self._refresh_progress_labels()
        self._log("Step 1 started.")

    def _finish_step1(self):
        if self._calibrator is None:
            return
        try:
            self._calibrator.finish_step1()
        except RuntimeError as exc:
            self._error("Step 1", str(exc))
            return
        self.s1_progress.setValue(100)
        self.s1_finish_btn.setEnabled(False)
        self.s1_start_btn.setEnabled(False)
        self.s2_start_btn.setEnabled(True)
        self.s2_finish_btn.setEnabled(False)
        self.status_lbl.setText(
            "Step 1 done — ask wearer to bend knee ~90° and press Start Step 2."
        )
        self._log(f"Step 1 finished "
                  f"({self._calibrator.step1_sample_count} frames captured).")

    def _start_step2(self):
        if self._calibrator is None:
            self._error("Step 2", "Please complete Step 1 first.")
            return
        if self._calibrator.phase != "STEP1_DONE":
            self._error(
                "Step 2",
                "Step 1 has not been finished yet — press Finish Step 1 first.",
            )
            return
        if not self._live_ok:
            self._warn_no_live()
            return
        try:
            self._calibrator.start_step2()
        except RuntimeError as exc:
            self._error("Step 2", str(exc))
            return
        self.s2_progress.setValue(0)
        self.s2_start_btn.setEnabled(False)
        self.s2_finish_btn.setEnabled(False)   # enabled once min_samples reached
        self.status_lbl.setText("Capturing Step 2 — hold knee bent ~90°.")
        self._refresh_progress_labels()
        self._log("Step 2 started.")

    def _finish_step2(self):
        if self._calibrator is None:
            return
        try:
            self._calibrator.finish_step2()
        except RuntimeError as exc:
            self._error("Step 2", str(exc))
            return
        self.s2_progress.setValue(100)
        self.s2_finish_btn.setEnabled(False)
        self.s2_start_btn.setEnabled(False)

        subject = self.subject_in.text().strip() or "default"
        if not self.subject_in.text().strip():
            self.subject_in.setText(subject)   # persist auto-fill to UI
        try:
            self._last_preview = self._calibrator.finalize(subject=subject)
        except RuntimeError as exc:
            self._error("Finalize", str(exc))
            return

        self.preview.setPlainText(self._last_preview.summary())
        self.save_btn.setEnabled(True)
        self.status_lbl.setText(
            "Calibration ready — review min/max above, then click Save."
        )
        self._log(f"Step 2 finished "
                  f"({self._calibrator.step2_sample_count} frames captured). "
                  f"Subject = {subject!r}.  Click Save to persist + reload.")

    def _save(self):
        if self._last_preview is None:
            self._error("Save", "Nothing to save — finish Step 2 first.")
            return
        try:
            abs_path = self._last_preview.save_json(self.JSON_PATH)
        except Exception as exc:  # noqa: BLE001
            self._error("Save", f"Could not write JSON: {exc}")
            return
        self.calibration_saved.emit(abs_path)
        self._log(f"Calibration saved → {abs_path}")
        self._log(self._last_preview.summary())
        QMessageBox.information(
            self, "Calibration saved",
            f"Saved personal calibration to:\n{abs_path}\n\n"
            "The live recognizer has been reloaded; new frames use these "
            "personal ranges immediately.",
        )
        self._refresh_existing_calibration_status()
        self.save_btn.setEnabled(False)

    def _reset(self):
        self._calibrator = None
        self._last_preview = None
        self._step_ready_notified.clear()
        for bar in (self.s1_progress, self.s2_progress):
            bar.setValue(0)
        self._apply_initial_button_state()
        self.preview.clear()
        self.log.clear()
        self.s1_count_lbl.setText("frames captured: 0 / 0")
        self.s2_count_lbl.setText("frames captured: 0 / 0")
        self._refresh_existing_calibration_status()
        self._log("Wizard reset.  Press Start Step 1 when live data is flowing.")

    # ═══════════════════ HELPERS ══════════════════════════════════════
    def _apply_initial_button_state(self):
        self.s1_start_btn.setEnabled(True)
        self.s1_finish_btn.setEnabled(False)
        self.s2_start_btn.setEnabled(False)
        self.s2_finish_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _refresh_progress_labels(self):
        if self._calibrator is None:
            return
        s1_n = self._calibrator.step1_sample_count
        s1_t = self._calibrator.step1_target_samples
        s2_n = self._calibrator.step2_sample_count
        s2_t = self._calibrator.step2_target_samples
        self.s1_count_lbl.setText(
            f"frames captured: {s1_n} / {s1_t}  "
            f"(minimum {self._calibrator.step1_min_samples} to finish)"
        )
        self.s2_count_lbl.setText(
            f"frames captured: {s2_n} / {s2_t}  "
            f"(minimum {self._calibrator.step2_min_samples} to finish)"
        )
        phase = self._calibrator.phase
        if phase == "STEP1_STANDING":
            pct = int(round(100.0 * min(1.0, s1_n / max(1, s1_t))))
            self.s1_progress.setValue(pct)
            # Allow manual finish as soon as the minimum sample count is hit.
            if s1_n >= self._calibrator.step1_min_samples:
                self.s1_finish_btn.setEnabled(True)
        elif phase == "STEP2_KNEE_BEND":
            pct = int(round(100.0 * min(1.0, s2_n / max(1, s2_t))))
            self.s2_progress.setValue(pct)
            if s2_n >= self._calibrator.step2_min_samples:
                self.s2_finish_btn.setEnabled(True)

    def _warn_no_live(self):
        QMessageBox.warning(
            self, "No live data",
            "No live bilateral data yet.  "
            "Please start both MCU streams first:\n\n"
            "  1. Click Start on the top-right of the main window.\n"
            "  2. Wait until the two status dots turn green AND the live "
            "sensor readout is updating.\n"
            "  3. Then come back here and press Start Step 1.",
        )
        self._log("Blocked: no live bilateral data yet.")

    def _error(self, title: str, msg: str):
        QMessageBox.warning(self, title, msg)
        self._log(f"{title}: {msg}")

    def _log(self, msg: str):
        # Prepend newest entries so users don't have to scroll; keep short.
        stamped = time.strftime("%H:%M:%S") + "  " + msg
        existing = self.log.toPlainText()
        combined = stamped + ("\n" + existing if existing else "")
        # cap log length
        if len(combined) > 8000:
            combined = combined[:8000]
        self.log.setPlainText(combined)

    def _refresh_existing_calibration_status(self):
        if not os.path.isfile(self.JSON_PATH):
            self.status_lbl.setText(
                f"No calibration on disk yet.  Will write to {self.JSON_PATH} on save."
            )
            return
        try:
            existing = personal_calib.PersonalCalibration.load_json(self.JSON_PATH)
            self.status_lbl.setText(
                f"Loaded existing calibration (source={existing.source}, "
                f"subject={existing.subject}).  Re-running this wizard will "
                "overwrite the JSON on save."
            )
            if not self.preview.toPlainText():
                self.preview.setPlainText(existing.summary())
        except Exception as exc:  # noqa: BLE001
            self.status_lbl.setText(f"(existing JSON unreadable: {exc})")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW — slim live display + cascade HUD + label strip + calibration tab
# ═══════════════════════════════════════════════════════════════════════════

_LABEL_DEFS: list[tuple[str, str]] = [
    ("Fwd Walk",    "WALKING_FORWARD"),
    ("Bwd Walk",    "WALKING_BACKWARD"),
    ("Stairs Up",   "STAIRS_UP"),
    ("Stairs Down", "STAIRS_DOWN"),
    ("Sitting",     "SITTING_NORMAL"),
    ("Cross-Leg",   "SITTING_CROSSLEGGED"),
    ("Sit→Stand",   "SIT_TO_STAND"),
    ("Upright",     "STANDING_UPRIGHT"),
    ("Lean Left",   "STANDING_LEFT_LEAN"),
    ("Lean Right",  "STANDING_RIGHT_LEAN"),
    ("Unknown",     "UNKNOWN"),
]

_STATE_COLORS = {
    "WALKING_FORWARD":      "#ff6b6b",
    "WALKING_BACKWARD":     "#ff4757",
    "STAIRS_UP":            "#a78bfa",
    "STAIRS_DOWN":          "#8b6fe0",
    "SITTING_NORMAL":       "#4ecdc4",
    "SITTING_CROSSLEGGED":  "#45b7aa",
    "SIT_TO_STAND":         "#ff9f43",
    "STANDING_UPRIGHT":     "#ffd93d",
    "STANDING_LEFT_LEAN":   "#f0c929",
    "STANDING_RIGHT_LEAN":  "#e6b800",
    "UNKNOWN":              "#606080",
}


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Foot Pressure — Slim Live Monitor")
        # The Calibration tab is the tallest content in the app; use a larger
        # minimum so none of its buttons ever get clipped on Windows at
        # 100–150% display scaling.  The panel is also wrapped in a scroll
        # area so going below this is still usable (just with a scrollbar).
        self.setMinimumSize(1200, 820)
        self.resize(1320, 900)
        self.setStyleSheet(DARK_STYLE)

        # Flipped to True by _on_socket_data on first synced bilateral frame,
        # and pushed into the CalibrationPanel so its Start buttons are only
        # effective when real data is flowing.
        self._live_bilateral_ok: bool = False

        # TCP + CSV state
        self._thread_l: Optional[SocketThread] = None
        self._thread_r: Optional[SocketThread] = None
        self._csv_labeled_f = None
        self._csv_writer_labeled = None
        self._csv_raw_f = None
        self._csv_writer_raw = None
        self._csv_labeled_path = ""
        self._csv_raw_path = ""

        self._data_n = 0
        self.current_label = "UNKNOWN"
        self.recognizer = OnlineRecognizer()

        # Latest synced frame for each foot
        self._last_left: Optional[FootFour] = None
        self._last_right: Optional[FootFour] = None

        # Bilateral pairing buffers: (pc_mono_s, ts_mcu, toe, forefoot, heel, knee)
        self.left_buffer: deque = deque(maxlen=200)
        self.right_buffer: deque = deque(maxlen=200)

        self._conn_l = "disconnected"
        self._conn_r = "disconnected"

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background-color:#0a0a1a;")
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(16, 12, 16, 8)
        vbox.setSpacing(10)

        vbox.addLayout(self._build_top_bar())

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFixedHeight(1)
        sep.setStyleSheet("background:#1c1c3a;")
        vbox.addWidget(sep)

        vbox.addWidget(self._build_live_panel())
        # Cascade HUD: natural size, doesn't need to grow.
        vbox.addWidget(self._build_cascade_panel(), stretch=0)
        # Tabs (Data capture / Inference / Calibration) get the remaining
        # vertical space so Calibration always has room for every control.
        vbox.addWidget(self._build_tabs(), stretch=1)

        self.statusBar().showMessage("Ready — set IP / ports, then Start.")

    def _build_top_bar(self) -> QHBoxLayout:
        top = QHBoxLayout(); top.setSpacing(10)

        title = QLabel("Foot Pressure — Dual MCU (L:5000 / R:6000)")
        title.setStyleSheet("font-size:18px; font-weight:bold; color:#ff6b6b;")
        top.addWidget(title)
        top.addStretch()

        top.addWidget(self._tiny("L:"))
        self.dot_l = StatusDot(); top.addWidget(self.dot_l)
        top.addWidget(self._tiny("R:"))
        self.dot_r = StatusDot(); top.addWidget(self.dot_r)

        self.conn_lbl = QLabel("L:off  R:off")
        self.conn_lbl.setStyleSheet(
            "color:#ff4455; font-size:12px; margin-right:10px;"
        )
        top.addWidget(self.conn_lbl)

        top.addWidget(self._tiny("IP:"))
        self.ip_in = QLineEdit("172.20.10.9")
        self.ip_in.setFixedWidth(130); top.addWidget(self.ip_in)

        top.addWidget(self._tiny("Port L:"))
        self.port_left_in = QLineEdit("5000")
        self.port_left_in.setFixedWidth(56); top.addWidget(self.port_left_in)

        top.addWidget(self._tiny("Port R:"))
        self.port_right_in = QLineEdit("6000")
        self.port_right_in.setFixedWidth(56); top.addWidget(self.port_right_in)

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedWidth(88)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self._start)
        top.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setFixedWidth(74)
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        top.addWidget(self.stop_btn)

        return top

    def _build_live_panel(self) -> QFrame:
        """Live ADC readout for all 8 channels (updated only on synced frames)."""
        box = QFrame(); box.setObjectName("card")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        head = QLabel("Live sensor stream (raw ADC, updated on bilateral sync)")
        head.setStyleSheet("color:#9090b0; font-size:11px; font-weight:bold;")
        lay.addWidget(head)

        grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(2)

        left_tag = QLabel("LEFT")
        left_tag.setStyleSheet("color:#ff6b6b; font-size:12px; font-weight:bold;")
        grid.addWidget(left_tag, 0, 0, Qt.AlignCenter)
        self.ch_l_toe      = ChannelReadout("Toe",      "#ffd93d")
        self.ch_l_forefoot = ChannelReadout("Forefoot", "#ff6b6b")
        self.ch_l_heel     = ChannelReadout("Heel",     "#4ecdc4")
        self.ch_l_knee     = ChannelReadout("Knee",     "#a78bfa", is_knee=True)
        grid.addWidget(self.ch_l_toe,      0, 1)
        grid.addWidget(self.ch_l_forefoot, 0, 2)
        grid.addWidget(self.ch_l_heel,     0, 3)
        grid.addWidget(self.ch_l_knee,     0, 4)

        right_tag = QLabel("RIGHT")
        right_tag.setStyleSheet("color:#ff6b6b; font-size:12px; font-weight:bold;")
        grid.addWidget(right_tag, 1, 0, Qt.AlignCenter)
        self.ch_r_toe      = ChannelReadout("Toe",      "#ffd93d")
        self.ch_r_forefoot = ChannelReadout("Forefoot", "#ff6b6b")
        self.ch_r_heel     = ChannelReadout("Heel",     "#4ecdc4")
        self.ch_r_knee     = ChannelReadout("Knee",     "#a78bfa", is_knee=True)
        grid.addWidget(self.ch_r_toe,      1, 1)
        grid.addWidget(self.ch_r_forefoot, 1, 2)
        grid.addWidget(self.ch_r_heel,     1, 3)
        grid.addWidget(self.ch_r_knee,     1, 4)

        for col in range(1, 5):
            grid.setColumnStretch(col, 1)

        lay.addLayout(grid)

        self.stream_meta_lbl = QLabel(
            "Packets: 0  |  Bilateral: no  |  CSV: not recording"
        )
        self.stream_meta_lbl.setStyleSheet("color:#707090; font-size:11px;")
        lay.addWidget(self.stream_meta_lbl)

        return box

    def _build_cascade_panel(self) -> QFrame:
        """Shows Layer-1 (knee gate), Layer-2 (motion/static), final RF label + step counters."""
        box = QFrame(); box.setObjectName("card")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(8)

        self.state_lbl = QLabel("STATE: —")
        self.state_lbl.setAlignment(Qt.AlignCenter)
        self.state_lbl.setStyleSheet(
            "color:#ff6b6b; font-size:22px; font-weight:bold;"
        )
        lay.addWidget(self.state_lbl)

        self.layer1_lbl = QLabel("Layer 1 (knee gate): —")
        self.layer1_lbl.setAlignment(Qt.AlignCenter)
        self.layer1_lbl.setStyleSheet(
            "color:#9ad0ff; font-size:13px; font-weight:bold;"
        )
        lay.addWidget(self.layer1_lbl)

        self.layer2_lbl = QLabel("Layer 2 (motion / static): —")
        self.layer2_lbl.setAlignment(Qt.AlignCenter)
        self.layer2_lbl.setStyleSheet(
            "color:#ffd43b; font-size:13px; font-weight:bold;"
        )
        lay.addWidget(self.layer2_lbl)

        self.rf_lbl = QLabel("RF: —")
        self.rf_lbl.setAlignment(Qt.AlignCenter)
        self.rf_lbl.setStyleSheet(
            "color:#a0ffa0; font-size:12px;"
        )
        lay.addWidget(self.rf_lbl)

        self.counters_lbl = QLabel(
            "Total: 0   Fwd: 0 | Bwd: 0 | Up: 0 | Down: 0   Sit→Stand: —"
        )
        self.counters_lbl.setAlignment(Qt.AlignCenter)
        self.counters_lbl.setStyleSheet(
            "color:#d0d0e0; font-size:12px;"
        )
        lay.addWidget(self.counters_lbl)

        return box

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        # Guarantee a sensible minimum height for whichever tab is selected,
        # so the Calibration panel's content always has room to render the
        # Step 1/2 buttons and progress bars without being clipped.
        tabs.setMinimumHeight(360)
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Data-capture tab with label strip
        cap_tab = QWidget()
        cap_lay = QVBoxLayout(cap_tab)
        cap_lay.setContentsMargins(10, 8, 10, 8)
        cap_lay.setSpacing(6)

        self.capture_label_line = QLabel("Current Label: UNKNOWN")
        self.capture_label_line.setAlignment(Qt.AlignCenter)
        self.capture_label_line.setStyleSheet(
            "color:#ff6b6b; font-size:14px; font-weight:bold;"
        )
        cap_lay.addWidget(self.capture_label_line)

        hint = QLabel(
            "Select a label, then press Start on the top bar.  This writes both "
            "labeled and raw CSVs into saving_data/ at 10 Hz bilateral-synced rate."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#707090; font-size:11px;")
        cap_lay.addWidget(hint)

        lbl_grid = QGridLayout(); lbl_grid.setSpacing(6)
        for i, (btn_text, lbl_value) in enumerate(_LABEL_DEFS):
            b = QPushButton(btn_text)
            b.setObjectName("labelBtn")
            b.setMinimumHeight(28)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _chk, v=lbl_value: self.set_label(v))
            lbl_grid.addWidget(b, i // 6, i % 6)
        cap_lay.addLayout(lbl_grid)
        tabs.addTab(cap_tab, "Data capture")

        # Inference tab (just explanatory)
        inf_tab = QWidget()
        inf_lay = QVBoxLayout(inf_tab)
        inf_hint = QLabel(
            "Inference mode — the cascade HUD above updates on every synced frame.\n"
            "Starting from this tab streams without writing any CSV."
        )
        inf_hint.setWordWrap(True)
        inf_hint.setStyleSheet("color:#9090b0; font-size:12px;")
        inf_lay.addWidget(inf_hint)
        inf_lay.addStretch()
        tabs.addTab(inf_tab, "Inference")

        # Calibration tab
        self.calib_panel = CalibrationPanel()
        self.calib_panel.calibration_saved.connect(self._on_calibration_saved)
        tabs.addTab(self.calib_panel, "Calibration")

        self._mode_tabs = tabs
        return tabs

    # ── helpers ─────────────────────────────────────────────────
    @staticmethod
    def _tiny(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#707090; font-size:13px;")
        return lbl

    def set_label(self, label: str):
        self.current_label = label
        self.capture_label_line.setText(f"Current Label: {label}")

    # ── start / stop ────────────────────────────────────────────
    def _start(self):
        host = self.ip_in.text().strip()
        try:
            port_l = int(self.port_left_in.text().strip())
            port_r = int(self.port_right_in.text().strip())
        except ValueError:
            self.statusBar().showMessage("Port L / Port R must be integers")
            return

        # CSV only when launched from Data capture tab
        self._csv_labeled_path = ""
        self._csv_raw_path = ""
        self._csv_labeled_f = self._csv_raw_f = None
        self._csv_writer_labeled = self._csv_writer_raw = None

        if self._mode_tabs.currentIndex() == 0:
            os.makedirs("saving_data", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_labeled_path = f"saving_data/sensor_data_dual_labeled_{ts}.csv"
            self._csv_raw_path = f"saving_data/sensor_data_dual_raw_{ts}.csv"
            self._csv_labeled_f = open(self._csv_labeled_path, "w", newline="", encoding="utf-8")
            self._csv_raw_f     = open(self._csv_raw_path,     "w", newline="", encoding="utf-8")
            self._csv_writer_labeled = csv.writer(self._csv_labeled_f)
            self._csv_writer_raw     = csv.writer(self._csv_raw_f)
            hdr_8 = [
                "Timestamp",
                "L_Toe", "L_Forefoot", "L_Heel", "L_Knee",
                "R_Toe", "R_Forefoot", "R_Heel", "R_Knee",
            ]
            self._csv_writer_labeled.writerow([*hdr_8, "Label"])
            self._csv_writer_raw.writerow(hdr_8)

        self._last_left = None
        self._last_right = None
        self._data_n = 0
        self.left_buffer.clear()
        self.right_buffer.clear()
        # Calibration panel should only allow progress advance once real
        # bilateral packets arrive — reset the flag on every Start.
        self._live_bilateral_ok = False
        if getattr(self, "calib_panel", None) is not None:
            self.calib_panel.set_live_state(False)

        self._thread_l = SocketThread(host, port_l, "L", parent=self)
        self._thread_r = SocketThread(host, port_r, "R", parent=self)
        for th in (self._thread_l, self._thread_r):
            th.data_received.connect(self._on_socket_data)
            th.status_changed.connect(self._on_status)
        self._thread_l.connection_state.connect(self._on_conn_state_l)
        self._thread_r.connection_state.connect(self._on_conn_state_r)
        self._thread_l.start()
        self._thread_r.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.ip_in.setEnabled(False)
        self.port_left_in.setEnabled(False)
        self.port_right_in.setEnabled(False)

        if self._csv_writer_labeled is not None:
            self.statusBar().showMessage(
                f"Recording labeled: {self._csv_labeled_path}  "
                f"raw: {self._csv_raw_path}"
            )
        else:
            self.statusBar().showMessage(
                "Streaming (Inference) — no CSV; switch to Data capture to record."
            )

    def _stop(self):
        for th in (self._thread_l, self._thread_r):
            if th is not None:
                th.stop()
                th.wait(3000)
        self._thread_l = self._thread_r = None

        for fobj in (self._csv_labeled_f, self._csv_raw_f):
            if fobj is not None:
                try:
                    fobj.close()
                except Exception:
                    pass
        self._csv_labeled_f = self._csv_raw_f = None
        self._csv_writer_labeled = self._csv_writer_raw = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.ip_in.setEnabled(True)
        self.port_left_in.setEnabled(True)
        self.port_right_in.setEnabled(True)
        self.dot_l.set_state("disconnected")
        self.dot_r.set_state("disconnected")
        self._conn_l = self._conn_r = "disconnected"
        self._sync_conn_label()
        self._live_bilateral_ok = False
        if getattr(self, "calib_panel", None) is not None:
            self.calib_panel.set_live_state(False)
        self.statusBar().showMessage("Stopped listening (both MCUs).")

    # ── socket slots ────────────────────────────────────────────
    def _on_socket_data(
        self,
        side: str,
        ts_ms: object,
        toe: float,
        forefoot: float,
        heel: float,
        knee: float,
    ) -> None:
        """Pair L/R by PC receive time; only synced pairs drive UI + recognizer + CSV."""
        try:
            ts_ms_i = int(ts_ms)
        except (TypeError, ValueError):
            return

        self._data_n += 1

        try:
            pc_now = time.monotonic()
            sample = (pc_now, ts_ms_i, toe, forefoot, heel, knee)
            synced = False
            l4 = r4 = None
            ts_avg_mcu = 0.0

            if side == "L":
                self.left_buffer.append(sample)
                best, best_d = None, float("inf")
                for r in self.right_buffer:
                    d = abs(r[0] - pc_now)
                    if d < best_d:
                        best_d, best = d, r
                if best is not None and best_d <= SYNC_PC_MAX_S:
                    self.left_buffer.pop()
                    _deque_remove_one(self.right_buffer, best)
                    ts_avg_mcu = (ts_ms_i + best[1]) / 2.0
                    l4 = (toe, forefoot, heel, knee)
                    r4 = (best[2], best[3], best[4], best[5])
                    synced = True
            else:
                self.right_buffer.append(sample)
                best, best_d = None, float("inf")
                for le in self.left_buffer:
                    d = abs(le[0] - pc_now)
                    if d < best_d:
                        best_d, best = d, le
                if best is not None and best_d <= SYNC_PC_MAX_S:
                    self.right_buffer.pop()
                    _deque_remove_one(self.left_buffer, best)
                    ts_avg_mcu = (best[1] + ts_ms_i) / 2.0
                    l4 = (best[2], best[3], best[4], best[5])
                    r4 = (toe, forefoot, heel, knee)
                    synced = True

            if synced and l4 is not None and r4 is not None:
                self._last_left = l4
                self._last_right = r4
                # Flip the live flag once — tell the Calibration panel it is
                # now safe to accept Start Step 1 / Step 2 and that its
                # progress bar will advance against real data.
                if not self._live_bilateral_ok:
                    self._live_bilateral_ok = True
                    if self.calib_panel is not None:
                        self.calib_panel.set_live_state(True)
                self._append_csv_synced(ts_avg_mcu, l4, r4)
                self._update_live_readouts(l4, r4)
                if self.calib_panel is not None:
                    self.calib_panel.feed(l4, r4)
                self._run_recognizer_and_hud()
                self._refresh_stream_meta()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"data error: {exc}")
            traceback.print_exc()

    def _append_csv_synced(
        self,
        ts_ms: float,
        l4: Tuple[float, float, float, float],
        r4: Tuple[float, float, float, float],
    ) -> None:
        if self._csv_writer_labeled is None or self._csv_writer_raw is None:
            return
        ts_str = str(int(round(ts_ms)))
        row8 = [
            ts_str,
            str(l4[0]), str(l4[1]), str(l4[2]), str(l4[3]),
            str(r4[0]), str(r4[1]), str(r4[2]), str(r4[3]),
        ]
        self._csv_writer_labeled.writerow([*row8, self.current_label])
        self._csv_writer_raw.writerow(row8)
        if self._csv_labeled_f:
            self._csv_labeled_f.flush()
        if self._csv_raw_f:
            self._csv_raw_f.flush()

    def _update_live_readouts(
        self,
        l4: Tuple[float, float, float, float],
        r4: Tuple[float, float, float, float],
    ) -> None:
        self.ch_l_toe.set_raw(l4[0])
        self.ch_l_forefoot.set_raw(l4[1])
        self.ch_l_heel.set_raw(l4[2])
        self.ch_l_knee.set_raw(l4[3])
        self.ch_r_toe.set_raw(r4[0])
        self.ch_r_forefoot.set_raw(r4[1])
        self.ch_r_heel.set_raw(r4[2])
        self.ch_r_knee.set_raw(r4[3])

    def _refresh_stream_meta(self):
        bilat = "yes" if (self._last_left and self._last_right) else "no"
        if self._csv_writer_labeled is None:
            csv_part = "CSV: not recording"
        else:
            csv_part = f"CSV: {os.path.basename(self._csv_labeled_path)}"
        self.stream_meta_lbl.setText(
            f"Packets: {self._data_n}  |  Bilateral: {bilat}  |  {csv_part}"
        )

    # ── recognizer / HUD ────────────────────────────────────────
    def _run_recognizer_and_hud(self):
        l4 = foot_tuple_for_recognizer(self._last_left)
        r4 = foot_tuple_for_recognizer(self._last_right)
        if l4 is not None and r4 is not None:
            out = self.recognizer.update_bilateral(l4, r4)
        elif l4 is not None:
            out = self.recognizer.update(*l4)
        elif r4 is not None:
            out = self.recognizer.update(*r4)
        else:
            return

        state = out["state"]
        counters = out["counters"]
        debug = out.get("debug", {})

        clr = _STATE_COLORS.get(state, "#d0d0e0")
        self.state_lbl.setText(f"STATE: {state}")
        self.state_lbl.setStyleSheet(
            f"color:{clr}; font-size:22px; font-weight:bold;"
        )

        l1 = debug.get("layer1_branch", "—")
        ph = debug.get("knee_gate_phase", "—")
        min_raw = debug.get("knee_min_raw", "—")
        th = debug.get("knee_gate_straight_th", "—")
        self.layer1_lbl.setText(
            f"Layer 1: {l1}   phase={ph}   "
            f"min_raw={min_raw}   straight_th<{th}"
        )

        l2 = debug.get("layer2_subbranch", "—")
        l2r = debug.get("layer2_reason", "—")
        self.layer2_lbl.setText(f"Layer 2: {l2}   ({l2r})")

        brk = debug.get("branch_rf_key", "—")
        rf_proba = debug.get("rf_proba", "—")
        rf_reject = debug.get("rf_reject", "—")
        ml_label = debug.get("ml_label", "—")
        self.rf_lbl.setText(
            f"RF: {brk}   predicted={ml_label}   "
            f"proba={rf_proba}   rejected={rf_reject}"
        )

        sts = out["sts_last_duration_s"]
        sts_text = f"{sts:.2f} s" if sts is not None else "—"
        self.counters_lbl.setText(
            f"Total: {counters['total_steps']}   "
            f"Fwd: {counters['forward_steps']} | "
            f"Bwd: {counters['backward_steps']} | "
            f"Up: {counters['up_steps']} | "
            f"Down: {counters['down_steps']}   "
            f"Sit→Stand: {sts_text}"
        )

    # ── connection status ──────────────────────────────────────
    def _on_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def _sync_conn_label(self):
        m = {"disconnected": "off", "waiting": "wait", "connected": "ok"}
        self.conn_lbl.setText(
            f"L:{m.get(self._conn_l, self._conn_l)}  "
            f"R:{m.get(self._conn_r, self._conn_r)}"
        )

    def _on_conn_state_l(self, state: str):
        self.dot_l.set_state(state)
        self._conn_l = state
        self._sync_conn_label()

    def _on_conn_state_r(self, state: str):
        self.dot_r.set_state(state)
        self._conn_r = state
        self._sync_conn_label()

    # ── calibration hot-reload ─────────────────────────────────
    def _on_calibration_saved(self, json_path: str):
        """Build a fresh recognizer with the just-saved calibration so it is live instantly."""
        try:
            self.recognizer = OnlineRecognizer(calibration=json_path)
            self.statusBar().showMessage(
                f"Reloaded recognizer with calibration {os.path.basename(json_path)}"
            )
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Reload recognizer failed: {exc}")
            traceback.print_exc()

    # ── cleanup ────────────────────────────────────────────────
    def closeEvent(self, event):
        for th in (self._thread_l, self._thread_r):
            if th is not None:
                th.stop()
                th.wait(3000)
        for fobj in (self._csv_labeled_f, self._csv_raw_f):
            if fobj is not None:
                try:
                    fobj.close()
                except Exception:
                    pass
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    palette = app.palette()
    palette.setColor(palette.Window,          QColor("#0a0a1a"))
    palette.setColor(palette.WindowText,      QColor("#e0e0e0"))
    palette.setColor(palette.Base,            QColor("#10102a"))
    palette.setColor(palette.AlternateBase,   QColor("#14143a"))
    palette.setColor(palette.ToolTipBase,     QColor("#10102a"))
    palette.setColor(palette.ToolTipText,     QColor("#e0e0e0"))
    palette.setColor(palette.Text,            QColor("#e0e0e0"))
    palette.setColor(palette.Button,          QColor("#1a1a3a"))
    palette.setColor(palette.ButtonText,      QColor("#e0e0e0"))
    palette.setColor(palette.Highlight,       QColor("#ff6b6b"))
    palette.setColor(palette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
