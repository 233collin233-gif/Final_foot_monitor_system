"""PyQt: dual TCP insoles, optional CSV capture, calibration wizard, live inference."""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
import socket
import sys
import time
import traceback
import warnings
from collections import deque
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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

SENSOR_MAX = 4095.0
SYNC_PC_MAX_S = 0.15

TAB_LISTEN = 0
TAB_CAPTURE = 1
TAB_INFERENCE = 2
TAB_CALIBRATION = 3


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


FootThree = Tuple[Optional[float], Optional[float], Optional[float]]


def parse_foot_json_line(line: str) -> Optional[FootThree]:
    """Parse one MCU line: JSON {forefoot,heel,knee} or three comma floats."""
    s = line.strip().lstrip("\ufeff")
    if not s:
        return None
    if s.startswith("{"):
        try:
            d = json.loads(s)
        except json.JSONDecodeError:
            return None
        def _f(key: str) -> Optional[float]:
            for k, v in d.items():
                if str(k).lower() == key:
                    try:
                        x = float(v)
                        if math.isnan(x) or math.isinf(x):
                            return None
                        return x
                    except (TypeError, ValueError):
                        return None
            return None
        ff, h, k = _f("forefoot"), _f("heel"), _f("knee")
        if ff is None or h is None or k is None:
            return None
        return (ff, h, k)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 3:
        return None
    vals: list[float] = []
    for p in parts[:3]:
        try:
            v = float(p)
            if math.isnan(v) or math.isinf(v):
                return None
            vals.append(v)
        except ValueError:
            return None
    return (vals[0], vals[1], vals[2])


def _deque_remove_one(d: deque, item: tuple) -> bool:
    before = len(d)
    ml = d.maxlen
    tmp = deque((x for x in d if x != item), maxlen=ml)
    if len(tmp) == before:
        return False
    d.clear()
    d.extend(tmp)
    return True


def foot_tuple_for_recognizer(tup: Optional[FootThree]) -> Optional[Tuple[float, float, float]]:
    if tup is None:
        return None
    return tuple(SENSOR_MAX if x is None else float(x) for x in tup)


class SocketThread(QThread):

    data_received = pyqtSignal(str, float, float, float)
    status_changed = pyqtSignal(str)
    connection_state = pyqtSignal(str)

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
                            if lo.startswith("{") and "forefoot" in lo:
                                header_skipped = True
                                continue
                            if "forefoot" in lo and "knee" in lo and not lo.startswith("{"):
                                header_skipped = True
                                continue
                            header_skipped = True
                        parsed = parse_foot_json_line(line)
                        if parsed is not None:
                            s_ff, s_heel, s_knee = parsed
                            self.data_received.emit(
                                self.side, s_ff, s_heel, s_knee,
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


class StatusDot(QWidget):
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


class ForeHeelHeatmapBlock(QWidget):
    def __init__(self, fore_color: str, heel_color: str, parent=None):
        super().__init__(parent)
        self._fore_color = QColor(fore_color)
        self._heel_color = QColor(heel_color)
        self._i_fore = -1.0
        self._i_heel = -1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._strip = _ForeHeelHeatStrip(self._fore_color, self._heel_color, parent=self)
        self._strip.setMinimumHeight(56)
        self._strip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(self._strip, 0)

        self._detail = QLabel("Forefoot —\nHeel —")
        self._detail.setTextFormat(Qt.PlainText)
        self._detail.setStyleSheet("color:#a8a8c8; font-size:10px; line-height:115%;")
        self._detail.setWordWrap(True)
        root.addWidget(self._detail)

    def set_sensors(
        self,
        raw_forefoot: Optional[float],
        raw_heel: Optional[float],
    ) -> None:
        def intensity(raw: Optional[float]) -> float:
            if raw is None:
                return -1.0
            return max(0.0, min(1.0, (SENSOR_MAX - float(raw)) / SENSOR_MAX))

        a, b = intensity(raw_forefoot), intensity(raw_heel)
        if a != self._i_fore or b != self._i_heel:
            self._i_fore, self._i_heel = a, b
            self._strip.set_intensity(a, b)

        if raw_forefoot is None and raw_heel is None:
            text = "Forefoot —\nHeel —"
        else:
            lines: list[str] = []
            for label, raw, i in (
                ("Forefoot", raw_forefoot, a),
                ("Heel", raw_heel, b),
            ):
                if raw is None:
                    lines.append(f"{label} —")
                else:
                    ri = int(round(float(raw)))
                    pct = int(round(100.0 * max(0.0, i))) if i >= 0 else 0
                    lines.append(f"{label}  {ri}  load {pct:3d}%")
            text = "\n".join(lines)
        if text != getattr(self, "_last_detail", None):
            self._last_detail = text
            self._detail.setText(text)


class _ForeHeelHeatStrip(QWidget):
    def __init__(self, c_fore: QColor, c_heel: QColor, parent=None):
        super().__init__(parent)
        self._c_fore = c_fore
        self._c_heel = c_heel
        self._a = -1.0
        self._b = -1.0

    def set_intensity(self, a: float, b: float) -> None:
        if a == self._a and b == self._b:
            return
        self._a, self._b = a, b
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        gap = 5
        half_w = max(8, (w - gap) // 2)
        bg = QColor("#141428")
        border = QColor("#2a2a48")
        for idx, (intens, c, x0) in enumerate(
            (
                (self._a, self._c_fore, 0),
                (self._b, self._c_heel, half_w + gap),
            )
        ):
            p.setPen(border)
            p.setBrush(QBrush(bg))
            p.drawRoundedRect(x0, 0, half_w, h, 7, 7)
            if intens < 0.0:
                p.setPen(QColor("#505070"))
                p.setBrush(Qt.NoBrush)
                p.drawText(
                    x0 + 4,
                    0,
                    half_w - 8,
                    h,
                    Qt.AlignCenter,
                    "—" if idx == 0 else "—",
                )
                continue
            fill = QColor(c)
            fill.setAlpha(35 + int(220 * intens))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(fill))
            inset = 3
            p.drawRoundedRect(
                x0 + inset,
                inset,
                half_w - 2 * inset,
                h - 2 * inset,
                5,
                5,
            )
        p.end()


class ChannelReadout(QLabel):
    def __init__(self, title: str, accent: str, is_knee: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        self._accent = accent
        self._is_knee = is_knee
        self.setAlignment(Qt.AlignCenter)
        self.setTextFormat(Qt.PlainText)
        self.setMinimumWidth(110)
        self.setStyleSheet(
            "color:#ffffff; font-size:13px; font-weight:bold; line-height:115%;"
        )
        self._last_text: Optional[str] = None
        self.set_raw(None)

    def set_raw(self, raw: Optional[float]):
        if raw is None:
            text = f"{self._title}\n—\nno data"
        else:
            raw_i = int(round(float(raw)))
            if self._is_knee:
                bend = max(0.0, (SENSOR_MAX - raw)) / SENSOR_MAX * 100.0
                tag = f"bend {bend:4.0f}%"
            else:
                load = max(0.0, (SENSOR_MAX - raw)) / SENSOR_MAX * 100.0
                tag = f"load {load:4.0f}%"
            text = f"{self._title}\n{raw_i}\n{tag}"
        if text != self._last_text:
            self._last_text = text
            self.setText(text)


class CalibrationPanel(QWidget):
    STEP1_SECONDS = 5.0
    STEP2_SECONDS = 5.0
    JSON_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        personal_calib.DEFAULT_CALIBRATION_FILENAME,
    )

    calibration_saved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calibrator: Optional[personal_calib.OnlineCalibrator] = None
        self._last_preview: Optional[personal_calib.PersonalCalibration] = None
        self._live_ok: bool = False
        self._step_ready_notified: set[str] = set()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._build_ui()
        self._apply_initial_button_state()
        self._refresh_existing_calibration_status()

    def _build_ui(self):
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

        banner = QLabel(
            "Usage: connect MCUs → on <b>Listen</b>, <b>Data capture</b>, or <b>Inference</b> "
            "tab press <b>Start</b> → wait for bilateral packets → open this <b>Calibration</b> "
            "tab → Step 1 → Step 2 → Save"
        )
        banner.setTextFormat(Qt.RichText)
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel{background:#241830;border:1px solid #ff9060;border-radius:6px;"
            "color:#ffd0a0;font-size:12px;padding:8px 10px;}"
        )
        root.addWidget(banner)

        header = QLabel(
            "Personal calibration wizard — aligns per-channel [min, max] ADC to this "
            "wearer.  Layer-1 knee 4095 rule is unaffected; only ML feature scale changes."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color:#d0d0e0; font-size:12px;")
        root.addWidget(header)

        self.status_lbl = QLabel("—")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#9090b0; font-size:11px;")
        root.addWidget(self.status_lbl)

        self.live_lbl = QLabel(
            "Live data: NO.  Press Start on the top bar and wait for packets."
        )
        self.live_lbl.setStyleSheet(
            "color:#ff8080; background:#1a0a14; border:1px solid #602030;"
            "border-radius:5px; padding:4px 8px; font-size:11px;"
        )
        self.live_lbl.setWordWrap(True)
        root.addWidget(self.live_lbl)

        subj_row = QHBoxLayout()
        subj_row.addWidget(self._small("Subject:"))
        self.subject_in = QLineEdit("default")
        self.subject_in.setFixedWidth(200)
        subj_row.addWidget(self.subject_in)
        subj_row.addStretch()
        root.addLayout(subj_row)

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

    def feed(self, raw_l: tuple, raw_r: tuple) -> None:
        if self._calibrator is None or not self._live_ok:
            return
        phase = self._calibrator.phase
        if phase not in ("STEP1_STANDING", "STEP2_KNEE_BEND"):
            return
        if raw_l is None or raw_r is None:
            return
        raw6 = [
            float(raw_l[0]), float(raw_l[1]), float(raw_l[2]),
            float(raw_r[0]), float(raw_r[1]), float(raw_r[2]),
        ]
        self._calibrator.feed(raw6)
        self._refresh_progress_labels()
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
        self.s1_finish_btn.setEnabled(False)
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
        self.s2_finish_btn.setEnabled(False)
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
            self.subject_in.setText(subject)
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
        stamped = time.strftime("%H:%M:%S") + "  " + msg
        existing = self.log.toPlainText()
        combined = stamped + ("\n" + existing if existing else "")
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
        self.setWindowTitle("Foot Pressure — Live Monitor")
        self.setMinimumSize(1200, 820)
        self.resize(1320, 900)
        self.setStyleSheet(DARK_STYLE)

        self._live_bilateral_ok: bool = False

        self._thread_l: Optional[SocketThread] = None
        self._thread_r: Optional[SocketThread] = None
        self._csv_labeled_f = None
        self._csv_writer_labeled = None
        self._csv_raw_f = None
        self._csv_writer_raw = None
        self._csv_labeled_path = ""
        self._csv_raw_path = ""

        self._data_n = 0
        self._csv_flush_every = 20
        self._csv_since_flush = 0
        self.current_label = "UNKNOWN"
        self.recognizer = OnlineRecognizer()

        self._ui_refresh_min_interval = 0.20
        self._last_ui_refresh_t = 0.0
        self._last_hud_state = None

        self._last_left: Optional[FootThree] = None
        self._last_right: Optional[FootThree] = None

        self.left_buffer: deque = deque(maxlen=200)
        self.right_buffer: deque = deque(maxlen=200)

        self._conn_l = "disconnected"
        self._conn_r = "disconnected"
        self._session_mode: str = "idle"

        self._build_ui()

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

        self.mode_banner = QLabel(
            "Mode: idle — choose <b>Listen</b>, <b>Inference</b>, or <b>Data capture</b>, then Start."
        )
        self.mode_banner.setTextFormat(Qt.RichText)
        self.mode_banner.setWordWrap(True)
        self.mode_banner.setStyleSheet(
            "color:#8a8ab8; font-size:12px; padding:4px 2px; background:transparent;"
        )
        vbox.addWidget(self.mode_banner)

        vbox.addWidget(self._build_live_panel())
        vbox.addWidget(self._build_cascade_panel(), stretch=0)
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
        box = QFrame()
        box.setObjectName("card")
        self.live_sensor_card = box
        lay = QVBoxLayout(box)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        head = QLabel(
            "Live sensors (bilateral) — top row = LEFT foot, bottom row = RIGHT foot | "
            "forefoot/heel heatmap + knee (raw 0–4095 when synced)"
        )
        head.setStyleSheet("color:#9090b0; font-size:11px; font-weight:bold;")

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        left_tag = QLabel("LEFT")
        left_tag.setStyleSheet("color:#ff6b6b; font-size:12px; font-weight:bold;")
        grid.addWidget(left_tag, 0, 0, Qt.AlignTop)
        self.heat_l = ForeHeelHeatmapBlock("#ff6b6b", "#4ecdc4")
        grid.addWidget(self.heat_l, 0, 1)
        self.ch_l_knee = ChannelReadout("Knee", "#a78bfa", is_knee=True)
        grid.addWidget(self.ch_l_knee, 0, 2, Qt.AlignTop)

        right_tag = QLabel("RIGHT")
        right_tag.setStyleSheet("color:#ff6b6b; font-size:12px; font-weight:bold;")
        grid.addWidget(right_tag, 1, 0, Qt.AlignTop)
        self.heat_r = ForeHeelHeatmapBlock("#ff6b6b", "#4ecdc4")
        grid.addWidget(self.heat_r, 1, 1)
        self.ch_r_knee = ChannelReadout("Knee", "#a78bfa", is_knee=True)
        grid.addWidget(self.ch_r_knee, 1, 2, Qt.AlignTop)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        self.inference_live_block = QWidget()
        ib_lay = QVBoxLayout(self.inference_live_block)
        ib_lay.setContentsMargins(0, 0, 0, 0)
        ib_lay.setSpacing(6)
        ib_lay.addWidget(head)
        ib_lay.addLayout(grid)
        lay.addWidget(self.inference_live_block)

        self.stream_meta_lbl = QLabel(
            "Packets: 0  |  Bilateral: no  |  CSV: not recording"
        )
        self.stream_meta_lbl.setStyleSheet("color:#707090; font-size:11px;")
        lay.addWidget(self.stream_meta_lbl)

        return box

    def _build_cascade_panel(self) -> QFrame:
        box = QFrame()
        box.setObjectName("card")
        self.hud_card = box
        lay = QVBoxLayout(box)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(8)

        self.state_lbl = QLabel("STATE: —")
        self.state_lbl.setAlignment(Qt.AlignCenter)
        self.state_lbl.setStyleSheet(
            "color:#ff6b6b; font-size:22px; font-weight:bold;"
        )
        lay.addWidget(self.state_lbl)

        self.layer1_lbl = QLabel("Knee mode: —")
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
        tabs.setMinimumHeight(360)
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        listen_tab = QWidget()
        listen_lay = QVBoxLayout(listen_tab)
        listen_lay.setContentsMargins(10, 8, 10, 8)
        listen_hint = QLabel(
            "<b>Listen</b> — verify TCP and bilateral ADC. Uses the live sensor card "
            "above only: <b>no</b> activity recognition and <b>no</b> CSV. "
            "Lowest CPU. Stop before switching to Inference or Data capture."
        )
        listen_hint.setTextFormat(Qt.RichText)
        listen_hint.setWordWrap(True)
        listen_hint.setStyleSheet("color:#9090b0; font-size:12px;")
        listen_lay.addWidget(listen_hint)
        listen_lay.addStretch()
        tabs.addTab(listen_tab, "Listen")

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
            "<b>Data capture</b> — pick a label, then <b>Start</b>. Writes labeled + raw CSV "
            "into saving_data/ at 10 Hz. <b>Stop</b> before switching to Listen or Inference."
        )
        hint.setTextFormat(Qt.RichText)
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

        inf_tab = QWidget()
        inf_lay = QVBoxLayout(inf_tab)
        inf_hint = QLabel(
            "<b>Inference</b> — activity model + step counts. <b>No CSV.</b> "
            "Default: STATE + counters in the card above. Optional: full sensors + RF debug."
        )
        inf_hint.setTextFormat(Qt.RichText)
        inf_hint.setWordWrap(True)
        inf_hint.setStyleSheet("color:#9090b0; font-size:12px;")
        inf_lay.addWidget(inf_hint)

        self.infer_show_extras = QCheckBox(
            "Show foot heatmaps, packet line, Knee / Layer2 / RF (heavier UI)"
        )
        self.infer_show_extras.setChecked(False)
        self.infer_show_extras.setStyleSheet("color:#c0c0d8; font-size:12px;")
        self.infer_show_extras.toggled.connect(self._on_inference_display_toggled)
        inf_lay.addWidget(self.infer_show_extras)

        inf_lay.addStretch()
        tabs.addTab(inf_tab, "Inference")

        self.calib_panel = CalibrationPanel()
        self.calib_panel.calibration_saved.connect(self._on_calibration_saved)
        tabs.addTab(self.calib_panel, "Calibration")

        self._mode_tabs = tabs
        return tabs

    @staticmethod
    def _tiny(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#707090; font-size:13px;")
        return lbl

    def set_label(self, label: str):
        self.current_label = label
        self.capture_label_line.setText(f"Current Label: {label}")

    def _streaming(self) -> bool:
        return self._thread_l is not None

    def _on_inference_display_toggled(self, _checked: bool = False) -> None:
        if not self._streaming() or self._session_mode != "inference":
            return
        self._apply_inference_display_prefs()

    def _lock_mode_tabs_for_session(self) -> None:
        cur = self._mode_tabs.currentIndex()
        for i in (TAB_LISTEN, TAB_CAPTURE, TAB_INFERENCE):
            self._mode_tabs.setTabEnabled(i, i == cur)
        self._mode_tabs.setTabEnabled(TAB_CALIBRATION, True)

    def _unlock_mode_tabs(self) -> None:
        for i in range(self._mode_tabs.count()):
            self._mode_tabs.setTabEnabled(i, True)

    def _apply_full_hud_visibility(self) -> None:
        self.hud_card.setVisible(True)
        self.live_sensor_card.setVisible(True)
        self.inference_live_block.setVisible(True)
        self.stream_meta_lbl.setVisible(True)
        self.layer1_lbl.setVisible(True)
        self.layer2_lbl.setVisible(True)
        self.rf_lbl.setVisible(True)

    def _apply_inference_display_prefs(self) -> None:
        self.hud_card.setVisible(True)
        extras = self.infer_show_extras.isChecked()
        self.live_sensor_card.setVisible(extras)
        self.layer1_lbl.setVisible(extras)
        self.layer2_lbl.setVisible(extras)
        self.rf_lbl.setVisible(extras)

    def _apply_session_layout_after_start(self) -> None:
        if self._session_mode == "listen":
            self.hud_card.setVisible(False)
            self.live_sensor_card.setVisible(True)
            self.inference_live_block.setVisible(True)
            self.stream_meta_lbl.setVisible(True)
        elif self._session_mode == "inference":
            self._apply_inference_display_prefs()
        else:
            self._apply_full_hud_visibility()

    def _start(self):
        host = self.ip_in.text().strip()
        try:
            port_l = int(self.port_left_in.text().strip())
            port_r = int(self.port_right_in.text().strip())
        except ValueError:
            self.statusBar().showMessage("Port L / Port R must be integers")
            return

        tab_i = self._mode_tabs.currentIndex()
        if tab_i == TAB_CALIBRATION:
            QMessageBox.information(
                self,
                "Choose a stream mode",
                "Start streaming from the **Listen**, **Data capture**, or **Inference** tab.\n\n"
                "Use **Calibration** while data is already streaming to run the wizard.",
            )
            return

        if tab_i == TAB_LISTEN:
            self._session_mode = "listen"
        elif tab_i == TAB_CAPTURE:
            self._session_mode = "capture"
        else:
            self._session_mode = "inference"

        self._csv_labeled_path = ""
        self._csv_raw_path = ""
        self._csv_labeled_f = self._csv_raw_f = None
        self._csv_writer_labeled = self._csv_writer_raw = None

        if self._session_mode == "capture":
            os.makedirs("saving_data/labeled", exist_ok=True)
            os.makedirs("saving_data/raw", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_labeled_path = f"saving_data/labeled/sensor_data_dual_labeled_{ts}.csv"
            self._csv_raw_path = f"saving_data/raw/sensor_data_dual_raw_{ts}.csv"
            self._csv_labeled_f = open(self._csv_labeled_path, "w", newline="", encoding="utf-8")
            self._csv_raw_f = open(self._csv_raw_path, "w", newline="", encoding="utf-8")
            self._csv_writer_labeled = csv.writer(self._csv_labeled_f)
            self._csv_writer_raw = csv.writer(self._csv_raw_f)
            hdr_6 = [
                "L_Forefoot", "L_Heel", "L_Knee",
                "R_Forefoot", "R_Heel", "R_Knee",
            ]
            self._csv_writer_labeled.writerow([*hdr_6, "Label"])
            self._csv_writer_raw.writerow(hdr_6)

        if self._session_mode == "listen":
            self.mode_banner.setText(
                "Active: <b>Listen</b> — bilateral ADC above · <b>no</b> CSV · <b>no</b> ML "
                "(stop, then switch tab to run inference or recording)."
            )
        elif self._session_mode == "inference":
            self.mode_banner.setText(
                "Active: <b>Inference</b> — STATE + steps · optional sensor/RF via checkbox · "
                "<b>no</b> CSV."
            )
        else:
            self.mode_banner.setText(
                "Active: <b>Data capture</b> — writing labeled + raw CSV · full HUD + ML."
            )

        if self._session_mode in ("inference", "capture"):
            self.recognizer = OnlineRecognizer()

        self._last_left = None
        self._last_right = None
        self._data_n = 0
        self.left_buffer.clear()
        self.right_buffer.clear()
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

        self._lock_mode_tabs_for_session()
        self._apply_session_layout_after_start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.ip_in.setEnabled(False)
        self.port_left_in.setEnabled(False)
        self.port_right_in.setEnabled(False)

        if self._session_mode == "capture":
            self.statusBar().showMessage(
                f"Recording: {self._csv_labeled_path}  |  raw: {self._csv_raw_path}"
            )
        elif self._session_mode == "inference":
            self.statusBar().showMessage("Inference — ML on, no CSV.")
        else:
            self.statusBar().showMessage("Listen — live data only, ML off, no CSV.")

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
        self._unlock_mode_tabs()
        self._session_mode = "idle"
        self._apply_full_hud_visibility()
        self.mode_banner.setText(
            "Mode: idle — choose <b>Listen</b> (data only), <b>Inference</b> (ML), or "
            "<b>Data capture</b> (CSV), then Start."
        )
        self.statusBar().showMessage("Stopped.")

    def _on_socket_data(
        self,
        side: str,
        forefoot: float,
        heel: float,
        knee: float,
    ) -> None:
        self._data_n += 1

        try:
            pc_now = time.monotonic()
            sample = (pc_now, forefoot, heel, knee)
            synced = False
            l3 = r3 = None

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
                    l3 = (forefoot, heel, knee)
                    r3 = (best[1], best[2], best[3])
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
                    l3 = (best[1], best[2], best[3])
                    r3 = (forefoot, heel, knee)
                    synced = True

            if synced and l3 is not None and r3 is not None:
                self._last_left = l3
                self._last_right = r3
                if not self._live_bilateral_ok:
                    self._live_bilateral_ok = True
                    if self.calib_panel is not None:
                        self.calib_panel.set_live_state(True)
                if self._session_mode == "capture":
                    self._append_csv_synced(l3, r3)
                if self.calib_panel is not None:
                    self.calib_panel.feed(l3, r3)

                hud_out = None
                if self._session_mode in ("inference", "capture"):
                    hud_out = self._run_recognizer(l3, r3)

                now = time.monotonic()
                state_changed = (
                    hud_out is not None
                    and hud_out["state"] != self._last_hud_state
                )
                if (
                    state_changed
                    or (now - self._last_ui_refresh_t) >= self._ui_refresh_min_interval
                ):
                    self._last_ui_refresh_t = now
                    if self.live_sensor_card.isVisible():
                        self._update_live_readouts(l3, r3)
                    if hud_out is not None:
                        self._update_hud(hud_out)
                        self._last_hud_state = hud_out["state"]
                    elif self._session_mode == "listen":
                        self._last_hud_state = None
                    self._refresh_stream_meta()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"data error: {exc}")
            traceback.print_exc()

    def _append_csv_synced(
        self,
        l3: Tuple[float, float, float],
        r3: Tuple[float, float, float],
    ) -> None:
        if self._csv_writer_labeled is None or self._csv_writer_raw is None:
            return
        row6 = [
            str(l3[0]), str(l3[1]), str(l3[2]),
            str(r3[0]), str(r3[1]), str(r3[2]),
        ]
        self._csv_writer_labeled.writerow([*row6, self.current_label])
        self._csv_writer_raw.writerow(row6)
        self._csv_since_flush += 1
        if self._csv_since_flush >= self._csv_flush_every:
            self._csv_since_flush = 0
            if self._csv_labeled_f:
                self._csv_labeled_f.flush()
            if self._csv_raw_f:
                self._csv_raw_f.flush()

    def _update_live_readouts(
        self,
        l3: Tuple[float, float, float],
        r3: Tuple[float, float, float],
    ) -> None:
        self.heat_l.set_sensors(l3[0], l3[1])
        self.heat_r.set_sensors(r3[0], r3[1])
        self.ch_l_knee.set_raw(l3[2])
        self.ch_r_knee.set_raw(r3[2])

    def _refresh_stream_meta(self):
        bilat = "yes" if (self._last_left and self._last_right) else "no"
        sm = self._session_mode if self._streaming() else "idle"
        if sm == "capture" and self._csv_writer_labeled is not None:
            csv_part = f"CSV: {os.path.basename(self._csv_labeled_path)}"
        elif sm == "listen":
            csv_part = "CSV: off"
        elif sm == "inference":
            csv_part = "CSV: off"
        else:
            csv_part = "CSV: —"
        self.stream_meta_lbl.setText(
            f"[{sm}]  Packets: {self._data_n}  |  Bilateral: {bilat}  |  {csv_part}"
        )

    def _run_recognizer(self, l3_raw, r3_raw):
        l3 = foot_tuple_for_recognizer(l3_raw)
        r3 = foot_tuple_for_recognizer(r3_raw)
        if l3 is not None and r3 is not None:
            return self.recognizer.update_bilateral(l3, r3)
        if l3 is not None:
            return self.recognizer.update_single(*l3)
        if r3 is not None:
            return self.recognizer.update_single(*r3)
        return None

    def _update_hud(self, out: dict):
        state = out["state"]
        counters = out["counters"]
        debug = out.get("debug", {})

        clr = _STATE_COLORS.get(state, "#d0d0e0")
        self.state_lbl.setText(f"STATE: {state}")
        self.state_lbl.setStyleSheet(
            f"color:{clr}; font-size:22px; font-weight:bold;"
        )

        km = debug.get("knee_mode", "—")
        br = debug.get("branch", km)
        self.layer1_lbl.setText(
            f"Knee mode: {km}   branch={br}"
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

    def _on_calibration_saved(self, json_path: str):
        try:
            self.recognizer = OnlineRecognizer(calibration=json_path)
            self.statusBar().showMessage(
                f"Reloaded recognizer with calibration {os.path.basename(json_path)}"
            )
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Reload recognizer failed: {exc}")
            traceback.print_exc()

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


def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    try:
        os.chdir(_here)
    except OSError as exc:
        print("Warning: could not chdir to script folder:", _here, exc, file=sys.stderr)

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
