"""
Per-subject range calibration for the 8 raw ADC channels
(L_Toe, L_Forefoot, L_Heel, L_Knee, R_Toe, R_Forefoot, R_Heel, R_Knee).

Why a dedicated module:
    The 6 pressure pads have wildly different sensitivity per foot / per person
    (heel is usually heaviest, toe is lightest; foot weight, insole fit, socks…),
    and the 2 knee stretch sensors have person-specific slack.
    Feeding a *global* linear map (e.g. raw/4095) into the RF bakes this
    subject bias into the tree splits.  A one-shot personal calibration gives
    every subject the same dynamic range [0, 4095] in their *own* usable band.

Two cooperating code paths share the exact same output schema:

    A.  **Offline auto-calibration** from existing labelled CSVs
        (``OfflineAutoCalibrator``).  Uses the raw data we already have:
          - pressure pads  →  peaks during WALKING_* / STAIRS_* / STANDING_*
                              (min-raw = peak load = personal full load)
                              and upper percentile of idle / sitting frames
                              (max-raw = unloaded reference).
          - knee stretch   →  deepest raw low during SITTING_* frames
                              (min-raw = fully bent)
                              and 4095 ceiling   (max-raw = straight).

    B.  **Online UI calibration** (``OnlineCalibrator``) driven by the upper-
        computer GUI.  Two guided phases asked of the wearer:
          Step 1  —  stand still (a few seconds) → captures *pressure
                     working range* (load distribution under body weight).
          Step 2  —  fully bend the knee ≈ 90° → captures *stretch minimum*
                     (maximum elongation for this subject).

Both paths produce a :class:`PersonalCalibration` with identical fields,
so downstream code (feature engineering, training, inference, UI) never
needs to know which source was used.  Drop-in switch = change the JSON file
path or call one function instead of the other.

The 4 random forests NEVER look at anything else than the output of this
module + the EWMA ``adaptive_preprocessing``.  The Layer-1 knee gate
(``KNEE_RAW_STRAIGHT_TH = 3500`` in ``realtime_recognizer.py``) still reads
*raw* ADC so the strict 4095 rule is unchanged by calibration.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

# Keep this list *identical* to adaptive_preprocessing.CHANNEL_NAMES_DUAL —
# several modules depend on this exact order (raw → 8-vector indexing).
CHANNEL_NAMES_DUAL = (
    "L_Toe", "L_Forefoot", "L_Heel", "L_Knee",
    "R_Toe", "R_Forefoot", "R_Heel", "R_Knee",
)
PRESSURE_IDX = (0, 1, 2, 4, 5, 6)       # toe/ff/heel, both feet
KNEE_IDX = (3, 7)                       # L_Knee, R_Knee
SENSOR_MAX = 4095.0

# ── Tunables for the offline auto-calibrator ────────────────────────────────

# TODO_PARAM: upper percentile on raw used as "unloaded / idle reference"
#             for pressure pads.  Raw is higher when the pad is *released*.
OFFLINE_PRESSURE_MAX_PCT = 97.0
# TODO_PARAM: lower percentile on raw used as "fully loaded personal peak".
#             Raw is lower when the pad is *pressed hard*.
OFFLINE_PRESSURE_MIN_PCT = 1.0
# TODO_PARAM: minimum span required on a pressure channel.  If the CSV shows
#             almost no dynamic range (e.g. sensor dead), we fall back to
#             the safe default [0, 4095] for that channel.
OFFLINE_PRESSURE_MIN_SPAN = 120.0

# TODO_PARAM: knee-stretch "deep bend" floor — percentile on SITTING frames.
OFFLINE_KNEE_MIN_PCT = 1.0
# TODO_PARAM: safety clamp so the knee personal range always reaches the rail.
OFFLINE_KNEE_MAX = 4095.0
# TODO_PARAM: minimum span required on the knee channel (avoid degenerate calib).
OFFLINE_KNEE_MIN_SPAN = 300.0

# Labels considered "loaded" for each sensor family during offline fitting.
_LOADED_LABELS_PRESSURE = frozenset({
    "WALKING_FORWARD", "WALKING_BACKWARD",
    "STAIRS_UP", "STAIRS_DOWN",
    "STANDING_UPRIGHT", "STANDING_LEFT_LEAN", "STANDING_RIGHT_LEAN",
})
_LOADED_LABELS_KNEE = frozenset({
    "SITTING_NORMAL", "SITTING_CROSSLEGGED",
    "STAIRS_UP", "STAIRS_DOWN",
})
_IDLE_LABELS_PRESSURE = frozenset({
    "SITTING_NORMAL", "SITTING_CROSSLEGGED",
})


# ── Core dataclass ──────────────────────────────────────────────────────────

CalibrationSource = Literal["offline_auto_csv", "ui_online_two_step", "manual"]


@dataclass
class PersonalCalibration:
    """Per-channel personal working range **plus** global statistics.

    Legacy fields (always present):

    min_raw : np.ndarray, shape (8,)
        Lower ADC bound per channel, **inclusive**.
          * pressure pads → raw at full personal load (heavy step / stand).
          * knee stretch → raw at maximum bend.
    max_raw : np.ndarray, shape (8,)
        Upper ADC bound per channel, **inclusive**.
          * pressure pads → raw when pad is unloaded (foot lifted).
          * knee stretch → raw when leg is straight (usually 4095 rail).

    **Global statistics** (may be ``None`` → fall back to online EWMA).  When
    present, they freeze the :class:`adaptive_preprocessing.AdaptiveSensorPreprocessor`'s
    learned state so *every* CSV / every live frame is preprocessed against
    the exact same numbers.  All are shape ``(8,)``.

    baseline_raw : np.ndarray
        Per-channel *reference* raw (pad unloaded / knee straight).  Used as
        the fixed baseline in ``press_mag = baseline_raw - raw`` and
        ``baseline_removed = raw - baseline_raw``.
    press_min, press_max : np.ndarray
        Global min / max of ``press_mag``, used as the denominator in the
        ``relative_pressure_ratio`` instead of a per-trajectory rolling span.
    press_mean, press_std : np.ndarray
        Global mean / std of ``press_mag``, used in
        ``adaptive_zscore = (press_mag - press_mean) / press_std`` instead of
        the online EWMA estimate.

    ``has_global_stats`` returns True iff all five global fields are set —
    the adaptive bank treats that as "freeze everything, use these numbers".
    """

    min_raw: np.ndarray
    max_raw: np.ndarray
    source: CalibrationSource = "manual"
    subject: str = "default"
    created_at: float = field(default_factory=time.time)
    notes: str = ""
    baseline_raw: Optional[np.ndarray] = None
    press_min: Optional[np.ndarray] = None
    press_max: Optional[np.ndarray] = None
    press_mean: Optional[np.ndarray] = None
    press_std: Optional[np.ndarray] = None

    # ── construction helpers ────────────────────────────────────────────
    def __post_init__(self) -> None:
        self.min_raw = np.asarray(self.min_raw, dtype=np.float64).reshape(8)
        self.max_raw = np.asarray(self.max_raw, dtype=np.float64).reshape(8)
        self._sanity_clamp()
        for name in ("baseline_raw", "press_min", "press_max",
                     "press_mean", "press_std"):
            val = getattr(self, name)
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).reshape(8)
                setattr(self, name, arr)

    def _sanity_clamp(self) -> None:
        """Guarantee ``max > min + 1`` on every channel."""
        for i in range(8):
            lo = float(self.min_raw[i])
            hi = float(self.max_raw[i])
            lo = max(0.0, min(lo, SENSOR_MAX - 1.0))
            hi = max(lo + 1.0, min(hi, SENSOR_MAX))
            self.min_raw[i] = lo
            self.max_raw[i] = hi

    # ── global-stats helpers ───────────────────────────────────────────
    @property
    def has_global_stats(self) -> bool:
        """True iff every channel has a globally-computed baseline / range / moments."""
        return all(
            getattr(self, k) is not None
            for k in ("baseline_raw", "press_min", "press_max",
                      "press_mean", "press_std")
        )

    def to_channel_seeds(self) -> Optional[list]:
        """Return one ``adaptive_preprocessing.ChannelSeed`` per channel, or ``None``.

        Returns ``None`` when :attr:`has_global_stats` is ``False`` so the
        caller can transparently fall back to the EWMA-only bank.
        """
        if not self.has_global_stats:
            return None
        # Lazy import to avoid a circular dep at module load time.
        from adaptive_preprocessing import ChannelSeed  # noqa: WPS433
        seeds = []
        for i in range(8):
            seeds.append(ChannelSeed(
                baseline_raw=float(self.baseline_raw[i]),     # type: ignore[index]
                press_min=float(self.press_min[i]),           # type: ignore[index]
                press_max=float(self.press_max[i]),           # type: ignore[index]
                press_mean=float(self.press_mean[i]),         # type: ignore[index]
                press_std=float(self.press_std[i]),           # type: ignore[index]
            ))
        return seeds

    @classmethod
    def identity(cls) -> "PersonalCalibration":
        """``[0, 4095]`` for all 8 channels — equivalent to *no* calibration."""
        return cls(
            min_raw=np.zeros(8, dtype=np.float64),
            max_raw=np.full(8, SENSOR_MAX, dtype=np.float64),
            source="manual",
            subject="identity",
            notes="identity mapping — downstream features see raw ADC unchanged",
        )

    # ── forward normalization  ──────────────────────────────────────────
    def normalize_to_unit(self, raw_8: np.ndarray) -> np.ndarray:
        """Map each channel of ``raw_8`` into ``[0, 1]`` using its personal range.

        Accepts ``(8,)`` or ``(T, 8)``; returns same shape, clipped.
        """
        r = np.asarray(raw_8, dtype=np.float64)
        lo = self.min_raw
        hi = self.max_raw
        out = (r - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)

    def normalize_to_adc(self, raw_8: np.ndarray) -> np.ndarray:
        """Same as :meth:`normalize_to_unit` but re-scaled to ``[0, 4095]``.

        We keep the ADC-like scale because the downstream EWMA preprocessor
        in ``adaptive_preprocessing`` has thresholds (``IDLE_PRESS_GATE_RAW``,
        ``LOW_DYNAMIC_RANGE_TH``) that were tuned against raw ADC amplitudes.
        Staying in that scale means we do *not* have to retune those.
        """
        return self.normalize_to_unit(raw_8) * SENSOR_MAX

    # ── JSON round-trip (one JSON per subject) ──────────────────────────
    def to_dict(self) -> dict:
        out = {
            "channels": list(CHANNEL_NAMES_DUAL),
            "min_raw": [float(x) for x in self.min_raw],
            "max_raw": [float(x) for x in self.max_raw],
            "source": self.source,
            "subject": self.subject,
            "created_at": float(self.created_at),
            "notes": self.notes,
        }
        # Only serialise global stats if we actually have them; older JSONs
        # without these keys continue to load fine (has_global_stats==False).
        for name in ("baseline_raw", "press_min", "press_max",
                     "press_mean", "press_std"):
            val = getattr(self, name)
            if val is not None:
                out[name] = [float(x) for x in np.asarray(val).reshape(8)]
        return out

    def save_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return os.path.abspath(path)

    @classmethod
    def load_json(cls, path: str) -> "PersonalCalibration":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        ch = list(d.get("channels", CHANNEL_NAMES_DUAL))
        if tuple(ch) != CHANNEL_NAMES_DUAL:
            raise ValueError(
                f"Channel layout in {path} does not match expected "
                f"{CHANNEL_NAMES_DUAL}; got {ch}"
            )

        def _maybe_array(key: str) -> Optional[np.ndarray]:
            return (np.asarray(d[key], dtype=np.float64)
                    if key in d and d[key] is not None else None)

        return cls(
            min_raw=np.asarray(d["min_raw"], dtype=np.float64),
            max_raw=np.asarray(d["max_raw"], dtype=np.float64),
            source=d.get("source", "manual"),
            subject=d.get("subject", "default"),
            created_at=float(d.get("created_at", time.time())),
            notes=d.get("notes", ""),
            baseline_raw=_maybe_array("baseline_raw"),
            press_min=_maybe_array("press_min"),
            press_max=_maybe_array("press_max"),
            press_mean=_maybe_array("press_mean"),
            press_std=_maybe_array("press_std"),
        )

    # ── readable summary for notebook / UI ─────────────────────────────
    def summary(self) -> str:
        lines = [
            f"PersonalCalibration(subject={self.subject!r}, source={self.source!r})",
            "  channel         min_raw   max_raw   span",
        ]
        for i, name in enumerate(CHANNEL_NAMES_DUAL):
            lo = float(self.min_raw[i])
            hi = float(self.max_raw[i])
            lines.append(f"  {name:<13} {lo:8.1f}  {hi:8.1f}  {hi - lo:8.1f}")
        if self.has_global_stats:
            lines.append("")
            lines.append("  GLOBAL STATS (used to freeze EWMA baseline / range / z-score):")
            lines.append(
                "  channel         baseline   p_min     p_max     p_mean    p_std"
            )
            for i, name in enumerate(CHANNEL_NAMES_DUAL):
                lines.append(
                    f"  {name:<13} {float(self.baseline_raw[i]):8.1f} "   # type: ignore[index]
                    f"{float(self.press_min[i]):8.1f} "                  # type: ignore[index]
                    f"{float(self.press_max[i]):8.1f} "                  # type: ignore[index]
                    f"{float(self.press_mean[i]):8.1f} "                 # type: ignore[index]
                    f"{float(self.press_std[i]):8.1f}"                   # type: ignore[index]
                )
        else:
            lines.append("")
            lines.append("  (no global stats → adaptive bank will fall back to EWMA)")
        return "\n".join(lines)


# ── Path A: offline auto-calibration from labelled CSV ──────────────────────


@dataclass
class OfflineAutoCalibrator:
    """Scan a labelled ``(T, 8) raw_adc`` + ``labels[T]`` sequence and
    derive a :class:`PersonalCalibration` without any user interaction.

    Pressure pads (toe / ff / heel on both feet):
        * ``min_raw`` ← ``OFFLINE_PRESSURE_MIN_PCT``-percentile of raw
          restricted to *loaded* labels (walking, stairs, standing).
          These are the user's personal heavy-step peaks.
        * ``max_raw`` ← ``OFFLINE_PRESSURE_MAX_PCT``-percentile of raw
          restricted to *idle* labels (sitting) — pad unloaded.

    Knee stretch pads (L_Knee, R_Knee):
        * ``min_raw`` ← ``OFFLINE_KNEE_MIN_PCT``-percentile of raw
          restricted to SITTING_* / STAIRS_* — the deepest personal bend.
        * ``max_raw`` ← ``OFFLINE_KNEE_MAX`` (rail 4095 — straight leg).

    If a label family is absent in the CSV, we fall back to a safe default
    (``[0, 4095]``) for that channel and log a warning in ``.warnings``.
    """

    pressure_min_pct: float = OFFLINE_PRESSURE_MIN_PCT
    pressure_max_pct: float = OFFLINE_PRESSURE_MAX_PCT
    pressure_min_span: float = OFFLINE_PRESSURE_MIN_SPAN
    knee_min_pct: float = OFFLINE_KNEE_MIN_PCT
    knee_min_span: float = OFFLINE_KNEE_MIN_SPAN
    warnings: list[str] = field(default_factory=list)

    def fit(
        self,
        data_t8: np.ndarray,
        labels: Sequence[str],
        *,
        subject: str = "offline_auto",
        notes: str = "",
    ) -> PersonalCalibration:
        data_t8 = np.asarray(data_t8, dtype=np.float64)
        if data_t8.ndim != 2 or data_t8.shape[1] != 8:
            raise ValueError(f"data_t8 must be (T, 8); got {data_t8.shape}")
        if len(labels) != data_t8.shape[0]:
            raise ValueError("len(labels) must equal T")

        labels = np.asarray([str(x).strip().upper() for x in labels])
        min_raw = np.zeros(8, dtype=np.float64)
        max_raw = np.full(8, SENSOR_MAX, dtype=np.float64)

        mask_load_p = np.isin(labels, sorted(_LOADED_LABELS_PRESSURE))
        mask_idle_p = np.isin(labels, sorted(_IDLE_LABELS_PRESSURE))
        mask_knee = np.isin(labels, sorted(_LOADED_LABELS_KNEE))

        for ch in PRESSURE_IDX:
            col = data_t8[:, ch]
            lo_src = col[mask_load_p] if mask_load_p.any() else col
            hi_src = col[mask_idle_p] if mask_idle_p.any() else col
            if lo_src.size < 20:
                self.warnings.append(
                    f"[{CHANNEL_NAMES_DUAL[ch]}] loaded samples <20; "
                    "falling back to global percentile"
                )
                lo_src = col
            if hi_src.size < 20:
                self.warnings.append(
                    f"[{CHANNEL_NAMES_DUAL[ch]}] idle samples <20; "
                    "falling back to global percentile"
                )
                hi_src = col
            lo = float(np.percentile(lo_src, self.pressure_min_pct))
            hi = float(np.percentile(hi_src, self.pressure_max_pct))
            if hi - lo < self.pressure_min_span:
                self.warnings.append(
                    f"[{CHANNEL_NAMES_DUAL[ch]}] personal span "
                    f"{hi - lo:.1f} < {self.pressure_min_span}; "
                    "expanding to default [0, 4095]"
                )
                lo, hi = 0.0, SENSOR_MAX
            min_raw[ch] = lo
            max_raw[ch] = hi

        for ch in KNEE_IDX:
            col = data_t8[:, ch]
            src = col[mask_knee] if mask_knee.any() else col
            if src.size < 20:
                self.warnings.append(
                    f"[{CHANNEL_NAMES_DUAL[ch]}] sitting/stairs samples <20; "
                    "using full column"
                )
                src = col
            lo = float(np.percentile(src, self.knee_min_pct))
            hi = OFFLINE_KNEE_MAX
            if hi - lo < self.knee_min_span:
                self.warnings.append(
                    f"[{CHANNEL_NAMES_DUAL[ch]}] knee span {hi - lo:.1f} < "
                    f"{self.knee_min_span}; expanding to [0, 4095]"
                )
                lo = 0.0
            min_raw[ch] = lo
            max_raw[ch] = hi

        # ── Global statistics across the FULL concatenated dataset ────────
        # baseline_raw := max_raw (unloaded pad / straight knee reference).
        # press_mag   := baseline_raw - raw  (≥ 0 when loaded / bent).
        # We use the WHOLE (T, 8) matrix here — no label mask, no per-file
        # split — so one subject's calibrator sees *exactly the same* stats
        # every CSV is going to be preprocessed against.
        baseline_raw = max_raw.copy()
        press_mag = baseline_raw[None, :] - data_t8        # (T, 8)
        # Clip at zero for stability; pads that read *higher* than baseline
        # during take-off would otherwise inject negative tails into the mean.
        press_mag_nonneg = np.clip(press_mag, 0.0, None)
        press_min = np.percentile(press_mag_nonneg, 1.0, axis=0)
        press_max = np.percentile(press_mag_nonneg, 99.0, axis=0)
        # Ensure non-degenerate range so the downstream divisor stays safe.
        degenerate = (press_max - press_min) < 1.0
        if np.any(degenerate):
            for ch in np.where(degenerate)[0]:
                self.warnings.append(
                    f"[{CHANNEL_NAMES_DUAL[ch]}] global press span too "
                    "narrow; widening to [0, 4095]."
                )
                press_min[ch] = 0.0
                press_max[ch] = SENSOR_MAX
        press_mean = press_mag_nonneg.mean(axis=0)
        press_std = press_mag_nonneg.std(axis=0, ddof=0)
        press_std = np.maximum(press_std, 1.0)   # std floor → no /0 downstream

        return PersonalCalibration(
            min_raw=min_raw,
            max_raw=max_raw,
            source="offline_auto_csv",
            subject=subject,
            notes=notes,
            baseline_raw=baseline_raw,
            press_min=press_min,
            press_max=press_max,
            press_mean=press_mean,
            press_std=press_std,
        )


def auto_calibrate_from_csv_dir(
    csv_dir: str,
    *,
    subject: str = "offline_auto_population",
    save_to: Optional[str] = None,
) -> PersonalCalibration:
    """Convenience wrapper: walk ``saving_data/sensor_data_dual_labeled_*.csv``
    (via :func:`ml_activity_features.load_csv_files`), run
    :class:`OfflineAutoCalibrator`, optionally dump JSON.
    """
    # Lazy import to avoid a hard cycle (``ml_activity_features`` already
    # imports ``adaptive_preprocessing`` but not this file).
    from ml_activity_features import load_csv_files  # noqa: WPS433

    data, labels, _subjects, mode = load_csv_files(
        csv_dir, labeled_only=True, raw_adc=True,
    )
    if mode != "dual" or data.size == 0:
        raise RuntimeError(
            f"No labelled dual-foot CSVs found under {csv_dir!r}"
        )
    calib = OfflineAutoCalibrator().fit(
        data, labels, subject=subject,
        notes=f"auto-fit from {csv_dir} ({data.shape[0]} frames)",
    )
    if save_to:
        calib.save_json(save_to)
    return calib


# ── Path B: UI-driven two-step online calibration ───────────────────────────

OnlinePhase = Literal["IDLE", "STEP1_STANDING", "STEP1_DONE",
                      "STEP2_KNEE_BEND", "DONE"]


@dataclass
class _PhaseBuffer:
    raw: list[np.ndarray] = field(default_factory=list)
    t0: float = 0.0
    min_samples: int = 20   # default 20 frames ≈ 2 s @ 10 Hz
    target_samples: int = 50  # default 50 frames ≈ 5 s @ 10 Hz


class OnlineCalibrator:
    """Two-step wearer-guided calibration, designed for the PyQt UI.

    Intended flow (UI side):

        cal = OnlineCalibrator(sample_hz=10)
        cal.start_step1()                     # ← UI shows "stand still"
        for frame in stream:                  # push every 100 ms
            progress = cal.feed(frame)
            if cal.step1_ready:               # ≥ target_samples collected
                break
        cal.finish_step1()

        cal.start_step2()                     # ← UI shows "bend knee 90°"
        for frame in stream:
            cal.feed(frame)
            if cal.step2_ready:
                break
        calib = cal.finalize(subject="alice")  # PersonalCalibration
        calib.save_json("personal_calibration.json")

    Only *this* function writes calibration state; the recognizer just
    calls :meth:`PersonalCalibration.normalize_to_adc` per frame.
    """

    def __init__(
        self,
        *,
        sample_hz: int = 10,
        step1_seconds: float = 5.0,
        step2_seconds: float = 5.0,
        min_seconds: float = 2.0,
    ) -> None:
        self.sample_hz = int(sample_hz)
        self._step1 = _PhaseBuffer(
            target_samples=max(10, int(round(step1_seconds * sample_hz))),
            min_samples=max(5, int(round(min_seconds * sample_hz))),
        )
        self._step2 = _PhaseBuffer(
            target_samples=max(10, int(round(step2_seconds * sample_hz))),
            min_samples=max(5, int(round(min_seconds * sample_hz))),
        )
        self.phase: OnlinePhase = "IDLE"

    # ── step 1: natural standing (pressure range) ───────────────────────
    def start_step1(self) -> None:
        self._step1 = _PhaseBuffer(
            target_samples=self._step1.target_samples,
            min_samples=self._step1.min_samples,
            t0=time.time(),
        )
        self.phase = "STEP1_STANDING"

    def finish_step1(self) -> None:
        if self.phase != "STEP1_STANDING":
            raise RuntimeError("Not currently in STEP1_STANDING")
        if len(self._step1.raw) < self._step1.min_samples:
            raise RuntimeError(
                f"Step 1 needs at least {self._step1.min_samples} frames; "
                f"got {len(self._step1.raw)}"
            )
        self.phase = "STEP1_DONE"

    # ── step 2: knee bend 90° (stretch range) ──────────────────────────
    def start_step2(self) -> None:
        if self.phase != "STEP1_DONE":
            raise RuntimeError("Must finish step 1 before starting step 2")
        self._step2 = _PhaseBuffer(
            target_samples=self._step2.target_samples,
            min_samples=self._step2.min_samples,
            t0=time.time(),
        )
        self.phase = "STEP2_KNEE_BEND"

    def finish_step2(self) -> None:
        if self.phase != "STEP2_KNEE_BEND":
            raise RuntimeError("Not currently in STEP2_KNEE_BEND")
        if len(self._step2.raw) < self._step2.min_samples:
            raise RuntimeError(
                f"Step 2 needs at least {self._step2.min_samples} frames; "
                f"got {len(self._step2.raw)}"
            )
        self.phase = "DONE"

    # ── sample ingestion ───────────────────────────────────────────────
    def feed(self, raw_8: Sequence[float]) -> float:
        """Feed one dual-foot raw frame.  Returns current-phase progress 0..1."""
        arr = np.asarray(raw_8, dtype=np.float64).reshape(8)
        if self.phase == "STEP1_STANDING":
            self._step1.raw.append(arr)
            return min(1.0, len(self._step1.raw) / max(1, self._step1.target_samples))
        if self.phase == "STEP2_KNEE_BEND":
            self._step2.raw.append(arr)
            return min(1.0, len(self._step2.raw) / max(1, self._step2.target_samples))
        return 0.0

    @property
    def step1_ready(self) -> bool:
        return len(self._step1.raw) >= self._step1.target_samples

    @property
    def step2_ready(self) -> bool:
        return len(self._step2.raw) >= self._step2.target_samples

    # ── thin UI-only accessors (do not change state machine behaviour) ──
    @property
    def step1_sample_count(self) -> int:
        """Number of raw frames collected for step 1 so far (UI read-only)."""
        return len(self._step1.raw)

    @property
    def step2_sample_count(self) -> int:
        """Number of raw frames collected for step 2 so far (UI read-only)."""
        return len(self._step2.raw)

    @property
    def step1_target_samples(self) -> int:
        """Target number of frames for step 1 before ``step1_ready`` turns True."""
        return self._step1.target_samples

    @property
    def step2_target_samples(self) -> int:
        """Target number of frames for step 2 before ``step2_ready`` turns True."""
        return self._step2.target_samples

    @property
    def step1_min_samples(self) -> int:
        """Minimum frames required by ``finish_step1()`` — below this it raises."""
        return self._step1.min_samples

    @property
    def step2_min_samples(self) -> int:
        """Minimum frames required by ``finish_step2()`` — below this it raises."""
        return self._step2.min_samples

    # ── final calibration  ─────────────────────────────────────────────
    def finalize(
        self,
        *,
        subject: str = "ui_online",
        notes: str = "",
    ) -> PersonalCalibration:
        """Collapse the two sample buffers into a :class:`PersonalCalibration`.

        Pressure channels (idx 0,1,2,4,5,6):
            * ``min_raw`` = percentile-1 of *step 1* frames → peak load
              produced by the subject's own body weight while standing.
            * ``max_raw`` = rail 4095 → unloaded reference (pad in air).

        Knee channels (idx 3, 7):
            * ``min_raw`` = percentile-1 of *step 2* frames → deepest bend.
            * ``max_raw`` = rail 4095 → straight leg.
        """
        if self.phase != "DONE":
            raise RuntimeError(
                f"finalize() requires phase == DONE; current={self.phase!r}"
            )
        s1 = np.stack(self._step1.raw, axis=0)   # (N1, 8) natural standing
        s2 = np.stack(self._step2.raw, axis=0)   # (N2, 8) knee bend ≈ 90°

        min_raw = np.zeros(8, dtype=np.float64)
        max_raw = np.full(8, SENSOR_MAX, dtype=np.float64)

        # Pressure — use step 1 (subject is standing, all pads loaded)
        for ch in PRESSURE_IDX:
            min_raw[ch] = float(np.percentile(s1[:, ch], 1.0))
            max_raw[ch] = SENSOR_MAX

        # Knee — use step 2 (subject is bending → low raw)
        for ch in KNEE_IDX:
            min_raw[ch] = float(np.percentile(s2[:, ch], 1.0))
            max_raw[ch] = SENSOR_MAX

        # ── Global stats from the UI buffers, matching the offline schema ──
        # Combine step1 + step2 so the moments cover both loaded-pads and
        # bent-knee regimes.  baseline_raw == max_raw as offline.
        all_frames = np.vstack([s1, s2])          # (N1+N2, 8)
        baseline_raw = max_raw.copy()
        press_mag = np.clip(baseline_raw[None, :] - all_frames, 0.0, None)
        press_min = np.percentile(press_mag, 1.0,  axis=0)
        press_max = np.percentile(press_mag, 99.0, axis=0)
        degenerate = (press_max - press_min) < 1.0
        press_min = np.where(degenerate, 0.0, press_min)
        press_max = np.where(degenerate, SENSOR_MAX, press_max)
        press_mean = press_mag.mean(axis=0)
        press_std = np.maximum(press_mag.std(axis=0, ddof=0), 1.0)

        return PersonalCalibration(
            min_raw=min_raw,
            max_raw=max_raw,
            source="ui_online_two_step",
            subject=subject,
            notes=notes or (
                f"step1 N={s1.shape[0]} frames, step2 N={s2.shape[0]} frames"
            ),
            baseline_raw=baseline_raw,
            press_min=press_min,
            press_max=press_max,
            press_mean=press_mean,
            press_std=press_std,
        )


# ── Default filename for the on-disk JSON (read by recognizer / UI) ─────────

DEFAULT_CALIBRATION_FILENAME = "personal_calibration.json"


def load_default_calibration(
    search_dirs: Sequence[str] = (".",),
) -> Optional[PersonalCalibration]:
    """Look for ``personal_calibration.json`` in the given dirs, in order.

    Returns ``None`` if none is found (caller can fall back to identity).
    """
    for d in search_dirs:
        p = os.path.join(d, DEFAULT_CALIBRATION_FILENAME)
        if os.path.isfile(p):
            return PersonalCalibration.load_json(p)
    return None
