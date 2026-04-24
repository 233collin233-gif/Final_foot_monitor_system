"""Per-user 6ch ADC min/max and optional frozen press-magnitude stats. Used for normalize_to_adc and adaptive preprocessor seeds."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

CHANNEL_NAMES_DUAL = (
    "L_Forefoot", "L_Heel", "L_Knee",
    "R_Forefoot", "R_Heel", "R_Knee",
)
N_CH = 6
PRESSURE_IDX = (0, 1, 3, 4)
KNEE_IDX = (2, 5)
SENSOR_MAX = 4095.0

OFFLINE_PRESSURE_MAX_PCT = 97.0
OFFLINE_PRESSURE_MIN_PCT = 1.0
OFFLINE_PRESSURE_MIN_SPAN = 120.0

OFFLINE_KNEE_MIN_PCT = 1.0
OFFLINE_KNEE_MAX = 4095.0
OFFLINE_KNEE_MIN_SPAN = 300.0

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


CalibrationSource = Literal["offline_auto_csv", "ui_online_two_step", "manual"]


@dataclass
class PersonalCalibration:
    """Per-channel min_raw/max_raw for linear scaling to [0,4095].
    Optional press-mag globals freeze adaptive_preprocessing when has_global_stats is true."""

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

    def __post_init__(self) -> None:
        self.min_raw = np.asarray(self.min_raw, dtype=np.float64).reshape(N_CH)
        self.max_raw = np.asarray(self.max_raw, dtype=np.float64).reshape(N_CH)
        self._sanity_clamp()
        for name in ("baseline_raw", "press_min", "press_max",
                     "press_mean", "press_std"):
            val = getattr(self, name)
            if val is not None:
                arr = np.asarray(val, dtype=np.float64).reshape(N_CH)
                setattr(self, name, arr)

    def _sanity_clamp(self) -> None:
        """Enforce min/max in [0,4095] and span >= 1."""
        for i in range(N_CH):
            lo = float(self.min_raw[i])
            hi = float(self.max_raw[i])
            lo = max(0.0, min(lo, SENSOR_MAX - 1.0))
            hi = max(lo + 1.0, min(hi, SENSOR_MAX))
            self.min_raw[i] = lo
            self.max_raw[i] = hi

    @property
    def has_global_stats(self) -> bool:
        """All five press-mag globals present for frozen preprocessing."""
        return all(
            getattr(self, k) is not None
            for k in ("baseline_raw", "press_min", "press_max",
                      "press_mean", "press_std")
        )

    def to_channel_seeds(self) -> Optional[list]:
        """List of ChannelSeed, or None if has_global_stats is false."""
        if not self.has_global_stats:
            return None
        from adaptive_preprocessing import ChannelSeed  # noqa: WPS433
        seeds = []
        for i in range(N_CH):
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
        """Full-span identity mapping (no per-user warping)."""
        return cls(
            min_raw=np.zeros(N_CH, dtype=np.float64),
            max_raw=np.full(N_CH, SENSOR_MAX, dtype=np.float64),
            source="manual",
            subject="identity",
            notes="identity mapping — downstream features see raw ADC unchanged",
        )

    def normalize_to_unit(self, raw_8: np.ndarray) -> np.ndarray:
        """Linear map each channel to [0,1] from min_raw..max_raw."""
        r = np.asarray(raw_8, dtype=np.float64)
        lo = self.min_raw
        hi = self.max_raw
        out = (r - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)

    def normalize_to_adc(self, raw_8: np.ndarray) -> np.ndarray:
        """Like normalize_to_unit but scaled to 0..4095 for downstream raw-level thresholds."""
        return self.normalize_to_unit(raw_8) * SENSOR_MAX

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
        for name in ("baseline_raw", "press_min", "press_max",
                     "press_mean", "press_std"):
            val = getattr(self, name)
            if val is not None:
                out[name] = [float(x) for x in np.asarray(val).reshape(N_CH)]
        return out

    def save_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return os.path.abspath(path)

    @staticmethod
    def _strip_legacy_toe_arrays(arr8: np.ndarray) -> np.ndarray:
        a = np.asarray(arr8, dtype=np.float64).ravel()
        if a.size != 8:
            return a
        return np.array(
            [a[1], a[2], a[3], a[5], a[6], a[7]], dtype=np.float64,
        )

    @classmethod
    def load_json(cls, path: str) -> "PersonalCalibration":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        ch = list(d.get("channels", CHANNEL_NAMES_DUAL))
        legacy8 = len(ch) == 8 and "L_Toe" in ch

        def _maybe_array(key: str) -> Optional[np.ndarray]:
            return (np.asarray(d[key], dtype=np.float64)
                    if key in d and d[key] is not None else None)

        def _maybe_strip(key: str) -> Optional[np.ndarray]:
            v = _maybe_array(key)
            return None if v is None else cls._strip_legacy_toe_arrays(v)

        if legacy8:
            min_r = cls._strip_legacy_toe_arrays(d["min_raw"])
            max_r = cls._strip_legacy_toe_arrays(d["max_raw"])
            br = _maybe_strip("baseline_raw")
            pmn = _maybe_strip("press_min")
            pmx = _maybe_strip("press_max")
            pme = _maybe_strip("press_mean")
            pst = _maybe_strip("press_std")
            notes = (d.get("notes") or "") + " [migrated 8ch→6ch, toe dropped]"
        else:
            if tuple(ch) != CHANNEL_NAMES_DUAL:
                raise ValueError(
                    f"Channel layout in {path} does not match {CHANNEL_NAMES_DUAL}; got {ch}",
                )
            min_r = np.asarray(d["min_raw"], dtype=np.float64)
            max_r = np.asarray(d["max_raw"], dtype=np.float64)
            br = _maybe_array("baseline_raw")
            pmn = _maybe_array("press_min")
            pmx = _maybe_array("press_max")
            pme = _maybe_array("press_mean")
            pst = _maybe_array("press_std")
            notes = d.get("notes", "")

        return cls(
            min_raw=min_r,
            max_raw=max_r,
            source=d.get("source", "manual"),
            subject=d.get("subject", "default"),
            created_at=float(d.get("created_at", time.time())),
            notes=notes,
            baseline_raw=br,
            press_min=pmn,
            press_max=pmx,
            press_mean=pme,
            press_std=pst,
        )

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




@dataclass
class OfflineAutoCalibrator:
    """Fit min_raw/max_raw from label-conditioned percentiles on stacked T×6 ADC, plus global press-mag stats."""

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
        if data_t8.ndim == 2 and data_t8.shape[1] == 8:
            data_t8 = np.column_stack([data_t8[:, 1:4], data_t8[:, 5:8]])
        if data_t8.ndim != 2 or data_t8.shape[1] != N_CH:
            raise ValueError(f"data must be (T, {N_CH}) or legacy (T, 8); got {data_t8.shape}")
        if len(labels) != data_t8.shape[0]:
            raise ValueError("len(labels) must equal T")

        labels = np.asarray([str(x).strip().upper() for x in labels])
        min_raw = np.zeros(N_CH, dtype=np.float64)
        max_raw = np.full(N_CH, SENSOR_MAX, dtype=np.float64)

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

        baseline_raw = max_raw.copy()
        press_mag = baseline_raw[None, :] - data_t8
        press_mag_nonneg = np.clip(press_mag, 0.0, None)
        press_min = np.percentile(press_mag_nonneg, 1.0, axis=0)
        press_max = np.percentile(press_mag_nonneg, 99.0, axis=0)
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
        press_std = np.maximum(press_std, 1.0)

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
    """Load all dual labeled CSVs from csv_dir, fit OfflineAutoCalibrator, optionally save JSON."""
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



OnlinePhase = Literal["IDLE", "STEP1_STANDING", "STEP1_DONE",
                      "STEP2_KNEE_BEND", "DONE"]


@dataclass
class _PhaseBuffer:
    raw: list[np.ndarray] = field(default_factory=list)
    t0: float = 0.0
    min_samples: int = 20
    target_samples: int = 50


class OnlineCalibrator:
    """UI two-step capture (stand, then knee bend) to build a PersonalCalibration."""

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

    def feed(self, raw_6: Sequence[float]) -> float:
        """One 6ch frame; returns step progress in [0,1]."""
        arr = np.asarray(raw_6, dtype=np.float64).reshape(N_CH)
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

    @property
    def step1_sample_count(self) -> int:
        return len(self._step1.raw)

    @property
    def step2_sample_count(self) -> int:
        return len(self._step2.raw)

    @property
    def step1_target_samples(self) -> int:
        return self._step1.target_samples

    @property
    def step2_target_samples(self) -> int:
        return self._step2.target_samples

    @property
    def step1_min_samples(self) -> int:
        return self._step1.min_samples

    @property
    def step2_min_samples(self) -> int:
        return self._step2.min_samples

    def finalize(
        self,
        *,
        subject: str = "ui_online",
        notes: str = "",
    ) -> PersonalCalibration:
        """Merge step buffers into min_raw/max_raw and press-mag globals."""
        if self.phase != "DONE":
            raise RuntimeError(
                f"finalize() requires phase == DONE; current={self.phase!r}"
            )
        s1 = np.stack(self._step1.raw, axis=0)
        s2 = np.stack(self._step2.raw, axis=0)

        min_raw = np.zeros(N_CH, dtype=np.float64)
        max_raw = np.full(N_CH, SENSOR_MAX, dtype=np.float64)

        for ch in PRESSURE_IDX:
            min_raw[ch] = float(np.percentile(s1[:, ch], 1.0))
            max_raw[ch] = SENSOR_MAX

        for ch in KNEE_IDX:
            min_raw[ch] = float(np.percentile(s2[:, ch], 1.0))
            max_raw[ch] = SENSOR_MAX

        all_frames = np.vstack([s1, s2])
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



DEFAULT_CALIBRATION_FILENAME = "personal_calibration.json"


def load_default_calibration(
    search_dirs: Sequence[str] = (".",),
) -> Optional[PersonalCalibration]:
    """Load personal_calibration.json from the first existing path in search_dirs, else None."""
    for d in search_dirs:
        p = os.path.join(d, DEFAULT_CALIBRATION_FILENAME)
        if os.path.isfile(p):
            return PersonalCalibration.load_json(p)
    return None
