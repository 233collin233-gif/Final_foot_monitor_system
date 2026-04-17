"""
Online adaptive preprocessing for heterogeneous piezoresistive ADC channels.

Why raw ADC must not feed RandomForest directly:
  - Different pads (heel / toe / forefoot / knee) have different sensitivity and
    usable ADC sub-ranges (e.g. 3800–4095 vs 0–4095).
  - Baseline and span drift over time; a global linear map (e.g. /4095) bakes in
    hardware bias that the RF would have to memorize instead of learning gait.

This module maps each channel into a domain-invariant feature triple (per tick):
  1) baseline_removed  — raw − baseline_raw (signed; as specified for RF/debug).
  2) relative_pressure_ratio — (baseline_raw − raw) normalized by an online
     estimate of dynamic span on the *press magnitude* signal, clipped to [0, 1].
     (Uses per-channel adaptive range, not a global 4095.)
  3) adaptive_zscore — press magnitude standardized with online mean / std.

Each physical channel must use its own AdaptiveSensorPreprocessor instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

# ── Tunables (all marked TODO_PARAM) ───────────────────────────────────────

# TODO_PARAM: EWMA for baseline when channel is idle (small press magnitude)
BASELINE_ALPHA_SLOW = 0.002
# TODO_PARAM: Faster baseline blend only in clearly idle windows (optional boost)
BASELINE_ALPHA_IDLE = 0.015
# TODO_PARAM: During unload (raw rises above baseline), recover baseline faster.
BASELINE_ALPHA_RECOVER = 0.03
# TODO_PARAM: During sustained press, strongly freeze downward baseline drift.
BASELINE_ALPHA_HEAVY_FREEZE = 0.0002
# TODO_PARAM: Number of consecutive pressed samples before freeze mode.
BASELINE_FREEZE_AFTER_SAMPLES = 20
# TODO_PARAM: Decay / tracking speed for recursive min/max on raw ADC
RANGE_ALPHA = 0.01
# TODO_PARAM: Online mean/std smoothing for z-score (EWMA-style)
ZSCORE_ALPHA = 0.05
# TODO_PARAM: |signal − mean| beyond this many std counts as outlier for range stats
RANGE_OUTLIER_SIGMA = 4.0
# TODO_PARAM: Minimum std estimate for division
ZSCORE_EPS = 1e-4
# TODO_PARAM: Floor on adaptive range (raw span and press span); below → low confidence
LOW_DYNAMIC_RANGE_TH = 8.0
# TODO_PARAM: Idle gate: |baseline_raw − raw| below this → allow baseline tracking
IDLE_PRESS_GATE_RAW = 25.0
# TODO_PARAM: Heavy press: baseline update strongly attenuated above this ratio
HEAVY_RATIO_FOR_BASELINE_FREEZE = 0.55
# TODO_PARAM: EMA for optional smoothed ratio (debug / extra feature)
RATIO_EMA_ALPHA = 0.2

# Three-state classifier (operates on relative_pressure_ratio)
# TODO_PARAM: Enter LIGHT from NO when ratio crosses up; leave LIGHT down
LIGHT_TH_UP = 0.12
LIGHT_TH_DOWN = 0.06
# TODO_PARAM: Enter HEAVY from LIGHT when ratio crosses up; leave HEAVY down
HEAVY_TH_UP = 0.45
HEAVY_TH_DOWN = 0.32
# TODO_PARAM: Require this many consecutive raw states before publishing stable state
STATE_CONFIRM_SAMPLES = 4


PressureState = Literal["NO_PRESSURE", "LIGHT_PRESSURE", "HEAVY_PRESSURE"]


@dataclass
class ChannelSeed:
    """Frozen per-channel statistics produced by :class:`PersonalCalibration`.

    When an :class:`AdaptiveSensorPreprocessor` is given a seed, it
    **suspends** its EWMA updates on ``baseline_raw``, ``press_min/max``,
    ``run_mean`` and ``run_var`` — every frame is normalised against the
    globally-learned numbers instead.  This is what guarantees that the
    exact same ``(raw_8, seed)`` pair always produces the same
    ``(baseline_removed, relative_pressure_ratio, adaptive_zscore)`` triple,
    regardless of which CSV it came from or how many frames have been seen
    so far.

    Attributes
    ----------
    baseline_raw
        ADC value when the pad is unloaded (or knee is straight).
    press_min, press_max
        Global min / max of ``press_mag = baseline_raw - raw`` across the
        full training population.  Used as the ``[0, 1]`` denominator in
        ``relative_pressure_ratio``.
    press_mean, press_std
        Global mean / std of ``press_mag``.  Used in
        ``adaptive_zscore = (press_mag - press_mean) / press_std``.
    """

    baseline_raw: float
    press_min: float
    press_max: float
    press_mean: float
    press_std: float


@dataclass
class AdaptiveChannelSnapshot:
    """One frame of per-channel outputs for RF, rules, and debug."""

    raw: float
    baseline_raw: float
    baseline_removed: float
    relative_pressure_ratio: float
    ratio_ema: float
    adaptive_zscore: float
    delta_raw: float
    dynamic_min_raw: float
    dynamic_max_raw: float
    adaptive_range_raw: float
    press_range: float
    confidence: float
    stable_state: PressureState
    state_candidate: PressureState


class AdaptiveSensorPreprocessor:
    """Per-channel online normalization + feature triple + confidence.

    Two operating modes:

    * **EWMA mode** (``seed is None``) — the historical behaviour: baseline,
      raw / press range, running mean and variance are all learned online
      from scratch each time the preprocessor is reset.

    * **Frozen mode** (``seed`` provided and ``freeze_when_seeded=True``) —
      ``baseline_raw``, ``press_min/max``, ``run_mean`` and ``run_var`` are
      pinned to the seed values and never updated.  Every frame produces
      the same ``(baseline_removed, relative_pressure_ratio, adaptive_zscore)``
      against the global statistics, regardless of which CSV / subject /
      time-slice it was seen in.  This is what the user-facing "global
      normalisation, trained once across all 33 CSVs" guarantee relies on.

    In frozen mode the online raw-envelope (``dynamic_min_raw``,
    ``dynamic_max_raw``) and the three-state classifier still live-update
    because they are diagnostic outputs that nobody's feature vector depends
    on.  Turning this off entirely (``update_ranges=False``) is available
    too, for a fully deterministic path.
    """

    def __init__(
        self,
        name: str = "",
        *,
        seed: Optional[ChannelSeed] = None,
        freeze_when_seeded: bool = True,
    ) -> None:
        self.name = name
        self._seed = seed
        self._freeze = bool(freeze_when_seeded and seed is not None)
        self._prev_raw: float | None = None
        self._baseline_raw: float | None = None
        self._dynamic_min_raw: float | None = None
        self._dynamic_max_raw: float | None = None
        self._dyn_min_press: float | None = None
        self._dyn_max_press: float | None = None
        self._run_mean = 0.0
        self._run_var = 0.0
        self._ratio_ema: float | None = None
        self._n_updates = 0
        self._pressed_run = 0
        self._classifier = ThreeStatePressureClassifier()
        if seed is not None:
            self._apply_seed(seed)

    def _apply_seed(self, seed: ChannelSeed) -> None:
        self._baseline_raw = float(seed.baseline_raw)
        self._dyn_min_press = float(seed.press_min)
        self._dyn_max_press = float(seed.press_max)
        self._run_mean = float(seed.press_mean)
        std = float(max(seed.press_std, ZSCORE_EPS))
        self._run_var = std * std

    def reset(self) -> None:
        self.__init__(
            self.name,
            seed=self._seed,
            freeze_when_seeded=bool(self._freeze or self._seed is not None),
        )

    def update(self, raw: float) -> AdaptiveChannelSnapshot:
        raw = float(raw)
        self._n_updates += 1
        delta_raw = 0.0 if self._prev_raw is None else raw - self._prev_raw
        self._prev_raw = raw

        if self._baseline_raw is None:
            self._baseline_raw = raw
        if self._dynamic_min_raw is None:
            self._dynamic_min_raw = raw
        if self._dynamic_max_raw is None:
            self._dynamic_max_raw = raw

        # Press magnitude: high ADC = low pressure on typical voltage-divider insoles
        press_mag = self._baseline_raw - raw

        # Raw envelope is purely diagnostic; we keep it live-updating even
        # in frozen mode because the confidence metric below reads it.
        if self._n_updates <= 3:
            self._dynamic_min_raw = min(self._dynamic_min_raw, raw)
            self._dynamic_max_raw = max(self._dynamic_max_raw, raw)
        else:
            self._update_raw_range(raw)

        adaptive_range_raw = max(
            float(self._dynamic_max_raw - self._dynamic_min_raw),
            LOW_DYNAMIC_RANGE_TH,
        )

        idle = abs(press_mag) < IDLE_PRESS_GATE_RAW
        if idle:
            self._pressed_run = 0
        else:
            self._pressed_run += 1

        if self._freeze:
            # FROZEN: baseline, press range, mean, var are all globally fixed.
            # No asymmetric EWMA, no range tracker updates — the triple below
            # is a deterministic function of (raw, seed).
            press_mag = self._baseline_raw - raw
            assert self._dyn_min_press is not None and self._dyn_max_press is not None
            press_range = max(
                float(self._dyn_max_press - self._dyn_min_press),
                LOW_DYNAMIC_RANGE_TH,
            )
            relative_pressure_ratio = float(
                np.clip((press_mag - self._dyn_min_press) / press_range, 0.0, 1.0),
            )
            std = float(np.sqrt(max(self._run_var, ZSCORE_EPS)))
            adaptive_zscore = float((press_mag - self._run_mean) / (std + ZSCORE_EPS))
        else:
            heavy = False
            if self._dyn_max_press is not None and self._dyn_min_press is not None:
                pr = max(
                    float(self._dyn_max_press - self._dyn_min_press),
                    LOW_DYNAMIC_RANGE_TH,
                )
                r_est = np.clip(
                    (press_mag - self._dyn_min_press) / pr, 0.0, 1.0,
                )
                heavy = r_est >= HEAVY_RATIO_FOR_BASELINE_FREEZE

            # Baseline update (asymmetric):
            # - Idle: normal tracking.
            # - Pressed for long enough: freeze downward drift so prolonged
            #   standing / sitting does not "eat" the historical baseline.
            # - Unload (raw > baseline): recover quickly to new no-load level.
            if raw >= self._baseline_raw:
                a = BASELINE_ALPHA_RECOVER if not idle else BASELINE_ALPHA_IDLE
            else:
                if idle:
                    a = BASELINE_ALPHA_IDLE
                elif heavy or self._pressed_run >= BASELINE_FREEZE_AFTER_SAMPLES:
                    a = BASELINE_ALPHA_HEAVY_FREEZE
                else:
                    a = BASELINE_ALPHA_SLOW
            self._baseline_raw = (1.0 - a) * self._baseline_raw + a * raw
            press_mag = self._baseline_raw - raw

            self._update_press_range(press_mag)
            press_range = max(
                float(self._dyn_max_press - self._dyn_min_press),
                LOW_DYNAMIC_RANGE_TH,
            )
            relative_pressure_ratio = float(
                np.clip((press_mag - self._dyn_min_press) / press_range, 0.0, 1.0),
            )

            # Online mean / variance (EWMA) on press_mag
            d = press_mag - self._run_mean
            self._run_mean += ZSCORE_ALPHA * d
            self._run_var = (1.0 - ZSCORE_ALPHA) * self._run_var + ZSCORE_ALPHA * (
                (press_mag - self._run_mean) ** 2
            )
            std = float(np.sqrt(max(self._run_var, ZSCORE_EPS)))
            adaptive_zscore = float((press_mag - self._run_mean) / (std + ZSCORE_EPS))

        if self._ratio_ema is None:
            self._ratio_ema = relative_pressure_ratio
        else:
            self._ratio_ema += RATIO_EMA_ALPHA * (
                relative_pressure_ratio - self._ratio_ema
            )

        baseline_removed = raw - float(self._baseline_raw)

        # Confidence: narrow raw span or narrow press span → not trustworthy
        raw_span_ok = adaptive_range_raw - LOW_DYNAMIC_RANGE_TH
        press_span_ok = press_range - LOW_DYNAMIC_RANGE_TH
        confidence = float(
            np.clip(min(raw_span_ok, press_span_ok) / 50.0, 0.0, 1.0)
            * np.clip(self._n_updates / 200.0, 0.0, 1.0),
        )

        cand = self._classifier.update(relative_pressure_ratio)
        stable = self._classifier.stable_state

        return AdaptiveChannelSnapshot(
            raw=raw,
            baseline_raw=float(self._baseline_raw),
            baseline_removed=baseline_removed,
            relative_pressure_ratio=relative_pressure_ratio,
            ratio_ema=float(self._ratio_ema),
            adaptive_zscore=adaptive_zscore,
            delta_raw=delta_raw,
            dynamic_min_raw=float(self._dynamic_min_raw),
            dynamic_max_raw=float(self._dynamic_max_raw),
            adaptive_range_raw=adaptive_range_raw,
            press_range=press_range,
            confidence=confidence,
            stable_state=stable,
            state_candidate=cand,
        )

    def _update_raw_range(self, raw: float) -> None:
        lo = self._dynamic_min_raw
        hi = self._dynamic_max_raw
        assert lo is not None and hi is not None
        span = max(hi - lo, LOW_DYNAMIC_RANGE_TH)
        out_lo = raw < lo - RANGE_OUTLIER_SIGMA * span
        out_hi = raw > hi + RANGE_OUTLIER_SIGMA * span
        if not out_hi:
            if raw > hi:
                self._dynamic_max_raw = raw
            else:
                self._dynamic_max_raw = hi + RANGE_ALPHA * (raw - hi)
        if not out_lo:
            if raw < lo:
                self._dynamic_min_raw = raw
            else:
                self._dynamic_min_raw = lo + RANGE_ALPHA * (raw - lo)

    def _update_press_range(self, press_mag: float) -> None:
        if self._dyn_min_press is None:
            self._dyn_min_press = min(0.0, press_mag)
            self._dyn_max_press = max(0.0, press_mag)
            return
        lo = self._dyn_min_press
        hi = self._dyn_max_press
        span = max(hi - lo, LOW_DYNAMIC_RANGE_TH)
        mean = self._run_mean
        std = float(np.sqrt(max(self._run_var, ZSCORE_EPS)))
        gate = RANGE_OUTLIER_SIGMA * max(std, 1.0)
        if abs(press_mag - mean) > gate and self._n_updates > 30:
            return
        if press_mag < lo:
            self._dyn_min_press = press_mag
        else:
            self._dyn_min_press = lo + RANGE_ALPHA * (press_mag - lo)
        if press_mag > hi:
            self._dyn_max_press = press_mag
        else:
            self._dyn_max_press = hi + RANGE_ALPHA * (press_mag - hi)


class ThreeStatePressureClassifier:
    """Hysteresis + debounced three-state output from relative_pressure_ratio."""

    def __init__(self) -> None:
        self._internal: PressureState = "NO_PRESSURE"
        self.stable_state: PressureState = "NO_PRESSURE"
        self._pending: PressureState = "NO_PRESSURE"
        self._count = 0

    def reset(self) -> None:
        self.__init__()

    def update(self, ratio: float) -> PressureState:
        if self._internal == "NO_PRESSURE":
            if ratio >= HEAVY_TH_UP:
                self._internal = "HEAVY_PRESSURE"
            elif ratio >= LIGHT_TH_UP:
                self._internal = "LIGHT_PRESSURE"
        elif self._internal == "LIGHT_PRESSURE":
            if ratio >= HEAVY_TH_UP:
                self._internal = "HEAVY_PRESSURE"
            elif ratio <= LIGHT_TH_DOWN:
                self._internal = "NO_PRESSURE"
        else:
            if ratio <= HEAVY_TH_DOWN:
                self._internal = "LIGHT_PRESSURE"

        # Debounce stable output
        if self._internal == self._pending:
            self._count += 1
        else:
            self._pending = self._internal
            self._count = 1
        if self._count >= STATE_CONFIRM_SAMPLES:
            self.stable_state = self._pending
        return self._internal


def stack_channel_features(snapshots: list[AdaptiveChannelSnapshot]) -> np.ndarray:
    """Flatten (8, 3) core features for one time step: per ch [br, ratio, z]."""
    row: list[float] = []
    for s in snapshots:
        row.extend([s.baseline_removed, s.relative_pressure_ratio, s.adaptive_zscore])
    return np.array(row, dtype=np.float64)


CHANNEL_NAMES_DUAL = [
    "L_Toe",
    "L_Forefoot",
    "L_Heel",
    "L_Knee",
    "R_Toe",
    "R_Forefoot",
    "R_Heel",
    "R_Knee",
]


class DualFootAdaptiveBank:
    """Eight independent preprocessors (L/R × toe, forefoot, heel, knee).

    Parameters
    ----------
    seeds
        Either ``None`` (each channel learns its own EWMA, legacy behaviour)
        or a length-8 sequence of :class:`ChannelSeed` that freezes every
        channel to the same globally-computed statistics — matches the order
        of :data:`CHANNEL_NAMES_DUAL`.
    freeze_when_seeded
        When ``True`` (default) and ``seeds`` is provided, EWMA updates are
        disabled so the outputs are deterministic w.r.t. ``(raw, seed)``.
    """

    def __init__(
        self,
        seeds: Optional[Sequence[ChannelSeed]] = None,
        *,
        freeze_when_seeded: bool = True,
    ) -> None:
        if seeds is not None and len(seeds) != 8:
            raise ValueError(
                f"seeds must have length 8 (one per channel); got {len(seeds)}",
            )
        self.channels: list[AdaptiveSensorPreprocessor] = [
            AdaptiveSensorPreprocessor(
                name=n,
                seed=(None if seeds is None else seeds[i]),
                freeze_when_seeded=freeze_when_seeded,
            )
            for i, n in enumerate(CHANNEL_NAMES_DUAL)
        ]

    def reset(self) -> None:
        for c in self.channels:
            c.reset()

    def update(self, raw8: np.ndarray | list[float]) -> tuple[np.ndarray, list[AdaptiveChannelSnapshot]]:
        arr = np.asarray(raw8, dtype=np.float64).ravel()
        if arr.size != 8:
            raise ValueError(f"Expected 8 raw channels, got {arr.size}")
        snaps: list[AdaptiveChannelSnapshot] = []
        for i, pre in enumerate(self.channels):
            snaps.append(pre.update(float(arr[i])))
        feat24 = stack_channel_features(snaps)
        return feat24, snaps
