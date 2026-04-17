"""
Real-time gait / posture recognizer for dual-foot pressure + knee-stretch sensors.
===================================================================================
Hierarchical pipeline (strict 4095 rule):

  1) Adaptive preprocessing (``DualFootAdaptiveBank``) → sliding window.

  2) **Layer 1 — kneepad ACTIVE / INACTIVE gate (strict 4095 rule).**

     Rule:
       * If *either* knee-stretch raw ADC is **not at the 4095 rail**
         (i.e. ``min(raw_knee_l, raw_knee_r) < KNEE_RAW_STRAIGHT_TH``)
         → Layer-1 candidate = ACTIVE.
       * Only if **both** knee raw ADC are at / very near 4095 simultaneously
         → Layer-1 candidate = INACTIVE.

     Debouncing:
       * Sliding-window sustain / release counters (``KNEE_GATE_SUSTAIN_SAMPLES``
         / ``KNEE_GATE_RELEASE_SAMPLES``) plus a minimum-hold timer
         (``KNEE_GATE_MIN_HOLD_S``) so the branch cannot flicker on tiny
         noise spikes while standing still.

  3) **Layer 2 — MOTION / STATIC sub-branch.**

     * ACTIVE branch:
        - MOTION  → STAIRS_UP / STAIRS_DOWN   (periodic 4095 spikes + step events)
        - STATIC  → SITTING_NORMAL / SITTING_CROSSLEGGED (no 4095 at all)

     * INACTIVE branch (stricter; user spec):
        - Must see ``WALK_ENTER_MIN_STEPS`` *real* foot-lift step events
          (bilateral foot-contact detector, not pressure amplitude alone) and
          ``WALK_ENTER_MIN_STEPS`` recent knee swings that reached ≥ 4095 rail
          before allowing STATIC → MOTION (WALKING_FWD / WALKING_BWD).
        - Small body tremor / weight shift while standing is suppressed.

  4) **Branch RF:** exactly one of four fully-independent models is invoked:
        ``rf_active_motion.joblib``   → STAIRS_UP / STAIRS_DOWN
        ``rf_active_static.joblib``   → SITTING_NORMAL / SITTING_CROSSLEGGED
        ``rf_inactive_motion.joblib`` → WALKING_FORWARD / WALKING_BACKWARD
        ``rf_inactive_static.joblib`` → STANDING_UPRIGHT / LEFT_LEAN / RIGHT_LEAN

Gait-signature scalars are appended as auxiliary features (same layout as training,
see ``ml_branch_models.build_auxiliary_vector``).

``SIT_TO_STAND`` is rule-only, not an RF class.

Dependencies: numpy, collections.deque, time  (NO scipy)
"""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

from ml_activity_features import WINDOW_SIZE as ML_WINDOW_SIZE
from ml_branch_models import (
    BranchRFEnsemble,
    build_auxiliary_vector,
    build_full_feature_vector,
    BRANCH_TO_FILE,
)
from adaptive_preprocessing import CHANNEL_NAMES_DUAL, DualFootAdaptiveBank

# ═══════════════════════════════════════════════════════════════════════════
#  TUNEABLE PARAMETERS  (all marked TODO_PARAM)
# ═══════════════════════════════════════════════════════════════════════════

SENSOR_MAX = 4095.0

# TODO_PARAM: Must match MCU firmware line rate. Current hardware sends one dual-foot
# frame every 100 ms (verified from CSV timestamp deltas), i.e. 10 Hz. Keep this in
# sync with ml_activity_features.SAMPLE_HZ and retrain the four RFs on any change.
SAMPLE_HZ = 10

# TODO_PARAM: EMA smoothing factor, 0 < α ≤ 1
EMA_ALPHA = 0.25

# ── Heel step detection (legacy fallback) ─────────────────────────────────

# TODO_PARAM: Second EMA on heel (after per-channel EMA) for step detection only
HEEL_STEP_EMA_ALPHA = 0.35

# TODO_PARAM: Initial Schmitt LOW/HIGH (auto-adjusted after ≥3 steps from recent peaks/troughs)
STEP_INIT_LOW = 0.25
STEP_INIT_HIGH = 0.45

# TODO_PARAM: Per-foot cool-down (s).  0.30 s → max 200 steps/min per foot.
STEP_COOLDOWN_S = 0.30
STEP_COOLDOWN_SAMPLES = max(4, int(round(SAMPLE_HZ * STEP_COOLDOWN_S)))

# TODO_PARAM: Adaptive threshold uses last N peaks/troughs
ADAPTIVE_HISTORY = 10

# TODO_PARAM: Minimum swing (peak-trough) to consider signal valid for step detection
ADAPTIVE_MIN_SWING = 0.04

# TODO_PARAM: Fraction of swing for LOW / HIGH thresholds
ADAPTIVE_LOW_FRAC = 0.30
ADAPTIVE_HIGH_FRAC = 0.60

# TODO_PARAM: Hysteresis for peak/trough tracking (fraction of current swing)
PEAK_TROUGH_HYST = 0.08

# ── Foot-contact step detection (primary bilateral method) ────────────────

# TODO_PARAM: per-foot total load (toe+ff+heel sum) below this → foot considered lifted
FOOT_OFF_GROUND_TH = 0.10

# TODO_PARAM: per-foot total load above this → foot considered on ground
FOOT_ON_GROUND_TH = 0.20

# TODO_PARAM: minimum time (s) between two valid steps from same foot
STEP_MIN_GAP_S = 0.30

# TODO_PARAM: consecutive samples below off threshold before declaring lifted
FOOT_OFF_MIN_SAMPLES = 3

# TODO_PARAM: consecutive samples above on threshold before declaring landed
FOOT_ON_MIN_SAMPLES = 3

# ── direction detection ──────────────────────────────────────────────────

# TODO_PARAM: Observation window (s) after a step for peak comparison
WINDOW_SECONDS = 0.6

# TODO_PARAM: Recent steps used in majority vote
VOTE_K_STEPS = 3

# ── layer-2 motion vs static (MOTION_BRANCH / STATIC_BRANCH) ─────────────

# TODO_PARAM: Reject motion sub-branch if overall motion amplitude is too small
MOTION_MIN_AMPLITUDE = 0.10

# TODO_PARAM: Consecutive frames with step+amplitude evidence before MOTION_BRANCH
MOTION_CONFIRM_FRAMES = 4

# TODO_PARAM: Minimum time in MOTION_BRANCH before allowing return to STATIC_BRANCH
MOTION_MIN_HOLD_S = 1.0

# TODO_PARAM: Consecutive low-evidence frames (in MOTION) before STATIC transition
LAYER2_STATIC_CONFIRM_FRAMES = 6

# ── walking-entry strong guard (INACTIVE branch only, per spec) ──────────
# Prevents standing micro-tremor from being misclassified as walking.
# To flip STATIC → MOTION under INACTIVE branch we additionally require:
#   (a) at least WALK_ENTER_MIN_STEPS real foot-lift step events, AND
#   (b) at least WALK_ENTER_MIN_STEPS knee-extension witnesses within the
#       last WALK_EVIDENCE_WINDOW_S seconds (i.e. the stretch sensor has
#       actually reached the 4095 rail during swing phases).

# TODO_PARAM: minimum real foot-lift steps required before entering WALKING
WALK_ENTER_MIN_STEPS = 2
# TODO_PARAM: rolling window (s) to accumulate step + knee-extend evidence
WALK_EVIDENCE_WINDOW_S = 2.0
# TODO_PARAM: raw ADC knee value counting as "extended" swing-phase witness
WALK_KNEE_EXTEND_RAIL_TH = 4080.0

# ── state machine ────────────────────────────────────────────────────────

# TODO_PARAM: Minimum hold time (s) before state can change
STATE_MIN_DURATION_S = 1.0

# TODO_PARAM: Seconds after last step before leaving WALKING/STAIRS
WALK_TIMEOUT_S = 2.0

# ── sitting detection ────────────────────────────────────────────────────

# TODO_PARAM: p_sum < this → sitting
SIT_PSUM_TH = 0.08

# TODO_PARAM: If foot pressure std > this while sitting → cross-legged
SIT_CROSSLEG_STD_TH = 0.04

# TODO_PARAM: If knee deviation from baseline > this while sitting → cross-legged
SIT_CROSSLEG_KNEE_TH = 0.15

# ── layer-1 kneepad gate (ACTIVE_BRANCH vs INACTIVE_BRANCH only) ─────────
# Strict 4095 rule (see module docstring):
#   • Instant ACTIVE  ← min(raw_knee_l, raw_knee_r) <  KNEE_RAW_STRAIGHT_TH
#     (at least one knee is clearly bent, i.e. not at the 4095 rail)
#   • Instant INACTIVE ← min(raw_knee_l, raw_knee_r) >= KNEE_RAW_STRAIGHT_TH
#     (both knee stretch sensors are at / very near the 4095 rail)

# TODO_PARAM: raw ADC value above which a knee is treated as "straight" (rail = 4095).
#   Raise towards 4095 to make ACTIVE more sensitive (reacts to smaller bends);
#   lower (e.g. 3000) to ignore small walking-stance dips and only react to stairs / sitting.
KNEE_RAW_STRAIGHT_TH = 3500.0

# TODO_PARAM: consecutive frames where instant ACTIVE rule holds before switching to ACTIVE_BRANCH.
#   Larger → more robust to single-frame bending artefacts, slower response to a true bend.
KNEE_GATE_SUSTAIN_SAMPLES = max(3, int(round(SAMPLE_HZ * 0.40)))
# TODO_PARAM: consecutive frames failing instant rule before dropping back to INACTIVE_BRANCH.
#   Larger → will stay in ACTIVE longer through brief straightening (e.g. between stair steps).
KNEE_GATE_RELEASE_SAMPLES = max(3, int(round(SAMPLE_HZ * 0.35)))
# TODO_PARAM: after any L1 transition, hold the new branch at least this long (s)
#   before allowing the next transition — state-lock to suppress standing / sitting flicker.
KNEE_GATE_MIN_HOLD_S = 0.6

# ── standing detection ───────────────────────────────────────────────────

# TODO_PARAM: p_sum ≥ this (and no steps) → standing
STAND_PSUM_TH = 0.15

# TODO_PARAM: bilateral left-right lean threshold on lr_ratio
LEAN_LR_TH = 0.12

# ── sit-to-stand ─────────────────────────────────────────────────────────

# TODO_PARAM: Sit-to-stand trigger/confirm thresholds (total foot load sum, pressure domain)
STS_TRIGGER_TH = 0.15
STS_CONFIRM_TH = 0.25
STS_CONFIRM_S = 0.8

# TODO_PARAM: knee stretch (pressure) must exceed sitting baseline by this delta to arm STS
STS_KNEE_DELTA_TH = 0.08

# TODO_PARAM: min rise of total load over STS_TREND_SAMPLES (short trend, pressure domain)
STS_MIN_PSUM_RISE = 0.02

# TODO_PARAM: samples for p_sum trend (sliding min→max within this many samples)
STS_TREND_SAMPLES = 10

# TODO_PARAM: When single-foot stream feeds an 8-ch adaptive model, pad missing foot raw ADC
_SINGLE_FOOT_PAD_RAW = 2048.0


def _snapshot_to_adaptive_debug_dict(s) -> dict:
    return {
        "raw": round(float(s.raw), 2),
        "baseline_raw": round(float(s.baseline_raw), 2),
        "baseline_removed": round(float(s.baseline_removed), 4),
        "relative_pressure_ratio": round(float(s.relative_pressure_ratio), 4),
        "adaptive_zscore": round(float(s.adaptive_zscore), 4),
        "current_state": s.stable_state,
        "confidence": round(float(s.confidence), 4),
        "dynamic_min_raw": round(float(s.dynamic_min_raw), 2),
        "dynamic_max_raw": round(float(s.dynamic_max_raw), 2),
    }

# ═══════════════════════════════════════════════════════════════════════════
#  LAYER-2 MOTION / STATIC SUB-BRANCH
# ═══════════════════════════════════════════════════════════════════════════


class _Layer2MotionStatic:
    """MOTION_BRANCH vs STATIC_BRANCH using step evidence + amplitude hysteresis.

    Under layer1 = INACTIVE branch (standing ↔ walking) we require an *extra*
    ``WALK_ENTER_MIN_STEPS`` real foot-lift step events **and** the same number
    of knee-extension witnesses inside a rolling ``WALK_EVIDENCE_WINDOW_S``
    window — this suppresses body-sway-induced false walking transitions.
    """

    def __init__(self) -> None:
        self._sub = "STATIC_BRANCH"
        self._up_cnt = 0
        self._down_cnt = 0
        self._motion_enter_t = -999.0
        self._last_evidence_t = -999.0
        self._reason = "init"
        self._step_events: deque[float] = deque()
        self._knee_extend_events: deque[float] = deque()

    def _trim(self, dq: deque[float], t: float) -> None:
        cutoff = t - WALK_EVIDENCE_WINDOW_S
        while dq and dq[0] < cutoff:
            dq.popleft()

    def update(
        self,
        raw_step: bool,
        recent_step: bool,
        amp: float,
        t: float,
        *,
        layer1_branch: str = "INACTIVE_BRANCH",
        knee_extended_now: bool = False,
    ) -> tuple[str, str]:
        if raw_step:
            self._step_events.append(t)
        if knee_extended_now:
            self._knee_extend_events.append(t)
        self._trim(self._step_events, t)
        self._trim(self._knee_extend_events, t)

        evidence = (raw_step or recent_step) and amp >= MOTION_MIN_AMPLITUDE
        if evidence:
            self._last_evidence_t = t

        # Extra guard only when layer-1 is Inactive: must see real steps + knee swings.
        walk_guard_ok = True
        walk_guard_reason = ""
        if layer1_branch == "INACTIVE_BRANCH":
            need = WALK_ENTER_MIN_STEPS
            n_steps = len(self._step_events)
            n_knee = len(self._knee_extend_events)
            walk_guard_ok = (n_steps >= need) and (n_knee >= need)
            if not walk_guard_ok:
                walk_guard_reason = (
                    f"walk_guard_waiting(steps={n_steps}/{need},"
                    f"knee_ext={n_knee}/{need})"
                )

        if self._sub == "STATIC_BRANCH":
            if evidence and walk_guard_ok:
                self._up_cnt += 1
                self._down_cnt = 0
                if self._up_cnt >= MOTION_CONFIRM_FRAMES:
                    self._sub = "MOTION_BRANCH"
                    self._motion_enter_t = t
                    self._up_cnt = 0
                    self._reason = "motion_confirmed"
            else:
                self._up_cnt = 0
                self._reason = walk_guard_reason or "static_no_evidence"
            return self._sub, self._reason

        # MOTION_BRANCH
        if evidence:
            self._down_cnt = 0
            self._reason = "motion_sustained"
            return self._sub, self._reason

        self._down_cnt += 1
        self._reason = "motion_cooling"
        hold_ok = (t - self._motion_enter_t) >= MOTION_MIN_HOLD_S
        if self._down_cnt >= LAYER2_STATIC_CONFIRM_FRAMES and hold_ok:
            self._sub = "STATIC_BRANCH"
            self._down_cnt = 0
            self._reason = "static_hold"
        elif (t - self._last_evidence_t) > WALK_TIMEOUT_S and hold_ok:
            self._sub = "STATIC_BRANCH"
            self._down_cnt = 0
            self._reason = "static_timeout"
        return self._sub, self._reason


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER
# ═══════════════════════════════════════════════════════════════════════════

def raw_to_pressure(raw: float) -> float:
    return float(np.clip((SENSOR_MAX - raw) / SENSOR_MAX, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  EMA FILTER
# ═══════════════════════════════════════════════════════════════════════════

class _EMAFilter:
    def __init__(self, alpha: float = EMA_ALPHA):
        self._a = alpha
        self._v: float | None = None

    def update(self, x: float) -> float:
        if self._v is None:
            self._v = x
        else:
            self._v += self._a * (x - self._v)
        return self._v

    @property
    def value(self) -> float:
        return 0.0 if self._v is None else self._v


# ═══════════════════════════════════════════════════════════════════════════
#  ADAPTIVE STEP DETECTOR — per-foot, self-tuning Schmitt thresholds
#  (legacy heel-based; kept as fallback / debug)
# ═══════════════════════════════════════════════════════════════════════════

class _AdaptiveStepDetector:
    """Per-foot step detector with adaptive Schmitt thresholds.

    Tracks recent heel-pressure peaks and troughs.  After ≥ 3 steps the
    LOW / HIGH thresholds are recomputed from the mean of the last
    ``ADAPTIVE_HISTORY`` peaks / troughs, so the detector works regardless
    of which resistors are soldered in.
    """

    def __init__(self) -> None:
        self._low = STEP_INIT_LOW
        self._high = STEP_INIT_HIGH
        self._armed = True
        self._cooldown = 0
        self._prev = 0.0

        # peak / trough tracker
        self._recent_peaks: deque[float] = deque(maxlen=ADAPTIVE_HISTORY)
        self._recent_troughs: deque[float] = deque(maxlen=ADAPTIVE_HISTORY)
        self._tracking_val = 0.0
        self._phase = "seek_peak"   # "seek_peak" | "seek_trough"

    # ── public ────────────────────────────────────────────────────────
    def update(self, heel_smooth: float) -> bool:
        self._track_peaks_troughs(heel_smooth)

        step = False
        if self._cooldown > 0:
            self._cooldown -= 1
        else:
            if (
                self._armed
                and self._prev < self._high
                and heel_smooth >= self._high
            ):
                step = True
                self._armed = False
                self._cooldown = STEP_COOLDOWN_SAMPLES
        if heel_smooth <= self._low:
            self._armed = True
        self._prev = heel_smooth
        return step

    @property
    def thresholds(self) -> tuple[float, float]:
        return self._low, self._high

    # ── internal ──────────────────────────────────────────────────────
    def _track_peaks_troughs(self, v: float) -> None:
        hyst = max(PEAK_TROUGH_HYST, (self._high - self._low) * 0.25)
        if self._phase == "seek_peak":
            if v > self._tracking_val:
                self._tracking_val = v
            elif v < self._tracking_val - hyst:
                self._recent_peaks.append(self._tracking_val)
                self._tracking_val = v
                self._phase = "seek_trough"
                self._recalc()
        else:
            if v < self._tracking_val:
                self._tracking_val = v
            elif v > self._tracking_val + hyst:
                self._recent_troughs.append(self._tracking_val)
                self._tracking_val = v
                self._phase = "seek_peak"
                self._recalc()

    def _recalc(self) -> None:
        if len(self._recent_peaks) < 3 or len(self._recent_troughs) < 3:
            return
        avg_pk = float(np.mean(list(self._recent_peaks)))
        avg_tr = float(np.mean(list(self._recent_troughs)))
        swing = avg_pk - avg_tr
        if swing < ADAPTIVE_MIN_SWING:
            return
        self._low  = avg_tr + ADAPTIVE_LOW_FRAC * swing
        self._high = avg_tr + ADAPTIVE_HIGH_FRAC * swing


class _StepDetectorV2(_AdaptiveStepDetector):
    """Backward-compatible alias so external code that imported this still works."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  FOOT-CONTACT STEP DETECTOR  (per foot, lift → land event counting)
# ═══════════════════════════════════════════════════════════════════════════

class _FootContactStepDetector:
    """Per-foot step detector based on total foot load (toe + forefoot + heel).

    State machine:
      ON_GROUND  --( load < FOOT_OFF_GROUND_TH for N samples )--> LIFTED
      LIFTED     --( load > FOOT_ON_GROUND_TH  for N samples )--> ON_GROUND + step event

    Debounced by consecutive-sample counters and minimum inter-step gap.
    """

    def __init__(self) -> None:
        self._phase = "ON_GROUND"   # "ON_GROUND" | "LIFTED"
        self._off_count = 0
        self._on_count = 0
        self._last_step_t = -999.0

    def update(self, load: float, t: float) -> bool:
        """Feed one sample; returns True when a valid step (foot re-contact) is detected."""
        step = False
        if self._phase == "ON_GROUND":
            if load < FOOT_OFF_GROUND_TH:
                self._off_count += 1
                if self._off_count >= FOOT_OFF_MIN_SAMPLES:
                    self._phase = "LIFTED"
                    self._on_count = 0
            else:
                self._off_count = 0
        elif self._phase == "LIFTED":
            if load > FOOT_ON_GROUND_TH:
                self._on_count += 1
                if self._on_count >= FOOT_ON_MIN_SAMPLES:
                    self._phase = "ON_GROUND"
                    self._off_count = 0
                    if (t - self._last_step_t) >= STEP_MIN_GAP_S:
                        step = True
                        self._last_step_t = t
            else:
                self._on_count = 0
        return step

    @property
    def phase(self) -> str:
        return self._phase


# ═══════════════════════════════════════════════════════════════════════════
#  DIRECTION DETECTOR  (per-step toe-vs-heel peak timing)
# ═══════════════════════════════════════════════════════════════════════════

class _DirectionDetector:
    """heel peaks first → forward ;  toe peaks first → backward"""

    def __init__(self):
        self._collecting = False
        self._t0 = 0.0
        self._buf: list[tuple[float, float, float]] = []

    def on_step(self, t: float):
        self._collecting = True
        self._t0 = t
        self._buf.clear()

    def feed(self, t: float, toe_p: float, heel_p: float) -> str | None:
        if not self._collecting:
            return None
        self._buf.append((t, toe_p, heel_p))
        if (t - self._t0) < WINDOW_SECONDS:
            return None
        self._collecting = False
        if not self._buf:
            return "unknown"
        t_toe = max(self._buf, key=lambda r: r[1])[0]
        t_heel = max(self._buf, key=lambda r: r[2])[0]
        if abs(t_heel - t_toe) < 0.02:
            return "unknown"
        return "forward" if t_heel < t_toe else "backward"


# ═══════════════════════════════════════════════════════════════════════════
#  DIRECTION VOTING  (majority of last K steps)
# ═══════════════════════════════════════════════════════════════════════════

class _DirectionVoting:
    def __init__(self, k: int = VOTE_K_STEPS):
        self._buf: deque[str] = deque(maxlen=k)

    def push(self, d: str):
        if d != "unknown":
            self._buf.append(d)

    @property
    def result(self) -> str:
        if not self._buf:
            return "unknown"
        fwd = sum(1 for d in self._buf if d == "forward")
        bwd = len(self._buf) - fwd
        if fwd > bwd:
            return "forward"
        if bwd > fwd:
            return "backward"
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
#  SIT-TO-STAND DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class _SitToStandDetector:
    """Sit→stand from rules only (load + knee vs baseline + short p_sum trend)."""

    def __init__(self) -> None:
        self._phase = "idle"
        self._trigger_t = 0.0
        self._confirm_t = 0.0
        self.last_duration: float | None = None
        self._psum_ring: deque[float] = deque(maxlen=max(3, STS_TREND_SAMPLES))
        self._last_p_sum = 0.0

    def update(
        self,
        p_sum: float,
        p_knee: float,
        sm_state: str,
        knee_baseline: float | None,
        t: float,
    ) -> tuple[bool, bool]:
        """
        Returns (force_sit_to_stand, post_complete_standing).
        If post_complete_standing, ``last_duration`` was just set; caller should
        propose standing on this frame.
        """
        self._last_p_sum = p_sum
        self._psum_ring.append(p_sum)
        trend_ok = True
        if len(self._psum_ring) >= 3:
            trend_ok = (max(self._psum_ring) - min(self._psum_ring)) >= STS_MIN_PSUM_RISE

        is_sitting = sm_state.startswith("SITTING")
        knee_ok = True
        if knee_baseline is not None:
            knee_ok = (p_knee - knee_baseline) >= STS_KNEE_DELTA_TH

        if self._phase == "idle":
            if (
                is_sitting
                and p_sum >= STS_TRIGGER_TH
                and knee_ok
                and trend_ok
            ):
                self._phase = "triggered"
                self._trigger_t = t
            return (False, False)

        if self._phase == "triggered":
            if p_sum < STS_TRIGGER_TH * 0.5:
                self._phase = "idle"
                return (False, False)
            if p_sum >= STS_CONFIRM_TH:
                self._phase = "confirming"
                self._confirm_t = t
            return (True, False)

        if self._phase == "confirming":
            if p_sum < STS_CONFIRM_TH:
                self._phase = "triggered"
                return (True, False)
            if (t - self._confirm_t) >= STS_CONFIRM_S:
                self.last_duration = t - self._trigger_t
                self._phase = "idle"
                self._psum_ring.clear()
                return (False, True)
            return (True, False)

        self._phase = "idle"
        return (False, False)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def trigger_level(self) -> float:
        th = STS_TRIGGER_TH if STS_TRIGGER_TH > 1e-9 else 1e-9
        return float(self._last_p_sum / th)

    @property
    def confirm_level(self) -> float:
        th = STS_CONFIRM_TH if STS_CONFIRM_TH > 1e-9 else 1e-9
        return float(self._last_p_sum / th)


# ═══════════════════════════════════════════════════════════════════════════
#  STATE MACHINE  (debounced, minimum-hold)
# ═══════════════════════════════════════════════════════════════════════════

ALL_STATES = {
    "WALKING_FORWARD", "WALKING_BACKWARD",
    "STAIRS_UP", "STAIRS_DOWN",
    "SITTING_NORMAL", "SITTING_CROSSLEGGED",
    "SIT_TO_STAND",
    "STANDING_UPRIGHT", "STANDING_LEFT_LEAN", "STANDING_RIGHT_LEAN",
    "UNKNOWN",
}

# RF training targets (subset of ALL_STATES): no SIT_TO_STAND
RF_TRAINING_LABELS = ALL_STATES - {"SIT_TO_STAND"}


class _StateMachine:
    def __init__(self):
        self.state = "UNKNOWN"
        self._entered_t = 0.0

    def propose(self, candidate: str, t: float, *, immediate: bool = False) -> str:
        if candidate == self.state:
            return self.state
        if immediate or (t - self._entered_t) >= STATE_MIN_DURATION_S:
            self.state = candidate
            self._entered_t = t
        return self.state


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN CLASS: OnlineRecognizer
# ═══════════════════════════════════════════════════════════════════════════

class OnlineRecognizer:
    """Call ``update()`` or ``update_bilateral()`` once per sample.

    Returns a dict with keys:
      state, step_event, walk_dir, counters, sts_last_duration_s, debug
    """

    _STEP_STATES = {"WALKING_FORWARD", "WALKING_BACKWARD", "STAIRS_UP", "STAIRS_DOWN"}

    def __init__(
        self,
        calibration: "object | str | None" = "auto",
    ):
        """
        Parameters
        ----------
        calibration : PersonalCalibration | str (path) | "auto" | None
            * :class:`personal_calibration.PersonalCalibration` — use as-is.
            * ``str``                — path to a JSON file produced by either the
              offline auto-calibrator or the UI online two-step wizard.
            * ``"auto"`` (default)   — try to load
              ``personal_calibration.json`` from the current working directory;
              if missing, fall back to ``None`` silently.
            * ``None``               — disable personal calibration entirely
              (legacy behaviour; the EWMA bank sees raw ADC).

            The Layer-1 knee gate (strict 4095 rule, ``KNEE_RAW_STRAIGHT_TH``)
            always reads the **original** raw ADC before any calibration
            rescaling, so the "single leg raw ≠ 4095 → ACTIVE" rule is
            unaffected by this knob.
        """
        # ── Personal normalization (optional) ───────────────────────────
        self._calibration = self._resolve_calibration(calibration)
        # Left-foot EMA filters
        self._f_toe  = _EMAFilter(EMA_ALPHA)
        self._f_ff   = _EMAFilter(EMA_ALPHA)
        self._f_heel = _EMAFilter(EMA_ALPHA)
        self._f_knee = _EMAFilter(EMA_ALPHA)

        # Right-foot EMA filters (update_bilateral only)
        self._f_toe_r  = _EMAFilter(EMA_ALPHA)
        self._f_ff_r   = _EMAFilter(EMA_ALPHA)
        self._f_heel_r = _EMAFilter(EMA_ALPHA)
        self._f_knee_r = _EMAFilter(EMA_ALPHA)

        # Heel-based step detectors (legacy fallback / debug)
        self._heel_step_l = _EMAFilter(HEEL_STEP_EMA_ALPHA)
        self._heel_step_r = _EMAFilter(HEEL_STEP_EMA_ALPHA)
        self._step_det_l  = _AdaptiveStepDetector()
        self._step_det_r  = _AdaptiveStepDetector()
        self._step_det    = _AdaptiveStepDetector()   # single-foot fallback

        # Foot-contact step detectors (primary bilateral method)
        self._contact_det_l      = _FootContactStepDetector()
        self._contact_det_r      = _FootContactStepDetector()
        self._contact_det_single = _FootContactStepDetector()

        # Direction
        self._dir_det  = _DirectionDetector()
        self._dir_vote = _DirectionVoting(VOTE_K_STEPS)
        self._last_valid_walk_dir: str | None = None

        # Sit-to-stand
        self._sts_det = _SitToStandDetector()

        # State machine
        self._sm = _StateMachine()

        # Four-branch RF ensemble (project working directory is expected to hold *.joblib)
        self._branch_models = BranchRFEnsemble()
        self._layer2 = _Layer2MotionStatic()
        self._last_adaptive_snaps: list | None = None

        self._counters = {
            "forward_steps":  0,
            "backward_steps": 0,
            "up_steps":       0,
            "down_steps":     0,
            "total_steps":    0,
        }
        self._last_step_t = 0.0

        # Knee baseline for cross-legged detection (learned from first N samples)
        self._knee_baseline: float | None = None
        self._knee_init_buf: list[float] = []

        # Online adaptive preprocessors (8 channels; single-foot path pads missing side).
        # If the active calibration carries global statistics, freeze every channel to
        # the globally-learned baseline / press range / mean / std so that inference
        # sees the same numbers training did.  Fall back to EWMA if only
        # [min_raw, max_raw] are present (legacy or identity calibration).
        _calib = self._calibration
        _seeds = None
        if _calib is not None and getattr(_calib, "has_global_stats", False):
            _seeds = _calib.to_channel_seeds()
        self._adapt_bank = DualFootAdaptiveBank(seeds=_seeds)

        # Pressure / feature history for ML sliding window:
        # legacy → 4 or 8 floats (pressure); adaptive_v2 → 24 floats (8×[br, ratio, z])
        self._p_hist: deque[np.ndarray] = deque(
            maxlen=max(ML_WINDOW_SIZE, int(SAMPLE_HZ * 2)),
        )
        self._motion_amp_hist: deque[float] = deque(maxlen=max(8, int(SAMPLE_HZ * 0.6)))
        self._zone_contact_state = {"toe": False, "forefoot": False, "heel": False}
        self._zone_event_hist: deque[tuple[str, str, float]] = deque(maxlen=24)

        # Layer-1 knee gate (strict 4095 rule, debounced + min-hold)
        self._layer1_branch = "INACTIVE_BRANCH"
        self._knee_gate_sustain_cnt = 0
        self._knee_gate_release_cnt = 0
        self._layer1_last_switch_t = -999.0

    def _knee_gate_instant_active(self, min_raw_knee: float) -> bool:
        """Strict 4095 rule: any knee NOT at the rail → instant ACTIVE.

        ``min_raw_knee`` = min(raw_knee_l, raw_knee_r); in the single-foot
        code path the missing side is passed as 4095 so the rule reduces to
        one knee only. A knee is treated as "straight" when its raw ADC
        is >= ``KNEE_RAW_STRAIGHT_TH`` (i.e. essentially on the 4095 rail).
        """
        return float(min_raw_knee) < KNEE_RAW_STRAIGHT_TH

    def _update_layer1_knee_gate(
        self, min_raw_knee: float, t: float,
    ) -> tuple[str, dict[str, object]]:
        """Debounced Layer-1 gate with sustain + release + minimum-hold timer."""
        cond = self._knee_gate_instant_active(min_raw_knee)
        phase = "inactive"
        hold_ok = (t - self._layer1_last_switch_t) >= KNEE_GATE_MIN_HOLD_S

        if self._layer1_branch == "INACTIVE_BRANCH":
            if cond:
                self._knee_gate_sustain_cnt += 1
                self._knee_gate_release_cnt = 0
                if self._knee_gate_sustain_cnt >= KNEE_GATE_SUSTAIN_SAMPLES and hold_ok:
                    self._layer1_branch = "ACTIVE_BRANCH"
                    self._knee_gate_sustain_cnt = 0
                    self._layer1_last_switch_t = t
                    phase = "active"
                else:
                    phase = "arming" if hold_ok else "locked_inactive"
            else:
                self._knee_gate_sustain_cnt = 0
                self._knee_gate_release_cnt = 0
                phase = "inactive"
        else:
            if not cond:
                self._knee_gate_release_cnt += 1
                self._knee_gate_sustain_cnt = 0
                if self._knee_gate_release_cnt >= KNEE_GATE_RELEASE_SAMPLES and hold_ok:
                    self._layer1_branch = "INACTIVE_BRANCH"
                    self._knee_gate_release_cnt = 0
                    self._layer1_last_switch_t = t
                    phase = "inactive"
                else:
                    phase = "releasing" if hold_ok else "locked_active"
            else:
                self._knee_gate_release_cnt = 0
                phase = "active"

        info: dict[str, object] = {
            "knee_gate_phase": phase,
            "knee_gate_min_raw": round(float(min_raw_knee), 1),
            "knee_gate_straight_th": float(KNEE_RAW_STRAIGHT_TH),
            "knee_gate_sustain_cnt": int(self._knee_gate_sustain_cnt),
            "knee_gate_sustain_need": int(KNEE_GATE_SUSTAIN_SAMPLES),
            "knee_gate_release_cnt": int(self._knee_gate_release_cnt),
            "knee_gate_release_need": int(KNEE_GATE_RELEASE_SAMPLES),
            "knee_gate_min_hold_s": float(KNEE_GATE_MIN_HOLD_S),
        }
        return self._layer1_branch, info

    def _compute_min_knee_raw(
        self,
        raw_knee_l: float,
        raw_knee_r: float,
        *,
        single_foot: bool = False,
    ) -> float:
        """Layer-1 metric: minimum of the two knee raw ADCs (the bent-most knee)."""
        if single_foot:
            return float(raw_knee_l)
        return float(min(raw_knee_l, raw_knee_r))

    @staticmethod
    def _resolve_calibration(arg):
        """Turn the ``calibration=`` constructor argument into a
        :class:`PersonalCalibration` or ``None``.  Shared by offline and
        online code paths so both behave identically.
        """
        if arg is None:
            return None
        try:
            import personal_calibration as _pc
        except Exception as exc:       # pragma: no cover  — defensive
            print(f"[OnlineRecognizer] personal_calibration unavailable: {exc}")
            return None
        if isinstance(arg, _pc.PersonalCalibration):
            return arg
        if arg == "auto":
            calib = _pc.load_default_calibration((".",))
            if calib is not None:
                print(f"[OnlineRecognizer] loaded calibration "
                      f"(source={calib.source!r}, subject={calib.subject!r})")
            return calib
        if isinstance(arg, str):
            return _pc.PersonalCalibration.load_json(arg)
        raise TypeError(f"Unsupported calibration argument type: {type(arg)!r}")

    def _calibrate_raw8(self, raw8: np.ndarray) -> np.ndarray:
        """Apply personal [min,max] → [0,4095] per-channel rescaling if
        a calibration is loaded.  Layer-1 knee gate is *NOT* fed from this
        output; it reads the original raw values from ``raw_*`` inputs
        (strict 4095 rule must survive calibration)."""
        if self._calibration is None:
            return raw8
        return np.asarray(
            self._calibration.normalize_to_adc(raw8), dtype=np.float64,
        )

    def _branch_key_from_layers(self, active: bool, motion_sub: str) -> str:
        if active:
            return (
                "active_motion"
                if motion_sub == "MOTION_BRANCH"
                else "active_static"
            )
        return (
            "inactive_motion"
            if motion_sub == "MOTION_BRANCH"
            else "inactive_static"
        )

    def _hierarchical_rf_predict(
        self,
        branch_key: str,
        sign: dict[str, object],
        left_load: float,
        right_load: float,
        lr_ratio: float,
        p_sum: float,
    ) -> tuple[str, float, str, str]:
        fname = BRANCH_TO_FILE.get(branch_key, "")
        b = self._branch_models.bundle(branch_key)
        if not b.available:
            return "UNKNOWN", 0.0, "model_missing", fname
        aux = build_auxiliary_vector(sign, left_load, right_load, lr_ratio, p_sum)
        win = np.array(list(self._p_hist)[-ML_WINDOW_SIZE:])
        if win.ndim != 2 or win.shape[1] != 24:
            return "UNKNOWN", 0.0, "window_not_adaptive_v2", fname
        feat = build_full_feature_vector(win, aux)
        if feat is None:
            return "UNKNOWN", 0.0, "feature_error", fname
        lab, pr, rj = b.predict(feat)
        return lab, float(pr), rj, fname

    # ── public API ───────────────────────────────────────────────────────

    def update(
        self,
        raw_toe: float,
        raw_forefoot: float,
        raw_heel: float,
        raw_knee: float,
        t: float | None = None,
    ) -> dict:
        """Single-foot update (4 channels)."""
        if t is None:
            t = time.monotonic()

        adaptive_dbg: dict[str, dict] = {}
        raw8 = np.array(
            [
                raw_toe,
                raw_forefoot,
                raw_heel,
                raw_knee,
                _SINGLE_FOOT_PAD_RAW,
                _SINGLE_FOOT_PAD_RAW,
                _SINGLE_FOOT_PAD_RAW,
                _SINGLE_FOOT_PAD_RAW,
            ],
            dtype=np.float64,
        )
        raw8_calib = self._calibrate_raw8(raw8)
        _flat24, snaps = self._adapt_bank.update(raw8_calib)
        self._last_adaptive_snaps = list(snaps)
        p_toe = self._f_toe.update(snaps[0].relative_pressure_ratio)
        p_ff = self._f_ff.update(snaps[1].relative_pressure_ratio)
        p_heel = self._f_heel.update(snaps[2].relative_pressure_ratio)
        p_knee = self._f_knee.update(snaps[3].relative_pressure_ratio)
        p_sum = p_toe + p_ff + p_heel
        self._p_hist.append(_flat24)
        for nm, sn in zip(CHANNEL_NAMES_DUAL, snaps):
            adaptive_dbg[nm] = _snapshot_to_adaptive_debug_dict(sn)

        if self._knee_baseline is None:
            self._knee_init_buf.append(p_knee)
            if len(self._knee_init_buf) >= 30:
                self._knee_baseline = float(np.mean(self._knee_init_buf))
        min_raw_knee = self._compute_min_knee_raw(raw_knee, 0.0, single_foot=True)
        layer1, knee_gate_info = self._update_layer1_knee_gate(min_raw_knee, t)
        knee_extended_now = float(raw_knee) >= WALK_KNEE_EXTEND_RAIL_TH
        self._update_zone_sequence(p_toe, p_ff, p_heel, t)

        contact_step = self._contact_det_single.update(p_sum, t)
        step_source: str | None = "single_contact" if contact_step else None
        heel_smooth = self._heel_step_l.update(p_heel)
        heel_step = self._step_det.update(heel_smooth)
        raw_step = contact_step
        if not raw_step and heel_step:
            raw_step = True
            step_source = "heel_fallback"

        if raw_step:
            self._dir_det.on_step(t)
            self._last_step_t = t

        per_step_dir = self._dir_det.feed(t, p_toe, p_heel)
        if per_step_dir is not None:
            self._dir_vote.push(per_step_dir)
        walk_dir = self._dir_vote.result
        if walk_dir in ("forward", "backward"):
            self._last_valid_walk_dir = walk_dir

        recent_step = (t - self._last_step_t) < WALK_TIMEOUT_S
        amp = self._estimate_motion_amplitude(p_toe, p_ff, p_heel, p_knee)
        layer2, layer2_reason = self._layer2.update(
            raw_step,
            recent_step,
            amp,
            t,
            layer1_branch=layer1,
            knee_extended_now=knee_extended_now,
        )
        sign = self._gait_signature_from_pressures(
            p_toe, p_ff, p_heel, p_knee,
        )
        br_key = self._branch_key_from_layers(
            layer1 == "ACTIVE_BRANCH", layer2,
        )
        cand, rf_p, rf_rej, rf_name = self._hierarchical_rf_predict(
            br_key,
            sign,
            p_sum,
            0.0,
            0.0,
            p_sum,
        )

        force_sts, post_sts = self._sts_det.update(
            p_sum, p_knee, self._sm.state, self._knee_baseline, t,
        )
        if post_sts:
            candidate = "STANDING_UPRIGHT"
        elif force_sts:
            candidate = "SIT_TO_STAND"
        else:
            candidate = cand

        immediate = force_sts or post_sts
        state = self._sm.propose(candidate, t, immediate=immediate)

        step_event = False
        if raw_step and state in self._STEP_STATES:
            step_event = True
            self._counters["total_steps"] += 1
            if state == "STAIRS_UP":
                self._counters["up_steps"] += 1
            elif state == "STAIRS_DOWN":
                self._counters["down_steps"] += 1
            elif state == "WALKING_FORWARD":
                self._counters["forward_steps"] += 1
            elif state == "WALKING_BACKWARD":
                self._counters["backward_steps"] += 1

        dbg = {
            "p_toe":  round(p_toe, 3),
            "p_ff":   round(p_ff, 3),
            "p_heel": round(p_heel, 3),
            "p_knee": round(p_knee, 3),
            "p_sum":  round(p_sum, 3),
            "left_load":          round(p_sum, 3),
            "right_load":         0.0,
            "lr_ratio":           0.0,
            "last_valid_walk_dir": self._last_valid_walk_dir,
            "step_source":        step_source,
            "left_foot_phase":    self._contact_det_single.phase,
            "right_foot_phase":   None,
            "raw_dir":  per_step_dir,
            "layer1_branch":      layer1,
            "layer2_subbranch":   layer2,
            "branch_rf_key":      br_key,
            "branch_rf_file":     rf_name,
            "ml_label":           cand,
            "knee_min_raw":       round(min_raw_knee, 1),
            "layer2_reason":    layer2_reason,
            "rf_proba":           round(rf_p, 4),
            "rf_reject":          rf_rej,
            "sts_phase": self._sts_det.phase,
            "sts_trigger_level": round(self._sts_det.trigger_level, 3),
            "sts_confirm_level": round(self._sts_det.confirm_level, 3),
            "ml_feature_mode": "branch_adaptive_v2",
        }
        dbg.update(knee_gate_info)
        dbg.update(sign)
        if adaptive_dbg:
            dbg["adaptive"] = adaptive_dbg
        return {
            "state":              state,
            "step_event":         step_event,
            "walk_dir":           walk_dir,
            "counters":           dict(self._counters),
            "sts_last_duration_s": self._sts_det.last_duration,
            "debug": dbg,
        }

    def update_bilateral(
        self,
        raw_left: tuple[float, float, float, float],
        raw_right: tuple[float, float, float, float],
        t: float | None = None,
    ) -> dict:
        """
        Bilateral fusion: left (toe, forefoot, heel, knee) + right same order.
        Primary step detection via foot-contact (lift→land) per foot.
        """
        if t is None:
            t = time.monotonic()

        lt, lf, lh, lk = raw_left
        rt, rf, rh, rk = raw_right

        adaptive_dbg: dict[str, dict] = {}
        raw8 = np.array([lt, lf, lh, lk, rt, rf, rh, rk], dtype=np.float64)
        raw8_calib = self._calibrate_raw8(raw8)
        _flat24, snaps = self._adapt_bank.update(raw8_calib)
        self._last_adaptive_snaps = list(snaps)
        p_toe_l = self._f_toe.update(snaps[0].relative_pressure_ratio)
        p_ff_l = self._f_ff.update(snaps[1].relative_pressure_ratio)
        p_heel_l = self._f_heel.update(snaps[2].relative_pressure_ratio)
        p_knee_l = self._f_knee.update(snaps[3].relative_pressure_ratio)
        p_toe_r = self._f_toe_r.update(snaps[4].relative_pressure_ratio)
        p_ff_r = self._f_ff_r.update(snaps[5].relative_pressure_ratio)
        p_heel_r = self._f_heel_r.update(snaps[6].relative_pressure_ratio)
        p_knee_r = self._f_knee_r.update(snaps[7].relative_pressure_ratio)
        self._p_hist.append(_flat24)
        for nm, sn in zip(CHANNEL_NAMES_DUAL, snaps):
            adaptive_dbg[nm] = _snapshot_to_adaptive_debug_dict(sn)

        left_load = p_toe_l + p_ff_l + p_heel_l
        right_load = p_toe_r + p_ff_r + p_heel_r
        p_sum = left_load + right_load
        p_knee_avg = 0.5 * (p_knee_l + p_knee_r)
        lr_ratio = (left_load - right_load) / (left_load + right_load + 1e-9)

        if self._knee_baseline is None:
            self._knee_init_buf.append(p_knee_avg)
            if len(self._knee_init_buf) >= 30:
                self._knee_baseline = float(np.mean(self._knee_init_buf))
        min_raw_knee = self._compute_min_knee_raw(lk, rk)
        layer1, knee_gate_info = self._update_layer1_knee_gate(min_raw_knee, t)
        # Knee-extension witness: either knee reaches the 4095 rail (swing phase)
        knee_extended_now = max(float(lk), float(rk)) >= WALK_KNEE_EXTEND_RAIL_TH
        dom_left = left_load >= right_load
        dom_toe = p_toe_l if dom_left else p_toe_r
        dom_ff = p_ff_l if dom_left else p_ff_r
        dom_heel = p_heel_l if dom_left else p_heel_r
        self._update_zone_sequence(dom_toe, dom_ff, dom_heel, t)

        # 2. Foot-contact step detection (primary)
        step_l = self._contact_det_l.update(left_load, t)
        step_r = self._contact_det_r.update(right_load, t)
        contact_step = step_l or step_r
        step_source: str | None = None
        if step_l:
            step_source = "left_contact"
        elif step_r:
            step_source = "right_contact"

        # Heel-based step detection (fallback / debug)
        sl = self._heel_step_l.update(p_heel_l)
        sr = self._heel_step_r.update(p_heel_r)
        heel_step_l = self._step_det_l.update(sl)
        heel_step_r = self._step_det_r.update(sr)
        heel_step = heel_step_l or heel_step_r

        raw_step = contact_step
        if not raw_step and heel_step:
            raw_step = True
            step_source = "heel_fallback"

        if raw_step:
            self._last_step_t = t
            self._dir_det.on_step(t)

        # 3. Direction
        per_step_dir = self._dir_det.feed(
            t, max(p_toe_l, p_toe_r), max(p_heel_l, p_heel_r),
        )
        if per_step_dir is not None:
            self._dir_vote.push(per_step_dir)
        walk_dir = self._dir_vote.result
        if walk_dir in ("forward", "backward"):
            self._last_valid_walk_dir = walk_dir

        recent_step = (t - self._last_step_t) < WALK_TIMEOUT_S
        amp = self._estimate_motion_amplitude(
            max(p_toe_l, p_toe_r),
            max(p_ff_l, p_ff_r),
            max(p_heel_l, p_heel_r),
            p_knee_avg,
        )
        layer2, layer2_reason = self._layer2.update(
            raw_step,
            recent_step,
            amp,
            t,
            layer1_branch=layer1,
            knee_extended_now=knee_extended_now,
        )
        sign = self._gait_signature_from_pressures(
            dom_toe, dom_ff, dom_heel, p_knee_avg,
        )
        br_key = self._branch_key_from_layers(
            layer1 == "ACTIVE_BRANCH", layer2,
        )
        cand, rf_p, rf_rej, rf_name = self._hierarchical_rf_predict(
            br_key,
            sign,
            left_load,
            right_load,
            lr_ratio,
            p_sum,
        )

        force_sts, post_sts = self._sts_det.update(
            p_sum, p_knee_avg, self._sm.state, self._knee_baseline, t,
        )
        if post_sts:
            candidate = self._classify_standing_bilateral(left_load, right_load)
        elif force_sts:
            candidate = "SIT_TO_STAND"
        else:
            candidate = cand

        immediate = force_sts or post_sts
        state = self._sm.propose(candidate, t, immediate=immediate)

        # Step counting — ONLY in locomotion states
        step_event = False
        if raw_step and state in self._STEP_STATES:
            step_event = True
            self._counters["total_steps"] += 1
            if state == "STAIRS_UP":
                self._counters["up_steps"] += 1
            elif state == "STAIRS_DOWN":
                self._counters["down_steps"] += 1
            elif state == "WALKING_FORWARD":
                self._counters["forward_steps"] += 1
            elif state == "WALKING_BACKWARD":
                self._counters["backward_steps"] += 1

        th_l = self._step_det_l.thresholds
        th_r = self._step_det_r.thresholds
        dbg = {
            "p_toe_l":          round(p_toe_l, 3),
            "p_ff_l":           round(p_ff_l, 3),
            "p_heel_l":         round(p_heel_l, 3),
            "p_knee_l":         round(p_knee_l, 3),
            "p_toe_r":          round(p_toe_r, 3),
            "p_ff_r":           round(p_ff_r, 3),
            "p_heel_r":         round(p_heel_r, 3),
            "p_knee_r":         round(p_knee_r, 3),
            "p_sum":            round(p_sum, 3),
            "left_load":        round(left_load, 3),
            "right_load":       round(right_load, 3),
            "lr_ratio":         round(lr_ratio, 3),
            "last_valid_walk_dir": self._last_valid_walk_dir,
            "step_source":      step_source,
            "left_foot_phase":  self._contact_det_l.phase,
            "right_foot_phase": self._contact_det_r.phase,
            "heel_combined":    round(max(sl, sr), 3),
            "heel_smooth_l":    round(sl, 3),
            "heel_smooth_r":    round(sr, 3),
            "step_th_l":        (round(th_l[0], 3), round(th_l[1], 3)),
            "step_th_r":        (round(th_r[0], 3), round(th_r[1], 3)),
            "raw_dir":          per_step_dir,
            "layer1_branch":    layer1,
            "layer2_subbranch": layer2,
            "branch_rf_key":    br_key,
            "branch_rf_file":   rf_name,
            "ml_label":         cand,
            "knee_min_raw":     round(min_raw_knee, 1),
            "layer2_reason":    layer2_reason,
            "rf_proba":         round(rf_p, 4),
            "rf_reject":        rf_rej,
            "sts_phase":        self._sts_det.phase,
            "sts_trigger_level": round(self._sts_det.trigger_level, 3),
            "sts_confirm_level": round(self._sts_det.confirm_level, 3),
            "ml_feature_mode":  "branch_adaptive_v2",
        }
        dbg.update(knee_gate_info)
        dbg.update(sign)
        if adaptive_dbg:
            dbg["adaptive"] = adaptive_dbg
        return {
            "state":              state,
            "step_event":         step_event,
            "walk_dir":           walk_dir,
            "counters":           dict(self._counters),
            "sts_last_duration_s": self._sts_det.last_duration,
            "debug": dbg,
        }

    # ── internal helpers ─────────────────────────────────────────────────

    def _resolve_walk_direction(self) -> str | None:
        """Return effective walk direction ('forward'/'backward') or None."""
        d = self._dir_vote.result
        if d in ("forward", "backward"):
            return d
        return self._last_valid_walk_dir

    def _is_motion_state(self, s: str) -> bool:
        return s in self._STEP_STATES

    def _estimate_motion_amplitude(
        self,
        toe: float,
        ff: float,
        heel: float,
        knee: float,
    ) -> float:
        vals = [toe, ff, heel, knee]
        inst = float(max(vals) - min(vals))
        self._motion_amp_hist.append(inst)
        if not self._motion_amp_hist:
            return inst
        return float(max(self._motion_amp_hist) - min(self._motion_amp_hist))

    def _gait_signature_from_pressures(
        self,
        toe: float,
        ff: float,
        heel: float,
        knee: float,
    ) -> dict[str, object]:
        total = max(toe + ff + heel, 1e-9)
        heel_impact = heel / total
        forefoot_dom = (toe + ff) / total
        knee_activity = (
            abs(knee - self._knee_baseline)
            if self._knee_baseline is not None
            else knee
        )

        initial, contact_order, release_order, complete = self._extract_gait_orders()

        return {
            "initial_contact_zone": initial,
            "contact_order": contact_order,
            "release_order": release_order,
            "heel_impact_score": float(heel_impact),
            "forefoot_dominance_score": float(forefoot_dom),
            "knee_activity_level": float(knee_activity),
            "gait_signature_complete": bool(complete),
        }

    def _update_zone_sequence(self, toe: float, ff: float, heel: float, t: float) -> None:
        vals = {"toe": toe, "forefoot": ff, "heel": heel}
        for zone, v in vals.items():
            prev = self._zone_contact_state[zone]
            now = prev
            if prev:
                if v <= FOOT_OFF_GROUND_TH:
                    now = False
            else:
                if v >= FOOT_ON_GROUND_TH:
                    now = True
            if now != prev:
                self._zone_contact_state[zone] = now
                self._zone_event_hist.append(("on" if now else "off", zone, float(t)))

    def _extract_order(self, kind: str) -> list[str]:
        seq: list[str] = []
        seen: set[str] = set()
        for evt, zone, _t in reversed(self._zone_event_hist):
            if evt != kind:
                continue
            if zone in seen:
                continue
            seq.append(zone)
            seen.add(zone)
            if len(seq) >= 3:
                break
        seq.reverse()
        return seq

    def _extract_gait_orders(self) -> tuple[str, str, str, bool]:
        contact = self._extract_order("on")
        release = self._extract_order("off")
        if len(contact) < 3 or len(release) < 3:
            return "unknown", "unknown", "unknown", False
        contact_order = "->".join(contact)
        release_order = "->".join(release)
        initial = contact[0]
        complete = len(set(contact)) == 3 and len(set(release)) == 3
        return initial, contact_order, release_order, complete

    def _classify_standing_bilateral(
        self, left_load: float, right_load: float,
    ) -> str:
        """Left-right weight distribution → upright / left lean / right lean."""
        lr_ratio = (left_load - right_load) / (left_load + right_load + 1e-9)
        if abs(lr_ratio) < LEAN_LR_TH:
            return "STANDING_UPRIGHT"
        if lr_ratio > LEAN_LR_TH:
            return "STANDING_LEFT_LEAN"
        return "STANDING_RIGHT_LEAN"
