"""Live 6ch stream: adaptive bank → knee/motion routing → branch RF → debounced pose state."""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

from adaptive_preprocessing import CHANNEL_NAMES_DUAL, DualFootAdaptiveBank
from ml_activity_features import WINDOW_SIZE as ML_WINDOW_SIZE
from ml_branch_models import (
    BRANCH_TO_FILE,
    BranchRFEnsemble,
    branch_base_proba_threshold,
    build_full_feature_vector_and_g8_from_window,
    build_full_feature_vector_from_window,
)
from gait_knee_features import (
    KNEE_MODES,
    eight_named_gait_features,
    knee_mode_argmax,
    knee_soft_scores_3,
)

SENSOR_MAX = 4095.0
SAMPLE_HZ = 10
EMA_ALPHA = 0.25
HEEL_STEP_EMA_ALPHA = 0.35
STEP_INIT_LOW = 0.25
STEP_INIT_HIGH = 0.45
STEP_COOLDOWN_S = 0.30
STEP_COOLDOWN_SAMPLES = max(4, int(round(SAMPLE_HZ * STEP_COOLDOWN_S)))
ADAPTIVE_HISTORY = 10
ADAPTIVE_MIN_SWING = 0.04
ADAPTIVE_LOW_FRAC = 0.30
ADAPTIVE_HIGH_FRAC = 0.60
PEAK_TROUGH_HYST = 0.08
FOOT_OFF_GROUND_TH = 0.10
FOOT_ON_GROUND_TH = 0.20
STEP_MIN_GAP_S = 0.30
FOOT_OFF_MIN_SAMPLES = 3
FOOT_ON_MIN_SAMPLES = 3
MOTION_MIN_AMPLITUDE = 0.10
MOTION_MIN_AMPLITUDE_MAX = 0.25
MOTION_ADAPTIVE_WINDOW_S = 6.0
MOTION_ADAPTIVE_QUANTILE = 0.70
MOTION_ADAPTIVE_UPPER_QUANTILE = 0.90
MOTION_ADAPTIVE_GAIN = 0.90
MOTION_ADAPTIVE_EMA_ALPHA = 0.25
MOTION_CONFIRM_FRAMES = 4
MOTION_MIN_HOLD_S = 1.0
LAYER2_STATIC_CONFIRM_FRAMES = 6
WALK_ENTER_MIN_STEPS = 2
WALK_EVIDENCE_WINDOW_S = 2.0
WALK_KNEE_RATIO_EXTEND_TH = 0.78
STATE_MIN_DURATION_S = 1.0
WALK_TIMEOUT_S = 2.0
MOTION_SET = frozenset({
    "WALKING_FORWARD", "WALKING_BACKWARD", "STAIRS_UP", "STAIRS_DOWN",
})
STATIC_SET = frozenset({
    "STANDING_UPRIGHT", "STANDING_LEFT_LEAN", "STANDING_RIGHT_LEAN",
    "SITTING_NORMAL", "SITTING_CROSSLEGGED",
})
STAIRS_LABELS = frozenset({"STAIRS_UP", "STAIRS_DOWN"})
MOTION_ENTER_VAR_TH = 0.12
MOTION_ENTER_AMP_TH = 0.15
MOTION_ENTER_CONFIRM_FRAMES = 6
MOTION_STOP_RELEASE_S = 3.0
MOTION_STOP_VAR_TH = 0.06
MOTION_STOP_AMP_TH = 0.08
STATIC_CONFIRM_FRAMES = 8
MOTION_LABEL_SWITCH_S = 1.2
STAIRS_SWITCH_HOLD_S = 1.5
STAIRS_FLIP_MIN_PROBA = 0.62
STAIRS_FLIP_CONFIRM_FRAMES = 2
WALK_DIR_VOTE_STEPS = 5
WALK_DIR_LOCK_MIN_STEPS = 3
WALK_DIR_SWITCH_CONFIRM_STEPS = 4
WALK_DIR_CONF_MIN = 0.35
WALK_DIR_HOLD_S = 1.0
KNEE_MODE_SUSTAIN_FRAMES = 5
KNEE_MODE_MIN_HOLD_S = 0.5
STAND_PSUM_TH = 0.15
STS_TRIGGER_TH = 0.15
STS_CONFIRM_TH = 0.25
STS_CONFIRM_S = 0.8
STS_KNEE_DELTA_TH = 0.08
STS_MIN_PSUM_RISE = 0.02
STS_TREND_SAMPLES = 10
_PAD_UNLOADED = SENSOR_MAX

RF_TRAINING_LABELS = {
    "WALKING_FORWARD", "WALKING_BACKWARD", "STAIRS_UP", "STAIRS_DOWN",
    "SITTING_NORMAL", "SITTING_CROSSLEGGED", "STANDING_UPRIGHT",
    "STANDING_LEFT_LEAN", "STANDING_RIGHT_LEAN", "UNKNOWN",
}
FALLBACK_BRANCH_MARGIN = 0.03


def _snapshot_to_adaptive_debug_dict(s) -> dict:
    return {
        "raw": round(float(s.raw), 2),
        "relative_pressure_ratio": round(float(s.relative_pressure_ratio), 4),
        "adaptive_zscore": round(float(s.adaptive_zscore), 4),
        "current_state": s.stable_state,
    }


class _Layer2MotionStatic:
    def __init__(self) -> None:
        self._sub = "STATIC_BRANCH"
        self._up_cnt = 0
        self._down_cnt = 0
        self._motion_enter_t = -999.0
        self._last_evidence_t = -999.0
        self._reason = "init"
        self._step_events: deque[float] = deque()
        self._knee_extend_events: deque[float] = deque()
        self._motion_amp_hist: deque[float] = deque(
            maxlen=max(8, int(round(SAMPLE_HZ * MOTION_ADAPTIVE_WINDOW_S))),
        )
        self._last_amp_th = MOTION_MIN_AMPLITUDE

    def _adaptive_motion_th(self, amp: float) -> float:
        if not np.isfinite(amp):
            amp = 0.0
        self._motion_amp_hist.append(float(max(0.0, amp)))
        if len(self._motion_amp_hist) < 8:
            self._last_amp_th = MOTION_MIN_AMPLITUDE
            return self._last_amp_th
        hist = np.asarray(self._motion_amp_hist, dtype=np.float64)
        q = float(np.quantile(hist, MOTION_ADAPTIVE_QUANTILE))
        q_hi = float(np.quantile(hist, MOTION_ADAPTIVE_UPPER_QUANTILE))
        q_robust = min(q, q_hi)
        target = np.clip(
            q_robust * MOTION_ADAPTIVE_GAIN,
            MOTION_MIN_AMPLITUDE,
            MOTION_MIN_AMPLITUDE_MAX,
        )
        self._last_amp_th += MOTION_ADAPTIVE_EMA_ALPHA * (float(target) - self._last_amp_th)
        return self._last_amp_th

    def _trim(self, dq: deque[float], t: float) -> None:
        cutoff = t - WALK_EVIDENCE_WINDOW_S
        while dq and dq[0] < cutoff:
            dq.popleft()

    def update(
        self,
        raw_step: bool,
        recent_step: bool,
        motion_amp: float,
        t: float,
        *,
        knee_low_activity: bool = True,
        knee_extended_now: bool = False,
    ) -> tuple[str, str]:
        if raw_step:
            self._step_events.append(t)
        if knee_extended_now:
            self._knee_extend_events.append(t)
        self._trim(self._step_events, t)
        self._trim(self._knee_extend_events, t)

        amp_th = self._adaptive_motion_th(motion_amp)
        evidence = (raw_step or recent_step) and motion_amp >= amp_th
        if evidence:
            self._last_evidence_t = t

        walk_guard_ok = True
        walk_guard_reason = ""
        if knee_low_activity:
            need = WALK_ENTER_MIN_STEPS
            n_steps = len(self._step_events)
            n_knee = len(self._knee_extend_events)
            walk_guard_ok = (n_steps >= need) and (n_knee >= need)
            if not walk_guard_ok:
                walk_guard_reason = f"walk_guard(steps={n_steps}/{need},knee={n_knee}/{need})"

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
                self._reason = walk_guard_reason or f"static_no_evidence(amp<{amp_th:.3f})"
            return self._sub, self._reason

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

    @property
    def current_motion_amp_threshold(self) -> float:
        return float(self._last_amp_th)


class _KneeModeRouter:
    """Map knee_soft_scores_3 to KNEE_MODES with hold time."""

    def __init__(self) -> None:
        self._mode: str = "KNEE_LOW_ACTIVITY"
        self._since_switch = -1e9
        self._pending: str | None = None
        self._cnt = 0

    def update(self, w63: np.ndarray, t: float) -> tuple[str, dict[str, float]]:
        sb, sd, sl = knee_soft_scores_3(w63)
        idx = knee_mode_argmax((sb, sd, sl))
        target = KNEE_MODES[idx]
        info = {
            "knee_soft_bent": float(sb),
            "knee_soft_dynamic": float(sd),
            "knee_soft_low": float(sl),
        }
        if target == self._mode:
            self._pending = None
            self._cnt = 0
            return self._mode, info
        if self._pending != target:
            self._pending = target
            self._cnt = 1
        else:
            self._cnt += 1
        if (
            self._cnt >= KNEE_MODE_SUSTAIN_FRAMES
            and (t - self._since_switch) >= KNEE_MODE_MIN_HOLD_S
        ):
            self._mode = target
            self._since_switch = t
            self._pending = None
            self._cnt = 0
        return self._mode, info


class _EMAFilter:
    def __init__(self, alpha: float = EMA_ALPHA):
        self._a = alpha
        self._v: float | None = None

    def update(self, x: float) -> float:
        if self._v is None:
            self._v = x
        else:
            self._v += self._a * (x - self._v)
        return float(self._v)


class _AdaptiveStepDetector:
    def __init__(self) -> None:
        self._low = STEP_INIT_LOW
        self._high = STEP_INIT_HIGH
        self._armed = True
        self._cooldown = 0
        self._prev = 0.0
        self._recent_peaks: deque[float] = deque(maxlen=ADAPTIVE_HISTORY)
        self._recent_troughs: deque[float] = deque(maxlen=ADAPTIVE_HISTORY)
        self._tracking_val = 0.0
        self._phase = "seek_peak"

    def update(self, heel_smooth: float) -> bool:
        hyst = max(PEAK_TROUGH_HYST, (self._high - self._low) * 0.25)
        if self._phase == "seek_peak":
            if heel_smooth > self._tracking_val:
                self._tracking_val = heel_smooth
            elif heel_smooth < self._tracking_val - hyst:
                self._recent_peaks.append(self._tracking_val)
                self._tracking_val = heel_smooth
                self._phase = "seek_trough"
                self._recalc()
        else:
            if heel_smooth < self._tracking_val:
                self._tracking_val = heel_smooth
            elif heel_smooth > self._tracking_val + hyst:
                self._recent_troughs.append(self._tracking_val)
                self._tracking_val = heel_smooth
                self._phase = "seek_peak"
                self._recalc()
        step = False
        if self._cooldown > 0:
            self._cooldown -= 1
        else:
            if self._armed and self._prev < self._high and heel_smooth >= self._high:
                step = True
                self._armed = False
                self._cooldown = STEP_COOLDOWN_SAMPLES
        if heel_smooth <= self._low:
            self._armed = True
        self._prev = heel_smooth
        return step

    def _recalc(self) -> None:
        if len(self._recent_peaks) < 3 or len(self._recent_troughs) < 3:
            return
        avg_pk = float(np.mean(list(self._recent_peaks)))
        avg_tr = float(np.mean(list(self._recent_troughs)))
        swing = avg_pk - avg_tr
        if swing < ADAPTIVE_MIN_SWING:
            return
        self._low = avg_tr + ADAPTIVE_LOW_FRAC * swing
        self._high = avg_tr + ADAPTIVE_HIGH_FRAC * swing


class _FootContactStepDetector:
    def __init__(self) -> None:
        self._phase = "ON_GROUND"
        self._off_count = 0
        self._on_count = 0
        self._last_step_t = -999.0

    def update(self, load: float, t: float) -> bool:
        step = False
        if self._phase == "ON_GROUND":
            if load < FOOT_OFF_GROUND_TH:
                self._off_count += 1
                if self._off_count >= FOOT_OFF_MIN_SAMPLES:
                    self._phase = "LIFTED"
                    self._on_count = 0
            else:
                self._off_count = 0
        else:
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


class _WalkDirectionTracker:
    """Fwd/bwd lock from gait votes (step UI only, not RF class)."""

    def __init__(self) -> None:
        self.locked: str = "UNKNOWN"
        self._votes: deque[str] = deque(maxlen=max(2, WALK_DIR_VOTE_STEPS))
        self._consec_opp: int = 0
        self._last_switch_t: float = -1e9
        self._switch_pending: bool = False
        self._last_single: str = "N"
        self._last_f: int = 0
        self._last_b: int = 0
        self._last_conf: float = 0.0

    @staticmethod
    def _single_step_vote(g8: dict) -> str:
        """Single-step F/B/N vote from eight_named_gait_features."""
        t_h = float(g8.get("heel_peak_time", 0.5))
        t_ff = float(g8.get("forefoot_peak_time", 0.5))
        h1 = float(g8.get("heel_first_score", 0.0))
        f1 = float(g8.get("forefoot_first_score", 0.0))
        c = float(g8.get("contact_order_confidence", 0.0))
        if c < WALK_DIR_CONF_MIN:
            return "N"
        th = WALK_DIR_CONF_MIN
        if t_h < t_ff and h1 >= th:
            return "F"
        if t_ff < t_h and f1 >= th:
            return "B"
        return "N"

    @staticmethod
    def _plurality(f: int, b: int) -> str:
        if f > b:
            return "F"
        if b > f:
            return "B"
        return "N"

    def on_step(
        self,
        t: float,
        g8: dict,
        *,
        use_vote: bool,
    ) -> None:
        if not use_vote:
            return
        sv = self._single_step_vote(g8)
        self._last_single = sv
        self._votes.append(sv)
        f_cnt = sum(1 for v in self._votes if v == "F")
        b_cnt = sum(1 for v in self._votes if v == "B")
        self._last_f, self._last_b = f_cnt, b_cnt
        denom = f_cnt + b_cnt
        self._last_conf = (max(f_cnt, b_cnt) / denom) if denom else 0.0

        if self.locked == "UNKNOWN":
            self._consec_opp = 0
            self._switch_pending = False
            if f_cnt > b_cnt and f_cnt >= WALK_DIR_LOCK_MIN_STEPS:
                self.locked = "FORWARD"
                self._last_switch_t = t
            elif b_cnt > f_cnt and b_cnt >= WALK_DIR_LOCK_MIN_STEPS:
                self.locked = "BACKWARD"
                self._last_switch_t = t
            return

        hold_ok = (t - self._last_switch_t) >= WALK_DIR_HOLD_S
        pl = self._plurality(f_cnt, b_cnt)

        if self.locked == "FORWARD":
            if sv == "B":
                self._consec_opp += 1
            else:
                self._consec_opp = 0
            want = (
                pl == "B"
                and b_cnt > f_cnt
                and self._consec_opp >= WALK_DIR_SWITCH_CONFIRM_STEPS
                and hold_ok
            )
        elif self.locked == "BACKWARD":
            if sv == "F":
                self._consec_opp += 1
            else:
                self._consec_opp = 0
            want = (
                pl == "F"
                and f_cnt > b_cnt
                and self._consec_opp >= WALK_DIR_SWITCH_CONFIRM_STEPS
                and hold_ok
            )
        else:
            want = False
            self._consec_opp = 0

        self._switch_pending = bool(
            self.locked in ("FORWARD", "BACKWARD")
            and self._consec_opp > 0
            and self._consec_opp < WALK_DIR_SWITCH_CONFIRM_STEPS
        )

        if not want:
            return
        if self.locked == "FORWARD":
            self.locked = "BACKWARD"
        else:
            self.locked = "FORWARD"
        self._last_switch_t = t
        self._consec_opp = 0

    def to_walk_dir(self) -> str:
        if self.locked == "FORWARD":
            return "forward"
        if self.locked == "BACKWARD":
            return "backward"
        return "unknown"

    @property
    def vote_forward(self) -> int:
        return self._last_f

    @property
    def vote_backward(self) -> int:
        return self._last_b

    @property
    def path_confidence(self) -> float:
        return self._last_conf

    @property
    def switch_pending(self) -> bool:
        return self._switch_pending


class _SitToStandDetector:
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
            if is_sitting and p_sum >= STS_TRIGGER_TH and knee_ok and trend_ok:
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


class _StateMachine:
    def __init__(self) -> None:
        self.state = "UNKNOWN"
        self._entered_t = 0.0

    def propose(self, candidate: str, t: float, *, immediate: bool = False) -> str:
        if candidate == self.state:
            return self.state
        if immediate or (t - self._entered_t) >= STATE_MIN_DURATION_S:
            self.state = candidate
            self._entered_t = t
        return self.state


def _sensor_variation_from_w63(w63: np.ndarray) -> float:
    """Scalar activity from ratio channels: std, |diff|, p2p blend."""
    if w63.ndim != 3 or w63.shape[0] < 2 or w63.shape[1] < 6:
        return 0.0
    r = w63[:, :6, 1].astype(np.float64)  # relative_pressure_ratio
    std_m = float(np.mean(np.std(r, axis=0)))
    d1 = float(np.mean(np.abs(np.diff(r, axis=0)))) if r.shape[0] >= 2 else 0.0
    p2p = float(np.mean(np.max(r, axis=0) - np.min(r, axis=0)))
    v = 0.35 * std_m + 0.40 * d1 * 1.4 + 0.25 * p2p
    return float(np.clip(v, 0.0, 1.5))


class _EvidenceStateStabilizer:
    """Holds or switches STATIC vs MOTION RF labels with timing rules."""

    def __init__(self) -> None:
        self._mode: str = "STATIC"
        self._enter_motion_cnt: int = 0
        self._stop_start_t: float | None = None
        self._motion_held: str | None = None
        self._motion_pend: str | None = None
        self._motion_pend_t0: float = 0.0
        self._static_pend: str | None = None
        self._static_cnt: int = 0
        self._static_out: str = "UNKNOWN"

    def _reset_motion_track(self) -> None:
        self._motion_held = None
        self._motion_pend = None
        self._motion_pend_t0 = 0.0

    def _reset_static_track(self) -> None:
        self._static_pend = None
        self._static_cnt = 0
        self._static_out = "UNKNOWN"

    def _normalize_motion_cand(self, c: str) -> str:
        if c in MOTION_SET:
            return c
        if self._motion_held in MOTION_SET:
            return self._motion_held
        return c if c in (MOTION_SET | STATIC_SET) else "UNKNOWN"

    @staticmethod
    def _switch_hold_s(cur: str, pend: str) -> float:
        if cur in STAIRS_LABELS or pend in STAIRS_LABELS:
            return STAIRS_SWITCH_HOLD_S
        return MOTION_LABEL_SWITCH_S

    def _stop_elapsed(self, t: float) -> float:
        if self._stop_start_t is None:
            return 0.0
        return float(t - self._stop_start_t)

    def _pack_dbg(
        self,
        *,
        reason: str,
        sensor_variation: float,
        motion_activity_score: float,
        motion_evidence: bool,
        stop_candidate: bool,
        t: float,
    ) -> dict:
        md = "MOTION_MODE" if self._mode == "MOTION" else "STATIC_MODE"
        stop_elapsed_s = self._stop_elapsed(t)
        mp = self._motion_pend
        if mp is None:
            motion_pending_state = "—"
        else:
            motion_pending_state = f"{mp}@{t - self._motion_pend_t0:.2f}s"
        sp = self._static_pend
        if sp is None:
            static_pending_state = "—"
        else:
            static_pending_state = f"{sp}({self._static_cnt}/{STATIC_CONFIRM_FRAMES})"
        return {
            "mode_detector_state": md,
            "final_state_stabilizer_reason": reason,
            "sensor_variation": round(sensor_variation, 4),
            "motion_activity_score": round(motion_activity_score, 4),
            "motion_evidence": bool(motion_evidence),
            "stop_candidate": bool(stop_candidate),
            "stop_elapsed_s": round(stop_elapsed_s, 3),
            "motion_pending_state": motion_pending_state,
            "static_pending_state": static_pending_state,
        }

    def update(
        self,
        t: float,
        amp: float,
        sensor_var: float,
        motion_evidence: bool,
        stop_candidate: bool,
        static_candidate: str,
        motion_candidate: str,
        *,
        fsts: bool,
        psts: bool,
        motion_activity_score: float,
    ) -> tuple[str, dict]:
        if fsts:
            self._mode = "STATIC"
            self._enter_motion_cnt = 0
            self._stop_start_t = None
            self._reset_motion_track()
            self._reset_static_track()
            return "SIT_TO_STAND", self._pack_dbg(
                reason="sit_to_stand_immediate",
                sensor_variation=sensor_var,
                motion_activity_score=motion_activity_score,
                motion_evidence=motion_evidence,
                stop_candidate=stop_candidate,
                t=t,
            )

        if psts:
            self._enter_motion_cnt = 0
            self._stop_start_t = None
            self._reset_motion_track()
            if motion_candidate in MOTION_SET:
                self._mode = "MOTION"
                self._motion_held = motion_candidate
            else:
                self._mode = "STATIC"
            return motion_candidate, self._pack_dbg(
                reason="psts_bypass",
                sensor_variation=sensor_var,
                motion_activity_score=motion_activity_score,
                motion_evidence=motion_evidence,
                stop_candidate=stop_candidate,
                t=t,
            )

        if self._mode == "MOTION":
            if stop_candidate:
                if self._stop_start_t is None:
                    self._stop_start_t = t
                if self._stop_elapsed(t) >= MOTION_STOP_RELEASE_S:
                    self._mode = "STATIC"
                    self._stop_start_t = None
                    self._reset_motion_track()
                    self._enter_motion_cnt = 0
                    self._reset_static_track()
            else:
                self._stop_start_t = None

        if self._mode == "MOTION":
            m_in = self._normalize_motion_cand(motion_candidate)
            reason: str
            if m_in not in MOTION_SET:
                reason = (
                    "hold_motion_non_motion_candidate"
                    if m_in not in (MOTION_SET | STATIC_SET)
                    else "hold_motion_static_like_rf"
                )
                out = self._motion_held if self._motion_held in MOTION_SET else m_in
                return out, self._pack_dbg(
                    reason=reason,
                    sensor_variation=sensor_var,
                    motion_activity_score=motion_activity_score,
                    motion_evidence=motion_evidence,
                    stop_candidate=stop_candidate,
                    t=t,
                )

            if self._motion_held is None or self._motion_held not in MOTION_SET:
                self._motion_held = m_in
                self._motion_pend = None
                reason = "motion_accept_initial"
                return self._motion_held, self._pack_dbg(
                    reason=reason,
                    sensor_variation=sensor_var,
                    motion_activity_score=motion_activity_score,
                    motion_evidence=motion_evidence,
                    stop_candidate=stop_candidate,
                    t=t,
                )

            if m_in == self._motion_held:
                self._motion_pend = None
                reason = "motion_hold_same"
                return self._motion_held, self._pack_dbg(
                    reason=reason,
                    sensor_variation=sensor_var,
                    motion_activity_score=motion_activity_score,
                    motion_evidence=motion_evidence,
                    stop_candidate=stop_candidate,
                    t=t,
                )

            hold_s = self._switch_hold_s(self._motion_held, m_in)
            if self._motion_pend != m_in:
                self._motion_pend = m_in
                self._motion_pend_t0 = t
                reason = "motion_switch_timer_reset"
                return self._motion_held, self._pack_dbg(
                    reason=reason,
                    sensor_variation=sensor_var,
                    motion_activity_score=motion_activity_score,
                    motion_evidence=motion_evidence,
                    stop_candidate=stop_candidate,
                    t=t,
                )

            if (t - self._motion_pend_t0) >= hold_s:
                self._motion_held = m_in
                self._motion_pend = None
                reason = "motion_switch_confirmed"
                return self._motion_held, self._pack_dbg(
                    reason=reason,
                    sensor_variation=sensor_var,
                    motion_activity_score=motion_activity_score,
                    motion_evidence=motion_evidence,
                    stop_candidate=stop_candidate,
                    t=t,
                )

            reason = "motion_switch_hold"
            return self._motion_held, self._pack_dbg(
                reason=reason,
                sensor_variation=sensor_var,
                motion_activity_score=motion_activity_score,
                motion_evidence=motion_evidence,
                stop_candidate=stop_candidate,
                t=t,
            )

        enter_gate = motion_evidence and (
            sensor_var >= MOTION_ENTER_VAR_TH or amp >= MOTION_ENTER_AMP_TH
        )
        if enter_gate:
            self._enter_motion_cnt += 1
        else:
            self._enter_motion_cnt = 0

        if self._enter_motion_cnt >= MOTION_ENTER_CONFIRM_FRAMES:
            self._mode = "MOTION"
            self._enter_motion_cnt = 0
            self._stop_start_t = None
            m0 = self._normalize_motion_cand(motion_candidate)
            if m0 in MOTION_SET:
                self._motion_held = m0
            else:
                self._motion_held = None
            self._motion_pend = None
            if self._motion_held in MOTION_SET:
                return self._motion_held, self._pack_dbg(
                    reason="entered_motion_from_static",
                    sensor_variation=sensor_var,
                    motion_activity_score=motion_activity_score,
                    motion_evidence=motion_evidence,
                    stop_candidate=stop_candidate,
                    t=t,
                )
            reason = "entered_motion_await_rf_motion_label"
            return "UNKNOWN", self._pack_dbg(
                reason=reason,
                sensor_variation=sensor_var,
                motion_activity_score=motion_activity_score,
                motion_evidence=motion_evidence,
                stop_candidate=stop_candidate,
                t=t,
            )

        if static_candidate == self._static_pend:
            self._static_cnt += 1
        else:
            self._static_pend = static_candidate
            self._static_cnt = 1
        if self._static_cnt >= STATIC_CONFIRM_FRAMES and self._static_pend in STATIC_SET:
            self._static_out = self._static_pend
        if self._static_cnt < STATIC_CONFIRM_FRAMES:
            out = self._static_out if self._static_out in STATIC_SET else (
                self._static_pend if self._static_pend in STATIC_SET else "UNKNOWN"
            )
            reason = "static_confirm_in_progress"
        else:
            out = self._static_out if self._static_out in STATIC_SET else self._static_pend
            reason = "static_confirmed"
        if out not in STATIC_SET and self._static_pend in STATIC_SET:
            out = self._static_pend
        return out, self._pack_dbg(
            reason=reason,
            sensor_variation=sensor_var,
            motion_activity_score=motion_activity_score,
            motion_evidence=motion_evidence,
            stop_candidate=stop_candidate,
            t=t,
        )


class OnlineRecognizer:
    _STEP_STATES = {"WALKING_FORWARD", "WALKING_BACKWARD", "STAIRS_UP", "STAIRS_DOWN"}

    def __init__(self, calibration: object | str | None = "auto") -> None:
        self._calibration = self._resolve_calibration(calibration)
        self._f_lf = _EMAFilter()
        self._f_lh = _EMAFilter()
        self._f_lk = _EMAFilter()
        self._f_rf = _EMAFilter()
        self._f_rh = _EMAFilter()
        self._f_rk = _EMAFilter()
        self._heel_step_l = _EMAFilter(HEEL_STEP_EMA_ALPHA)
        self._heel_step_r = _EMAFilter(HEEL_STEP_EMA_ALPHA)
        self._step_det_l = _AdaptiveStepDetector()
        self._step_det_r = _AdaptiveStepDetector()
        self._step_det = _AdaptiveStepDetector()
        self._contact_l = _FootContactStepDetector()
        self._contact_r = _FootContactStepDetector()
        self._walk_dir = _WalkDirectionTracker()
        self._last_valid_walk_dir: str | None = None
        self._sts_det = _SitToStandDetector()
        self._sm = _StateMachine()
        self._branch_models = BranchRFEnsemble()
        self._layer2 = _Layer2MotionStatic()
        self._counters = {
            "forward_steps": 0, "backward_steps": 0, "up_steps": 0,
            "down_steps": 0, "total_steps": 0,
        }
        self._last_step_t = 0.0
        self._knee_baseline: float | None = None
        self._knee_init_buf: list[float] = []
        _calib = self._calibration
        _seeds = None
        if _calib is not None and getattr(_calib, "has_global_stats", False):
            _seeds = _calib.to_channel_seeds()
        self._adapt_bank = DualFootAdaptiveBank(seeds=_seeds)
        self._p_hist: deque[np.ndarray] = deque(
            maxlen=max(ML_WINDOW_SIZE, int(SAMPLE_HZ * 2)),
        )
        self._ff_heel_motion_hist: deque[float] = deque(
            maxlen=max(8, int(SAMPLE_HZ * 0.6)),
        )
        self._zone_state = {"forefoot": False, "heel": False}
        self._zone_event_hist: deque[tuple[str, str, float]] = deque(maxlen=24)
        self._knee_router = _KneeModeRouter()
        self._mode_stabil = _EvidenceStateStabilizer()
        self._state_gate_blocked_total = 0
        self._state_gate_pass_total = 0
        self._stairs_flip_pending: str | None = None
        self._stairs_flip_pending_cnt = 0

    def _resolve_calibration(self, arg):
        if arg is None:
            return None
        import personal_calibration as pc
        if isinstance(arg, pc.PersonalCalibration):
            return arg
        if arg == "auto":
            base = os.path.dirname(os.path.abspath(__file__))
            c = pc.load_default_calibration((".", base))
            return c
        if isinstance(arg, str):
            return pc.PersonalCalibration.load_json(arg)
        raise TypeError(f"bad calibration type {type(arg)!r}")

    def _calibrate_raw6(self, raw6: np.ndarray) -> np.ndarray:
        if self._calibration is None:
            return raw6
        return np.asarray(self._calibration.normalize_to_adc(raw6), dtype=np.float64)

    def _branch_key(self, knee_mode: str, motion_sub: str) -> str:
        if knee_mode == "KNEE_STATIC_BENT":
            return "sitting"
        if knee_mode == "KNEE_DYNAMIC_ACTIVE":
            return "stairs"
        if motion_sub == "MOTION_BRANCH":
            return "walking"
        return "standing"

    def _rf_predict(self, br: str) -> tuple[str, float, str, str]:
        """Predict from deque history; prefer _rf_predict_from_feat in hot path."""
        win = np.array(list(self._p_hist)[-ML_WINDOW_SIZE:])
        if win.ndim != 2 or win.shape[1] != 18 or win.shape[0] < ML_WINDOW_SIZE:
            return "UNKNOWN", 0.0, "window_not_ready", ""
        w63 = win.reshape(ML_WINDOW_SIZE, 6, 3)
        feat = build_full_feature_vector_from_window(w63)
        if feat is None:
            return "UNKNOWN", 0.0, "feature_error", ""
        return self._rf_predict_from_feat(br, feat)

    def _rf_predict_from_feat(
        self, br: str, feat: np.ndarray
    ) -> tuple[str, float, str, str]:
        b = self._branch_models.bundle(br)
        if not b.available:
            return "UNKNOWN", 0.0, "model_missing", BRANCH_TO_FILE.get(br, "")
        lab, pr, rj = b.predict(feat)
        return lab, float(pr), rj, b.path or ""

    def _ff_heel_only_amplitude(
        self,
        pff_l: float,
        ph_l: float,
        pff_r: float,
        ph_r: float,
    ) -> float:
        inst = max(pff_l, ph_l, pff_r, ph_r) - min(pff_l, ph_l, pff_r, ph_r)
        self._ff_heel_motion_hist.append(float(inst))
        if not self._ff_heel_motion_hist:
            return float(inst)
        return float(
            max(self._ff_heel_motion_hist) - min(self._ff_heel_motion_hist),
        )

    def _update_zones_dominant(self, pff: float, ph: float, t: float) -> None:
        for zone, v in (("forefoot", pff), ("heel", ph)):
            prev = self._zone_state[zone]
            now = prev
            if prev:
                if v <= FOOT_OFF_GROUND_TH:
                    now = False
            else:
                if v >= FOOT_ON_GROUND_TH:
                    now = True
            if now != prev:
                self._zone_state[zone] = now
                self._zone_event_hist.append(("on" if now else "off", zone, float(t)))

    def update_bilateral(
        self,
        left: tuple[float, float, float],
        right: tuple[float, float, float],
        t: float | None = None,
    ) -> dict:
        """One bilateral frame: returns state, counters, debug."""
        if t is None:
            t = time.monotonic()
        lf, lh, lk = (float(x) for x in left)
        rf, rh, rk = (float(x) for x in right)
        raw6 = np.array([lf, lh, lk, rf, rh, rk], dtype=np.float64)
        raw6 = np.nan_to_num(raw6, nan=SENSOR_MAX, posinf=SENSOR_MAX, neginf=0.0)
        raw6 = np.clip(raw6, 0.0, SENSOR_MAX)
        raw6c = self._calibrate_raw6(raw6)
        raw6c = np.asarray(raw6c, dtype=np.float64)
        raw6c = np.nan_to_num(raw6c, nan=SENSOR_MAX, posinf=SENSOR_MAX, neginf=0.0)
        raw6c = np.clip(raw6c, 0.0, SENSOR_MAX)
        flat18, snaps = self._adapt_bank.update(raw6c)
        self._p_hist.append(flat18)

        pff_l = self._f_lf.update(snaps[0].relative_pressure_ratio)
        ph_l = self._f_lh.update(snaps[1].relative_pressure_ratio)
        pkl = self._f_lk.update(snaps[2].relative_pressure_ratio)
        pff_r = self._f_rf.update(snaps[3].relative_pressure_ratio)
        ph_r = self._f_rh.update(snaps[4].relative_pressure_ratio)
        pkr = self._f_rk.update(snaps[5].relative_pressure_ratio)

        left_load = pff_l + ph_l
        right_load = pff_r + ph_r
        p_sum = left_load + right_load
        p_knee_avg = 0.5 * (pkl + pkr)
        lr_ratio = (left_load - right_load) / (p_sum + 1e-9)

        if self._knee_baseline is None:
            self._knee_init_buf.append(p_knee_avg)
            if len(self._knee_init_buf) >= 30:
                self._knee_baseline = float(np.mean(self._knee_init_buf))

        n_hist = len(self._p_hist)
        if n_hist < 1:
            w63 = np.zeros((ML_WINDOW_SIZE, 6, 3), dtype=np.float64)
        else:
            take = min(n_hist, ML_WINDOW_SIZE)
            win2 = np.array(list(self._p_hist)[-take:])
            if win2.shape[0] < ML_WINDOW_SIZE:
                pad = np.tile(win2[0:1], (ML_WINDOW_SIZE - win2.shape[0], 1))
                win2 = np.vstack([pad, win2])
            w63 = win2.reshape(ML_WINDOW_SIZE, 6, 3)

        knee_mode, knee_soft = self._knee_router.update(w63, t)
        knee_ext = max(pkl, pkr) >= WALK_KNEE_RATIO_EXTEND_TH
        dom_left = left_load >= right_load
        dff = pff_l if dom_left else pff_r
        dheel = ph_l if dom_left else ph_r
        self._update_zones_dominant(dff, dheel, t)

        step_l = self._contact_l.update(left_load, t)
        step_r = self._contact_r.update(right_load, t)
        cstep = step_l or step_r
        st_src: str | None = "left" if step_l else ("right" if step_r else None)
        sl = self._heel_step_l.update(ph_l)
        sr = self._heel_step_r.update(ph_r)
        hstep = self._step_det_l.update(sl) or self._step_det_r.update(sr)
        raw_step = bool(cstep)
        if not raw_step and hstep:
            raw_step = True
            st_src = st_src or "heel_fb"

        if raw_step:
            self._last_step_t = t

        recent = (t - self._last_step_t) < WALK_TIMEOUT_S
        amp = self._ff_heel_only_amplitude(pff_l, ph_l, pff_r, ph_r)
        knee_low = knee_mode == "KNEE_LOW_ACTIVITY"
        if knee_low:
            l2, l2r = self._layer2.update(
                bool(raw_step), recent, amp, t,
                knee_low_activity=True, knee_extended_now=knee_ext,
            )
        elif knee_mode == "KNEE_STATIC_BENT":
            l2, l2r = "STATIC_BRANCH", "knee_static_bent"
        else:
            l2, l2r = "MOTION_BRANCH", "knee_dynamic_active"
        brk = self._branch_key(knee_mode, l2)
        static_brk = "sitting" if knee_mode == "KNEE_STATIC_BENT" else "standing"
        fallback_reason = "not_available"
        if len(self._p_hist) < ML_WINDOW_SIZE:
            feat = None
            g8 = eight_named_gait_features(w63)
            cand, rfp, rfrj, rfn = "UNKNOWN", 0.0, "window_not_ready", ""
            scand, s_rfp = "UNKNOWN", 0.0
            fallback_reason = "window_not_ready"
            rf_dynamic_th = None
        else:
            feat, g8 = build_full_feature_vector_and_g8_from_window(w63)
            if feat is None:
                g8 = eight_named_gait_features(w63)
                cand, rfp, rfrj, rfn = "UNKNOWN", 0.0, "feature_error", ""
                scand, s_rfp = "UNKNOWN", 0.0
                fallback_reason = "feature_error"
                rf_dynamic_th = None
            else:
                cand, rfp, rfrj, rfn = self._rf_predict_from_feat(brk, feat)
                rf_dynamic_th = None
                if isinstance(rfrj, str) and "thr=" in rfrj:
                    try:
                        rf_dynamic_th = float(rfrj.split("thr=", 1)[1].rstrip(")"))
                    except Exception:
                        rf_dynamic_th = None
                if static_brk == brk:
                    scand, s_rfp = cand, rfp
                else:
                    scand, s_rfp, _, _ = self._rf_predict_from_feat(static_brk, feat)
                fallback_used = False
                fallback_reason = "none"
                if (
                    static_brk != brk
                    and cand == "UNKNOWN"
                    and isinstance(rfrj, str)
                    and rfrj.startswith("low_proba")
                    and scand != "UNKNOWN"
                ):
                    base_th = branch_base_proba_threshold(static_brk)
                    if float(s_rfp) >= max(0.20, base_th - FALLBACK_BRANCH_MARGIN):
                        cand = scand
                        rfp = float(s_rfp)
                        rfrj = f"fallback_from_{brk}"
                        fallback_used = True
                        fallback_reason = f"static_branch:{static_brk}"
                if not fallback_used:
                    fallback_reason = "not_triggered"
                if (
                    brk == "stairs"
                    and self._sm.state in STAIRS_LABELS
                    and cand in STAIRS_LABELS
                    and cand != self._sm.state
                ):
                    if float(rfp) < STAIRS_FLIP_MIN_PROBA:
                        cand = self._sm.state
                        rfrj = f"stairs_flip_hold(p<{STAIRS_FLIP_MIN_PROBA:.2f})"
                        fallback_reason = "stairs_flip_hold_low_proba"
                        self._stairs_flip_pending = None
                        self._stairs_flip_pending_cnt = 0
                    else:
                        if self._stairs_flip_pending == cand:
                            self._stairs_flip_pending_cnt += 1
                        else:
                            self._stairs_flip_pending = cand
                            self._stairs_flip_pending_cnt = 1
                        if self._stairs_flip_pending_cnt < STAIRS_FLIP_CONFIRM_FRAMES:
                            cand = self._sm.state
                            rfrj = (
                                "stairs_flip_pending("
                                f"{self._stairs_flip_pending_cnt}/{STAIRS_FLIP_CONFIRM_FRAMES})"
                            )
                            fallback_reason = "stairs_flip_pending_confirm"
                        else:
                            fallback_reason = "stairs_flip_confirmed"
                            self._stairs_flip_pending = None
                            self._stairs_flip_pending_cnt = 0
                else:
                    self._stairs_flip_pending = None
                    self._stairs_flip_pending_cnt = 0
        sensor_var = _sensor_variation_from_w63(w63)
        motion_activity = float(
            0.5 * sensor_var + 0.5 * min(1.0, max(0.0, amp * 2.0)),
        )
        motion_evidence = bool(
            raw_step
            or (recent and amp >= MOTION_ENTER_AMP_TH)
            or knee_mode == "KNEE_DYNAMIC_ACTIVE"
        )
        stop_candidate = bool(
            (not recent)
            and amp <= MOTION_STOP_AMP_TH
            and sensor_var <= MOTION_STOP_VAR_TH
        )
        use_dir_vote = bool(
            raw_step
            and brk == "walking"
            and l2 == "MOTION_BRANCH"
        )
        if use_dir_vote:
            self._walk_dir.on_step(t, g8, use_vote=True)

        fsts, psts = self._sts_det.update(
            p_sum, p_knee_avg, self._sm.state, self._knee_baseline, t,
        )
        stabilized_state, mode_dbg = self._mode_stabil.update(
            t,
            amp,
            sensor_var,
            motion_evidence,
            stop_candidate,
            scand,
            cand,
            fsts=fsts,
            psts=psts,
            motion_activity_score=motion_activity,
        )
        prev_state = self._sm.state
        state = self._sm.propose(
            stabilized_state,
            t,
            immediate=bool(fsts or psts),
        )
        gate_blocked = (stabilized_state != prev_state) and (state == prev_state)
        gate_passed = (stabilized_state != prev_state) and (state == stabilized_state)
        if gate_blocked:
            self._state_gate_blocked_total += 1
        if gate_passed:
            self._state_gate_pass_total += 1
        wdir = self._walk_dir.to_walk_dir()
        if wdir in ("forward", "backward"):
            self._last_valid_walk_dir = wdir

        locked = self._walk_dir.locked
        step_ev = bool(raw_step and state in self._STEP_STATES)
        step_assigned_to = "—"
        step_assign_reason = "no_step_event"
        if step_ev:
            self._counters["total_steps"] += 1
            if state == "STAIRS_UP":
                self._counters["up_steps"] += 1
                step_assigned_to = "up_steps"
                step_assign_reason = "stair_state"
            elif state == "STAIRS_DOWN":
                self._counters["down_steps"] += 1
                step_assigned_to = "down_steps"
                step_assign_reason = "stair_state"
            elif state in ("WALKING_FORWARD", "WALKING_BACKWARD"):
                if locked == "FORWARD":
                    self._counters["forward_steps"] += 1
                    step_assigned_to = "forward_steps"
                    step_assign_reason = "locked_forward"
                elif locked == "BACKWARD":
                    self._counters["backward_steps"] += 1
                    step_assigned_to = "backward_steps"
                    step_assign_reason = "locked_backward"
                else:
                    step_assigned_to = "none"
                    step_assign_reason = "dir_unknown_skip_fb"
            else:
                step_assigned_to = "none"
                step_assign_reason = "unexpected_step_state"

        dbg: dict = {
            "p_ff_l": round(pff_l, 3), "p_heel_l": round(ph_l, 3), "p_knee_l": round(pkl, 3),
            "p_ff_r": round(pff_r, 3), "p_heel_r": round(ph_r, 3), "p_knee_r": round(pkr, 3),
            "p_sum": round(p_sum, 3), "ff_heel_motion_amp": round(amp, 4),
            "left_load": round(left_load, 3), "right_load": round(right_load, 3),
            "branch": knee_mode,
            "knee_mode": knee_mode,
            "knee_mode_soft": knee_soft,
            "layer2_subbranch": l2, "layer2_reason": l2r,
            "layer2_motion_amp_th": round(self._layer2.current_motion_amp_threshold, 4),
            "motion_flag": l2 == "MOTION_BRANCH",
            "branch_rf_key": brk, "ml_label": cand, "ml_label_static_branch": scand,
            "rf_proba": round(rfp, 4), "static_rf_proba": round(s_rfp, 4),
            "rf_reject": rfrj, "rf_model_name": os.path.basename(rfn) if rfn else "—",
            "rf_fallback_reason": fallback_reason,
            "stairs_flip_pending": self._stairs_flip_pending or "—",
            "stairs_flip_pending_cnt": int(self._stairs_flip_pending_cnt),
            "rf_dynamic_threshold": (round(float(rf_dynamic_th), 4) if rf_dynamic_th is not None else "—"),
            "heel_peak_time": g8["heel_peak_time"],
            "forefoot_peak_time": g8["forefoot_peak_time"],
            "heel_first_score": g8["heel_first_score"],
            "forefoot_first_score": g8["forefoot_first_score"],
            "contact_order_confidence": round(g8["contact_order_confidence"], 4),
            "walk_dir_vote_forward": self._walk_dir.vote_forward,
            "walk_dir_vote_backward": self._walk_dir.vote_backward,
            "walk_dir_locked": locked,
            "walk_dir_confidence": round(self._walk_dir.path_confidence, 4),
            "walk_dir_switch_pending": self._walk_dir.switch_pending,
            "step_assigned_to": step_assigned_to,
            "step_assign_reason": step_assign_reason,
            "state_gate_prev": prev_state,
            "state_gate_candidate": stabilized_state,
            "state_gate_immediate": bool(fsts or psts),
            "state_gate_min_duration_s": float(STATE_MIN_DURATION_S),
            "state_gate_blocked_now": bool(gate_blocked),
            "state_gate_passed_now": bool(gate_passed),
            "state_gate_blocked_total": int(self._state_gate_blocked_total),
            "state_gate_pass_total": int(self._state_gate_pass_total),
        }
        dbg.update(mode_dbg)
        dbg["adaptive"] = {n: _snapshot_to_adaptive_debug_dict(s) for n, s in zip(CHANNEL_NAMES_DUAL, snaps)}

        return {
            "state": state,
            "step_event": step_ev,
            "walk_dir": wdir,
            "counters": dict(self._counters),
            "sts_last_duration_s": self._sts_det.last_duration,
            "debug": dbg,
        }

    def update_single(
        self,
        fore: float,
        heel: float,
        knee: float,
        t: float | None = None,
    ) -> dict:
        """One foot only: other side padded to 4095 (unloaded)."""
        if t is None:
            t = time.monotonic()
        return self.update_bilateral(
            (fore, heel, knee),
            (_PAD_UNLOADED, _PAD_UNLOADED, _PAD_UNLOADED),
            t=t,
        )
