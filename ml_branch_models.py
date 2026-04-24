"""Load rf_*.joblib per branch, stack adaptive + 32d aux; thresholds and biasing for inference."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from gait_knee_features import eight_named_gait_features, knee_soft_scores_3
from ml_activity_features import (
    WINDOW_SIZE as ML_WINDOW_SIZE,
    extract_features_dual_adaptive,
)

RF_SITTING = "rf_sitting.joblib"
RF_STAIRS = "rf_stairs.joblib"
RF_WALKING = "rf_walking.joblib"
RF_STANDING = "rf_standing.joblib"

RF_ACTIVE_MOTION = RF_STAIRS
RF_ACTIVE_STATIC = RF_SITTING
RF_INACTIVE_MOTION = RF_WALKING
RF_INACTIVE_STATIC = RF_STANDING

BRANCH_TO_FILE: dict[str, str] = {
    "sitting": RF_SITTING,
    "stairs": RF_STAIRS,
    "walking": RF_WALKING,
    "standing": RF_STANDING,
}

BRANCH_RF_PROBA_MIN_DEFAULT = 0.42
BRANCH_RF_PROBA_MIN: dict[str, float] = {
    "sitting": 0.44,
    "stairs": 0.38,
    "walking": 0.38,
    "standing": 0.43,
}

STAIRS_UP_CLASS_BIAS = 0.20

RF_AUX_BLOCK_GAIN = 1.28

AUX_DIM = 32


def branch_base_proba_threshold(branch: str) -> float:
    return float(BRANCH_RF_PROBA_MIN.get(branch, BRANCH_RF_PROBA_MIN_DEFAULT))


def dynamic_proba_threshold(branch: str, proba_row: np.ndarray) -> float:
    """Adjust base accept threshold from proba margin and entropy."""
    base = branch_base_proba_threshold(branch)
    if proba_row.size <= 1:
        return base
    p = np.asarray(proba_row, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / max(float(p.sum()), 1e-12)
    top2 = np.partition(p, -2)[-2:]
    p1 = float(max(top2))
    p2 = float(min(top2))
    margin = max(0.0, p1 - p2)
    entropy = -float(np.sum(p * np.log(p))) / float(np.log(len(p)))
    uncertainty = 0.55 * (1.0 - margin) + 0.45 * entropy
    scale = 0.90 + 0.20 * uncertainty
    thr = base * scale
    lo = max(0.20, base - 0.05)
    hi = min(0.72, base + 0.12)
    return float(np.clip(thr, lo, hi))


def apply_branch_decision_bias(
    branch: str,
    classes: "list[str]",
    proba_row: np.ndarray,
) -> np.ndarray:
    """Re-weight proba (e.g. stairs up); renormalize."""
    p = np.asarray(proba_row, dtype=np.float64).copy()
    if p.size == 0:
        return p
    if branch == "stairs" and "STAIRS_UP" in classes:
        i_up = classes.index("STAIRS_UP")
        p[i_up] += float(STAIRS_UP_CLASS_BIAS)
    p = np.clip(p, 0.0, None)
    s = float(np.sum(p))
    if s > 0.0:
        p /= s
    return p


def build_aux_block(w63: np.ndarray, g8: dict | None = None) -> np.ndarray:
    """32d aux from (T,6,3) window; pass g8 to skip recomputing gait features."""
    vec = np.zeros(AUX_DIM, dtype=np.float64)
    if w63.ndim != 3 or w63.shape[0] < 2 or w63.shape[1] != 6 or w63.shape[2] != 3:
        return vec

    if g8 is None:
        g8 = eight_named_gait_features(w63)
    order = (
        "heel_first_score",
        "forefoot_first_score",
        "heel_peak_time",
        "forefoot_peak_time",
        "heel_impact_score",
        "forefoot_dominance_score",
        "knee_dynamic_active_score",
        "contact_order_confidence",
    )
    for i, k in enumerate(order):
        vec[i] = float(g8[k])
    sb, sd, sl = knee_soft_scores_3(w63)
    vec[8], vec[9], vec[10] = sb, sd, sl

    rat = w63[-1, :, 1].astype(np.float64)
    lf, lh, lk, rf, rh, rk = [float(rat[i]) for i in range(6)]
    ll, rl = lf + lh, rf + rh
    ps = ll + rl + 1e-9
    lr = (ll - rl) / ps
    vec[11], vec[12], vec[13], vec[14] = ll, rl, lr, ps
    vec[15] = float(g8["heel_first_score"] * g8["heel_impact_score"])
    vec[16] = float(g8["forefoot_first_score"] * g8["forefoot_dominance_score"])
    vec[17] = float(sb)
    vec[18] = float(sd)
    vec[19] = float(sl)
    N = int(w63.shape[0])
    half = max(1, N // 2)
    Lff = w63[:, 0, 1].astype(np.float64)
    Lh = w63[:, 1, 1].astype(np.float64)
    Rff = w63[:, 3, 1].astype(np.float64)
    Rh = w63[:, 4, 1].astype(np.float64)
    Ltot = Lff + Lh
    Rtot = Rff + Rh
    dom_left = bool(np.sum(Ltot) >= np.sum(Rtot))
    ff_d = Lff if dom_left else Rff
    h_d = Lh if dom_left else Rh
    ff_n = Rff if dom_left else Lff
    h_n = Rh if dom_left else Lh

    def _half_delta(x: np.ndarray) -> float:
        a = float(np.mean(x[:half]))
        b = float(np.mean(x[-half:]))
        return b - a

    def _signed_impulse(x: np.ndarray) -> float:
        early = float(np.sum(x[:half]))
        late = float(np.sum(x[-half:]))
        return (late - early) / max(early + late, 1e-9)

    vec[20] = _half_delta(ff_d)
    vec[21] = _half_delta(h_d)
    vec[22] = _half_delta(ff_d - h_d)
    vec[23] = _signed_impulse(ff_d)
    vec[24] = _signed_impulse(h_d)
    vec[25] = _half_delta(ff_n)
    vec[26] = _half_delta(h_n)
    vec[27] = _half_delta(ff_n - h_n)
    vec[28] = _half_delta(Ltot)
    vec[29] = _half_delta(Rtot)
    vec[30] = _half_delta(Ltot - Rtot)
    vec[31] = _signed_impulse(Ltot + Rtot)
    return vec


def build_full_feature_vector_from_window(w63: np.ndarray) -> np.ndarray | None:
    """Concat base adaptive stats + aux block; None if window too short."""
    feat, _ = build_full_feature_vector_and_g8_from_window(w63)
    return feat


def build_full_feature_vector_and_g8_from_window(
    w63: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, float] | None]:
    """Full RF vector and gait dict g8."""
    if w63.ndim != 3 or w63.shape[1] != 6 or w63.shape[2] != 3:
        return None, None
    n = w63.shape[0]
    if n < ML_WINDOW_SIZE:
        return None, None
    if n > ML_WINDOW_SIZE:
        w63 = w63[-ML_WINDOW_SIZE:].copy()
    g8 = eight_named_gait_features(w63)
    base = extract_features_dual_adaptive(w63).astype(np.float64)
    aux = build_aux_block(w63, g8=g8).astype(np.float64) * float(RF_AUX_BLOCK_GAIN)
    return np.concatenate([base, aux], axis=0), g8


def build_auxiliary_vector(
    sign: dict[str, Any] | None,
    left_load: float,
    right_load: float,
    lr_ratio: float,
    p_sum: float,
) -> np.ndarray:
    """Stub: zeros (unused API)."""
    del sign, left_load, right_load, lr_ratio, p_sum
    return np.zeros(AUX_DIM, dtype=np.float64)


def auxiliary_from_window_ratios(ratios_6: np.ndarray) -> np.ndarray:
    """Stub: zeros (unused API)."""
    del ratios_6
    return np.zeros(AUX_DIM, dtype=np.float64)


FEAT_PER_WINDOW_ROW = 18


def build_full_feature_vector(
    window_flat: np.ndarray,
    aux: np.ndarray,
) -> np.ndarray | None:
    """Old flat18+aux path."""
    if window_flat.ndim != 2 or window_flat.shape[1] != FEAT_PER_WINDOW_ROW:
        return None
    n = window_flat.shape[0]
    if n < ML_WINDOW_SIZE:
        return None
    w63 = window_flat[-ML_WINDOW_SIZE:].reshape(ML_WINDOW_SIZE, 6, 3)
    return build_full_feature_vector_from_window(w63)


class BranchRFBundle:
    def __init__(self, branch: str, path: str | None = None) -> None:
        self.branch = branch
        self.path = path or BRANCH_TO_FILE.get(branch, "")
        self.pipeline = None
        self.classes: list[str] = []
        self._load()

    def _load(self) -> None:
        p = self.path
        if not p or not os.path.isfile(p):
            return
        try:
            import joblib as jl

            obj = jl.load(p)
        except Exception:
            return
        if isinstance(obj, dict):
            self.pipeline = obj.get("pipeline")
            cl = obj.get("classes")
            if cl is not None:
                self.classes = [str(x) for x in list(cl)]
        else:
            self.pipeline = obj

    @property
    def available(self) -> bool:
        return self.pipeline is not None

    def predict(self, feat1d: np.ndarray) -> tuple[str, float, str]:
        if not self.available:
            return "UNKNOWN", 0.0, "model_missing"
        x = feat1d.reshape(1, -1)
        try:
            cl = getattr(self.pipeline, "classes_", None)
            if cl is None and self.classes:
                cl = np.asarray(self.classes)
            proba_row = self.pipeline.predict_proba(x)[0]
            if cl is None:
                cl = np.arange(len(proba_row))
            cls = [str(c) for c in list(cl)]
            proba_adj = apply_branch_decision_bias(self.branch, cls, proba_row)
            ji = int(np.argmax(proba_adj))
            lab = str(cl[ji])
            proba = float(proba_adj[ji])
        except Exception as e:
            return "UNKNOWN", 0.0, f"predict_error:{e}"
        thr = dynamic_proba_threshold(self.branch, np.asarray(proba_adj, dtype=np.float64))
        if proba < thr:
            return "UNKNOWN", proba, f"low_proba(thr={thr:.3f})"
        return lab, proba, ""


class BranchRFEnsemble:
    def __init__(self, models_dir: str | None = None) -> None:
        if models_dir is not None:
            root = models_dir
        else:
            root = os.path.dirname(os.path.abspath(__file__))
        self._bundles: dict[str, BranchRFBundle] = {}
        for br, fn in BRANCH_TO_FILE.items():
            self._bundles[br] = BranchRFBundle(br, os.path.join(root, fn))

    def bundle(self, branch: str) -> BranchRFBundle:
        return self._bundles.get(branch, BranchRFBundle(branch, ""))

    def any_available(self) -> bool:
        return any(b.available for b in self._bundles.values())
