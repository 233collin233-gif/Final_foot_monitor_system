"""
Four-branch RandomForest bundles (hierarchical recogniser).

Each bundle: ``dict`` with ``pipeline`` (sklearn), ``branch``, ``classes``,
``aux_dim``, ``feature_mode`` (``branch_adaptive_v2``).

Training and ``OnlineRecognizer`` must use the same
``extract_features_dual_adaptive`` + ``build_auxiliary_vector`` layout.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ml_activity_features import (
    WINDOW_SIZE as ML_WINDOW_SIZE,
    extract_features_dual_adaptive,
)

# TODO_PARAM: filenames (project root)
RF_ACTIVE_MOTION = "rf_active_motion.joblib"
RF_ACTIVE_STATIC = "rf_active_static.joblib"
RF_INACTIVE_MOTION = "rf_inactive_motion.joblib"
RF_INACTIVE_STATIC = "rf_inactive_static.joblib"

BRANCH_TO_FILE = {
    "active_motion": RF_ACTIVE_MOTION,
    "active_static": RF_ACTIVE_STATIC,
    "inactive_motion": RF_INACTIVE_MOTION,
    "inactive_static": RF_INACTIVE_STATIC,
}

# TODO_PARAM: min proba to accept RF output (else UNKNOWN)
BRANCH_RF_PROBA_MIN = 0.45

# TODO_PARAM: length of auxiliary vector appended to adaptive base features
AUX_DIM = 20


def _zone_token(s: str, z: str) -> float:
    return 1.0 if z in s else 0.0


def build_auxiliary_vector(
    sign: dict[str, Any] | None,
    left_load: float,
    right_load: float,
    lr_ratio: float,
    p_sum: float,
) -> np.ndarray:
    """
    Gait / stance hints for all four RFs (same dim; unused dims are 0).

    Order (must match training in ``ml_train_branch_rfs``):
      0 heel_impact   1 forefoot_dom   2 knee_activity   3 gait_complete
      4 init_heel     5 init_toe       6 init_forefoot
      7 fwd_contact_hint   8 bwd_contact_hint
      9 stairs_up_hint    10 stairs_down_hint
      11 left_load   12 right_load   13 lr_ratio   14 p_sum
      15-19 reserved / zeros
    """
    aux = np.zeros(AUX_DIM, dtype=np.float64)
    if not sign:
        aux[11] = left_load
        aux[12] = right_load
        aux[13] = lr_ratio
        aux[14] = p_sum
        return aux

    heel_imp = float(sign.get("heel_impact_score", 0.0))
    ff_dom = float(sign.get("forefoot_dominance_score", 0.0))
    knee_act = float(sign.get("knee_activity_level", 0.0))
    complete = 1.0 if sign.get("gait_signature_complete") else 0.0
    initial = str(sign.get("initial_contact_zone", "unknown"))
    aux[0] = heel_imp
    aux[1] = ff_dom
    aux[2] = knee_act
    aux[3] = complete
    aux[4] = 1.0 if initial == "heel" else 0.0
    aux[5] = 1.0 if initial == "toe" else 0.0
    aux[6] = 1.0 if initial == "forefoot" else 0.0

    co = str(sign.get("contact_order", ""))
    ro = str(sign.get("release_order", ""))
    # Heuristic flags (same semantics as former rule path)
    aux[7] = 1.0 if (
        initial == "heel"
        and co.startswith("heel")
        and "forefoot" in co
        and co.endswith("toe")
        and ro == "heel->forefoot->toe"
    ) else 0.0
    aux[8] = 1.0 if (
        initial in ("toe", "forefoot")
        and not co.startswith("heel")
        and co.endswith("heel")
    ) else 0.0
    aux[9] = 1.0 if (
        initial in ("toe", "forefoot")
        and ff_dom >= 0.45
    ) else 0.0
    aux[10] = 1.0 if initial == "heel" and heel_imp >= 0.35 else 0.0

    aux[11] = float(left_load)
    aux[12] = float(right_load)
    aux[13] = float(lr_ratio)
    aux[14] = float(p_sum)
    aux[15] = _zone_token(co, "toe")
    aux[16] = _zone_token(co, "forefoot")
    aux[17] = _zone_token(co, "heel")
    return aux


def auxiliary_from_window_ratios(ratios_8: np.ndarray) -> np.ndarray:
    """
    Training-only proxy from last frame (N,8) relative_pressure_ratio row
    or mean over window last row.
    """
    if ratios_8.size != 8:
        return np.zeros(AUX_DIM, dtype=np.float64)
    lt, lf, lh, lk, rt, rf, rh, rk = [float(x) for x in ratios_8]
    tl = lt + lf + lh + 1e-9
    tr = rt + rf + rh + 1e-9
    hi = 0.5 * (lh / tl + rh / tr)
    fd = 0.5 * ((lt + lf) / tl + (rt + rf) / tr)
    sign = {
        "heel_impact_score": hi,
        "forefoot_dominance_score": fd,
        "knee_activity_level": abs(lk - rk),
        "gait_signature_complete": 0.0,
        "initial_contact_zone": "unknown",
        "contact_order": "",
        "release_order": "",
    }
    left_load = lt + lf + lh
    right_load = rt + rf + rh
    s = left_load + right_load + 1e-9
    lr = (left_load - right_load) / s
    return build_auxiliary_vector(sign, left_load, right_load, lr, s)


def build_full_feature_vector(
    window_flat24: np.ndarray,
    aux: np.ndarray,
) -> np.ndarray | None:
    """``window_flat24`` shape (N, 24) adaptive rows."""
    if window_flat24.ndim != 2 or window_flat24.shape[1] != 24:
        return None
    n = window_flat24.shape[0]
    if n < ML_WINDOW_SIZE:
        return None
    if n > ML_WINDOW_SIZE:
        window_flat24 = window_flat24[-ML_WINDOW_SIZE:]
    w83 = window_flat24.reshape(ML_WINDOW_SIZE, 8, 3)
    base = extract_features_dual_adaptive(w83).astype(np.float64)
    if aux.shape[0] != AUX_DIM:
        raise ValueError(aux.shape)
    return np.concatenate([base, aux], axis=0)


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
        """
        Returns (label, max_proba, reject_reason).
        reject_reason empty on success.
        """
        if not self.available:
            return "UNKNOWN", 0.0, "model_missing"
        x = feat1d.reshape(1, -1)
        try:
            lab = str(self.pipeline.predict(x)[0])
            proba = float(np.max(self.pipeline.predict_proba(x)))
        except Exception as e:
            return "UNKNOWN", 0.0, f"predict_error:{e}"
        if proba < BRANCH_RF_PROBA_MIN:
            return "UNKNOWN", proba, "low_proba"
        return lab, proba, ""


class BranchRFEnsemble:
    """Loads up to four branch models from *models_dir* (default ``.``)."""

    def __init__(self, models_dir: str | None = None) -> None:
        root = models_dir or "."
        self._bundles: dict[str, BranchRFBundle] = {}
        for br, fn in BRANCH_TO_FILE.items():
            self._bundles[br] = BranchRFBundle(br, os.path.join(root, fn))

    def bundle(self, branch: str) -> BranchRFBundle:
        return self._bundles.get(branch, BranchRFBundle(branch, ""))

    def any_available(self) -> bool:
        return any(b.available for b in self._bundles.values())
