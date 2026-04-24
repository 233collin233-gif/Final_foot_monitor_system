"""Gait timing/impact scores and 3-way knee soft scores from (N,6,3) adaptive windows."""

from __future__ import annotations

import numpy as np

I_LK, I_RK = 2, 5


def knee_soft_scores_3(w63: np.ndarray) -> tuple[float, float, float]:
    if w63.ndim != 3 or w63.shape[1] < 6 or w63.shape[2] < 2:
        return 0.33, 0.33, 0.33
    Lk, Rk = w63[:, I_LK, 1].astype(np.float64), w63[:, I_RK, 1].astype(np.float64)
    km = 0.5 * (Lk + Rk)
    mu = float(np.clip(np.mean(km), 0.0, 1.0))
    sig = float(np.std(km) if len(km) > 1 else 0.0)
    asym = float(np.mean(np.abs(Lk - Rk)))
    static_bent = mu * (1.0 / (1.0 + 12.0 * sig)) * (1.0 + 0.35 * np.clip(asym, 0.0, 0.4))
    dynamic = (sig * 6.5 + asym * 2.7) * max(mu, 0.08)
    low = (1.0 - mu) * (1.0 / (1.0 + 8.5 * sig)) + 0.16 * (1.0 - asym)
    s1, s2, s3 = static_bent, dynamic, max(low, 1e-9)
    z = s1 + s2 + s3
    if z < 1e-9:
        return 0.33, 0.33, 0.33
    return s1 / z, s2 / z, s3 / z


def eight_named_gait_features(w63: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "heel_first_score": 0.0,
        "forefoot_first_score": 0.0,
        "heel_peak_time": 0.5,
        "forefoot_peak_time": 0.5,
        "heel_impact_score": 0.0,
        "forefoot_dominance_score": 0.0,
        "knee_dynamic_active_score": 0.0,
        "contact_order_confidence": 0.0,
    }
    if w63.ndim != 3 or w63.shape[0] < 4 or w63.shape[1] != 6 or w63.shape[2] != 3:
        return out

    N = w63.shape[0]
    Lf, Lh = w63[:, 0, 1], w63[:, 1, 1]
    Rf, Rh = w63[:, 3, 1], w63[:, 4, 1]
    sL, sR = (Lf + Lh).sum(), (Rf + Rh).sum()
    dom = 0 if sL >= sR else 1
    ff = w63[:, 0 + 3 * dom, 1]
    heel = w63[:, 1 + 3 * dom, 1]
    k_l, k_r = w63[:, I_LK, 1], w63[:, I_RK, 1]

    i_ff = int(np.argmax(ff))
    i_heel = int(np.argmax(heel))
    t_ff = i_ff / max(N - 1, 1)
    t_heel = i_heel / max(N - 1, 1)
    out["forefoot_peak_time"] = float(t_ff)
    out["heel_peak_time"] = float(t_heel)
    sep = abs(t_heel - t_ff)
    out["contact_order_confidence"] = float(np.clip(1.0 - 2.0 * sep, 0.0, 1.0))
    if t_heel < t_ff - 1.0 / N:
        out["heel_first_score"] = float(np.clip(0.5 + 0.5 * (t_ff - t_heel) * N / 3.0, 0.0, 1.0))
        out["forefoot_first_score"] = float(1.0 - out["heel_first_score"])
    elif t_ff < t_heel - 1.0 / N:
        out["forefoot_first_score"] = float(np.clip(0.5 + 0.5 * (t_heel - t_ff) * N / 3.0, 0.0, 1.0))
        out["heel_first_score"] = float(1.0 - out["forefoot_first_score"])
    else:
        out["heel_first_score"] = 0.5
        out["forefoot_first_score"] = 0.5

    dheel = np.diff(heel, prepend=heel[0])
    dff = np.diff(ff, prepend=ff[0])
    out["heel_impact_score"] = float(np.clip(np.max(np.abs(dheel)) * 2.0, 0.0, 1.0))
    tot = float(np.sum(ff) + np.sum(heel) + 1e-9)
    out["forefoot_dominance_score"] = float(np.sum(ff) / tot)
    s1, s2, s3 = knee_soft_scores_3(w63)
    out["knee_dynamic_active_score"] = float(s2)

    return out


def build_sign_from_window(
    w63: np.ndarray,
) -> dict[str, float | str | bool]:
    g8 = eight_named_gait_features(w63)
    N = w63.shape[0]
    dom = 0
    Lf, Lh = w63[:, 0, 1], w63[:, 1, 1]
    Rf, Rh = w63[:, 3, 1], w63[:, 4, 1]
    if (Rf + Rh).sum() > (Lf + Lh).sum():
        dom = 1
    ff = w63[:, 0 + 3 * dom, 1]
    heel = w63[:, 1 + 3 * dom, 1]
    t_ff = int(np.argmax(ff)) / max(N - 1, 1)
    t_h = int(np.argmax(heel)) / max(N - 1, 1)
    initial = "heel" if t_h < t_ff else "forefoot"
    return {
        "heel_impact_score": g8["heel_impact_score"],
        "forefoot_dominance_score": g8["forefoot_dominance_score"],
        "knee_activity_level": g8["knee_dynamic_active_score"],
        "gait_signature_complete": g8["contact_order_confidence"] > 0.35,
        "initial_contact_zone": initial,
        "contact_order": f"ff@{t_ff:.2f},heel@{t_h:.2f}",
        "release_order": "",
        "heel_first_score": g8["heel_first_score"],
        "forefoot_first_score": g8["forefoot_first_score"],
        "heel_peak_time": g8["heel_peak_time"],
        "forefoot_peak_time": g8["forefoot_peak_time"],
        "contact_order_confidence": g8["contact_order_confidence"],
    }


def knee_mode_argmax(soft3: tuple[float, float, float]) -> int:
    a, b, c = soft3
    m = max(a, b, c)
    if a == m:
        return 0
    if b == m:
        return 1
    return 2


KNEE_MODES = (
    "KNEE_STATIC_BENT",
    "KNEE_DYNAMIC_ACTIVE",
    "KNEE_LOW_ACTIVITY",
)
