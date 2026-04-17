"""
Shared feature extraction + CSV loading for RandomForest activity training and inference.
Dual-foot 8-channel CSV: L_Toe…L_Knee, R_Toe…R_Knee (see sensor_data_dual_*.csv).

Must stay in sync with realtime_recognizer / training notebook sample rate and windowing.
"""

from __future__ import annotations

import csv
import glob
import os
import numpy as np

SENSOR_MAX = 4095.0

# TODO_PARAM: Must match MCU firmware line rate and realtime_recognizer.SAMPLE_HZ.
# Hardware currently streams one frame every 100 ms (confirmed from CSV timestamps),
# i.e. 10 Hz. If the firmware is later bumped, update this number here AND in
# realtime_recognizer.py, then retrain all four branch RFs.
SAMPLE_HZ = 10

# TODO_PARAM: sliding-window duration (seconds). 1.0 s @ 10 Hz → 10 frames per window
# (tight window for lowest latency).  FFT still works (N=10 → 5 usable non-DC bins);
# the statistical / gait features are the primary signal, not the spectrum.
ML_WINDOW_DURATION_S = 1.0
# TODO_PARAM: Window length in samples — must be ≥ 4 or the FFT block below
# (_fft_channel_features) has fewer than 2 useful bins.
WINDOW_SIZE = max(4, int(round(ML_WINDOW_DURATION_S * SAMPLE_HZ)))   # → 10 frames

# TODO_PARAM: Training window stride.  Step = 2 frames at 10 Hz → 0.2 s hop,
# 80 % overlap, one new window per ~200 ms (matches UI refresh cadence).
WINDOW_STEP = 2

FEATURE_MODE_ADAPTIVE_V2 = "adaptive_v2"

# Hierarchical four-branch RF exports (see ``ml_train_branch_rfs.py``); sole deployable models.
RF_BRANCH_ACTIVE_MOTION = "rf_active_motion.joblib"
RF_BRANCH_ACTIVE_STATIC = "rf_active_static.joblib"
RF_BRANCH_INACTIVE_MOTION = "rf_inactive_motion.joblib"
RF_BRANCH_INACTIVE_STATIC = "rf_inactive_static.joblib"

# TODO_PARAM: CSV data directory
DATA_DIR = "saving_data"

VALID_LABELS = {
    "WALKING_FORWARD", "WALKING_BACKWARD",
    "STAIRS_UP", "STAIRS_DOWN",
    "SITTING_NORMAL", "SITTING_CROSSLEGGED",
    "SIT_TO_STAND",
    "STANDING_UPRIGHT", "STANDING_LEFT_LEAN", "STANDING_RIGHT_LEAN",
    "UNKNOWN",
}


def _raw_to_pressure(raw: float) -> float:
    """Legacy global linear map (not for RF v2). Prefer adaptive features for ML."""
    return float(np.clip((SENSOR_MAX - raw) / SENSOR_MAX, 0.0, 1.0))


def _fft_channel_features(col: np.ndarray, N: int) -> list[float]:
    fft_vals = np.abs(np.fft.rfft(col - col.mean()))
    freqs = np.fft.rfftfreq(N, d=1.0 / SAMPLE_HZ)
    spectral_energy = float(np.sum(fft_vals ** 2))
    feats = [spectral_energy]
    if len(fft_vals) > 1:
        idx = np.argmax(fft_vals[1:]) + 1
        feats.append(float(freqs[idx]))
    else:
        feats.append(0.0)
    psd = fft_vals ** 2
    psd_norm = psd / (psd.sum() + 1e-12)
    entropy = -float(np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    feats.append(entropy)
    return feats


def _time_features_for_column(col: np.ndarray) -> list[float]:
    feats: list[float] = []
    feats.append(float(col.mean()))
    feats.append(float(col.std()))
    feats.append(float(col.min()))
    feats.append(float(col.max()))
    feats.append(float(col.max() - col.min()))
    feats.append(float(np.median(col)))
    feats.append(float(np.sqrt(np.mean(col ** 2))))
    feats.append(float(np.mean(np.diff(col))))
    peaks = np.sum((col[1:-1] > col[:-2]) & (col[1:-1] > col[2:]))
    feats.append(float(peaks))
    feats.append(float(np.var(col)))
    return feats


def _cross_block_four(window4: np.ndarray) -> list[float]:
    N = len(window4)
    feats: list[float] = []
    p_sum = window4[:, :3].sum(axis=1)
    feats.append(float(p_sum.mean()))
    feats.append(float(p_sum.std()))
    feats.append(float(p_sum.max()))
    knee_range = float(window4[:, 3].max() - window4[:, 3].min())
    feats.append(knee_range)
    toe_s = window4[:, 0].sum()
    ff_s = window4[:, 1].sum()
    heel_s = window4[:, 2].sum()
    foot_total = toe_s + ff_s + heel_s + 1e-9
    feats.append(float(toe_s / foot_total))
    feats.append(float(heel_s / foot_total))
    feats.append(float(ff_s / foot_total))
    feats.append(float((toe_s + ff_s) / foot_total))
    if N > 2:
        with np.errstate(invalid="ignore"):
            c1 = np.corrcoef(window4[:, 0], window4[:, 2])[0, 1]
            c2 = np.corrcoef(window4[:, 0], window4[:, 3])[0, 1]
        feats.append(float(np.nan_to_num(c1, nan=0.0)))
        feats.append(float(np.nan_to_num(c2, nan=0.0)))
    else:
        feats.extend([0.0, 0.0])
    return feats


def extract_features(window: np.ndarray) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError("window must be 2D")
    c = window.shape[1]
    if c == 4:
        return _extract_features_single(window)
    if c == 8:
        return _extract_features_dual(window)
    raise ValueError(f"Expected 4 or 8 channels, got {c}")


def _extract_features_single(window: np.ndarray) -> np.ndarray:
    feats: list[float] = []
    N = len(window)
    for ch in range(4):
        col = window[:, ch]
        feats.extend(_time_features_for_column(col))
        feats.extend(_fft_channel_features(col, N))
    feats.extend(_cross_block_four(window))
    return np.array(feats, dtype=np.float64)


def _ratio_slice_foot(window_8x3: np.ndarray, start: int, end: int) -> np.ndarray:
    """Toe/forefoot/heel/knee relative_pressure_ratio slice (N, 4)."""
    return window_8x3[:, start:end, 1].astype(np.float64)


def extract_features_dual_adaptive(window_8x3: np.ndarray) -> np.ndarray:
    """
    RandomForest input for adaptive_v2: window shape (N, 8, 3) with per-channel
    [baseline_removed, relative_pressure_ratio, adaptive_zscore].
    Must match realtime_recognizer streaming preprocessor outputs.
    """
    if window_8x3.ndim != 3 or window_8x3.shape[1] != 8 or window_8x3.shape[2] != 3:
        raise ValueError(f"Expected (N, 8, 3) adaptive window, got {window_8x3.shape}")
    feats: list[float] = []
    N = len(window_8x3)
    for ch in range(8):
        for k in range(3):
            col = window_8x3[:, ch, k]
            feats.extend(_time_features_for_column(col))
            feats.extend(_fft_channel_features(col, N))
    wl = _ratio_slice_foot(window_8x3, 0, 4)
    wr = _ratio_slice_foot(window_8x3, 4, 8)
    feats.extend(_cross_block_four(wl))
    feats.extend(_cross_block_four(wr))
    hl, hr = wl[:, 2], wr[:, 2]
    kl, kr = wl[:, 3], wr[:, 3]
    feats.append(float(np.mean(np.abs(hl - hr))))
    if N > 2:
        with np.errstate(invalid="ignore"):
            chh = np.corrcoef(hl, hr)[0, 1]
            ck = np.corrcoef(kl, kr)[0, 1]
        feats.append(float(np.nan_to_num(chh, nan=0.0)))
        feats.append(float(np.nan_to_num(ck, nan=0.0)))
    else:
        feats.extend([0.0, 0.0])
    for ch in range(4):
        diff = wl[:, ch] - wr[:, ch]
        feats.append(float(np.mean(diff)))
        feats.append(float(np.std(diff)))
        feats.append(float(np.max(np.abs(diff))))
    return np.array(feats, dtype=np.float64)


def extract_features_single_adaptive(window_4x3: np.ndarray) -> np.ndarray:
    if window_4x3.ndim != 3 or window_4x3.shape[1] != 4 or window_4x3.shape[2] != 3:
        raise ValueError(f"Expected (N, 4, 3) adaptive window, got {window_4x3.shape}")
    feats: list[float] = []
    N = len(window_4x3)
    for ch in range(4):
        for k in range(3):
            col = window_4x3[:, ch, k]
            feats.extend(_time_features_for_column(col))
            feats.extend(_fft_channel_features(col, N))
    w4 = window_4x3[:, :, 1].astype(np.float64)
    feats.extend(_cross_block_four(w4))
    return np.array(feats, dtype=np.float64)


def _extract_features_dual(window: np.ndarray) -> np.ndarray:
    feats: list[float] = []
    N = len(window)
    for ch in range(8):
        col = window[:, ch]
        feats.extend(_time_features_for_column(col))
        feats.extend(_fft_channel_features(col, N))
    feats.extend(_cross_block_four(window[:, 0:4]))
    feats.extend(_cross_block_four(window[:, 4:8]))
    hl, hr = window[:, 2], window[:, 6]
    kl, kr = window[:, 3], window[:, 7]
    feats.append(float(np.mean(np.abs(hl - hr))))
    if N > 2:
        with np.errstate(invalid="ignore"):
            chh = np.corrcoef(hl, hr)[0, 1]
            ck = np.corrcoef(kl, kr)[0, 1]
        feats.append(float(np.nan_to_num(chh, nan=0.0)))
        feats.append(float(np.nan_to_num(ck, nan=0.0)))
    else:
        feats.extend([0.0, 0.0])
    for ch in range(4):
        diff = window[:, ch] - window[:, ch + 4]
        feats.append(float(np.mean(diff)))
        feats.append(float(np.std(diff)))
        feats.append(float(np.max(np.abs(diff))))
    return np.array(feats, dtype=np.float64)


FEATURE_DIM_SINGLE = 62
FEATURE_DIM_DUAL = 139

# Adaptive pipeline: per physical channel 3 streams × (10 time + 3 FFT) + cross blocks
# Dual: 8 * 3 * 13 + 10 + 10 + 3 + 12 = 347
FEATURE_DIM_SINGLE_ADAPTIVE = 4 * 3 * 13 + 10  # 166
FEATURE_DIM_DUAL_ADAPTIVE = 8 * 3 * 13 + 10 + 10 + 3 + 12  # 347


def _parse_cell(s: str) -> float | None:
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _find_col(fields: list[str], canonical: str) -> str | None:
    for fn in fields:
        if fn.strip().lower() == canonical.lower():
            return fn
    return None


def load_csv_files(
    data_dir: str = DATA_DIR,
    labeled_only: bool = False,
    *,
    raw_adc: bool = False,
):
    all_rows: list[np.ndarray] = []
    all_labels: list[str] = []
    all_subjects: list[str] = []

    if labeled_only:
        pattern = os.path.join(data_dir, "sensor_data_dual_labeled_*.csv")
    else:
        pattern = os.path.join(data_dir, "*.csv")

    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        parts = fname.replace(".csv", "").split("_")
        subject = "default"
        for p in parts:
            if p.lower().startswith("sub") or p.lower().startswith("person"):
                subject = p
                break

        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            fields = [x.strip() for x in reader.fieldnames]

            c_l_toe = _find_col(fields, "L_Toe")
            c_l_ff = _find_col(fields, "L_Forefoot")
            c_l_heel = _find_col(fields, "L_Heel")
            c_l_knee = _find_col(fields, "L_Knee")
            c_r_toe = _find_col(fields, "R_Toe")
            c_r_ff = _find_col(fields, "R_Forefoot")
            c_r_heel = _find_col(fields, "R_Heel")
            c_r_knee = _find_col(fields, "R_Knee")
            label_col = _find_col(fields, "Label")

            is_dual = all(
                [c_l_toe, c_l_ff, c_l_heel, c_l_knee, c_r_toe, c_r_ff, c_r_heel, c_r_knee]
            )

            if is_dual:
                for row in reader:
                    lbl = row.get(label_col, "").strip() if label_col else ""
                    lbl = lbl if lbl else "UNKNOWN"
                    vals = [
                        _parse_cell(row[c_l_toe]),
                        _parse_cell(row[c_l_ff]),
                        _parse_cell(row[c_l_heel]),
                        _parse_cell(row[c_l_knee]),
                        _parse_cell(row[c_r_toe]),
                        _parse_cell(row[c_r_ff]),
                        _parse_cell(row[c_r_heel]),
                        _parse_cell(row[c_r_knee]),
                    ]
                    if any(v is None for v in vals):
                        continue
                    if raw_adc:
                        arr8 = np.array([float(v) for v in vals], dtype=np.float64)
                    else:
                        arr8 = np.array([_raw_to_pressure(v) for v in vals], dtype=np.float64)
                    all_rows.append(arr8)
                    all_labels.append(lbl)
                    all_subjects.append(subject)
                continue

    if not all_rows:
        return np.empty((0, 8)), [], [], "dual"

    return np.vstack(all_rows), all_labels, all_subjects, "dual"


def build_dataset(
    data: np.ndarray,
    labels: list[str],
    subjects: list[str],
    *,
    exclude_if_any_label_in_window: set[str] | frozenset[str] | None = None,
):
    """
    Sliding windows with majority label ≥ 80% agreement.
    If exclude_if_any_label_in_window is set, drop any window that contains one of those row labels.
    """
    banned = exclude_if_any_label_in_window or set()
    X_list, y_list, subj_list = [], [], []
    for i in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_STEP):
        lbl_window = labels[i: i + WINDOW_SIZE]
        if banned and any(lbl in banned for lbl in lbl_window):
            continue
        majority = max(set(lbl_window), key=lbl_window.count)
        if lbl_window.count(majority) / len(lbl_window) < 0.8:
            continue

        window = data[i: i + WINDOW_SIZE]
        feat = extract_features(window)
        X_list.append(feat)
        y_list.append(majority)

        subj_window = subjects[i: i + WINDOW_SIZE]
        subj_list.append(max(set(subj_window), key=subj_window.count))

    if not X_list:
        return np.empty((0, FEATURE_DIM_DUAL)), np.array([]), []
    return np.array(X_list), np.array(y_list), subj_list


def simulate_adaptive_sequence_dual(
    raw_data: np.ndarray,
    calibration: "object | None" = None,
) -> np.ndarray:
    """One causal pass over recorded raw ADC: ``(T, 8) → (T, 8, 3)``.

    Parameters
    ----------
    raw_data : np.ndarray
        ``(T, 8)`` raw ADC counts, column order ``CHANNEL_NAMES_DUAL``.
    calibration : PersonalCalibration, optional
        If provided, raw frames are first linearly rescaled per channel
        from their personal ``[min_raw, max_raw]`` to the full ``[0, 4095]``
        ADC range before being fed to the EWMA bank.  This makes the
        downstream features roughly domain-invariant across subjects.
        Pass ``None`` (default) for the legacy behaviour (no personal
        rescaling — useful for back-compat sanity checks).

    The knee-gate in :mod:`realtime_recognizer` bypasses this function
    entirely and reads the true raw ADC, so the strict ``KNEE_RAW_STRAIGHT_TH``
    4095 rule is unaffected by whatever calibration you pass in here.

    Global-stats pathway
    --------------------
    If the provided ``calibration`` carries global statistics (i.e. the
    offline auto-calibrator has computed ``baseline_raw`` / ``press_min`` /
    ``press_max`` / ``press_mean`` / ``press_std`` across the full training
    population), those numbers are **frozen into every channel** of the
    ``DualFootAdaptiveBank`` — every window in every file sees the exact
    same ``(baseline_removed, relative_pressure_ratio, adaptive_zscore)``
    for the same ``raw``, so no per-file or per-burst local drift leaks
    into training features.  If only ``[min_raw, max_raw]`` are present
    (e.g. legacy JSON) the bank transparently falls back to its online
    EWMA behaviour — backwards-compatible either way.
    """
    from adaptive_preprocessing import DualFootAdaptiveBank

    seeds = None
    if calibration is not None:
        # `.normalize_to_adc` is a duck-typed contract shared by both
        # ``PersonalCalibration`` and anything else that exposes it.
        raw_data = np.asarray(calibration.normalize_to_adc(raw_data), dtype=np.float64)
        to_seeds = getattr(calibration, "to_channel_seeds", None)
        if callable(to_seeds):
            seeds = to_seeds()  # None → no global stats → legacy EWMA path

    bank = DualFootAdaptiveBank(seeds=seeds)
    t_max = int(raw_data.shape[0])
    out = np.zeros((t_max, 8, 3), dtype=np.float64)
    for t in range(t_max):
        flat8x3, _ = bank.update(raw_data[t])
        out[t] = flat8x3.reshape(8, 3)
    return out


def build_dataset_adaptive(
    raw_data: np.ndarray,
    labels: list[str],
    subjects: list[str],
    *,
    exclude_if_any_label_in_window: set[str] | frozenset[str] | None = None,
    calibration: "object | None" = None,
):
    """
    Sliding windows on adaptively preprocessed sequence (matches inference).
    ``raw_data`` must be ``(T, 8)`` raw ADC counts.

    ``calibration`` (optional) — a :class:`personal_calibration.PersonalCalibration`.
    If provided and it carries global statistics, the underlying
    ``simulate_adaptive_sequence_dual`` freezes the adaptive bank to those
    global values so every window is computed against the exact same numbers.
    If ``None``, the legacy EWMA behaviour is used (useful for ablation).
    """
    banned = exclude_if_any_label_in_window or set()
    seq = simulate_adaptive_sequence_dual(raw_data, calibration=calibration)
    X_list, y_list, subj_list = [], [], []
    for i in range(0, len(seq) - WINDOW_SIZE + 1, WINDOW_STEP):
        lbl_window = labels[i: i + WINDOW_SIZE]
        if banned and any(lbl in banned for lbl in lbl_window):
            continue
        majority = max(set(lbl_window), key=lbl_window.count)
        if lbl_window.count(majority) / len(lbl_window) < 0.8:
            continue
        window = seq[i: i + WINDOW_SIZE]
        feat = extract_features_dual_adaptive(window)
        X_list.append(feat)
        y_list.append(majority)
        subj_window = subjects[i: i + WINDOW_SIZE]
        subj_list.append(max(set(subj_window), key=subj_window.count))
    if not X_list:
        return np.empty((0, FEATURE_DIM_DUAL_ADAPTIVE)), np.array([]), []
    return np.array(X_list), np.array(y_list), subj_list
