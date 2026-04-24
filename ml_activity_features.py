"""RF training: load dual-foot labeled CSVs, sliding windows, adaptive (T,6,3) feature extraction."""

from __future__ import annotations

import csv
import glob
import hashlib
import os
import sys
import numpy as np

SENSOR_MAX = 4095.0

SAMPLE_HZ = 10
ML_WINDOW_DURATION_S = 1.0
WINDOW_SIZE = max(4, int(round(ML_WINDOW_DURATION_S * SAMPLE_HZ)))
WINDOW_STEP = 2

FEATURE_MODE_ADAPTIVE_V2 = "adaptive_v2"

RF_BRANCH_SITTING = "rf_sitting.joblib"
RF_BRANCH_STAIRS = "rf_stairs.joblib"
RF_BRANCH_WALKING = "rf_walking.joblib"
RF_BRANCH_STANDING = "rf_standing.joblib"

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_MODULE_DIR, "saving_data")
LABELED_CSV_GLOB = "*sensor_data_dual_labeled_*.csv"
IGNORED_DIR_NAMES = {"__MACOSX", ".ipynb_checkpoints"}
IGNORED_FILE_NAMES = {".DS_Store"}
IGNORED_FILE_PREFIXES = ("._",)


def _should_ignore_path(path: str) -> bool:
    sp = path.replace("\\", "/")
    parts = sp.split("/")
    if any(part in IGNORED_DIR_NAMES for part in parts):
        return True
    name = os.path.basename(path)
    if name in IGNORED_FILE_NAMES:
        return True
    if name.startswith(IGNORED_FILE_PREFIXES):
        return True
    return False


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _path_rank(path: str) -> tuple[int, int, str]:
    return (path.count(os.sep), len(path), path)


def labeled_csv_paths(data_dir: str | None = None) -> list[str]:
    """Glob *sensor_data_dual_labeled_*.csv under data_dir; dedupe by basename and content hash."""
    d = os.path.normpath(data_dir or DATA_DIR)
    if not os.path.isdir(d):
        return []
    rec = os.path.join(d, "**", LABELED_CSV_GLOB)
    found: set[str] = set()
    out: list[str] = []
    for p in glob.glob(rec, recursive=True):
        npath = os.path.normpath(os.path.realpath(p))
        if not os.path.isfile(npath):
            continue
        if npath in found:
            continue
        if _should_ignore_path(npath):
            continue
        found.add(npath)
        out.append(npath)
    by_base: dict[str, list[str]] = {}
    for p in out:
        by_base.setdefault(os.path.basename(p), []).append(p)

    chosen: list[str] = []
    for base, paths in by_base.items():
        if len(paths) == 1:
            chosen.append(paths[0])
            continue
        ranked = sorted(paths, key=_path_rank)
        hashes: dict[str, list[str]] = {}
        hash_failed = False
        for p in ranked:
            try:
                hs = _file_sha1(p)
            except OSError:
                hash_failed = True
                break
            hashes.setdefault(hs, []).append(p)
        if hash_failed:
            print(
                f"[labeled_csv_paths] WARNING: duplicate basename but hash failed, keep all: {base}",
                file=sys.stderr,
            )
            chosen.extend(ranked)
            continue
        if len(hashes) == 1:
            kept = ranked[0]
            print(
                f"[labeled_csv_paths] INFO: duplicate basename same content, keep shallow: {base} -> {kept}",
                file=sys.stderr,
            )
            chosen.append(kept)
            continue
        print(
            f"[labeled_csv_paths] WARNING: duplicate basename with DIFFERENT content, keep all: {base}",
            file=sys.stderr,
        )
        for p in ranked:
            print(f"  - {p}", file=sys.stderr)
        chosen.extend(ranked)

    dedup_by_hash: dict[str, str] = {}
    final_paths: list[str] = []
    for p in sorted(chosen, key=_path_rank):
        try:
            hs = _file_sha1(p)
        except OSError:
            final_paths.append(p)
            continue
        kept = dedup_by_hash.get(hs)
        if kept is None:
            dedup_by_hash[hs] = p
            final_paths.append(p)
            continue
        if _path_rank(p) < _path_rank(kept):
            dedup_by_hash[hs] = p
            final_paths = [x for x in final_paths if x != kept]
            final_paths.append(p)
            print(
                f"[labeled_csv_paths] INFO: cross-name duplicate content, prefer shallow: {kept} -> {p}",
                file=sys.stderr,
            )
            continue
        print(
            f"[labeled_csv_paths] INFO: cross-name duplicate content skipped: {p} (same as {kept})",
            file=sys.stderr,
        )

    return sorted(final_paths, key=lambda p: (os.path.basename(p).lower(), p))

VALID_LABELS = {
    "WALKING_FORWARD", "WALKING_BACKWARD",
    "STAIRS_UP", "STAIRS_DOWN",
    "SITTING_NORMAL", "SITTING_CROSSLEGGED",
    "SIT_TO_STAND",
    "STANDING_UPRIGHT", "STANDING_LEFT_LEAN", "STANDING_RIGHT_LEAN",
    "UNKNOWN",
}


def _raw_to_pressure(raw: float) -> float:
    """Legacy (4095-raw)/4095; not used for adaptive_v2 RF."""
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
    if c == 6:
        return _extract_features_dual_six(window)
    if c == 8:
        s8 = window.astype(np.float64)
        six = np.column_stack(
            [s8[:, 1:4], s8[:, 5:8]],
        )
        return _extract_features_dual_six(six)
    raise ValueError(f"Expected 4, 6, or 8 channels, got {c}")


def _extract_features_single(window: np.ndarray) -> np.ndarray:
    feats: list[float] = []
    N = len(window)
    for ch in range(4):
        col = window[:, ch]
        feats.extend(_time_features_for_column(col))
        feats.extend(_fft_channel_features(col, N))
    feats.extend(_cross_block_four(window))
    return np.array(feats, dtype=np.float64)


def _ratio_slice_foot(window_6x3: np.ndarray, start: int, end: int) -> np.ndarray:
    return window_6x3[:, start:end, 1].astype(np.float64)


def _cross_block_foot3(window3: np.ndarray) -> list[float]:
    """Stats for one foot's (N,3) ratio rows (ff,heel,knee)."""
    N = len(window3)
    feats: list[float] = []
    foot = window3[:, 0:2].sum(axis=1)
    feats.append(float(foot.mean()))
    feats.append(float(foot.std()))
    feats.append(float(foot.max()))
    knee_range = float(window3[:, 2].max() - window3[:, 2].min())
    feats.append(knee_range)
    ff_s = float(window3[:, 0].sum())
    h_s = float(window3[:, 1].sum())
    tot = ff_s + h_s + 1e-9
    feats.append(float(ff_s / tot))
    feats.append(float(h_s / tot))
    if N > 2:
        with np.errstate(invalid="ignore"):
            c0 = np.corrcoef(window3[:, 0], window3[:, 1])[0, 1]
            c1 = np.corrcoef(window3[:, 0], window3[:, 2])[0, 1]
        feats.append(float(np.nan_to_num(c0, nan=0.0)))
        feats.append(float(np.nan_to_num(c1, nan=0.0)))
    else:
        feats.extend([0.0, 0.0])
    return feats


def extract_features_dual_adaptive(window_6x3: np.ndarray) -> np.ndarray:
    """Hand-crafted stats from (N,6,3) adaptive window (time + FFT + cross-foot)."""
    if window_6x3.ndim != 3 or window_6x3.shape[1] != 6 or window_6x3.shape[2] != 3:
        raise ValueError(f"Expected (N, 6, 3) adaptive window, got {window_6x3.shape}")
    feats: list[float] = []
    N = len(window_6x3)
    for ch in range(6):
        for k in range(3):
            col = window_6x3[:, ch, k]
            feats.extend(_time_features_for_column(col))
            feats.extend(_fft_channel_features(col, N))
    wl = _ratio_slice_foot(window_6x3, 0, 3)
    wr = _ratio_slice_foot(window_6x3, 3, 6)
    feats.extend(_cross_block_foot3(wl))
    feats.extend(_cross_block_foot3(wr))
    hl, hr = wl[:, 1], wr[:, 1]
    kl, kr = wl[:, 2], wr[:, 2]
    feats.append(float(np.mean(np.abs(hl - hr))))
    if N > 2:
        with np.errstate(invalid="ignore"):
            chh = np.corrcoef(hl, hr)[0, 1]
            ck = np.corrcoef(kl, kr)[0, 1]
        feats.append(float(np.nan_to_num(chh, nan=0.0)))
        feats.append(float(np.nan_to_num(ck, nan=0.0)))
    else:
        feats.extend([0.0, 0.0])
    for ch in range(3):
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


def _extract_features_dual_six(window: np.ndarray) -> np.ndarray:
    """Legacy features from raw 6ch columns (no adaptive tensor)."""
    feats: list[float] = []
    N = len(window)
    for ch in range(6):
        col = window[:, ch]
        feats.extend(_time_features_for_column(col))
        feats.extend(_fft_channel_features(col, N))
    wl, wr = window[:, 0:3], window[:, 3:6]
    feats.extend(_cross_block_foot3(wl))
    feats.extend(_cross_block_foot3(wr))
    hl, hr = window[:, 1], window[:, 4]
    kl, kr = window[:, 2], window[:, 5]
    feats.append(float(np.mean(np.abs(hl - hr))))
    if N > 2:
        with np.errstate(invalid="ignore"):
            chh = np.corrcoef(hl, hr)[0, 1]
            ck = np.corrcoef(kl, kr)[0, 1]
        feats.append(float(np.nan_to_num(chh, nan=0.0)))
        feats.append(float(np.nan_to_num(ck, nan=0.0)))
    else:
        feats.extend([0.0, 0.0])
    for ch in range(3):
        diff = wl[:, ch] - wr[:, ch]
        feats.append(float(np.mean(diff)))
        feats.append(float(np.std(diff)))
        feats.append(float(np.max(np.abs(diff))))
    return np.array(feats, dtype=np.float64)


FEATURE_DIM_SINGLE = 62
FEATURE_DIM_DUAL = int(_extract_features_dual_six(np.zeros((WINDOW_SIZE, 6))).shape[0])

FEATURE_DIM_SINGLE_ADAPTIVE = 4 * 3 * 13 + 10
FEATURE_DIM_DUAL_ADAPTIVE = int(
    extract_features_dual_adaptive(np.zeros((WINDOW_SIZE, 6, 3))).shape[0]
)


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
        paths = labeled_csv_paths(data_dir)
    else:
        pattern = os.path.join(data_dir, "*.csv")
        paths = sorted(glob.glob(pattern))

    for path in paths:
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

            c_l_ff = _find_col(fields, "L_Forefoot")
            c_l_heel = _find_col(fields, "L_Heel")
            c_l_knee = _find_col(fields, "L_Knee")
            c_r_ff = _find_col(fields, "R_Forefoot")
            c_r_heel = _find_col(fields, "R_Heel")
            c_r_knee = _find_col(fields, "R_Knee")
            label_col = _find_col(fields, "Label")

            c_l_toe = _find_col(fields, "L_Toe")
            c_r_toe = _find_col(fields, "R_Toe")
            is_6 = all(
                [c_l_ff, c_l_heel, c_l_knee, c_r_ff, c_r_heel, c_r_knee]
            )
            is_8 = is_6 and c_l_toe and c_r_toe

            if is_6 and not c_l_toe:
                for row in reader:
                    lbl = row.get(label_col, "").strip() if label_col else ""
                    lbl = lbl if lbl else "UNKNOWN"
                    vals = [
                        _parse_cell(row[c_l_ff]),
                        _parse_cell(row[c_l_heel]),
                        _parse_cell(row[c_l_knee]),
                        _parse_cell(row[c_r_ff]),
                        _parse_cell(row[c_r_heel]),
                        _parse_cell(row[c_r_knee]),
                    ]
                    if any(v is None for v in vals):
                        continue
                    arr6 = np.array([float(x) for x in vals], dtype=np.float64)
                    if not raw_adc:
                        arr6 = np.array(
                            [_raw_to_pressure(v) for v in arr6], dtype=np.float64,
                        )
                    all_rows.append(arr6)
                    all_labels.append(lbl)
                    all_subjects.append(subject)
                continue

            if is_8:
                for row in reader:
                    lbl = row.get(label_col, "").strip() if label_col else ""
                    lbl = lbl if lbl else "UNKNOWN"
                    v8 = [
                        _parse_cell(row[c_l_toe]), _parse_cell(row[c_l_ff]),
                        _parse_cell(row[c_l_heel]), _parse_cell(row[c_l_knee]),
                        _parse_cell(row[c_r_toe]), _parse_cell(row[c_r_ff]),
                        _parse_cell(row[c_r_heel]), _parse_cell(row[c_r_knee]),
                    ]
                    if any(x is None for x in v8):
                        continue
                    a8 = np.array([float(x) for x in v8], dtype=np.float64)
                    arr6 = np.concatenate([a8[1:4], a8[5:8]]).astype(np.float64)
                    if not raw_adc:
                        arr6 = np.array(
                            [_raw_to_pressure(v) for v in arr6], dtype=np.float64,
                        )
                    all_rows.append(arr6)
                    all_labels.append(lbl)
                    all_subjects.append(subject)
                continue

    if labeled_only and paths and not all_rows:
        print(
            f"[load_csv_files] WARNING: {len(paths)} file(s) matched but 0 rows loaded. "
            f"Check headers in {data_dir!r} (expect 6ch L_Forefoot…R_Knee, or 8ch + L_Toe/R_Toe).",
            file=sys.stderr,
        )

    if not all_rows:
        return np.empty((0, 6)), [], [], "dual"

    return np.vstack(all_rows), all_labels, all_subjects, "dual"


def build_dataset(
    data: np.ndarray,
    labels: list[str],
    subjects: list[str],
    *,
    exclude_if_any_label_in_window: set[str] | frozenset[str] | None = None,
):
    """80% label-agreement windows; optional drop if any row label in banned set."""
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
    """Causal (T,6) ADC → (T,6,3) adaptive features. Optional PersonalCalibration: normalize_to_adc and frozen seeds if has_global_stats."""
    from adaptive_preprocessing import DualFootAdaptiveBank

    raw_data = np.asarray(raw_data, dtype=np.float64)
    if raw_data.ndim == 2 and raw_data.shape[1] == 8:
        raw_data = np.column_stack([raw_data[:, 1:4], raw_data[:, 5:8]])

    seeds = None
    if calibration is not None:
        raw_data = np.asarray(calibration.normalize_to_adc(raw_data), dtype=np.float64)
        to_seeds = getattr(calibration, "to_channel_seeds", None)
        if callable(to_seeds):
            seeds = to_seeds()
    bank = DualFootAdaptiveBank(seeds=seeds)
    t_max = int(raw_data.shape[0])
    out = np.zeros((t_max, 6, 3), dtype=np.float64)
    for t in range(t_max):
        flat18, _ = bank.update(raw_data[t])
        out[t] = flat18.reshape(6, 3)
    return out


def build_dataset_adaptive(
    raw_data: np.ndarray,
    labels: list[str],
    subjects: list[str],
    *,
    exclude_if_any_label_in_window: set[str] | frozenset[str] | None = None,
    calibration: "object | None" = None,
):
    """Sliding windows on simulate_adaptive_sequence_dual output (inference-consistent)."""
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
