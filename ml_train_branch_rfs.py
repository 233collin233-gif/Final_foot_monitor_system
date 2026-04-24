#!/usr/bin/env python3
"""Train four per-branch RFs from labeled saving_data CSVs, blocked/LOFO eval, write joblib."""

from __future__ import annotations

import argparse
import os
import sys

import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_activity_features import (
    DATA_DIR,
    WINDOW_SIZE,
    WINDOW_STEP,
    labeled_csv_paths,
    load_csv_files,
    simulate_adaptive_sequence_dual,
)
from ml_branch_models import (
    AUX_DIM,
    BRANCH_TO_FILE,
    apply_branch_decision_bias,
    build_full_feature_vector_from_window,
)

RANDOM_STATE = 42
RF_N_ESTIMATORS = 280
RF_MAX_DEPTH = 24
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = "sqrt"
RF_MAX_SAMPLES = 0.88
CALIBRATION_ENABLED = True
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_MIN_CLASS_SAMPLES = 12

TEST_FRAC = 0.20
VAL_FRAC_OF_REMAINING = 0.25
CV_N_SPLITS = 5
BLOCKED_N_FOLDS = 5
BLOCKED_PURGE = 5

RF_BASE_CONFIG: dict[str, object] = {
    "n_estimators": RF_N_ESTIMATORS,
    "max_depth": RF_MAX_DEPTH,
    "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
    "max_features": RF_MAX_FEATURES,
    "max_samples": RF_MAX_SAMPLES,
    "class_weight": "balanced_subsample",
}

BRANCH_RF_CANDIDATES: dict[str, list[dict[str, object]]] = {
    "sitting": [
        {},
        {"n_estimators": 360, "max_depth": 20, "min_samples_leaf": 1, "max_samples": 0.95},
    ],
    "standing": [
        {},
        {"n_estimators": 320, "max_depth": 18, "min_samples_leaf": 1, "max_samples": 0.95},
    ],
    "walking": [
        {},
        {"n_estimators": 420, "max_depth": 20, "min_samples_leaf": 1, "max_samples": 0.95},
        {"n_estimators": 500, "max_depth": 14, "min_samples_leaf": 2, "class_weight": "balanced"},
        {"n_estimators": 360, "max_depth": 12, "min_samples_leaf": 3, "max_features": None},
    ],
    "stairs": [
        {},
        {"n_estimators": 420, "max_depth": 16, "min_samples_leaf": 1, "max_samples": 0.95},
        {"n_estimators": 520, "max_depth": 10, "min_samples_leaf": 2, "class_weight": "balanced"},
        {"n_estimators": 360, "max_depth": None, "min_samples_leaf": 1, "max_features": None},
    ],
}


LABEL_TO_BRANCH: dict[str, str] = {
    "SITTING_NORMAL": "sitting",
    "SITTING_CROSSLEGGED": "sitting",
    "STAIRS_UP": "stairs",
    "STAIRS_DOWN": "stairs",
    "WALKING_FORWARD": "walking",
    "WALKING_BACKWARD": "walking",
    "STANDING_UPRIGHT": "standing",
    "STANDING_LEFT_LEAN": "standing",
    "STANDING_RIGHT_LEAN": "standing",
}


def _canonical_label(s: str) -> str:
    s = (s or "").strip()
    aliases = {
        "WALK_FWD": "WALKING_FORWARD",
        "WALK_BWD": "WALKING_BACKWARD",
        "STAIRS_UPWARDS": "STAIRS_UP",
        "STAIRS_DOWNWARDS": "STAIRS_DOWN",
    }
    return aliases.get(s.upper(), s.upper() if s else "UNKNOWN")


def _load_csv_per_file(
    data_dir: str = DATA_DIR,
) -> "list[tuple[str, np.ndarray, list[str]]]":
    """One (path, (T,6), labels) per file."""
    from ml_activity_features import _find_col  # type: ignore

    files = labeled_csv_paths(data_dir)
    out: list[tuple[str, np.ndarray, list[str]]] = []
    for path in files:
        rows: list[list[float]] = []
        labs: list[str] = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            fields = [x.strip() for x in reader.fieldnames]
            names_6 = (
                "L_Forefoot", "L_Heel", "L_Knee",
                "R_Forefoot", "R_Heel", "R_Knee",
            )
            cols6 = [_find_col(fields, n) for n in names_6]
            if not all(cols6):
                continue
            c_l_toe = _find_col(fields, "L_Toe")
            c_r_toe = _find_col(fields, "R_Toe")
            use_8 = bool(c_l_toe and c_r_toe)
            label_col = _find_col(fields, "Label")
            for row in reader:
                try:
                    if use_8:
                        v8 = [
                            float(row[c_l_toe]), float(row[cols6[0]]),
                            float(row[cols6[1]]), float(row[cols6[2]]),
                            float(row[c_r_toe]), float(row[cols6[3]]),
                            float(row[cols6[4]]), float(row[cols6[5]]),
                        ]
                        rowf = v8[1:4] + v8[5:8]
                    else:
                        rowf = [float(row[c]) for c in cols6]  # type: ignore[index]
                except (ValueError, KeyError):
                    continue
                rows.append(rowf)
                labs.append(row.get(label_col, "") if label_col else "")
        if rows:
            out.append((path, np.asarray(rows, dtype=np.float64), labs))
    return out


def _windows_for_branch_per_file(
    per_file: "list[tuple[str, np.ndarray, list[str]]]",
    calibration: "object | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build windows; segment_ids group contiguous same-(file,label) runs for CV."""
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    b_rows: list[str] = []
    g_rows: list[int] = []
    seg_rows: list[int] = []

    segment_counter = -1
    prev_key: "tuple[int, str] | None" = None

    for fidx, (_path, raw_t6, labels) in enumerate(per_file):
        seq = simulate_adaptive_sequence_dual(raw_t6, calibration=calibration)
        T = seq.shape[0]
        prev_key = None
        for i in range(0, T - WINDOW_SIZE + 1, WINDOW_STEP):
            lbl_w = [_canonical_label(labels[i + j]) for j in range(WINDOW_SIZE)]
            majority = max(set(lbl_w), key=lbl_w.count)
            if lbl_w.count(majority) / len(lbl_w) < 0.8:
                prev_key = None
                continue
            if majority not in LABEL_TO_BRANCH:
                prev_key = None
                continue
            br = LABEL_TO_BRANCH[majority]
            win = seq[i : i + WINDOW_SIZE]
            feat = build_full_feature_vector_from_window(win)
            if feat is None:
                prev_key = None
                continue
            key = (fidx, majority)
            if key != prev_key:
                segment_counter += 1
                prev_key = key
            X_rows.append(feat)
            y_rows.append(majority)
            b_rows.append(br)
            g_rows.append(fidx)
            seg_rows.append(segment_counter)

    if not X_rows:
        return (np.empty((0,)), np.array([]), np.array([]),
                np.array([]), np.array([]))
    return (
        np.stack(X_rows, axis=0),
        np.array(y_rows),
        np.array(b_rows),
        np.array(g_rows, dtype=np.int64),
        np.array(seg_rows, dtype=np.int64),
    )


def _assign_blocked_folds(
    segment_ids: np.ndarray,
    n_folds: int = BLOCKED_N_FOLDS,
) -> np.ndarray:
    """Assign fold id per window from segment chunks."""
    folds = np.full(len(segment_ids), -1, dtype=np.int64)
    for seg in np.unique(segment_ids):
        idx = np.where(segment_ids == seg)[0]
        n = len(idx)
        if n == 0:
            continue
        if n < n_folds:
            for k in range(n):
                folds[idx[k]] = k % n_folds
        else:
            boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
            for k in range(n_folds):
                folds[idx[boundaries[k] : boundaries[k + 1]]] = k
    assert (folds >= 0).all(), "every window should have a fold id"
    return folds


class BlockedIntraFileCV:
    """CV splitter: train/test from fold_ids, optional purge of boundary windows."""

    def __init__(
        self,
        fold_ids: np.ndarray,
        n_splits: int,
        purge: int = BLOCKED_PURGE,
    ) -> None:
        self.fold_ids = np.asarray(fold_ids)
        self.n_splits = int(n_splits)
        self.purge = int(purge)

    def split(self, X=None, y=None, groups=None):
        n = len(self.fold_ids)
        for k in range(self.n_splits):
            test_mask = self.fold_ids == k
            if not test_mask.any():
                yield np.where(~test_mask)[0], np.where(test_mask)[0]
                continue
            test_idx = np.where(test_mask)[0]
            train_mask = ~test_mask
            if self.purge > 0:
                widened = np.zeros(n, dtype=bool)
                for offset in range(-self.purge, self.purge + 1):
                    if offset == 0:
                        continue
                    shifted = np.roll(test_mask, offset)
                    if offset > 0:
                        shifted[:offset] = False
                    elif offset < 0:
                        shifted[offset:] = False
                    widened |= shifted
                train_mask &= ~widened
            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def _windows_for_branch(
    raw_t6: np.ndarray,
    labels: list[str],
    calibration: "object | None" = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One file's windows: features, label, branch key."""
    seq = simulate_adaptive_sequence_dual(raw_t6, calibration=calibration)
    T = seq.shape[0]
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    b_rows: list[str] = []
    for i in range(0, T - WINDOW_SIZE + 1, WINDOW_STEP):
        lbl_w = [_canonical_label(labels[i + j]) for j in range(WINDOW_SIZE)]
        majority = max(set(lbl_w), key=lbl_w.count)
        if lbl_w.count(majority) / len(lbl_w) < 0.8:
            continue
        if majority not in LABEL_TO_BRANCH:
            continue
        br = LABEL_TO_BRANCH[majority]
        win = seq[i : i + WINDOW_SIZE]
        feat = build_full_feature_vector_from_window(win)
        if feat is None:
            continue
        X_rows.append(feat)
        y_rows.append(majority)
        b_rows.append(br)
    if not X_rows:
        return np.empty((0,)), np.array([]), []
    return np.stack(X_rows, axis=0), np.array(y_rows), b_rows


def _format_confusion_matrix(
    cm: np.ndarray,
    classes: "list[str]",
    *,
    header: str,
) -> str:
    """Pretty-print cm with recall/precision."""
    w = max(10, max(len(c) for c in classes) + 2)
    lines = [f"  {header}", f"  {'(rows=true, cols=pred)'.ljust(4 + w)}"]
    head = " " * (w + 4) + " ".join(c.rjust(w) for c in classes)
    lines.append(head)
    for i, true_cls in enumerate(classes):
        row_sum = cm[i].sum()
        cells = [f"{int(cm[i, j]):>{w}d}" for j in range(len(classes))]
        suffix = f"  | n={int(row_sum)}"
        lines.append(f"  {true_cls.ljust(w)} | " + " ".join(cells) + suffix)
    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)
    diag = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(row_sums > 0, diag / np.maximum(row_sums, 1), 0.0)
        prec = np.where(col_sums > 0, diag / np.maximum(col_sums, 1), 0.0)
    lines.append("")
    lines.append(f"  {'class'.ljust(w)}   recall     precision")
    for i, cls in enumerate(classes):
        lines.append(f"  {cls.ljust(w)}   {recall[i]:.4f}     {prec[i]:.4f}")
    return "\n".join(lines)


def _save_confusion_png(
    cm: np.ndarray,
    classes: "list[str]",
    out_path: str,
    *,
    title: str,
) -> None:
    """Save confusion matrix PNG; no-op if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(classes) + 2),
                                     max(3.2, 0.9 * len(classes) + 1.5)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            frac = cm[i, j] / row_sums[i, 0]
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, f"{int(cm[i, j])}\n{frac*100:.1f}%",
                    ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _build_pipeline(rf_overrides: "dict[str, object] | None" = None) -> Pipeline:
    cfg = dict(RF_BASE_CONFIG)
    if rf_overrides:
        cfg.update(rf_overrides)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=int(cfg["n_estimators"]),
                    max_depth=cfg["max_depth"],
                    min_samples_leaf=int(cfg["min_samples_leaf"]),
                    max_features=cfg["max_features"],
                    max_samples=cfg["max_samples"],
                    random_state=RANDOM_STATE,
                    class_weight=cfg["class_weight"],
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _branch_candidate_overrides(branch: str) -> "list[dict[str, object]]":
    return BRANCH_RF_CANDIDATES.get(branch, [{}])


def _blocked_candidate_search(
    branch: str,
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    classes_ordered: "list[str]",
) -> "tuple[dict[str, object], dict[str, object]]":
    """Grid-search RF override dicts; maximize min per-class recall then OOF acc."""
    candidates = _branch_candidate_overrides(branch)
    cv = BlockedIntraFileCV(fold_ids, BLOCKED_N_FOLDS, purge=BLOCKED_PURGE)
    best_override: dict[str, object] = {}
    best_report: dict[str, object] = {
        "blocked_oof_accuracy": float("-inf"),
        "blocked_min_class_recall": float("-inf"),
        "candidate_index": -1,
    }
    for i, override in enumerate(candidates):
        pipe = _build_pipeline(override)
        oof_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        acc = float(accuracy_score(y, oof_pred))
        cm = confusion_matrix(y, oof_pred, labels=classes_ordered)
        recalls = np.divide(
            np.diag(cm).astype(np.float64),
            np.maximum(cm.sum(axis=1), 1),
            dtype=np.float64,
        )
        min_recall = float(np.min(recalls)) if recalls.size else 0.0
        cand_key = (min_recall, acc, -i)
        best_key = (
            float(best_report["blocked_min_class_recall"]),
            float(best_report["blocked_oof_accuracy"]),
            -int(best_report["candidate_index"]),
        )
        if cand_key > best_key:
            best_override = dict(override)
            best_report = {
                "blocked_oof_accuracy": acc,
                "blocked_min_class_recall": min_recall,
                "candidate_index": int(i),
            }
    return best_override, best_report


def _labels_from_proba_with_branch_bias(
    branch: str,
    classes_ordered: "list[str]",
    proba: np.ndarray,
) -> np.ndarray:
    cls = [str(c) for c in classes_ordered]
    out: list[str] = []
    for row in np.asarray(proba, dtype=np.float64):
        adj = apply_branch_decision_bias(branch, cls, row)
        ji = int(np.argmax(adj))
        out.append(cls[ji])
    return np.asarray(out, dtype=object)


def _train_one_branch(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    out_path: str,
    *,
    calibration_info: dict | None = None,
    save_plots_dir: str | None = None,
    groups: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
    eval_mode: str = "blocked_lofo",
) -> dict:
    """Fit one branch: blocked and/or LOFO and/or random split per eval_mode."""
    print(f"\n=== Branch: {name} ===")
    metrics: dict = {"branch": name, "n_samples": int(X.shape[0])}

    if X.shape[0] < 16:
        print(f"  skip: too few samples ({X.shape[0]})")
        metrics["skipped"] = True
        return metrics
    uniq, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip([str(u) for u in uniq], [int(c) for c in counts]))
    print("  class counts:", class_counts)
    metrics["class_counts"] = class_counts

    classes_ordered = sorted(set(y))
    selected_rf_overrides: dict[str, object] = {}
    pipe = _build_pipeline(selected_rf_overrides)

    run_blocked = eval_mode in ("blocked", "blocked_lofo")
    run_lofo = eval_mode in ("lofo", "blocked_lofo")
    use_blocked = (
        run_blocked
        and segment_ids is not None
        and len(segment_ids) == len(y)
    )
    use_lofo = (
        run_lofo
        and groups is not None
        and len(np.unique(groups)) >= 3
        and len(groups) == len(y)
    )

    if use_blocked:
        fold_ids = _assign_blocked_folds(segment_ids, n_folds=BLOCKED_N_FOLDS)
        n_segments = int(len(np.unique(segment_ids)))
        print(f"  [BLOCKED-CV] n_segments={n_segments}  "
              f"n_folds={BLOCKED_N_FOLDS}  purge={BLOCKED_PURGE} windows")

        for k in range(BLOCKED_N_FOLDS):
            test_lbls = set(y[fold_ids == k].tolist())
            missing = set(classes_ordered) - test_lbls
            if missing:
                print(f"    (warn) fold {k} test set missing classes: {sorted(missing)}")

        selected_rf_overrides, sel_report = _blocked_candidate_search(
            name, X, y, fold_ids, classes_ordered,
        )
        pipe = _build_pipeline(selected_rf_overrides)
        print(
            "  [model-select] blocked-CV best candidate="
            f"{sel_report['candidate_index']}  "
            f"min_class_recall={sel_report['blocked_min_class_recall']:.4f}  "
            f"oof_acc={sel_report['blocked_oof_accuracy']:.4f}"
        )
        if selected_rf_overrides:
            print(f"  [model-select] overrides: {selected_rf_overrides}")
        else:
            print("  [model-select] overrides: <baseline>")
        metrics["rf_selected_overrides"] = dict(selected_rf_overrides)

        cv = BlockedIntraFileCV(fold_ids, BLOCKED_N_FOLDS, purge=BLOCKED_PURGE)
        fold_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        oof_proba = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1, method="predict_proba")
        oof_pred = _labels_from_proba_with_branch_bias(name, classes_ordered, oof_proba)
        acc_oof = accuracy_score(y, oof_pred)
        cm_oof = confusion_matrix(y, oof_pred, labels=classes_ordered)

        print(f"  [BLOCKED per-fold]  mean = {fold_scores.mean():.4f}   "
              f"std = {fold_scores.std():.4f}   "
              f"min = {fold_scores.min():.4f}   max = {fold_scores.max():.4f}")
        print(f"  [BLOCKED aggregate OOF]  accuracy = {acc_oof:.4f}  "
              f"(over all {len(y)} windows)")
        print(_format_confusion_matrix(
            cm_oof, classes_ordered,
            header="Confusion matrix (blocked intra-file OOF aggregate):",
        ))
        print("\n  classification_report (BLOCKED OOF):")
        print(classification_report(y, oof_pred, zero_division=0,
                                     labels=classes_ordered))

        metrics.update({
            "evaluation": "blocked_intra_file",
            "blocked_n_folds": int(BLOCKED_N_FOLDS),
            "blocked_purge": int(BLOCKED_PURGE),
            "blocked_n_segments": n_segments,
            "blocked_fold_accuracies": [float(s) for s in fold_scores],
            "blocked_mean": float(fold_scores.mean()),
            "blocked_std": float(fold_scores.std()),
            "blocked_min": float(fold_scores.min()),
            "blocked_max": float(fold_scores.max()),
            "oof_accuracy": float(acc_oof),
            "classes": [str(c) for c in classes_ordered],
            "confusion_oof": cm_oof.tolist(),
        })

        if eval_mode == "blocked_lofo":
            if use_lofo:
                unique_files = np.unique(groups)
                print(f"  [LOFO-CV extra] {len(unique_files)} files → "
                      f"{len(unique_files)} folds (for cross-subject/session check)")
                logo = LeaveOneGroupOut()
                lofo_fold_scores = cross_val_score(
                    pipe, X, y, groups=groups, cv=logo, scoring="accuracy", n_jobs=-1,
                )
                lofo_oof_proba = cross_val_predict(
                    pipe, X, y, groups=groups, cv=logo, n_jobs=-1, method="predict_proba",
                )
                lofo_oof_pred = _labels_from_proba_with_branch_bias(
                    name, classes_ordered, lofo_oof_proba,
                )
                lofo_acc_oof = accuracy_score(y, lofo_oof_pred)
                lofo_cm_oof = confusion_matrix(y, lofo_oof_pred, labels=classes_ordered)
                print(f"  [LOFO extra per-file] mean = {lofo_fold_scores.mean():.4f}   "
                      f"std = {lofo_fold_scores.std():.4f}   "
                      f"min = {lofo_fold_scores.min():.4f}   max = {lofo_fold_scores.max():.4f}")
                print(f"  [LOFO extra aggregate OOF] accuracy = {lofo_acc_oof:.4f}")
                metrics.update({
                    "evaluation": "blocked_plus_lofo",
                    "lofo_n_files": int(len(unique_files)),
                    "lofo_fold_accuracies": [float(s) for s in lofo_fold_scores],
                    "lofo_mean": float(lofo_fold_scores.mean()),
                    "lofo_std": float(lofo_fold_scores.std()),
                    "lofo_min": float(lofo_fold_scores.min()),
                    "lofo_max": float(lofo_fold_scores.max()),
                    "lofo_oof_accuracy": float(lofo_acc_oof),
                    "lofo_confusion_oof": lofo_cm_oof.tolist(),
                })
                if save_plots_dir is not None:
                    os.makedirs(save_plots_dir, exist_ok=True)
                    _save_confusion_png(
                        lofo_cm_oof, [str(c) for c in classes_ordered],
                        os.path.join(save_plots_dir, f"confusion_{name}_lofo_oof.png"),
                        title=f"{name} · LOFO out-of-fold  "
                              f"(acc={lofo_acc_oof:.3f}, per-file {lofo_fold_scores.mean():.3f}±{lofo_fold_scores.std():.3f})",
                    )
                    print(f"  [plots] saved confusion_{name}_lofo_oof.png "
                          f"in {save_plots_dir}")
            else:
                print("  [LOFO-CV extra] skipped: not enough valid per-file groups.")
                metrics["lofo_skipped"] = True

        print(f"  [final fit] training deployed model on ALL {len(y)} windows...")
        deployed_model = pipe
        model_kind = "pipeline_rf"
        calib_note = "none"
        min_cls = int(np.min(counts)) if len(counts) else 0
        if CALIBRATION_ENABLED and min_cls >= CALIBRATION_MIN_CLASS_SAMPLES:
            try:
                n_cv = max(2, min(3, min_cls // 2))
                cal = CalibratedClassifierCV(
                    estimator=_build_pipeline(selected_rf_overrides),
                    method=CALIBRATION_METHOD,
                    cv=n_cv,
                )
                cal.fit(X, y)
                deployed_model = cal
                model_kind = "calibrated_cv"
                calib_note = f"{CALIBRATION_METHOD}_cv{n_cv}"
                print(f"  [calibration] enabled: {calib_note}")
            except Exception as exc:
                print(f"  [calibration] skipped due to error: {exc}")
                pipe.fit(X, y)
        else:
            pipe.fit(X, y)
            if CALIBRATION_ENABLED:
                print(
                    "  [calibration] skipped: insufficient per-class samples "
                    f"(min={min_cls}, need>={CALIBRATION_MIN_CLASS_SAMPLES})",
                )
        classes_attr = getattr(deployed_model, "classes_", None)
        if classes_attr is None:
            classes_attr = getattr(pipe.named_steps["rf"], "classes_", None)
        classes_final = [str(c) for c in list(classes_attr)] if classes_attr is not None else []

        if save_plots_dir is not None:
            os.makedirs(save_plots_dir, exist_ok=True)
            _save_confusion_png(
                cm_oof, [str(c) for c in classes_ordered],
                os.path.join(save_plots_dir, f"confusion_{name}_blocked_oof.png"),
                title=(f"{name} · blocked intra-file OOF  "
                       f"(acc={acc_oof:.3f}, per-fold "
                       f"{fold_scores.mean():.3f}±{fold_scores.std():.3f})"),
            )
            print(f"  [plots] saved confusion_{name}_blocked_oof.png "
                  f"in {save_plots_dir}")

    elif use_lofo:
        unique_files = np.unique(groups)
        print(f"  [LOFO-CV] {len(unique_files)} files → "
              f"{len(unique_files)} folds (each = hold out one whole CSV)")
        file_class_counts = {
            int(f): dict(zip(*np.unique(y[groups == f], return_counts=True)))
            for f in unique_files
        }
        missing_cls = [
            (int(f), sorted(set(classes_ordered) - set(file_class_counts[int(f)].keys())))
            for f in unique_files
        ]
        n_missing = sum(1 for _, m in missing_cls if m)
        if n_missing:
            print(f"  (note) {n_missing} / {len(unique_files)} files do not contain "
                  "every class — aggregated OOF still covers all classes.")

        logo = LeaveOneGroupOut()
        fold_scores = cross_val_score(
            pipe, X, y, groups=groups, cv=logo, scoring="accuracy", n_jobs=-1
        )
        oof_proba = cross_val_predict(
            pipe, X, y, groups=groups, cv=logo, n_jobs=-1, method="predict_proba",
        )
        oof_pred = _labels_from_proba_with_branch_bias(name, classes_ordered, oof_proba)
        acc_oof = accuracy_score(y, oof_pred)
        cm_oof = confusion_matrix(y, oof_pred, labels=classes_ordered)

        print(f"  [LOFO per-file]  mean = {fold_scores.mean():.4f}   "
              f"std = {fold_scores.std():.4f}   "
              f"min = {fold_scores.min():.4f}   max = {fold_scores.max():.4f}")
        print(f"  [LOFO aggregate OOF]  accuracy = {acc_oof:.4f}  "
              f"(over all {len(y)} windows)")
        print(_format_confusion_matrix(
            cm_oof, classes_ordered,
            header="Confusion matrix (LOFO out-of-fold aggregate):",
        ))
        print("\n  classification_report (LOFO OOF):")
        print(classification_report(y, oof_pred, zero_division=0,
                                     labels=classes_ordered))

        metrics.update({
            "evaluation": "leave_one_file_out",
            "lofo_n_files": int(len(unique_files)),
            "lofo_fold_accuracies": [float(s) for s in fold_scores],
            "lofo_mean": float(fold_scores.mean()),
            "lofo_std": float(fold_scores.std()),
            "lofo_min": float(fold_scores.min()),
            "lofo_max": float(fold_scores.max()),
            "oof_accuracy": float(acc_oof),
            "classes": [str(c) for c in classes_ordered],
            "confusion_oof": cm_oof.tolist(),
        })

        print(f"  [final fit] training deployed model on ALL {len(y)} windows...")
        deployed_model = pipe
        model_kind = "pipeline_rf"
        calib_note = "none"
        min_cls = int(np.min(counts)) if len(counts) else 0
        if CALIBRATION_ENABLED and min_cls >= CALIBRATION_MIN_CLASS_SAMPLES:
            try:
                n_cv = max(2, min(3, min_cls // 2))
                cal = CalibratedClassifierCV(
                    estimator=_build_pipeline(selected_rf_overrides),
                    method=CALIBRATION_METHOD,
                    cv=n_cv,
                )
                cal.fit(X, y)
                deployed_model = cal
                model_kind = "calibrated_cv"
                calib_note = f"{CALIBRATION_METHOD}_cv{n_cv}"
                print(f"  [calibration] enabled: {calib_note}")
            except Exception as exc:
                print(f"  [calibration] skipped due to error: {exc}")
                pipe.fit(X, y)
        else:
            pipe.fit(X, y)
            if CALIBRATION_ENABLED:
                print(
                    "  [calibration] skipped: insufficient per-class samples "
                    f"(min={min_cls}, need>={CALIBRATION_MIN_CLASS_SAMPLES})",
                )
        classes_attr = getattr(deployed_model, "classes_", None)
        if classes_attr is None:
            classes_attr = getattr(pipe.named_steps["rf"], "classes_", None)
        classes_final = [str(c) for c in list(classes_attr)] if classes_attr is not None else []

        if save_plots_dir is not None:
            os.makedirs(save_plots_dir, exist_ok=True)
            _save_confusion_png(
                cm_oof, [str(c) for c in classes_ordered],
                os.path.join(save_plots_dir, f"confusion_{name}_lofo_oof.png"),
                title=f"{name} · LOFO out-of-fold  "
                      f"(acc={acc_oof:.3f}, per-file {fold_scores.mean():.3f}±{fold_scores.std():.3f})",
            )
            print(f"  [plots] saved confusion_{name}_lofo_oof.png "
                  f"in {save_plots_dir}")

    else:
        print("  [evaluation] random 3-way split (groups not provided)")
        X_rem, X_te, y_rem, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, random_state=RANDOM_STATE, stratify=y,
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_rem, y_rem,
            test_size=VAL_FRAC_OF_REMAINING,
            random_state=RANDOM_STATE,
            stratify=y_rem,
        )
        n_splits = min(CV_N_SPLITS, int(np.min(counts)))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
            metrics["cv_mean"] = float(cv_scores.mean())
            metrics["cv_std"] = float(cv_scores.std())
        pipe.fit(X_tr, y_tr)
        deployed_model = pipe
        model_kind = "pipeline_rf"
        calib_note = "none"
        classes_final = [str(c) for c in pipe.named_steps["rf"].classes_]
        pred_val = pipe.predict(X_val)
        pred_te = pipe.predict(X_te)
        acc_val = accuracy_score(y_val, pred_val)
        acc_te = accuracy_score(y_te, pred_te)
        cm_val = confusion_matrix(y_val, pred_val, labels=classes_ordered)
        cm_te = confusion_matrix(y_te, pred_te, labels=classes_ordered)
        print(f"  val={acc_val:.4f}   test={acc_te:.4f}")
        print(_format_confusion_matrix(cm_val, classes_ordered,
                                         header="Confusion matrix (validation):"))
        print(_format_confusion_matrix(cm_te, classes_ordered,
                                         header="Confusion matrix (test):"))
        metrics.update({
            "evaluation": "random_3way_split",
            "classes": [str(c) for c in classes_ordered],
            "val_accuracy": float(acc_val),
            "test_accuracy": float(acc_te),
            "confusion_val": cm_val.tolist(),
            "confusion_test": cm_te.tolist(),
        })
        if save_plots_dir is not None:
            os.makedirs(save_plots_dir, exist_ok=True)
            _save_confusion_png(cm_val, [str(c) for c in classes_ordered],
                                 os.path.join(save_plots_dir, f"confusion_{name}_val.png"),
                                 title=f"{name} · validation  (acc={acc_val:.3f})")
            _save_confusion_png(cm_te, [str(c) for c in classes_ordered],
                                 os.path.join(save_plots_dir, f"confusion_{name}_test.png"),
                                 title=f"{name} · test  (acc={acc_te:.3f})")

    bundle = {
        "pipeline": deployed_model,
        "branch": name,
        "classes": classes_final,
        "aux_dim": AUX_DIM,
        "feature_mode": "branch_adaptive_v2",
        "model_kind": model_kind,
        "probability_calibration": calib_note,
        "calibration": calibration_info,
        "metrics": {
            k: v for k, v in metrics.items()
            if k not in ("confusion_val", "confusion_test")
        },
    }
    import joblib

    joblib.dump(bundle, out_path)
    print(f"  saved model: {out_path}  (trained on all {len(y)} windows)")
    return metrics


def _build_calibration_for_training(
    args: argparse.Namespace,
    raw: np.ndarray,
    labels: list[str],
    data_dir: str = DATA_DIR,
) -> "object | None":
    """Resolve PersonalCalibration: None, JSON path, or auto-fit on raw."""
    if args.no_calib:
        print("[calib] --no-calib → training without personal normalization.")
        return None

    import personal_calibration as pc

    if args.calib:
        if not os.path.isfile(args.calib):
            print(f"[calib] ERROR: file not found: {args.calib}")
            sys.exit(2)
        calib = pc.PersonalCalibration.load_json(args.calib)
        print(f"[calib] loaded {args.calib}  (source={calib.source!r}, "
              f"subject={calib.subject!r})")
        return calib

    calib = pc.OfflineAutoCalibrator().fit(
        raw, labels,
        subject="offline_auto_population",
        notes=f"auto-fit from {data_dir} during training",
    )
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            pc.DEFAULT_CALIBRATION_FILENAME)
    calib.save_json(out_path)
    print(f"[calib] auto-fit offline calibration saved → {out_path}")
    for w in calib.__dict__.get("warnings", []) or []:
        print("  warn:", w)
    print(calib.summary())
    return calib


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calib",
        default=None,
        help="Path to personal_calibration.json",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Disable personal calibration",
    )
    parser.add_argument(
        "--plots-dir",
        default="confusion_matrices",
        help="Directory to save confusion-matrix PNGs (set to empty string to skip).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("blocked_lofo", "blocked", "lofo", "random"),
        default="blocked_lofo",
        help="blocked_lofo | blocked | lofo | random",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder with *sensor_data_dual_labeled_*.csv. Default: ml_activity_features.DATA_DIR (project saving_data).",
    )
    args = parser.parse_args(argv)

    data_dir = (args.data_dir or DATA_DIR).strip() or DATA_DIR
    data_dir = os.path.normpath(os.path.abspath(data_dir))

    raw, labels, _subj, mode = load_csv_files(data_dir, labeled_only=True, raw_adc=True)
    if raw.size == 0 or mode != "dual":
        print("No dual-foot labeled CSV found under", data_dir)
        return 1

    calib = _build_calibration_for_training(args, raw, labels, data_dir=data_dir)
    calib_info = None if calib is None else {
        "source": calib.source,
        "subject": calib.subject,
        "min_raw": list(map(float, calib.min_raw)),
        "max_raw": list(map(float, calib.max_raw)),
    }

    per_file = _load_csv_per_file(data_dir)
    if not per_file:
        print("No labeled CSVs found under", data_dir)
        return 1
    print(f"Loaded {len(per_file)} CSV files (temporal order preserved).")

    X_all, y_all, b_all, g_all, s_all = _windows_for_branch_per_file(
        per_file, calibration=calib,
    )
    if X_all.shape[0] == 0:
        print("No sliding windows produced (check labels / WINDOW_SIZE).")
        return 1

    print("Window samples per branch (from majority label → bucket):")
    b_arr = np.array(b_all)
    for br in sorted(set(b_all)):
        mask = b_arr == br
        n = int(np.sum(mask))
        sub_y = y_all[mask]
        u, c = np.unique(sub_y, return_counts=True)
        files_in_br = sorted(set(int(x) for x in g_all[mask]))
        segs_in_br = int(len(np.unique(s_all[mask])))
        print(f"  {br}: n={n}  files_contributing={len(files_in_br)}  "
              f"segments={segs_in_br}  "
              f"labels={dict(zip([str(x) for x in u], [int(x) for x in c]))}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = None
    if args.plots_dir:
        plots_dir = os.path.join(out_dir, args.plots_dir)

    if args.eval_mode == "blocked_lofo":
        print(f"\n[eval] blocked_lofo: K={BLOCKED_N_FOLDS} purge={BLOCKED_PURGE}")
    elif args.eval_mode == "blocked":
        print(f"\n[eval] blocked: K={BLOCKED_N_FOLDS} purge={BLOCKED_PURGE}")
    elif args.eval_mode == "lofo":
        print("\n[eval] lofo")
    else:
        print("\n[eval] random split (ablation)")

    all_metrics: list[dict] = []
    for br, fname in BRANCH_TO_FILE.items():
        mask = np.array([x == br for x in b_all])
        if not np.any(mask):
            print(f"\n=== Branch {br}: no samples ===")
            continue
        groups_br = g_all[mask] if args.eval_mode in ("lofo", "blocked_lofo") else None
        seg_br = s_all[mask] if args.eval_mode in ("blocked", "blocked_lofo") else None
        m = _train_one_branch(
            br, X_all[mask], y_all[mask],
            os.path.join(out_dir, fname),
            calibration_info=calib_info,
            save_plots_dir=plots_dir,
            groups=groups_br,
            segment_ids=seg_br,
            eval_mode=args.eval_mode,
        )
        all_metrics.append(m)

    print("\n" + "=" * 94)
    print("SUMMARY")
    print("=" * 94)
    if args.eval_mode == "blocked_lofo":
        print(f"{'branch':<18}{'N':>7}{'segs':>7}{'b-mean':>10}{'b-std':>9}"
              f"{'b-oof':>10}{'files':>8}{'l-mean':>10}{'l-std':>9}{'l-oof':>10}")
        print("-" * 104)
        for m in all_metrics:
            if m.get("skipped"):
                print(f"{m['branch']:<18}{m.get('n_samples', 0):>7}  (skipped)")
                continue
            print(f"{m['branch']:<18}{m['n_samples']:>7}"
                  f"{m.get('blocked_n_segments', 0):>7}"
                  f"{m.get('blocked_mean', float('nan')):>10.4f}"
                  f"{m.get('blocked_std', float('nan')):>9.4f}"
                  f"{m.get('oof_accuracy', float('nan')):>10.4f}"
                  f"{m.get('lofo_n_files', 0):>8}"
                  f"{m.get('lofo_mean', float('nan')):>10.4f}"
                  f"{m.get('lofo_std', float('nan')):>9.4f}"
                  f"{m.get('lofo_oof_accuracy', float('nan')):>10.4f}")
    elif args.eval_mode == "blocked":
        print(f"{'branch':<20}{'N':>7}{'segs':>7}{'folds':>7}"
              f"{'mean':>11}{'std':>10}{'min':>10}{'OOF acc':>12}")
        print("-" * 94)
        for m in all_metrics:
            if m.get("skipped"):
                print(f"{m['branch']:<20}{m.get('n_samples', 0):>7}  (skipped)")
                continue
            print(f"{m['branch']:<20}{m['n_samples']:>7}"
                  f"{m.get('blocked_n_segments', 0):>7}"
                  f"{m.get('blocked_n_folds', 0):>7}"
                  f"{m.get('blocked_mean', float('nan')):>11.4f}"
                  f"{m.get('blocked_std', float('nan')):>10.4f}"
                  f"{m.get('blocked_min', float('nan')):>10.4f}"
                  f"{m.get('oof_accuracy', float('nan')):>12.4f}")
    elif args.eval_mode == "lofo":
        print(f"{'branch':<20}{'N':>7}{'files':>7}{'LOFO mean':>12}"
              f"{'LOFO std':>11}{'LOFO min':>11}{'OOF acc':>10}")
        print("-" * 94)
        for m in all_metrics:
            if m.get("skipped"):
                print(f"{m['branch']:<20}{m.get('n_samples', 0):>7}  (skipped)")
                continue
            print(f"{m['branch']:<20}{m['n_samples']:>7}"
                  f"{m.get('lofo_n_files', 0):>7}"
                  f"{m.get('lofo_mean', float('nan')):>12.4f}"
                  f"{m.get('lofo_std', float('nan')):>11.4f}"
                  f"{m.get('lofo_min', float('nan')):>11.4f}"
                  f"{m.get('oof_accuracy', float('nan')):>10.4f}")
    else:
        print(f"{'branch':<20}{'N':>7}{'CV mean':>12}{'CV std':>10}"
              f"{'val acc':>11}{'test acc':>11}")
        print("-" * 94)
        for m in all_metrics:
            if m.get("skipped"):
                print(f"{m['branch']:<20}{m.get('n_samples', 0):>7}   (skipped)")
                continue
            cv_mean = f"{m.get('cv_mean', float('nan')):.4f}" if 'cv_mean' in m else '  —  '
            cv_std = f"{m.get('cv_std', float('nan')):.4f}" if 'cv_std' in m else '  —  '
            print(f"{m['branch']:<20}{m['n_samples']:>7}{cv_mean:>12}{cv_std:>10}"
                  f"{m.get('val_accuracy', float('nan')):>11.4f}"
                  f"{m.get('test_accuracy', float('nan')):>11.4f}")
    print("=" * 94)
    if plots_dir:
        print(f"Confusion-matrix PNGs are in: {plots_dir}/")
    print(f"Deployed models were all fit on 100% of the {len(y_all)} windows "
          "(no data withheld for final training).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
