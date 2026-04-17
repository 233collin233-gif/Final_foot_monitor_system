#!/usr/bin/env python3
"""
Train four branch RandomForest models (bucketed hierarchical labels).

Buckets (by ground-truth ``Label`` in CSV, after canonicalisation):
  - active_motion:   STAIRS_UP, STAIRS_DOWN
  - active_static:   SITTING_NORMAL, SITTING_CROSSLEGGED
  - inactive_motion: WALKING_FORWARD, WALKING_BACKWARD
  - inactive_static: STANDING_UPRIGHT, STANDING_LEFT_LEAN, STANDING_RIGHT_LEAN

Features match runtime: ``extract_features_dual_adaptive`` + ``auxiliary_from_window_ratios``
(last frame ratios in the window) — same ``AUX_DIM`` as ``ml_branch_models.build_auxiliary_vector``
layout (training fills kinematic part from window proxy; stance slots from last frame loads).

Usage (from project root)::

    # 1) auto-derive a personal calibration from labelled CSVs and train with it
    python ml_train_branch_rfs.py

    # 2) reuse an existing JSON (typically produced by the UI online wizard)
    python ml_train_branch_rfs.py --calib personal_calibration.json

    # 3) skip personal calibration entirely (legacy / ablation)
    python ml_train_branch_rfs.py --no-calib

Requires ``saving_data/sensor_data_dual_labeled_*.csv`` with raw ADC columns.
"""

from __future__ import annotations

import argparse
import os
import sys

import csv
import glob

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_activity_features import (
    WINDOW_SIZE,
    WINDOW_STEP,
    load_csv_files,
    simulate_adaptive_sequence_dual,
)
from ml_branch_models import (
    AUX_DIM,
    BRANCH_TO_FILE,
    RF_ACTIVE_MOTION,
    RF_ACTIVE_STATIC,
    RF_INACTIVE_MOTION,
    RF_INACTIVE_STATIC,
    build_full_feature_vector,
    auxiliary_from_window_ratios,
)

# TODO_PARAM
DATA_DIR = "saving_data"
RANDOM_STATE = 42
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 22

# Three-way split (no extra dataset available → carve it out of the 33 CSVs).
# 60 / 20 / 20 is the default: train / validation / test.
TEST_FRAC = 0.20
VAL_FRAC_OF_REMAINING = 0.25         # 0.25 of 0.80 = 0.20 of the whole
CV_N_SPLITS = 5                      # Stratified K-Fold on the train portion

# Blocked CV parameters (default evaluation).
BLOCKED_N_FOLDS = 5
# Purge = number of adjacent windows to drop from TRAIN on each side of each
# TEST block, so the two sets do not share frames via the sliding-window overlap.
# WINDOW_SIZE=10 and WINDOW_STEP=2 → 5 windows of overlap span.
BLOCKED_PURGE = 5


LABEL_TO_BRANCH: dict[str, str] = {
    "STAIRS_UP": "active_motion",
    "STAIRS_DOWN": "active_motion",
    "SITTING_NORMAL": "active_static",
    "SITTING_CROSSLEGGED": "active_static",
    "WALKING_FORWARD": "inactive_motion",
    "WALKING_BACKWARD": "inactive_motion",
    "STANDING_UPRIGHT": "inactive_static",
    "STANDING_LEFT_LEAN": "inactive_static",
    "STANDING_RIGHT_LEAN": "inactive_static",
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
    """Iterate labelled CSVs **one by one**, preserving each file's temporal order.

    Returns ``[(filepath, raw_t8, labels), ...]`` where ``raw_t8`` is ``(T_i, 8)``
    ADC counts and ``labels`` is the aligned list of ground-truth labels.
    Uses the same column-resolution logic as :func:`ml_activity_features.load_csv_files`
    but never concatenates files — time continuity within each CSV is preserved.
    """
    # Import lazily to avoid a cycle at module import time.
    from ml_activity_features import _find_col  # type: ignore

    pattern = os.path.join(data_dir, "sensor_data_dual_labeled_*.csv")
    files = sorted(glob.glob(pattern))
    out: list[tuple[str, np.ndarray, list[str]]] = []
    for path in files:
        rows: list[list[float]] = []
        labs: list[str] = []
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            fields = [x.strip() for x in reader.fieldnames]
            cols = [
                _find_col(fields, n)
                for n in (
                    "L_Toe", "L_Forefoot", "L_Heel", "L_Knee",
                    "R_Toe", "R_Forefoot", "R_Heel", "R_Knee",
                )
            ]
            if not all(cols):
                continue
            label_col = _find_col(fields, "Label")
            for row in reader:
                try:
                    rows.append([float(row[c]) for c in cols])  # type: ignore[index]
                except (ValueError, KeyError):
                    continue
                labs.append(row.get(label_col, "") if label_col else "")
        if rows:
            out.append((path, np.asarray(rows, dtype=np.float64), labs))
    return out


def _windows_for_branch_per_file(
    per_file: "list[tuple[str, np.ndarray, list[str]]]",
    calibration: "object | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, branch, file_idx, segment_id) for each window.

    Each window is also tagged with a **segment id** — a sequential integer
    that only stays the same while the previous and current window share
    ``(file_idx, majority_label)``. A segment is therefore a *maximal
    contiguous run of same-label windows inside one CSV*. Later the blocked
    CV carves each segment into K contiguous chunks so every fold's test set
    covers every class.

    Because ``simulate_adaptive_sequence_dual`` is deterministic per-frame
    when fed a calibration with global stats (the bank is frozen), processing
    files independently yields bit-identical features to the concat path.
    """
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    b_rows: list[str] = []
    g_rows: list[int] = []
    seg_rows: list[int] = []

    segment_counter = -1
    prev_key: "tuple[int, str] | None" = None

    for fidx, (_path, raw_t8, labels) in enumerate(per_file):
        seq = simulate_adaptive_sequence_dual(raw_t8, calibration=calibration)
        T = seq.shape[0]
        # A new file always starts a fresh segment-counter context.
        prev_key = None
        for i in range(0, T - WINDOW_SIZE + 1, WINDOW_STEP):
            lbl_w = [_canonical_label(labels[i + j]) for j in range(WINDOW_SIZE)]
            majority = max(set(lbl_w), key=lbl_w.count)
            if lbl_w.count(majority) / len(lbl_w) < 0.8:
                # Mixed window → break segment continuity.
                prev_key = None
                continue
            if majority not in LABEL_TO_BRANCH:
                prev_key = None
                continue
            br = LABEL_TO_BRANCH[majority]
            win = seq[i : i + WINDOW_SIZE]
            ratios_last = win[-1, :, 1].astype(np.float64)
            aux = auxiliary_from_window_ratios(ratios_last)
            flat = win.reshape(WINDOW_SIZE, 24)
            feat = build_full_feature_vector(flat, aux)
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


# ─────────────────────────────────────────────────────────────────────────────
#  Blocked intra-file CV (default evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def _assign_blocked_folds(
    segment_ids: np.ndarray,
    n_folds: int = BLOCKED_N_FOLDS,
) -> np.ndarray:
    """Assign fold IDs (0..n_folds-1) to every window via contiguous chunks
    *within each label-segment*.

    A label-segment is a maximal contiguous run of windows sharing the same
    ``(file_idx, majority_label)`` (see ``_windows_for_branch_per_file``).
    Splitting each segment into ``n_folds`` contiguous pieces and using the
    k-th piece of every segment as fold k's test set guarantees:

      * Every fold's test set contains samples from every label that exists in
        the dataset (as long as that label occurs in at least one segment).
      * Every test block is contiguous in time, so local temporal order is
        preserved inside the block.
      * Very short segments (< n_folds windows) fall back to round-robin
        assignment; their contribution is small enough that this doesn't break
        the invariant in practice.
    """
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
    """A pre-computed, purged CV iterator compatible with sklearn.

    ``fold_ids[i]`` = the fold the i-th window belongs to. For each fold k:

      * test = windows with ``fold_ids == k``
      * train = windows with ``fold_ids != k`` **minus** any window within
        ``purge`` positions (array order) of a test window → prevents the
        sliding-window overlap at chunk boundaries from leaking information.
    """

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
                # Degenerate fold — yield empty test, full train; sklearn will
                # handle or skip it. In practice this only happens when every
                # segment was shorter than n_splits, which we guard against.
                yield np.where(~test_mask)[0], np.where(test_mask)[0]
                continue
            # Purge zone: exclude from TRAIN any window within `purge` positions
            # of any TEST window (in array order within this branch).
            test_idx = np.where(test_mask)[0]
            train_mask = ~test_mask
            if self.purge > 0:
                # Vectorised purge: widen the test mask by ±purge, subtract
                # from train.
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
    raw_t8: np.ndarray,
    labels: list[str],
    calibration: "object | None" = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns X (n, feat_dim), y labels, branch list for each row.

    Passing ``calibration`` (a :class:`personal_calibration.PersonalCalibration`)
    routes the raw stream through personal ``[min, max]`` normalization
    before the EWMA bank; the feature dimensionality is unchanged.
    """
    seq = simulate_adaptive_sequence_dual(raw_t8, calibration=calibration)
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
        ratios_last = win[-1, :, 1].astype(np.float64)
        aux = auxiliary_from_window_ratios(ratios_last)
        flat = win.reshape(WINDOW_SIZE, 24)
        feat = build_full_feature_vector(flat, aux)
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
    """ASCII confusion matrix. Rows = true label, columns = predicted label."""
    w = max(10, max(len(c) for c in classes) + 2)
    lines = [f"  {header}", f"  {'(rows=true, cols=pred)'.ljust(4 + w)}"]
    head = " " * (w + 4) + " ".join(c.rjust(w) for c in classes)
    lines.append(head)
    for i, true_cls in enumerate(classes):
        row_sum = cm[i].sum()
        cells = [f"{int(cm[i, j]):>{w}d}" for j in range(len(classes))]
        suffix = f"  | n={int(row_sum)}"
        lines.append(f"  {true_cls.ljust(w)} | " + " ".join(cells) + suffix)
    # per-class recall / precision
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
    """Save a labelled confusion-matrix heatmap. Silently skip if matplotlib missing."""
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
    # annotate counts
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


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    max_depth=RF_MAX_DEPTH,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


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
    eval_mode: str = "blocked",
) -> dict:
    """Train one branch RF.

    Evaluation strategies:

    * ``eval_mode="blocked"`` (default, recommended for time-series)
      Per-segment contiguous-block CV. Each **label-segment** (a maximal run
      of same-label windows inside one CSV) is sliced into ``BLOCKED_N_FOLDS``
      contiguous chunks; fold k tests on the k-th chunk of every segment. So
      every fold's test set covers every class present in the dataset, each
      test chunk is a contiguous run of frames (local time order preserved),
      and a purge gap is applied so train/test don't share frames via the
      sliding-window overlap.

    * ``eval_mode="lofo"``
      Leave-One-File-Out. Each fold holds out one whole CSV. Strictest in
      terms of subject/session independence, but many files contain only a
      subset of labels, which inflates std and underestimates per-class recall.

    * ``eval_mode="random"``
      60 / 20 / 20 stratified shuffle split — fast but leaks neighbour frames
      across train/val/test. Kept for ablation.

    The **deployed model is always fitted on every window** after evaluation.
    """
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

    pipe = _build_pipeline()
    classes_ordered = sorted(set(y))   # for consistent confusion-matrix axes

    use_blocked = (
        eval_mode == "blocked"
        and segment_ids is not None
        and len(segment_ids) == len(y)
    )
    use_lofo = (
        not use_blocked
        and eval_mode == "lofo"
        and groups is not None
        and len(np.unique(groups)) >= 3
        and len(groups) == len(y)
    )

    if use_blocked:
        # ── Blocked intra-file CV ───────────────────────────────────────────
        fold_ids = _assign_blocked_folds(segment_ids, n_folds=BLOCKED_N_FOLDS)
        n_segments = int(len(np.unique(segment_ids)))
        print(f"  [BLOCKED-CV] n_segments={n_segments}  "
              f"n_folds={BLOCKED_N_FOLDS}  purge={BLOCKED_PURGE} windows")

        # Sanity check: is every label represented in every fold's test set?
        for k in range(BLOCKED_N_FOLDS):
            test_lbls = set(y[fold_ids == k].tolist())
            missing = set(classes_ordered) - test_lbls
            if missing:
                print(f"    (warn) fold {k} test set missing classes: {sorted(missing)}")

        cv = BlockedIntraFileCV(fold_ids, BLOCKED_N_FOLDS, purge=BLOCKED_PURGE)
        # (1) Per-fold scores
        fold_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        # (2) Aggregated OOF predictions (fold-wise, so each window gets
        #     exactly one prediction from the fold where it was in the test).
        oof_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
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

        # ── Deployed model: fit on ALL windows ──────────────────────────────
        print(f"  [final fit] training deployed model on ALL {len(y)} windows...")
        pipe.fit(X, y)
        classes_final = [str(c) for c in pipe.named_steps["rf"].classes_]

        # ── Save PNG ────────────────────────────────────────────────────────
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
        # ── LOFO-CV ─────────────────────────────────────────────────────────
        unique_files = np.unique(groups)
        print(f"  [LOFO-CV] {len(unique_files)} files → "
              f"{len(unique_files)} folds (each = hold out one whole CSV)")
        # Warn if some files lack all classes for this branch (very common —
        # a CSV that only recorded "SITTING_NORMAL" has no "SITTING_CROSSLEGGED"
        # so the fold trained without it still predicts across all classes;
        # sklearn handles this gracefully).
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
        # (1) Per-fold accuracy (33 numbers)
        fold_scores = cross_val_score(
            pipe, X, y, groups=groups, cv=logo, scoring="accuracy", n_jobs=-1
        )
        # (2) Aggregated OOF predictions (one prediction per window)
        oof_pred = cross_val_predict(
            pipe, X, y, groups=groups, cv=logo, n_jobs=-1
        )
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

        # ── Deployed model: fit on ALL windows (no data wasted) ─────────────
        print(f"  [final fit] training deployed model on ALL {len(y)} windows...")
        pipe.fit(X, y)
        classes_final = [str(c) for c in pipe.named_steps["rf"].classes_]

        # ── Save PNG ────────────────────────────────────────────────────────
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
        # ── Legacy 60/20/20 stratified shuffle split (only when no groups) ──
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

    # ── Persist deployed model ──────────────────────────────────────────────────
    bundle = {
        "pipeline": pipe,
        "branch": name,
        "classes": classes_final,
        "aux_dim": AUX_DIM,
        "feature_mode": "branch_adaptive_v2",
        "calibration": calibration_info,
        "metrics": {
            k: v for k, v in metrics.items()
            # Drop only the very large per-fold arrays; keep confusion_oof so
            # notebooks can re-render it from the joblib bundle directly.
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
) -> "object | None":
    """Resolve which PersonalCalibration to use.

    Priority:
      1. ``--no-calib``            → no calibration (legacy mode)
      2. ``--calib <json>``        → load the JSON and use it verbatim
      3. default                   → auto-fit from the CSVs we already loaded
                                     and dump to ``personal_calibration.json``
    """
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
        notes=f"auto-fit from {DATA_DIR} during training",
    )
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            pc.DEFAULT_CALIBRATION_FILENAME)
    calib.save_json(out_path)
    print(f"[calib] auto-fit offline calibration saved → {out_path}")
    for w in calib.__dict__.get("warnings", []) or []:   # not used by dataclass
        print("  warn:", w)
    print(calib.summary())
    return calib


def main(argv: "list[str] | None" = None) -> int:
    """Entry point. Pass ``argv=[]`` when calling from a Jupyter notebook so
    argparse does not pick up the kernel's own CLI flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calib",
        default=None,
        help="Path to personal_calibration.json to reuse (e.g. produced by the UI wizard).",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Disable personal calibration entirely (ablation / legacy).",
    )
    parser.add_argument(
        "--plots-dir",
        default="confusion_matrices",
        help="Directory to save confusion-matrix PNGs (set to empty string to skip).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("blocked", "lofo", "random"),
        default="blocked",
        help="Evaluation strategy. "
             "'blocked' (default, recommended) = per-segment contiguous-block CV: every "
             "label-segment in every CSV is sliced into K contiguous chunks, fold k tests "
             "on the k-th chunk of every segment, so every fold's test set covers all "
             "classes while each test block stays temporally contiguous. "
             "'lofo' = Leave-One-File-Out (strict subject independence, but many held-out "
             "CSVs contain only 1-2 labels → high variance). "
             "'random' = legacy 60/20/20 stratified shuffle split (leaks neighbour frames).",
    )
    args = parser.parse_args(argv)

    # Still load the full concat once — the OfflineAutoCalibrator needs the
    # whole dataset to compute the 5 global stats (this does NOT shuffle; it
    # just concatenates file contents end-to-end so quantiles / mean / std
    # are derived from the whole population).
    raw, labels, _subj, mode = load_csv_files(DATA_DIR, labeled_only=True, raw_adc=True)
    if raw.size == 0 or mode != "dual":
        print("No dual-foot labeled CSV found under", DATA_DIR)
        return 1

    calib = _build_calibration_for_training(args, raw, labels)
    calib_info = None if calib is None else {
        "source": calib.source,
        "subject": calib.subject,
        "min_raw": list(map(float, calib.min_raw)),
        "max_raw": list(map(float, calib.max_raw)),
    }

    # ── Load the same CSVs file-by-file so we can tag each window with its
    #    origin file index → preserves per-file time integrity.
    per_file = _load_csv_per_file(DATA_DIR)
    if not per_file:
        print("No labeled CSVs found under", DATA_DIR)
        return 1
    print(f"Loaded {len(per_file)} CSV files in original temporal order "
          f"(no intra-file shuffling).")

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

    # ── Decide evaluation mode
    if args.eval_mode == "blocked":
        print(f"\n[eval] Using BLOCKED intra-file CV (K={BLOCKED_N_FOLDS}, "
              f"purge={BLOCKED_PURGE}). Every label-segment is split into "
              f"{BLOCKED_N_FOLDS} contiguous chunks; fold k tests on the k-th "
              "chunk of every segment. Every fold's test set therefore "
              "contains every class; each test block is a contiguous run of "
              "frames; final model fits all windows.")
    elif args.eval_mode == "lofo":
        print("\n[eval] Using Leave-One-File-Out cross-validation "
              "(held-out CSVs may contain only 1-2 labels — high variance).")
    else:
        print("\n[eval] Using legacy 60/20/20 stratified shuffle split "
              "(IGNORES time continuity, kept only for ablation).")

    all_metrics: list[dict] = []
    for br, fname in BRANCH_TO_FILE.items():
        mask = np.array([x == br for x in b_all])
        if not np.any(mask):
            print(f"\n=== Branch {br}: no samples ===")
            continue
        groups_br = g_all[mask] if args.eval_mode == "lofo" else None
        seg_br = s_all[mask] if args.eval_mode == "blocked" else None
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

    # ── Final summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 94)
    print("                          FINAL SUMMARY — four branch RFs")
    print("=" * 94)
    if args.eval_mode == "blocked":
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
