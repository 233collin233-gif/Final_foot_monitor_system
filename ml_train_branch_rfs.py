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

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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


def _train_one_branch(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    out_path: str,
    *,
    calibration_info: dict | None = None,
    save_plots_dir: str | None = None,
) -> dict:
    """Train a single branch RF with a 3-way split + K-fold CV + confusion matrices.

    Returns a metrics dict suitable for tabular summaries in a notebook.
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

    # ── 3-way split: train / val / test  (default 60 / 20 / 20, stratified) ───────
    # Step 1: hold out TEST_FRAC for test
    X_rem, X_te, y_rem, y_te = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=RANDOM_STATE, stratify=y,
    )
    # Step 2: out of the remaining 80%, take VAL_FRAC_OF_REMAINING as validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rem, y_rem,
        test_size=VAL_FRAC_OF_REMAINING,
        random_state=RANDOM_STATE,
        stratify=y_rem,
    )
    print(f"  [split] train N={len(y_tr)}   val N={len(y_val)}   test N={len(y_te)}")

    # ── Build pipeline ──────────────────────────────────────────────────────────
    pipe = Pipeline(
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

    # ── Stratified K-Fold CV on the TRAIN portion (never touches val / test) ────
    n_splits = min(CV_N_SPLITS, int(np.min(counts)))   # guard against small classes
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"  [{n_splits}-fold CV on train]  accuracy = "
              f"{cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}   "
              f"(folds: {', '.join(f'{s:.3f}' for s in cv_scores)})")
        metrics["cv_n_splits"] = int(n_splits)
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"] = float(cv_scores.std())
        metrics["cv_folds"] = [float(s) for s in cv_scores]
    else:
        print("  [CV] skipped (smallest class has <2 samples)")

    # ── Final fit on train, evaluate on val then test ───────────────────────────
    pipe.fit(X_tr, y_tr)
    classes_ = [str(c) for c in pipe.named_steps["rf"].classes_]

    pred_val = pipe.predict(X_val)
    acc_val = accuracy_score(y_val, pred_val)
    cm_val = confusion_matrix(y_val, pred_val, labels=classes_)
    print(f"  [validation]  accuracy = {acc_val:.4f}")
    print(_format_confusion_matrix(cm_val, classes_, header="Confusion matrix (validation):"))

    pred_te = pipe.predict(X_te)
    acc_te = accuracy_score(y_te, pred_te)
    cm_te = confusion_matrix(y_te, pred_te, labels=classes_)
    print(f"\n  [test]        accuracy = {acc_te:.4f}")
    print(_format_confusion_matrix(cm_te, classes_, header="Confusion matrix (test):"))
    print("\n  classification_report (test):")
    print(classification_report(y_te, pred_te, zero_division=0))

    metrics.update({
        "classes": classes_,
        "val_accuracy": float(acc_val),
        "test_accuracy": float(acc_te),
        "confusion_val": cm_val.tolist(),
        "confusion_test": cm_te.tolist(),
    })

    # ── Save PNG confusion matrices ─────────────────────────────────────────────
    if save_plots_dir is not None:
        os.makedirs(save_plots_dir, exist_ok=True)
        _save_confusion_png(
            cm_val, classes_,
            os.path.join(save_plots_dir, f"confusion_{name}_val.png"),
            title=f"{name} · validation  (acc={acc_val:.3f})",
        )
        _save_confusion_png(
            cm_te, classes_,
            os.path.join(save_plots_dir, f"confusion_{name}_test.png"),
            title=f"{name} · test  (acc={acc_te:.3f})",
        )
        print(f"  [plots] saved confusion_{name}_val.png / confusion_{name}_test.png "
              f"in {save_plots_dir}")

    # ── Persist trained model ───────────────────────────────────────────────────
    bundle = {
        "pipeline": pipe,
        "branch": name,
        "classes": classes_,
        "aux_dim": AUX_DIM,
        "feature_mode": "branch_adaptive_v2",
        "calibration": calibration_info,
        "metrics": {
            k: v for k, v in metrics.items()
            if k not in ("confusion_val", "confusion_test")  # keep the joblib small
        },
    }
    import joblib

    joblib.dump(bundle, out_path)
    print(f"  saved model: {out_path}")
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
    args = parser.parse_args(argv)

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

    X_all, y_all, b_all = _windows_for_branch(raw, labels, calibration=calib)
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
        print(f"  {br}: n={n}  labels={dict(zip([str(x) for x in u], [int(x) for x in c]))}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = None
    if args.plots_dir:
        plots_dir = os.path.join(out_dir, args.plots_dir)

    all_metrics: list[dict] = []
    for br, fname in BRANCH_TO_FILE.items():
        mask = np.array([x == br for x in b_all])
        if not np.any(mask):
            print(f"\n=== Branch {br}: no samples ===")
            continue
        m = _train_one_branch(
            br, X_all[mask], y_all[mask],
            os.path.join(out_dir, fname),
            calibration_info=calib_info,
            save_plots_dir=plots_dir,
        )
        all_metrics.append(m)

    # ── Final summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("                     FINAL SUMMARY — four branch RFs")
    print("=" * 78)
    print(f"{'branch':<20}{'N':>7}{'CV mean':>12}{'CV std':>10}{'val acc':>11}{'test acc':>11}")
    print("-" * 78)
    for m in all_metrics:
        if m.get("skipped"):
            print(f"{m['branch']:<20}{m.get('n_samples', 0):>7}   (skipped)")
            continue
        cv_mean = f"{m.get('cv_mean', float('nan')):.4f}" if 'cv_mean' in m else '  —  '
        cv_std = f"{m.get('cv_std', float('nan')):.4f}" if 'cv_std' in m else '  —  '
        print(f"{m['branch']:<20}{m['n_samples']:>7}{cv_mean:>12}{cv_std:>10}"
              f"{m['val_accuracy']:>11.4f}{m['test_accuracy']:>11.4f}")
    print("=" * 78)
    if plots_dir:
        print(f"Confusion-matrix PNGs are in: {plots_dir}/")
    print("Done. Place generated *.joblib next to foot_pressure_monitor / realtime_recognizer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
