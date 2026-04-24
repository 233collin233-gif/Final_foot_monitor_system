from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

TOE = ("L_Toe", "R_Toe")
SIX = ("L_Forefoot", "L_Heel", "L_Knee", "R_Forefoot", "R_Heel", "R_Knee")


def _list_target_csvs(data_dir: Path) -> list[Path]:
    out: list[Path] = []
    for pat in ("sensor_data_dual_labeled_*.csv", "sensor_data_dual_raw_*.csv"):
        for p in sorted(data_dir.glob(pat)):
            if p.is_file():
                out.append(p)
    return out


def _build_output_fieldnames(cols: list[str]) -> list[str] | None:
    """Return ordered header for 6ch CSV, or None if not an 8ch toe layout to strip."""
    cset = {x.strip() for x in cols}
    if not (TOE[0] in cset and TOE[1] in cset):
        if all(s in cset for s in SIX) and not (cset & set(TOE)):
            return None
        if any(t in cset for t in TOE):
            return None
        return None
    if "Timestamp" not in cset:
        return None
    out = ["Timestamp"]
    for s in SIX:
        if s not in cset:
            return None
        out.append(s)
    if "Label" in cset:
        out.append("Label")
    return out


def strip_toe_in_csv(path: Path, dry_run: bool) -> str:
    """Outcome tag: stripped | skipped | error."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  [err] cannot read {path}: {e}", file=sys.stderr)
        return "error"

    lines = text.splitlines()
    if not lines:
        return "skipped"

    r = csv.reader([lines[0]])
    try:
        header = next(r)
    except StopIteration:
        return "skipped"
    fields = [h.strip() for h in header]
    outnames = _build_output_fieldnames(fields)
    if outnames is None:
        return "skipped"

    if dry_run:
        print(f"  [dry-run] would rewrite: {path}  -> columns {outnames}")
        return "stripped"

    rows_out: list[list[str]] = [outnames]
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return "skipped"
        drows = list(reader)

    for drow in drows:
        line = []
        for name in outnames:
            v = drow.get(name, "")
            line.append(str(v) if v is not None else "")
        rows_out.append(line)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerows(rows_out)
    return "stripped"


def main_strip(data_dir: Path, dry_run: bool) -> int:
    if not data_dir.is_dir():
        print(f"data directory missing: {data_dir}", file=sys.stderr)
        return 1
    files = _list_target_csvs(data_dir)
    if not files:
        print(f"no {data_dir}/sensor_data_dual_labeled_*.csv or sensor_data_dual_raw_*.csv")
        return 0
    n_strip = 0
    n_skip = 0
    for p in files:
        r = strip_toe_in_csv(p, dry_run=dry_run)
        if r == "stripped":
            n_strip += 1
            if not dry_run:
                print(f"  stripped toe: {p.name}")
        elif r == "skipped":
            n_skip += 1
        else:
            return 1
    print(f"done: rewrote {n_strip} file(s), skipped {n_skip} (no toe or already 6ch).")
    return 0


def main() -> int:
    if "--" in sys.argv:
        i = sys.argv.index("--")
        argv_strip = sys.argv[1:i]
        train_argv = sys.argv[i + 1 :]
    else:
        argv_strip = sys.argv[1:]
        train_argv = []

    p = argparse.ArgumentParser(description="Drop toe columns from CSVs, then train RFs.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("saving_data"),
        help="Directory with sensor_data_dual_labeled_*.csv and sensor_data_dual_raw_*.csv",
    )
    p.add_argument("--dry-run", action="store_true", help="Print changes only, do not write")
    p.add_argument("--no-train", action="store_true", help="Strip columns only, skip training")
    a = p.parse_args(argv_strip)

    os.chdir(Path(__file__).resolve().parent)
    print("== 1) Remove L_Toe / R_Toe from CSVs ==")
    rc = main_strip(a.data_dir, dry_run=a.dry_run)
    if rc != 0:
        return rc
    if a.no_train or a.dry_run:
        if a.dry_run and not a.no_train:
            print("dry-run: training skipped; run without --dry-run to train.")
        return 0

    print("== 2) Train RandomForest branches and write joblib ==")
    from ml_train_branch_rfs import main as train_main

    return train_main(train_argv)


if __name__ == "__main__":
    raise SystemExit(main())
