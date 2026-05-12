#!/usr/bin/env python3
"""Append / finalize rows for PromptAD full-path run status CSV (used by run_promptad_raw.sh)."""
from __future__ import annotations

import argparse
import csv
import os
from typing import List

FIELDS: List[str] = [
    "dataset",
    "category",
    "shot",
    "seed",
    "train_status",
    "infer_status",
    "export_status",
    "per_sample_path",
    "error_message",
    "start_time",
    "end_time",
]


def _truncate_err(s: str, max_len: int = 2000) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " | ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def cmd_init(args: argparse.Namespace) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(args.path)) or ".", exist_ok=True)
    with open(args.path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
    return 0


def cmd_append(args: argparse.Namespace) -> int:
    err_raw = args.error_message or ""
    if getattr(args, "error_file", None) and args.error_file and os.path.isfile(args.error_file):
        with open(args.error_file, "r", encoding="utf-8", errors="replace") as ef:
            err_raw = ef.read()
    row = {
        "dataset": args.dataset,
        "category": args.category,
        "shot": str(args.shot),
        "seed": str(args.seed),
        "train_status": args.train_status,
        "infer_status": args.infer_status,
        "export_status": args.export_status,
        "per_sample_path": args.per_sample_path,
        "error_message": _truncate_err(err_raw),
        "start_time": args.start_time or "",
        "end_time": args.end_time or "",
    }
    with open(args.path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writerow(row)
    return 0


def cmd_finalize_export(args: argparse.Namespace) -> int:
    path = args.path
    if not os.path.isfile(path):
        return 0
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames != FIELDS and r.fieldnames is not None:
            # tolerate extra/missing only if same core
            pass
        for row in r:
            if row.get("export_status") == "pending":
                p = row.get("per_sample_path", "").strip()
                if p and os.path.isfile(p):
                    row["export_status"] = "ok"
                else:
                    row["export_status"] = "missing_per_sample"
            rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="truncate and write header")
    pi.add_argument("path")
    pi.set_defaults(func=cmd_init)

    pa = sub.add_parser("append", help="append one status row")
    pa.add_argument("path")
    pa.add_argument("--dataset", required=True)
    pa.add_argument("--category", required=True)
    pa.add_argument("--shot", required=True)
    pa.add_argument("--seed", required=True)
    pa.add_argument("--train-status", required=True, dest="train_status")
    pa.add_argument("--infer-status", required=True, dest="infer_status")
    pa.add_argument("--export-status", required=True, dest="export_status")
    pa.add_argument("--per-sample-path", required=True, dest="per_sample_path")
    pa.add_argument("--error-message", default="", dest="error_message")
    pa.add_argument(
        "--error-file",
        default="",
        dest="error_file",
        help="if set and file exists, overrides --error-message (raw log tail)",
    )
    pa.add_argument("--start-time", default="", dest="start_time")
    pa.add_argument("--end-time", default="", dest="end_time")
    pa.set_defaults(func=cmd_append)

    pf = sub.add_parser("finalize-export", help="set pending export_status from disk")
    pf.add_argument("path")
    pf.set_defaults(func=cmd_finalize_export)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
