#!/usr/bin/env python3
"""Concatenate PromptAD run status CSVs (same header) into one file."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="status CSV paths in order")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    rows: List[dict] = []
    fields: List[str] | None = None
    for p in args.inputs:
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if fields is None:
                fields = list(r.fieldnames or [])
            for row in r:
                rows.append(row)
    if not fields:
        raise SystemExit("no header")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} data rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
