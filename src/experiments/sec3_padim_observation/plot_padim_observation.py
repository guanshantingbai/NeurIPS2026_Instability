"""Section 3.1.2 figure outputs for the fast path (no full PaDiM rerun)."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    outdir = Path("outputs/figures/sec3_padim")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "fastpath_done.txt").write_text(
        "Fast path only: paper figures are not regenerated here.\n"
        "See outputs/cached_results/sec3_padim/ for bundled CSV stubs.\n"
        "Set FULL_RUN=1 in run.sh for full PaDiM seed-killer (long GPU run).\n",
        encoding="utf-8",
    )
    print(f"wrote {outdir / 'fastpath_done.txt'}")


if __name__ == "__main__":
    main()
