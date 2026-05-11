"""Section 3.1.1 figure outputs for the fast path (no full PromptAD rerun)."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    outdir = Path("outputs/figures/sec3_promptad")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "fastpath_done.txt").write_text(
        "Fast path only: figures in the paper are not regenerated here.\n"
        "See outputs/cached_results/sec3_promptad/ for bundled CSV stubs.\n"
        "Set FULL_RUN=1 in run.sh for full PromptAD pipelines (requires prior artifacts).\n",
        encoding="utf-8",
    )
    print(f"wrote {outdir / 'fastpath_done.txt'}")


if __name__ == "__main__":
    main()
