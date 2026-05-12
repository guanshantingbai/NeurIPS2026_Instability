"""Section 3.1.1 marker when using stub fallback only (SEC3_PROMPTAD_ALLOW_STUB=1)."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    outdir = Path("outputs/figures/sec3_promptad")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "fastpath_done.txt").write_text(
        "Stub fallback (SEC3_PROMPTAD_ALLOW_STUB=1): bundled CSV only — not empirical raw reproduction.\n"
        "For Section 3.1.1 from PromptAD Stage 1 export, omit the env var and provide\n"
        "outputs/cached_results/raw_scores/promptad/unified_raw_scores_wide.csv (or _long.csv).\n",
        encoding="utf-8",
    )
    print(f"wrote {outdir / 'fastpath_done.txt'}")


if __name__ == "__main__":
    main()
