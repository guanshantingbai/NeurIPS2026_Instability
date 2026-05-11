"""Section 4 fast-path markers (local stub compute only; not paper figures)."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    fig_dir = Path("outputs/figures/sec4_systematic")
    tbl_dir = Path("outputs/tables/sec4_systematic")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "fastpath_done.txt").write_text(
        "Fast path: local stub scripts ran; main-paper Fig 3–6 are not regenerated here.\n"
        "Set FULL_RUN=1 on sec4 run.sh to invoke external PromptAD strengthening (requires pilot data).\n",
        encoding="utf-8",
    )
    (tbl_dir / "fastpath_done.txt").write_text(
        "Fast path: no paper tables regenerated; see compute_*.py stubs for structure only.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
