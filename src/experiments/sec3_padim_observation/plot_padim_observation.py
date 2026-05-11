"""Section 3.1.2 figure markers / optional quick plot from Protocol B marginal CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    cache = Path("outputs/cached_results/sec3_padim")
    outdir = Path("outputs/figures/sec3_padim")
    outdir.mkdir(parents=True, exist_ok=True)

    marginal = cache / "marginal_protocol_b.csv"
    stub = cache / "marginal_stub.csv"
    if marginal.is_file():
        df = pd.read_csv(marginal)
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        if len(df) and "auroc" in df.columns and "instability" in df.columns:
            ax.scatter(df["auroc"].astype(float), df["instability"].astype(float), alpha=0.75, s=36)
            ax.set_xlabel("fused AUROC (Protocol B)")
            ax.set_ylabel("mean sample instability")
            ax.set_title("PaDiM Protocol B (FULL_RUN)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(outdir / "padim_marginal_scatter.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        msg = (
            f"FULL_RUN marginal from {marginal} ({len(df)} rows).\n"
            f"Scatter: {outdir / 'padim_marginal_scatter.png'}\n"
        )
    elif stub.is_file():
        df = pd.read_csv(stub)
        msg = f"Fast path marginal from {stub} ({len(df)} rows).\n"
    else:
        msg = "No marginal_protocol_b.csv or marginal_stub.csv under outputs/cached_results/sec3_padim.\n"

    (outdir / "fastpath_done.txt").write_text(
        msg
        + "Unified raw scores (FULL_RUN): outputs/cached_results/raw_scores/padim/\n"
        + "See docs/FULLPATH_PADIM.md.\n",
        encoding="utf-8",
    )
    print(f"wrote {outdir / 'fastpath_done.txt'}")


if __name__ == "__main__":
    main()
