#!/usr/bin/env python3
"""
Merge PaDiM Section-4 style panels (same layout as PromptAD):
  - fig3 & fig4 → fig3_4_merged_mechanism.{pdf,png}  (uses plot_mechanism_combined_section4.plot_mechanism_figure)
  - fig5 & fig6 → fig5_6_merged_failure.{pdf,png}   (uses plot_failure_merged_section4.plot_fig5_6_merged)

Inputs default to PaDiM/result_analysis/padim_section4_paper/*.csv
(build_padim_paper_figures_section4.py writes these).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=str, default="/home/zju/mywork/NeurIPS2026")
    p.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Directory with mechanism_chain_summary.csv, controlled_margin_analysis.csv, "
        "failure_gate_analysis.csv, failure_conditioned_signal_analysis.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="paper_figures/padim",
        help="Output directory under repo root",
    )
    args = p.parse_args()

    repo = Path(args.repo_root).resolve()
    padim = repo / "PaDiM-Anomaly-Detection-Localization-master"
    csv_dir = Path(args.csv_dir).resolve() if args.csv_dir else (padim / "result_analysis" / "padim_section4_paper")
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chain = csv_dir / "mechanism_chain_summary.csv"
    margin = csv_dir / "controlled_margin_analysis.csv"
    gate = csv_dir / "failure_gate_analysis.csv"
    fc = csv_dir / "failure_conditioned_signal_analysis.csv"
    for fp in (chain, margin, gate, fc):
        if not fp.is_file():
            raise SystemExit(f"Missing {fp}")

    mech_mod = _load_module(repo / "PromptAD" / "utils" / "plot_mechanism_combined_section4.py")
    fail_mod = _load_module(repo / "PromptAD" / "utils" / "plot_failure_merged_section4.py")

    mech_mod.plot_mechanism_figure(
        chain,
        margin,
        gate,
        out_dir / "fig3_4_merged_mechanism.pdf",
        out_dir / "fig3_4_merged_mechanism.png",
    )
    fail_mod.plot_fig5_6_merged(
        fc,
        gate,
        out_dir / "fig5_6_merged_failure.pdf",
        out_dir / "fig5_6_merged_failure.png",
    )
    print("Wrote", out_dir / "fig3_4_merged_mechanism.pdf")
    print("Wrote", out_dir / "fig5_6_merged_failure.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
