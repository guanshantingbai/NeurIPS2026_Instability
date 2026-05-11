#!/usr/bin/env python3
"""
Collect / generate Section-4 paper figures for PromptAD instability experiments.

Outputs: <repo>/paper_figures/fig*.png + fig*.pdf (>=300 dpi raster) and paper_figures.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _paper_rc() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_png_pdf(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _find_first(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def _collect_pngs(root: Path) -> List[Path]:
    out: List[Path] = []
    if not root.is_dir():
        return out
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".png"):
                out.append(Path(dirpath) / fn)
    return sorted(set(out))


def plot_controlled_margin(csv_path: Path, out_base: Path) -> Tuple[bool, str]:
    _paper_rc()
    if not csv_path.is_file():
        return False, f"missing {csv_path}"
    df = pd.read_csv(csv_path)
    order = ["low", "mid", "high"]
    rows = []
    for b in order:
        sub = df[df["margin_bucket"].astype(str).str.lower() == b]
        if len(sub) == 0:
            continue
        rows.append(sub.iloc[0])
    if not rows:
        return False, "no margin_bucket rows"
    g = pd.DataFrame(rows)
    x = np.arange(len(g))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - w / 2, g["error_low_I"], width=w, label=r"$I_{\mathrm{flip}}=0$")
    ax.bar(x + w / 2, g["error_high_I"], width=w, label=r"$I_{\mathrm{flip}}=1$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b).capitalize() for b in g["margin_bucket"].tolist()])
    ax.set_xlabel(r"$|\mathrm{margin}|$ tertile (pairwise)")
    ax.set_ylabel("Mean pairwise ranking error")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Controlled margin: error vs instability")
    _save_png_pdf(fig, out_base)
    return True, str(csv_path)


def plot_failure_gate(csv_path: Path, out_base: Path) -> Tuple[bool, str]:
    _paper_rc()
    if not csv_path.is_file():
        return False, f"missing {csv_path}"
    df = pd.read_csv(csv_path)
    # Weighted mean failure_rate by regime (all epsilon, coverage)
    rows = []
    for reg in ["low_I", "mid_I", "high_I"]:
        sub = df[df["regime"].astype(str) == reg]
        if sub.empty:
            rows.append({"regime": reg, "failure_rate": np.nan, "n_settings": 0})
            continue
        w = sub["n_settings"].to_numpy(dtype=np.float64)
        fr = sub["failure_rate"].to_numpy(dtype=np.float64)
        tot = float(np.sum(w))
        val = float(np.sum(fr * w) / tot) if tot > 0 else float("nan")
        rows.append({"regime": reg, "failure_rate": val, "n_settings": int(np.sum(sub["n_settings"]))})
    g = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(3)
    ax.bar(x, g["failure_rate"], color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_xticks(x)
    ax.set_xticklabels(["low_I", "mid_I", "high_I"])
    ax.set_xlabel(r"Instability regime ($I_{\mathrm{setting}}$ tertiles)")
    ax.set_ylabel("Failure rate (weighted)")
    ax.set_title("Failure gate by instability regime")
    ax.grid(axis="y", alpha=0.25)
    _save_png_pdf(fig, out_base)
    return True, str(csv_path)


def plot_failure_signal(csv_path: Path, out_base: Path) -> Tuple[bool, str]:
    _paper_rc()
    if not csv_path.is_file():
        return False, f"missing {csv_path}"
    df = pd.read_csv(csv_path)
    df = df[df["signal"].isin(["instability", "margin", "gap", "score_var"])]
    df = df[df["coverage"].isin([0.5, 0.7, 0.8])]
    if df.empty:
        return False, "filtered failure_conditioned empty"
    g = df.groupby("signal", as_index=False)[["mean_failure", "mean_non_failure"]].mean()
    order = ["instability", "gap", "score_var", "margin"]
    g["signal"] = pd.Categorical(g["signal"], categories=order, ordered=True)
    g = g.sort_values("signal")
    x = np.arange(len(g))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - w / 2, g["mean_failure"], width=w, label="failure")
    ax.bar(x + w / 2, g["mean_non_failure"], width=w, label="non-failure")
    ax.set_xticks(x)
    ax.set_xticklabels(g["signal"].astype(str).tolist())
    ax.set_xlabel("Signal (AUROC seed, setting-level)")
    ax.set_ylabel("Mean signal value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Failure-conditioned separation (cov 0.5/0.7/0.8)")
    _save_png_pdf(fig, out_base)
    return True, str(csv_path)


def _failure_rate_by_instability_regime(failure_gate_csv: Path) -> Tuple[np.ndarray, List[str]]:
    fg = pd.read_csv(failure_gate_csv)
    xs: List[float] = []
    labels_display = ["Low", "Mid", "High"]
    for reg in ["low_I", "mid_I", "high_I"]:
        sub = fg[fg["regime"].astype(str) == reg]
        if sub.empty:
            xs.append(float("nan"))
            continue
        w = sub["n_settings"].to_numpy(dtype=np.float64)
        fr = sub["failure_rate"].to_numpy(dtype=np.float64)
        tot = float(np.sum(w))
        xs.append(float(np.sum(fr * w) / tot) if tot > 0 else float("nan"))
    return np.asarray(xs, dtype=np.float64), labels_display


def plot_mechanism_chain(
    csv_path: Path, out_base: Path, failure_gate_csv: Optional[Path] = None
) -> Tuple[bool, str, str]:
    _paper_rc()
    if not csv_path.is_file():
        _placeholder(out_base, "Fig.4 mechanism chain\n(data: mechanism_chain_summary.csv missing)")
        return True, str(csv_path), "placeholder"
    df = pd.read_csv(csv_path)
    if df.empty:
        _placeholder(out_base, "Fig.4 mechanism chain\n(empty CSV)")
        return True, str(csv_path), "placeholder"
    order = ["low", "mid", "high"]
    df["margin_bucket"] = df["margin_bucket"].astype(str).str.lower()
    df = df.set_index("margin_bucket").reindex(order).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=False)
    axes[0].plot(df["margin_bucket"], df["mean_instability"], marker="o", color="#C44E52")
    axes[0].set_ylabel(r"Mean $I$")
    axes[0].set_title("Instability")
    axes[0].set_xlabel(r"$|\mathrm{margin}|$ bucket")
    axes[1].plot(df["margin_bucket"], df["error_rate"], marker="o", color="#4C72B0")
    axes[1].set_ylabel("Pairwise error rate")
    axes[1].set_title("Ranking error")
    axes[1].set_xlabel(r"$|\mathrm{margin}|$ bucket")
    if failure_gate_csv is not None and failure_gate_csv.is_file():
        fr_inst, inst_labels = _failure_rate_by_instability_regime(failure_gate_csv)
        xpos = np.arange(3)
        axes[2].plot(xpos, fr_inst, marker="o", color="#55A868")
        axes[2].set_xticks(xpos)
        axes[2].set_xticklabels(inst_labels)
        axes[2].set_xlabel("Instability bucket")
    else:
        axes[2].plot(df["margin_bucket"], df["failure_rate"], marker="o", color="#55A868")
        axes[2].set_xlabel(r"$|\mathrm{margin}|$ bucket")
    axes[2].set_ylabel("Setting failure rate")
    axes[2].set_title("AUROC-selection failure")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Mechanism chain sketch (margin → instability → error → failure)", y=1.02)
    fig.tight_layout()
    _save_png_pdf(fig, out_base)
    return True, str(csv_path), "generated"


def _placeholder(out_base: Path, text: str) -> None:
    _paper_rc()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, wrap=True)
    _save_png_pdf(fig, out_base)


def run(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve()
    ra = repo / "PromptAD" / "result_analysis"
    pilot = ra / "pilot_instability_selection"
    fail_d = pilot / "failure_driven"
    mech = ra / "mechanism_analysis"
    strengthen = ra / "promptad_strengthening"
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "paper_figures_dir": str(out_dir),
        "figures": [],
        "all_collected_png": [],
        "missing_or_failed": [],
    }

    all_png = []
    for d in [ra, pilot, fail_d, strengthen, mech]:
        all_png.extend(_collect_pngs(d))
    manifest["all_collected_png"] = [str(p) for p in sorted(set(all_png))]

    # --- fig2 same AUROC case ---
    src2 = _find_first(
        [
            pilot / "selection_case_scatter.png",
            ra / "section42_same_auroc" / "selection_case_scatter.png",
        ]
    )
    if src2:
        shutil.copy2(src2, out_dir / "fig2_same_auroc.png")
        pdf2 = src2.with_suffix(".pdf")
        if pdf2.is_file():
            shutil.copy2(pdf2, out_dir / "fig2_same_auroc.pdf")
        else:
            _paper_rc()
            im = plt.imread(str(src2))
            h, w = im.shape[0], im.shape[1]
            fig, ax = plt.subplots(figsize=(w / 300, h / 300), dpi=300)
            ax.imshow(im)
            ax.axis("off")
            fig.savefig(out_dir / "fig2_same_auroc.pdf", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        manifest["figures"].append(
            {"name": "fig2_same_auroc", "source_png": str(src2), "csv": "pilot selection_summary + per_seed (scatter script)", "status": "copied"}
        )
    else:
        manifest["missing_or_failed"].append("fig2_same_auroc: selection_case_scatter.png not found")

    # --- fig7 delta risk ---
    src7 = _find_first([pilot / "aggregate_delta_risk.png"])
    if src7:
        shutil.copy2(src7, out_dir / "fig7_delta_risk.png")
        pdf7 = src7.with_suffix(".pdf")
        if pdf7.is_file():
            shutil.copy2(pdf7, out_dir / "fig7_delta_risk.pdf")
        else:
            _paper_rc()
            im = plt.imread(str(src7))
            h, w = im.shape[0], im.shape[1]
            fig, ax = plt.subplots(figsize=(w / 300, h / 300), dpi=300)
            ax.imshow(im)
            ax.axis("off")
            fig.savefig(out_dir / "fig7_delta_risk.pdf", dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        manifest["figures"].append(
            {"name": "fig7_delta_risk", "source_png": str(src7), "csv": "selection_baseline_summary / aggregate script inputs", "status": "copied"}
        )
    else:
        manifest["missing_or_failed"].append("fig7_delta_risk: aggregate_delta_risk.png not found")

    # --- fig3 controlled margin (generate) ---
    cm_csv = _find_first(
        [
            mech / "controlled_margin_analysis.csv",
            strengthen / "controlled_margin_analysis.csv",
        ]
    )
    ok, src = (
        plot_controlled_margin(cm_csv, out_dir / "fig3_controlled_margin")
        if cm_csv is not None
        else (False, "controlled_margin_analysis.csv missing")
    )
    if ok:
        manifest["figures"].append({"name": "fig3_controlled_margin", "source_png": str(out_dir / "fig3_controlled_margin.png"), "csv": src, "status": "generated"})
    else:
        manifest["missing_or_failed"].append(f"fig3_controlled_margin: {src}")

    # --- fig6 failure regime ---
    fg_csv = mech / "failure_gate_analysis.csv"
    ok, src = plot_failure_gate(fg_csv, out_dir / "fig6_failure_regime") if fg_csv.is_file() else (False, str(fg_csv))
    if ok:
        manifest["figures"].append({"name": "fig6_failure_regime", "source_png": str(out_dir / "fig6_failure_regime.png"), "csv": src, "status": "generated"})
    else:
        manifest["missing_or_failed"].append(f"fig6_failure_regime: {src}")

    # --- fig5 failure signal ---
    fs_csv = strengthen / "failure_conditioned_signal_analysis.csv"
    ok, src = plot_failure_signal(fs_csv, out_dir / "fig5_failure_signal") if fs_csv.is_file() else (False, str(fs_csv))
    if ok:
        manifest["figures"].append({"name": "fig5_failure_signal", "source_png": str(out_dir / "fig5_failure_signal.png"), "csv": src, "status": "generated"})
    else:
        manifest["missing_or_failed"].append(f"fig5_failure_signal: {src}")

    # --- fig4 mechanism chain ---
    mc_csv = mech / "mechanism_chain_summary.csv"
    ok4, src4, st4 = plot_mechanism_chain(
        mc_csv, out_dir / "fig4_mechanism_chain", failure_gate_csv=mech / "failure_gate_analysis.csv"
    )
    if not ok4:
        manifest["missing_or_failed"].append("fig4_mechanism_chain: plot failed")
    manifest["figures"].append(
        {
            "name": "fig4_mechanism_chain",
            "source_png": str(out_dir / "fig4_mechanism_chain.png"),
            "csv": src4,
            "status": st4,
        }
    )

    # Copy key reference PNGs into paper_figures/supplementary/ for convenience
    sup = out_dir / "supplementary"
    sup.mkdir(exist_ok=True)
    for name in [
        "failure_instability_scatter.png",
        "conditional_gain.png",
        "oracle_gap_reduction.png",
    ]:
        p = _find_first([fail_d / name, strengthen / name, pilot / name])
        if p:
            shutil.copy2(p, sup / p.name)

    man_path = out_dir / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    zip_path = repo / "paper_figures.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(out_dir.rglob("*")):
            if fp.is_file():
                zf.write(fp, arcname=str(fp.relative_to(repo)))

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"ZIP: {zip_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=str, default="/home/zju/mywork/NeurIPS2026")
    p.add_argument("--out-dir", type=str, default="paper_figures")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
