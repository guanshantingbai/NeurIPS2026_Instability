#!/usr/bin/env python3
"""
Appendix Figure 1 style (1x3): PaDiM Protocol B across settings (dataset, category, backbone).

Protocol (locked — do not silently change):
  - **Protocol B only** (three marginal Mahalanobis views + shared fused Gaussian); see
    `padim_protocol_b_mvtec_multiclass.py` / `padim_instability_protocol_ab.py`.
  - **Canonical detector**: original fused PaDiM image-level score `s_fused` (same path as
    `canonical_fused_scores` / streaming equivalent). No Protocol A, no alternate fusion.
  - **Pair table** for ranking error: `padim_promptad_pairwise_tables.build_padim_protocol_b_pairwise_table`
    — flip from (z0,z1,z2) on marginals; `z_final` from fused `s_fused` only.
  - **Per-sample ranking error** e(x): same as PromptAD exp5 / `build_sample_ranking_error_table`
    (mean over opposite-label pairs of `1 - z_final`).
  - **Sample-level instability I(x)**: `sample_instability_Ix(s0,s1,s2,gt)` in Protocol B
    (mean over opposite-label pairs of Var(z0,z1,z2) with z from marginal scores).

**PromptAD difference (explicit):** PromptAD appendix Spearman uses **I_bin** = mean pairwise
`flip` per sample (semantic/visual/harmonic heads). PaDiM uses **I(x)** from marginal vote variance
as above. Rejection in this script sorts by **I(x)** (not by I_bin), matching the user spec
“按 sample-level instability 从高到低”.

Outputs:
  - `per_setting_metrics.csv` (+ optional `per_setting_metrics.json` rows)
  - `padim_figure1_evidence_row_abc.png` / `.pdf`
  - `summary.json`, `caption_suggested.tex`
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import zlib
from random import sample
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torchvision.models import resnet18, wide_resnet50_2

import datasets.mvtec as mvtec
import datasets.visa as visa
from padim_dataloader import enable_fast_gpu, make_feature_loader
from padim_promptad_pairwise_tables import (
    build_padim_protocol_b_pairwise_table,
    build_sample_instability_table,
    build_sample_ranking_error_table,
)
from padim_protocol_b_mvtec_multiclass import run_one_category, sample_instability_Ix

try:
    from scipy.stats import spearmanr
except ImportError as e:  # pragma: no cover
    raise SystemExit("Please install scipy: pip install scipy") from e

FIG_W, FIG_H = 14.0, 4.2
PANEL_HIST = (6.5, 4.5)  # per-subplot style reference (PromptAD appendix)
DPI = 300
EPS = 1e-12


def _maybe_cap_train_ds(train_ds, max_images: int | None, salt: int):
    if max_images is None:
        return train_ds
    n = len(train_ds)
    if n <= max_images:
        return train_ds
    rng = random.Random(salt)
    idx = sorted(rng.sample(range(n), max_images))
    return Subset(train_ds, idx)


def _row_key(dataset: str, category: str, backbone: str) -> tuple[str, str, str]:
    return (str(dataset), str(category), str(backbone))


def compute_setting_metrics(
    *,
    dataset: str,
    category: str,
    backbone: str,
    class_name: str,
    DatasetCls: type,
    root: str,
    model: torch.nn.Module,
    device: torch.device,
    outputs: list,
    idx_fused: torch.Tensor,
    marginal_groups: list,
    seed: int,
    batch_size: int | None,
    num_workers: int | None,
    max_train_images: int | None,
    cov_float32: bool,
) -> dict[str, Any]:
    train_ds = DatasetCls(root, class_name=class_name, is_train=True)
    cap_salt = (zlib.crc32(f"{dataset}\0{class_name}\0{backbone}".encode()) & 0x7FFFFFFF) ^ int(seed)
    train_ds = _maybe_cap_train_ds(train_ds, max_train_images, cap_salt)
    test_ds = DatasetCls(root, class_name=class_name, is_train=False)
    paths = [os.path.abspath(p) for p in test_ds.x]
    train_loader = make_feature_loader(train_ds, backbone, batch_size=batch_size, num_workers=num_workers)
    test_loader = make_feature_loader(test_ds, backbone, batch_size=batch_size, num_workers=num_workers)

    _compact, _full, s0, s1, s2, s_fused, gt = run_one_category(
        class_name,
        model,
        device,
        outputs,
        idx_fused,
        marginal_groups,
        train_loader,
        test_loader,
        cov_float32=cov_float32,
        return_scores=True,
    )

    if len(paths) != len(gt):
        raise RuntimeError(f"path/gt mismatch: {len(paths)} vs {len(gt)}")

    lab = np.where(gt == 1, "anomaly", "normal")
    per_df = pd.DataFrame(
        {
            "image_path": paths,
            "image_label": lab,
            "s0": s0.astype(np.float64),
            "s1": s1.astype(np.float64),
            "s2": s2.astype(np.float64),
            "s_fused": s_fused.astype(np.float64),
        }
    )
    pair_df = build_padim_protocol_b_pairwise_table(per_df)
    si = build_sample_instability_table(pair_df)
    per_h = pd.DataFrame(
        {
            "image_path": paths,
            "harmonic_score": s_fused.astype(np.float64),
        }
    )
    err_tbl = build_sample_ranking_error_table(pair_df, si, per_h)

    Ix = sample_instability_Ix(s0, s1, s2, gt).astype(np.float64)
    side = pd.DataFrame({"image_path": paths, "Ix": Ix})
    merged = err_tbl.merge(side, on="image_path", how="inner", validate="one_to_one")
    merged = merged.sort_values("image_path").reset_index(drop=True)

    ix = merged["Ix"].to_numpy(dtype=float)
    err = merged["error"].to_numpy(dtype=float)
    I_bin = merged["I_bin"].to_numpy(dtype=float)

    m = np.isfinite(ix) & np.isfinite(err)
    n_m = int(np.sum(m))
    spearman_rho = float("nan")
    spearman_reason = ""
    if n_m < 3:
        spearman_reason = "n_lt_3"
    elif float(np.nanstd(ix[m])) < EPS:
        spearman_reason = "constant_Ix"
    elif float(np.nanstd(err[m])) < EPS:
        spearman_reason = "constant_error"
    elif float(np.max(ix[m])) <= EPS:
        spearman_reason = "no_nonzero_instability"
    else:
        spearman_rho = float(spearmanr(ix[m], err[m]).correlation)
        if spearman_rho != spearman_rho:
            spearman_reason = "spearman_undefined"
            spearman_rho = float("nan")

    n = len(merged)
    k_rej = int(np.ceil(0.30 * n))
    k_rej = min(k_rej, n)
    sort_ix = merged.sort_values("Ix", ascending=False, kind="mergesort").reset_index(drop=True)
    e0 = float(sort_ix["error"].mean())
    e30 = float(sort_ix.iloc[k_rej:]["error"].mean()) if k_rej < n else float("nan")
    rel_drop = float("nan")
    if abs(e0) > EPS:
        rel_drop = (e0 - e30) / e0

    out: dict[str, Any] = {
        "dataset": dataset,
        "category": category,
        "backbone": backbone,
        "n_test": int(len(gt)),
        "n_pairs_ranking": int(len(pair_df)),
        "spearman_Ix_error": spearman_rho,
        "spearman_valid": bool(spearman_rho == spearman_rho and spearman_reason == ""),
        "spearman_exclude_reason": spearman_reason or ("ok" if spearman_rho == spearman_rho else ""),
        "mean_Ix": float(np.nanmean(Ix)),
        "max_Ix": float(np.nanmax(Ix)),
        "mean_I_bin": float(np.mean(I_bin)),
        "e0_mean_ranking_error": e0,
        "e30_mean_ranking_error_after_top30pct_Ix_rejection": e30,
        "rel_drop_30": rel_drop,
        "rel_drop_defined": bool(rel_drop == rel_drop and abs(e0) > EPS),
        "n_rejected_top30": int(k_rej),
    }

    del train_loader, test_loader, train_ds, test_ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def load_done_keys(csv_path: str) -> set[tuple[str, str, str]]:
    if not os.path.isfile(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return {_row_key(r["dataset"], r["category"], r["backbone"]) for _, r in df.iterrows()}


def _bool_mask(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    return s.astype(str).str.lower().isin(("true", "1", "yes"))


def plot_row_figure(df: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    """Native 1x3 matplotlib (style aligned with PromptAD appendix figure1 evidence)."""
    sp = df.loc[_bool_mask(df["spearman_valid"]), "spearman_Ix_error"].astype(float)
    rel = df.loc[_bool_mask(df["rel_drop_defined"]), "rel_drop_30"].astype(float)
    worsened = int((rel < 0).sum())
    improved = int((rel >= 0).sum())

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))

    # (a) Spearman
    ax = axes[0]
    if len(sp):
        nb = min(22, max(8, len(sp) // 3))
        ax.hist(sp, bins=nb, color="C0", edgecolor="0.35", linewidth=0.4)
        sm = float(sp.mean())
        ax.axvline(sm, color="0.1", linestyle="-", linewidth=1.6, zorder=5)
        ax.text(
            0.97,
            1.06,
            f"mean = {sm:.3f}\n(n = {len(sp)})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="0.1",
            zorder=6,
            clip_on=False,
        )
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=0.8, zorder=4)
    ax.set_xlabel(r"Spearman $\rho$ ($I(x)$, ranking error)")
    ax.set_ylabel("Number of settings")
    ax.set_title("(a) Spearman distribution", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) rel drop
    ax = axes[1]
    if len(rel):
        nb = min(24, max(8, len(rel) // 3))
        ax.hist(rel, bins=nb, color="C0", edgecolor="0.35", linewidth=0.4)
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Relative error drop at 30% rejection: $(e_0 - e_{30}) / e_0$")
    ax.set_ylabel("Number of settings")
    ax.set_title("(b) Relative error drop", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) counts
    ax = axes[2]
    labels = ["Rel. drop < 0\n(error increases)", r"Rel. drop $\geq$ 0" + "\n(same or lower error)"]
    counts = [worsened, improved]
    ax.bar(labels, counts, color=["C1", "C0"], edgecolor="0.2", linewidth=0.5)
    ax.set_ylabel("Number of settings (defined rel. drop)")
    ax.set_title("(c) Improved vs worsened", fontsize=11)
    ymax = max(counts) if counts else 1
    for i, c in enumerate(counts):
        ax.text(i, c + ymax * 0.02, str(c), ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    letters = ("a", "b", "c")
    for ax, letter in zip(axes, letters):
        ax.text(
            0.5,
            -0.14,
            f"({letter})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="0.05",
            clip_on=False,
        )

    plt.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.22, wspace=0.32)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_summary_and_caption(
    df: pd.DataFrame,
    summary_path: str,
    caption_path: str,
    n_total: int,
) -> None:
    n_sp = int(_bool_mask(df["spearman_valid"]).sum())
    n_sp_excl = n_total - n_sp
    rel_def = _bool_mask(df["rel_drop_defined"])
    n_rel = int(rel_def.sum())
    rel = df.loc[rel_def, "rel_drop_30"].astype(float)
    n_worse = int((rel < 0).sum())
    n_nonworse = int((rel >= 0).sum())

    summary = {
        "protocol": "PaDiM Protocol B; canonical fused PaDiM score; sample I(x)=sample_instability_Ix; "
        "rejection by I(x) descending top 30%; ranking error e(x) as exp5 build_sample_ranking_error_table",
        "n_settings_total": n_total,
        "n_spearman_valid": n_sp,
        "n_spearman_excluded": n_sp_excl,
        "n_rel_drop_defined": n_rel,
        "n_rel_drop_worse_lt_0": n_worse,
        "n_rel_drop_nonworse_ge_0": n_nonworse,
        "promptad_difference_note": "PromptAD appendix uses I_bin (mean flip) for Spearman/rejection CSV; "
        "this PaDiM figure uses I(x) variance-based mean per 3.1.2 user spec.",
    }
    if "spearman_exclude_reason" in df.columns:
        summary["spearman_exclude_reason_counts"] = (
            df.loc[~_bool_mask(df["spearman_valid"]), "spearman_exclude_reason"].value_counts().to_dict()
        )
    if n_sp:
        summary["spearman_mean_valid"] = float(
            df.loc[_bool_mask(df["spearman_valid"]), "spearman_Ix_error"].astype(float).mean()
        )
    if n_rel:
        summary["median_rel_drop_defined"] = float(rel.median())

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    cap = r"""\caption{PaDiM (Protocol B, MVTec+VisA, ResNet-18 and Wide-ResNet-50-2). 
\textbf{(a)} Distribution of Spearman $\rho$ between per-image instability $I(x)$ and per-image ranking error $e(x)$ across settings $(\mathrm{dataset},\mathrm{category},\mathrm{backbone})$; 
only settings with non-degenerate $I(x)$ and finite $\rho$ are included ($n$ in panel). 
\textbf{(b)} Histogram of relative ranking-error drop after removing the top 30\% of samples by $I(x)$, $(e_0-e_{30})/e_0$, with the vertical dashed line at no change. 
\textbf{(c)} Count of settings where 30\% rejection increases error vs.\ leaves it unchanged or decreases it. 
Overall, positive $\rho$ is common and rejection typically reduces mean error, though some settings worsen.}
"""
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(cap.strip() + "\n")


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo, "result_analysis", "appendix_figure1_evidence")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mvtec_path", type=str, default=os.path.expanduser("~/datasets/mvtec"))
    p.add_argument("--visa_path", type=str, default=os.path.expanduser("~/datasets/pro_visa"))
    p.add_argument("--out_dir", type=str, default=out_dir)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_train_images", type=int, default=128)
    p.add_argument("--no_visa", action="store_true", help="MVTec only (if VisA path missing).")
    p.add_argument("--figure_only", action="store_true", help="Only plot from existing per_setting_metrics.csv")
    p.add_argument("--limit", type=int, default=None, help="Max number of settings to compute (debug).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "per_setting_metrics.csv")
    png_path = os.path.join(args.out_dir, "padim_figure1_evidence_row_abc.png")
    pdf_path = os.path.join(args.out_dir, "padim_figure1_evidence_row_abc.pdf")
    summary_path = os.path.join(args.out_dir, "summary.json")
    caption_path = os.path.join(args.out_dir, "caption_suggested.tex")

    if args.figure_only:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        n_total = len(df)
        plot_row_figure(df, png_path, pdf_path)
        write_summary_and_caption(df, summary_path, caption_path, n_total)
        print("Wrote", png_path, pdf_path, summary_path, caption_path)
        return

    mvtec_path = os.path.expanduser(args.mvtec_path)
    visa_path = os.path.expanduser(args.visa_path)
    include_mvtec = os.path.isdir(mvtec_path)
    include_visa = (not args.no_visa) and os.path.isdir(visa_path)
    if not include_mvtec:
        raise FileNotFoundError(f"MVTec root not found: {mvtec_path}")
    if not args.no_visa and not include_visa:
        print(f"WARNING: VisA not found at {visa_path}; proceeding MVTec-only. Use --no_visa to silence.", flush=True)

    specs: list[tuple[str, type, str, list[str]]] = []
    if include_mvtec:
        specs.append(("mvtec", mvtec.MVTecDataset, mvtec_path, list(mvtec.CLASS_NAMES)))
    if include_visa:
        specs.append(("visa", visa.VISADataset, visa_path, list(visa.CLASS_NAMES)))

    arches = ("resnet18", "wide_resnet50_2")
    todo: list[tuple[str, str, str, type, str, list]] = []
    for ds_key, Cls, root, names in specs:
        for cat in names:
            for arch in arches:
                todo.append((ds_key, cat, arch, Cls, root, names))

    if args.limit is not None:
        todo = todo[: int(args.limit)]

    done = load_done_keys(csv_path) if os.path.isfile(csv_path) else set()
    rows: list[dict[str, Any]] = []
    if os.path.isfile(csv_path):
        rows = pd.read_csv(csv_path).to_dict("records")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    enable_fast_gpu()
    outputs: list = []

    def hook(module, inp, out):
        outputs.append(out)

    n_total = len(todo)

    for arch in arches:
        pending = [t for t in todo if t[2] == arch and _row_key(t[0], t[1], t[2]) not in done]
        if not pending:
            if not any(t[2] == arch for t in todo):
                print(f"[skip] backbone {arch}: not in current task list (--limit).", flush=True)
            else:
                print(f"[resume] skip backbone {arch} (all settings already in csv).", flush=True)
            continue

        if arch == "resnet18":
            model = resnet18(pretrained=True, progress=True)
            t_d, d_sub = 448, 100
        else:
            model = wide_resnet50_2(pretrained=True, progress=True)
            t_d, d_sub = 1792, 550

        model.to(device)
        model.eval()
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed_all(args.seed)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

        idx_fused = torch.tensor(sample(range(t_d), d_sub))
        sort_ord = np.argsort(idx_fused.numpy())
        marginal_groups = [t.tolist() for t in np.array_split(sort_ord, 3)]

        for ds_key, class_name, arch2, DatasetCls, root, _names in pending:
            assert arch2 == arch
            key = _row_key(ds_key, class_name, arch)
            if key in done:
                continue
            cov_f32 = ds_key == "visa" and arch == "wide_resnet50_2"
            print(f"  [{ds_key}/{arch}] {class_name} ...", flush=True)
            rec = compute_setting_metrics(
                dataset=ds_key,
                category=class_name,
                backbone=arch,
                class_name=class_name,
                DatasetCls=DatasetCls,
                root=root,
                model=model,
                device=device,
                outputs=outputs,
                idx_fused=idx_fused,
                marginal_groups=marginal_groups,
                seed=args.seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_train_images=args.max_train_images,
                cov_float32=cov_f32,
            )
            rows.append(rec)
            done.add(key)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            json_path = os.path.join(args.out_dir, "per_setting_metrics.json")
            pd.DataFrame(rows).to_json(json_path, orient="records", indent=2)
            print(f"    spearman={rec['spearman_Ix_error']} rel_drop={rec['rel_drop_30']}", flush=True)

        del model
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise SystemExit("No rows computed.")
    df.to_csv(csv_path, index=False)
    pd.DataFrame(rows).to_json(os.path.join(args.out_dir, "per_setting_metrics.json"), orient="records", indent=2)

    plot_row_figure(df, png_path, pdf_path)
    write_summary_and_caption(df, summary_path, caption_path, n_total=len(df))

    print("Wrote", csv_path)
    print("Wrote", png_path, pdf_path)
    print("Wrote", summary_path, caption_path)


if __name__ == "__main__":
    main()
