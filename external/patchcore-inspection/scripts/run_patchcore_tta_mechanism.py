#!/usr/bin/env python3
"""
PatchCore + TTA mechanism pipeline (Fig3-style controlled margin + Fig4-style mechanism chain).

Steps: (1) TTA scores -> patchcore_tta_scores.csv (checkpoint: one append per class; use --resume)
       (2) pairwise -> patchcore_pairwise_analysis.csv
       (3-4) controlled_margin + mechanism_chain CSVs
       (5) figures PNG+PDF

Performance: batched GPU forwards + GPU flip/rotate; CPU only does JPEG decode + resize/crop.
Use --inference-batch-size 48 (3090) or higher until OOM. --faiss-on-gpu if faiss-gpu is installed.

Resume example after interrupt:
  python ... --step scores --resume --out-dir ... --data-root ... --dataset visa --models-run ...
Then when all classes present:
  python ... --step analyze --out-dir ... --scores-csv .../patchcore_tta_scores.csv

Run from anywhere; PYTHONPATH is set via patchcore-inspection/src.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torchvision.transforms.functional as TVF
from tqdm import tqdm

_PATCHCORE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PATCHCORE_ROOT.parent
if str(_PATCHCORE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PATCHCORE_ROOT / "src"))

import patchcore.common  # noqa: E402
import patchcore.patchcore  # noqa: E402

from patchcore.datasets.mvtec import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    MVTecDataset,
    DatasetSplit as MVTecSplit,
    _CLASSNAMES as MVTEC_CLASSES,
)
from patchcore.datasets.visa import (  # noqa: E402
    VisADataset,
    DatasetSplit as VisaSplit,
    _CLASSNAMES as VISA_CLASSES,
)

LOGGER = logging.getLogger("patchcore_tta")

def _default_out_dir() -> Path:
    p = _REPO_ROOT / "result_analysis" / "patchcore_tta"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pil_to_tensor01(pil: PIL.Image.Image, resize: int, crop: int) -> torch.Tensor:
    """CPU: Resize -> CenterCrop -> float tensor [0,1], shape (3,H,W)."""
    x = TVF.resize(pil, (resize, resize), interpolation=TVF.InterpolationMode.BILINEAR)
    x = TVF.center_crop(x, (crop, crop))
    return TVF.to_tensor(x)


def _imagenet_norm_batch(x01: torch.Tensor) -> torch.Tensor:
    """x01 on device, values [0,1] -> ImageNet normalized."""
    mean = torch.tensor(IMAGENET_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def _rotate_batch_normed(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate each (C,H,W) on device; x is ImageNet-normalized batch (B,C,H,W)."""
    out = []
    for i in range(x.shape[0]):
        out.append(
            TVF.rotate(
                x[i],
                angle=angle_deg,
                interpolation=TVF.InterpolationMode.BILINEAR,
                expand=False,
            )
        )
    return torch.stack(out, dim=0)


def _completed_classes_from_scores_csv(out_csv: Path) -> Set[str]:
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return set()
    return set(pd.read_csv(out_csv, usecols=["classname"])["classname"].astype(str))


def _append_scores_csv(out_csv: Path, rows: List[Dict], *, header: bool) -> None:
    pd.DataFrame(rows).to_csv(out_csv, mode="a", index=False, header=header)


def _infer_resize(imagesize: int) -> int:
    """Match common PatchCore README presets."""
    if imagesize >= 320:
        return 366
    return 256


def _model_subdir(dataset: str, classname: str) -> str:
    return ("mvtec_" if dataset == "mvtec" else "visa_") + classname


def _load_input_shape(models_run: Path, classname: str, dataset: str) -> Tuple[int, int, int]:
    """Return (C, H, W) from saved patchcore_params."""
    load_path = models_run / _model_subdir(dataset, classname)
    pkl = load_path / "patchcore_params.pkl"
    with open(pkl, "rb") as f:
        params = pickle.load(f)
    shape = params["input_shape"]
    return int(shape[0]), int(shape[1]), int(shape[2])


def _score_batch(model: patchcore.patchcore.PatchCore, images: torch.Tensor) -> np.ndarray:
    """Image-level scores for a batch [B,3,H,W] on device."""
    scores, _ = model._predict(images)
    return np.asarray(scores, dtype=np.float64)


def collect_scores(
    *,
    data_root: Path,
    dataset: str,
    class_names: Sequence[str],
    models_run: Path,
    out_dir: Path,
    device: torch.device,
    faiss_on_gpu: bool,
    faiss_workers: int,
    resume: bool,
    inference_batch_size: int,
    max_classes: Optional[int],
) -> Path:
    """
    Batched GPU inference + GPU TTA (flip / rotate) after Resize+Crop on CPU.
    Checkpoint: append ``patchcore_tta_scores.csv`` after each class (resume-safe).
    """
    out_csv = out_dir / "patchcore_tta_scores.csv"
    completed: Set[str] = _completed_classes_from_scores_csv(out_csv) if resume else set()
    if resume and completed:
        LOGGER.info("Resume: skipping %d completed class(es): %s", len(completed), sorted(completed))

    pending = [c for c in class_names if c not in completed]
    if max_classes is not None:
        pending = pending[: max(0, int(max_classes))]

    write_header = not out_csv.exists() or out_csv.stat().st_size == 0

    for classname in pending:
        c, h, w = _load_input_shape(models_run, classname, dataset)
        _ = c
        resize = _infer_resize(h)

        load_path = models_run / _model_subdir(dataset, classname)
        if not load_path.is_dir():
            raise FileNotFoundError(f"Missing model dir: {load_path}")

        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_workers)
        pc = patchcore.patchcore.PatchCore(device)
        try:
            pc.load_from_path(str(load_path), device, nn_method)
        except Exception as e:  # noqa: BLE001
            if faiss_on_gpu:
                LOGGER.warning("FAISS GPU load failed (%s); retrying with faiss_on_gpu=False", e)
                nn_method = patchcore.common.FaissNN(False, faiss_workers)
                pc = patchcore.patchcore.PatchCore(device)
                pc.load_from_path(str(load_path), device, nn_method)
            else:
                raise

        if dataset == "mvtec":
            ds = MVTecDataset(
                str(data_root),
                classname=classname,
                resize=resize,
                imagesize=h,
                split=MVTecSplit.TEST,
            )
        else:
            ds = VisADataset(
                str(data_root),
                classname=classname,
                resize=resize,
                imagesize=h,
                split=VisaSplit.TEST,
            )

        n = len(ds)
        LOGGER.info(
            "Class %s: %d test images (batched TTA, batch=%d, device=%s)",
            classname,
            n,
            inference_batch_size,
            device,
        )

        class_rows: List[Dict] = []
        idxs = list(range(n))
        with torch.inference_mode():
            for start in tqdm(
                range(0, n, inference_batch_size),
                desc=f"tta[{classname}]",
                unit="batch",
                leave=False,
            ):
                batch_idx = idxs[start : start + inference_batch_size]
                meta: List[Tuple[str, int]] = []
                tensors01: List[torch.Tensor] = []
                for i in batch_idx:
                    cn_i, anomaly, image_path, _mp = ds.data_to_iterate[i]
                    assert cn_i == classname
                    label = 1 if anomaly != "good" else 0
                    pil = PIL.Image.open(image_path).convert("RGB")
                    tensors01.append(_pil_to_tensor01(pil, resize, h))
                    meta.append((str(image_path), label))

                x01 = torch.stack(tensors01, dim=0).to(device, non_blocking=True)
                x = _imagenet_norm_batch(x01)

                s_id = _score_batch(pc, x)
                s_flip = _score_batch(pc, torch.flip(x, dims=(-1,)))
                s_rp = _score_batch(pc, _rotate_batch_normed(x, 5.0))
                s_rn = _score_batch(pc, _rotate_batch_normed(x, -5.0))

                for j in range(len(batch_idx)):
                    path_j, lab_j = meta[j]
                    class_rows.append(
                        {
                            "classname": classname,
                            "image_path": path_j,
                            "label": lab_j,
                            "score_id": float(s_id[j]),
                            "score_flip": float(s_flip[j]),
                            "score_rot_pos": float(s_rp[j]),
                            "score_rot_neg": float(s_rn[j]),
                        }
                    )

        _append_scores_csv(out_csv, class_rows, header=write_header)
        write_header = False
        LOGGER.info("Checkpoint: appended %d rows for %s -> %s", len(class_rows), classname, out_csv)

        del pc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if out_csv.exists():
        LOGGER.info("Scores CSV %s total rows=%d", out_csv, len(pd.read_csv(out_csv)))
    return out_csv


def build_pairwise(scores_csv: Path, out_dir: Path) -> Path:
    df = pd.read_csv(scores_csv)
    need = {"classname", "image_path", "label", "score_id", "score_flip", "score_rot_pos", "score_rot_neg"}
    if not need.issubset(df.columns):
        raise ValueError(f"scores CSV missing columns: {need - set(df.columns)}")

    out_path = out_dir / "patchcore_pairwise_analysis.csv"
    pair_rows: List[Dict] = []

    for classname, g in df.groupby("classname"):
        pos = g[g["label"] == 1].reset_index(drop=True)
        neg = g[g["label"] == 0].reset_index(drop=True)
        if len(pos) == 0 or len(neg) == 0:
            LOGGER.warning("Skip class %s: pos=%d neg=%d", classname, len(pos), len(neg))
            continue

        sp = pos["score_id"].to_numpy(dtype=np.float64)
        sf_p = pos["score_flip"].to_numpy(dtype=np.float64)
        srpp = pos["score_rot_pos"].to_numpy(dtype=np.float64)
        srnp = pos["score_rot_neg"].to_numpy(dtype=np.float64)

        sn = neg["score_id"].to_numpy(dtype=np.float64)
        sf_n = neg["score_flip"].to_numpy(dtype=np.float64)
        srpn = neg["score_rot_pos"].to_numpy(dtype=np.float64)
        srmn = neg["score_rot_neg"].to_numpy(dtype=np.float64)

        z_id = (sp[:, None] > sn[None, :]).astype(np.float64)
        z_flip = (sf_p[:, None] > sf_n[None, :]).astype(np.float64)
        z_rp = (srpp[:, None] > srpn[None, :]).astype(np.float64)
        z_rn = (srnp[:, None] > srmn[None, :]).astype(np.float64)

        margin = sp[:, None] - sn[None, :]
        err = (sp[:, None] <= sn[None, :]).astype(np.float64)
        zs = np.stack([z_id, z_flip, z_rp, z_rn], axis=-1)
        inst = np.var(zs, axis=-1, ddof=0)

        pp = pos["image_path"].to_numpy()
        nn = neg["image_path"].to_numpy()
        ni, nj = inst.shape
        for i in range(ni):
            for j in range(nj):
                zrow = zs[i, j]
                pair_rows.append(
                    {
                        "classname": classname,
                        "path_pos": pp[i],
                        "path_neg": nn[j],
                        "margin": float(margin[i, j]),
                        "abs_margin": float(abs(margin[i, j])),
                        "error": float(err[i, j]),
                        "z_id": float(zrow[0]),
                        "z_flip": float(zrow[1]),
                        "z_rot_pos": float(zrow[2]),
                        "z_rot_neg": float(zrow[3]),
                        "instability": float(inst[i, j]),
                    }
                )

    out = pd.DataFrame(pair_rows)
    out.to_csv(out_path, index=False)
    LOGGER.info("Wrote %s (%d pairs)", out_path, len(out))
    return out_path


def _assign_margin_buckets(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Per-class tertiles on abs_margin (same cut as mechanism_driven_analysis)."""
    out = pair_df.copy()
    out["margin_bucket"] = ""
    for classname, g in out.groupby("classname"):
        idx = g.index
        abs_m = g["abs_margin"].to_numpy(dtype=np.float64)
        if abs_m.size < 3:
            out.loc[idx, "margin_bucket"] = "low"
            continue
        q1, q2 = np.quantile(abs_m, [1.0 / 3.0, 2.0 / 3.0])
        mb = np.where(abs_m > q2, "high", np.where(abs_m > q1, "mid", "low"))
        out.loc[idx, "margin_bucket"] = mb
    return out


def controlled_margin_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Within each margin_bucket (per-class tertile labels), pool all pairs and split by I."""
    df = _assign_margin_buckets(pair_df)
    order = ["low", "mid", "high"]
    rows = []
    for b in order:
        sub = df[df["margin_bucket"] == b]
        if sub.empty:
            rows.append({"margin_bucket": b, "error_low_I": np.nan, "error_high_I": np.nan})
            continue
        inst = sub["instability"].to_numpy(dtype=np.float64)
        err = sub["error"].to_numpy(dtype=np.float64)
        m0 = inst == 0.0
        m1 = inst > 0.0
        e0 = float(np.mean(err[m0])) if np.any(m0) else float("nan")
        e1 = float(np.mean(err[m1])) if np.any(m1) else float("nan")
        rows.append({"margin_bucket": b, "error_low_I": e0, "error_high_I": e1})
    return pd.DataFrame(rows)


def mechanism_chain_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    df = _assign_margin_buckets(pair_df)
    order = ["low", "mid", "high"]
    rows = []
    for b in order:
        sub = df[df["margin_bucket"] == b]
        if sub.empty:
            rows.append(
                {
                    "margin_bucket": b,
                    "mean_instability": float("nan"),
                    "error_rate": float("nan"),
                }
            )
        else:
            rows.append(
                {
                    "margin_bucket": b,
                    "mean_instability": float(sub["instability"].mean()),
                    "error_rate": float(sub["error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _savefig_both(fig: plt.Figure, out_no_ext: Path) -> None:
    plt.rcParams["pdf.fonttype"] = 42
    fig.savefig(str(out_no_ext) + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(out_no_ext) + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_controlled_margin(cm: pd.DataFrame, out_no_ext: Path) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})

    def _bar_val(v: float) -> float:
        return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    order = ["low", "mid", "high"]
    x = np.arange(len(order))
    w = 0.35
    e0 = [float(cm.loc[cm["margin_bucket"] == b, "error_low_I"].iloc[0]) if len(cm[cm["margin_bucket"] == b]) else np.nan for b in order]
    e1 = [float(cm.loc[cm["margin_bucket"] == b, "error_high_I"].iloc[0]) if len(cm[cm["margin_bucket"] == b]) else np.nan for b in order]
    e0 = [_bar_val(v) for v in e0]
    e1 = [_bar_val(v) for v in e1]
    ax.bar(x - w / 2, e0, width=w, label="stable (I=0)")
    ax.bar(x + w / 2, e1, width=w, label="unstable (I>0)")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("margin_bucket (|margin| tertile within class)")
    ax.set_ylabel("pairwise error rate")
    ax.legend()
    ax.set_title("PatchCore + TTA: controlled margin")
    ax.grid(axis="y", alpha=0.25)
    _savefig_both(fig, out_no_ext)


def plot_mechanism_chain(mc: pd.DataFrame, out_no_ext: Path) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    order = ["low", "mid", "high"]
    x = np.arange(len(order))

    def col(b: str, name: str) -> float:
        sub = mc[mc["margin_bucket"] == b]
        if sub.empty:
            return float("nan")
        return float(sub[name].iloc[0])

    y1 = [col(b, "mean_instability") for b in order]
    y2 = [col(b, "error_rate") for b in order]
    axes[0].bar(x, y1, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_xlabel("margin_bucket")
    axes[0].set_ylabel("mean instability")
    axes[0].set_title("Mean instability by margin bucket")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, y2, color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order)
    axes[1].set_xlabel("margin_bucket")
    axes[1].set_ylabel("mean pairwise error")
    axes[1].set_title("Error rate by margin bucket")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    _savefig_both(fig, out_no_ext)


def run_analyze(scores_csv: Path, out_dir: Path) -> None:
    pair_path = build_pairwise(scores_csv, out_dir)
    pair_df = pd.read_csv(pair_path)

    cm = controlled_margin_table(pair_df)
    cm_path = out_dir / "controlled_margin_analysis.csv"
    cm.to_csv(cm_path, index=False, na_rep="nan")

    mc = mechanism_chain_table(pair_df)
    mc_path = out_dir / "mechanism_chain_summary.csv"
    mc.to_csv(mc_path, index=False, na_rep="nan")

    plot_controlled_margin(cm, out_dir / "fig_patchcore_controlled_margin")
    plot_mechanism_chain(mc, out_dir / "fig_patchcore_mechanism_chain")

    print("\n=== PatchCore TTA mechanism outputs ===")
    print("pairwise:", pair_path)
    print("controlled_margin:", cm_path)
    print("mechanism_chain:", mc_path)
    print("figures:", out_dir / "fig_patchcore_controlled_margin.png", out_dir / "fig_patchcore_mechanism_chain.png")
    print("\n--- controlled_margin_analysis ---\n", cm.to_string(index=False))
    print("\n--- mechanism_chain_summary ---\n", mc.to_string(index=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=("all", "scores", "analyze"), default="all")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <repo>/result_analysis/patchcore_tta/")
    p.add_argument("--data-root", type=Path, required=True, help="MVTec parent (contains bottle/...) or VisA parent")
    p.add_argument("--dataset", choices=("mvtec", "visa"), default="mvtec")
    p.add_argument(
        "--models-run",
        type=Path,
        required=True,
        help="Directory containing mvtec_* / visa_* subdirs (e.g. .../IM224_.../models/)",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--classes", type=str, default="", help="Comma-separated; empty = all benchmark classes")
    p.add_argument(
        "--faiss-on-gpu",
        action="store_true",
        help="Use FAISS on GPU if available (falls back to CPU on failure).",
    )
    p.add_argument("--faiss-workers", type=int, default=8)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip classes already present in patchcore_tta_scores.csv; append new rows.",
    )
    p.add_argument(
        "--inference-batch-size",
        type=int,
        default=24,
        help="Batch size for GPU PatchCore forward + TTA (increase on 3090 until OOM).",
    )
    p.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Process at most this many not-yet-completed classes then exit (checkpoint-friendly).",
    )
    p.add_argument("--scores-csv", type=Path, default=None, help="For step=analyze: read this instead of recomputing")
    args = p.parse_args()

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "mvtec":
        default_classes = list(MVTEC_CLASSES)
    else:
        default_classes = list(VISA_CLASSES)

    if args.classes.strip():
        class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        class_names = default_classes

    import patchcore.utils as patchcore_utils

    device = patchcore_utils.set_torch_device([args.gpu])

    if args.step in ("all", "scores"):
        out_scores = out_dir / "patchcore_tta_scores.csv"
        if out_scores.exists() and not args.resume:
            raise SystemExit(
                f"Scores file already exists:\n  {out_scores}\n"
                "Use --resume to append new classes, or delete the file to start over."
            )

        scores_csv = collect_scores(
            data_root=args.data_root.resolve(),
            dataset=args.dataset,
            class_names=class_names,
            models_run=args.models_run.resolve(),
            out_dir=out_dir,
            device=device,
            faiss_on_gpu=args.faiss_on_gpu,
            faiss_workers=args.faiss_workers,
            resume=args.resume,
            inference_batch_size=max(1, int(args.inference_batch_size)),
            max_classes=args.max_classes,
        )
    else:
        scores_csv = args.scores_csv
        if scores_csv is None:
            scores_csv = out_dir / "patchcore_tta_scores.csv"
    scores_csv = scores_csv.resolve()

    if args.step in ("all", "analyze"):
        if args.step == "all" and args.max_classes is not None:
            LOGGER.warning(
                "Skipping analyze (--max-classes set). When all classes are scored, run:\n"
                "  python ... --step analyze --out-dir %s --data-root ... --dataset ... --models-run ... --scores-csv %s",
                out_dir,
                scores_csv,
            )
        else:
            run_analyze(scores_csv, out_dir)


if __name__ == "__main__":
    main()
