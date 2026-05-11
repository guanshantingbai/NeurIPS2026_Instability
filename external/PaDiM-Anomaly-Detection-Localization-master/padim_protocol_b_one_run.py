#!/usr/bin/env python3
"""
Run Protocol B for a single (dataset, class, arch, seed) and write artifacts for seed-killer pipeline.

Outputs under ``save_dir`` (basename should be the string form of ``seed`` so multi-seed runs never share ``temp_*/train_*.pkl`` from ``main.py``):

  - per_sample.csv  (image_path, image_label, s0..s_fused, semantic/visual/harmonic scores, sample_error, proxy_u_padim_marg)
  - summary.json               (fused_auroc, flip_rate_mean, mean_sample_instability, n_test, idx_fused_first_20, ...)
  - protocol_b_meta.json       (marginal_groups lens, arch, class_name, seed)

No pickle reuse from main.py; train Gaussian is fit fresh each run.

Optional ``--max-train-images`` subsamples the training set (deterministic per seed) to reduce RAM
for VisA×WR50; omit for the full training set (high memory).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset
from torchvision.models import resnet18, wide_resnet50_2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import datasets.mvtec as mvtec
import datasets.visa as visa
from padim_dataloader import enable_fast_gpu, make_feature_loader
from padim_promptad_pairwise_tables import (
    build_padim_protocol_b_pairwise_table,
    build_sample_instability_table,
    build_sample_ranking_error_table,
)
from padim_protocol_b_mvtec_multiclass import run_one_category


def _proxy_u_padim_marg(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Normalized marginal score spread (Protocol B interpretable u(x) proxy)."""
    s0 = np.asarray(s0, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    num = np.abs(s0 - s1) + np.abs(s1 - s2) + np.abs(s0 - s2)
    den = np.abs(s0) + np.abs(s1) + np.abs(s2) + 1e-6
    return num / den


def _labels_str(gt: np.ndarray) -> List[str]:
    return ["anomaly" if int(g) == 1 else "normal" for g in gt]


def _maybe_cap_train_ds(
    train_ds: torch.utils.data.Dataset, max_images: Optional[int], salt: int
) -> torch.utils.data.Dataset:
    """Subset training images for Gaussian RAM (same logic as ``padim_build_panel_a``)."""
    if max_images is None:
        return train_ds
    n = len(train_ds)
    if n <= max_images:
        return train_ds
    rng = random.Random(salt)
    idx = sorted(rng.sample(range(n), max_images))
    return Subset(train_ds, idx)


def run_protocol_b_one_setting(
    dataset: str,
    class_name: str,
    arch: str,
    seed: int,
    save_dir: str,
    data_path: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    cov_float32: bool = False,
    max_train_images: Optional[int] = None,
) -> Dict[str, Any]:
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if os.path.basename(save_dir.rstrip(os.sep)) != str(int(seed)):
        raise ValueError(
            f"save_dir basename must be '{seed}' (got {os.path.basename(save_dir)!r}) "
            "so each seed uses a distinct directory and never shares main.py temp/train_*.pkl caches."
        )

    if dataset == "mvtec":
        if class_name not in mvtec.CLASS_NAMES:
            raise ValueError(f"class_name must be in {mvtec.CLASS_NAMES}")
        DatasetCls = mvtec.MVTecDataset
    elif dataset == "visa":
        if class_name not in visa.CLASS_NAMES:
            raise ValueError(f"class_name must be in {visa.CLASS_NAMES}")
        DatasetCls = visa.VISADataset
    else:
        raise ValueError("dataset must be mvtec or visa")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    enable_fast_gpu()

    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    if arch == "resnet18":
        model = resnet18(pretrained=True, progress=True)
        t_d, d_sub = 448, 100
    elif arch == "wide_resnet50_2":
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d, d_sub = 1792, 550
    else:
        raise ValueError(arch)

    model.to(device)
    model.eval()

    idx_fused = torch.tensor(random.sample(range(t_d), d_sub), dtype=torch.long)
    sort_ord = np.argsort(idx_fused.numpy())
    marginal_groups = [t.tolist() for t in np.array_split(sort_ord, 3)]

    outputs: List[torch.Tensor] = []

    def hook(module, inp, out):
        outputs.append(out)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    train_ds = DatasetCls(data_path, class_name=class_name, is_train=True)
    train_ds = _maybe_cap_train_ds(train_ds, max_train_images, int(seed))
    test_ds = DatasetCls(data_path, class_name=class_name, is_train=False)
    train_loader = make_feature_loader(train_ds, arch, batch_size=batch_size, num_workers=num_workers)
    test_loader = make_feature_loader(test_ds, arch, batch_size=batch_size, num_workers=num_workers)

    base_compact, base_full, s0, s1, s2, s_fused, gt = run_one_category(
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

    paths = list(test_ds.x)
    if len(paths) != len(gt):
        raise RuntimeError("test path count mismatch vs scores")

    lab_str = _labels_str(gt)
    per = pd.DataFrame(
        {
            "image_path": paths,
            "image_label": lab_str,
            "s0": s0,
            "s1": s1,
            "s2": s2,
            "s_fused": s_fused,
            "semantic_score": s0,
            "visual_score": s1,
            "harmonic_score": s_fused,
        }
    )
    pair_df = build_padim_protocol_b_pairwise_table(per)
    sample_inst = build_sample_instability_table(pair_df)
    rank_tbl = build_sample_ranking_error_table(pair_df, sample_inst, per[["image_path", "harmonic_score"]])
    err_by_path = rank_tbl.set_index("image_path")["error"].to_dict()
    per["sample_error"] = per["image_path"].map(err_by_path).astype(float)
    per["proxy_u_padim_marg"] = _proxy_u_padim_marg(
        per["s0"].to_numpy(), per["s1"].to_numpy(), per["s2"].to_numpy()
    )

    csv_path = os.path.join(save_dir, "per_sample.csv")
    per.to_csv(csv_path, index=False)

    y = gt.astype(int)
    auc = float(roc_auc_score(y, s_fused)) if len(np.unique(y)) >= 2 else float("nan")
    flip_mean = float(pair_df["flip"].mean())

    summary = {
        "dataset": dataset,
        "class_name": class_name,
        "arch": arch,
        "seed": int(seed),
        "sklearn_auroc_final": auc,
        "fused_auroc": auc,
        "flip_rate_mean": flip_mean,
        "mean_sample_instability": float(base_compact.get("mean_sample_instability", float("nan"))),
        "n_test": int(len(gt)),
        "n_anomaly": int(np.sum(gt == 1)),
        "n_normal": int(np.sum(gt == 0)),
        "n_pairs": int(len(pair_df)),
        "per_sample_csv": os.path.abspath(csv_path),
    }
    meta = {
        "idx_fused_first_20": idx_fused[:20].tolist(),
        "marginal_dim_groups": [len(g) for g in marginal_groups],
        "save_dir": os.path.abspath(save_dir),
        "max_train_images": max_train_images,
        "n_train_used": int(len(train_ds)),
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, **meta}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(save_dir, "protocol_b_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {"csv_path": csv_path, "summary": summary}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["mvtec", "visa"], default="mvtec")
    p.add_argument("--class_name", type=str, required=True)
    p.add_argument("--arch", type=str, choices=["resnet18", "wide_resnet50_2"], default="resnet18")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--cov-float32", action="store_true")
    p.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Cap training images for Gaussian fit (deterministic subsample per seed). "
        "Recommended for VisA×WR50 on limited RAM; omit for full training set.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    dp = args.data_path or (
        os.path.expanduser("~/datasets/pro_visa") if args.dataset == "visa" else os.path.expanduser("~/datasets/mvtec")
    )
    out = run_protocol_b_one_setting(
        dataset=args.dataset,
        class_name=args.class_name,
        arch=args.arch,
        seed=int(args.seed),
        save_dir=os.path.abspath(args.save_dir),
        data_path=os.path.expanduser(dp),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cov_float32=bool(args.cov_float32),
        max_train_images=args.max_train_images,
    )
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
