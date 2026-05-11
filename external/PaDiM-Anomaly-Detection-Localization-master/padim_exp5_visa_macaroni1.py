"""
Exp5-style CSVs for VisA macaroni1 + ResNet-18 (Protocol B): sample I_bin, ranking error, rejection curve.

Writes under result_analysis/exp5_visa_macaroni1_r18/experiments/:
  sample_instability_table.csv
  per_sample_harmonic.csv
  exp5_sample_ranking_error.csv
  exp5_instability_rejection.csv
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import zlib
from random import sample

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torchvision.models import resnet18

import datasets.visa as visa
from padim_dataloader import enable_fast_gpu, make_feature_loader
from padim_promptad_pairwise_tables import (
    build_padim_protocol_b_pairwise_table,
    build_sample_instability_table,
    run_instability_rejection_experiment,
)
from padim_protocol_b_mvtec_multiclass import run_one_category


def _maybe_cap_train_ds(train_ds, max_images: int | None, salt: int):
    if max_images is None:
        return train_ds
    n = len(train_ds)
    if n <= max_images:
        return train_ds
    rng = random.Random(salt)
    idx = sorted(rng.sample(range(n), max_images))
    return Subset(train_ds, idx)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--visa_path", type=str, default="~/datasets/pro_visa")
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_train_images", type=int, default=None)
    args = p.parse_args()

    visa_path = os.path.expanduser(args.visa_path)
    repo = os.path.dirname(os.path.abspath(__file__))
    out_root = args.out_root or os.path.join(repo, "result_analysis", "exp5_visa_macaroni1_r18")
    exp_dir = os.path.join(out_root, "experiments")
    os.makedirs(exp_dir, exist_ok=True)

    class_name = "macaroni1"
    arch = "resnet18"
    if class_name not in visa.CLASS_NAMES:
        raise SystemExit(f"unknown class {class_name}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    enable_fast_gpu()

    model = resnet18(pretrained=True, progress=True)
    t_d, d_sub = 448, 100
    model.to(device)
    model.eval()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    outputs: list = []

    def hook(module, inp, out):
        outputs.append(out)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    idx_fused = torch.tensor(sample(range(t_d), d_sub))
    sort_ord = np.argsort(idx_fused.numpy())
    marginal_groups = [t.tolist() for t in np.array_split(sort_ord, 3)]

    train_ds = visa.VISADataset(visa_path, class_name=class_name, is_train=True)
    cap_salt = (zlib.crc32(f"visa\0{class_name}\0{arch}".encode()) & 0x7FFFFFFF) ^ int(args.seed)
    train_ds = _maybe_cap_train_ds(train_ds, args.max_train_images, cap_salt)
    test_ds = visa.VISADataset(visa_path, class_name=class_name, is_train=False)
    train_loader = make_feature_loader(train_ds, arch, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = make_feature_loader(test_ds, arch, batch_size=args.batch_size, num_workers=args.num_workers)

    compact, _full, s0, s1, s2, s_fused, gt = run_one_category(
        class_name,
        model,
        device,
        outputs,
        idx_fused,
        marginal_groups,
        train_loader,
        test_loader,
        cov_float32=False,
        return_scores=True,
    )

    paths = [os.path.abspath(x) for x in test_ds.x]
    if len(paths) != len(gt):
        raise RuntimeError("path list / gt length mismatch")

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
    si_path = os.path.join(exp_dir, "sample_instability_table.csv")
    si.to_csv(si_path, index=False)

    per_h = pd.DataFrame(
        {
            "image_path": paths,
            "image_label": lab,
            "harmonic_score": s_fused.astype(np.float64),
        }
    )
    per_path = os.path.join(exp_dir, "per_sample_harmonic.csv")
    per_h.to_csv(per_path, index=False)

    p_sample, p_rej = run_instability_rejection_experiment(pair_df, si_path, per_path, exp_dir)

    print("Protocol B compact:", {k: compact[k] for k in ("fused_auroc", "mean_pairwise_I", "fraction_I_gt0")})
    print("Wrote", si_path)
    print("Wrote", per_path)
    print("Wrote", p_sample)
    print("Wrote", p_rej)

    del model, train_loader, test_loader
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
