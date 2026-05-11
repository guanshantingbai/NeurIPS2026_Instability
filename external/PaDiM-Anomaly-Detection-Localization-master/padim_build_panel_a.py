"""
PaDiM panel (a): AUROC vs mean pairwise instability.
Settings: {MVTec + VisA, category, backbone} with backbone in {resnet18, wide_resnet50_2}.

Mirrors PromptAD draw_panel_a_counterexample logic (gray cloud + blue/red gap in AUROC band).
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import shutil
import tempfile
import zlib
from random import sample

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
from padim_protocol_b_mvtec_multiclass import run_one_category

TITLE_FS = 10
AXIS_FS = 10
TICK_FS = 9
ANNOT_FS_PANEL_A = 8
PNG_DPI = 300

ALL_ARCHES = ("resnet18", "wide_resnet50_2")
ARCH_SHORT = {"resnet18": "R18", "wide_resnet50_2": "WR50"}
DATASET_TAG = {"mvtec": "M", "visa": "V"}


def _mem_status_lines() -> list[str]:
    """Lightweight RSS + CUDA stats for OOM post-mortem (137 ≈ Linux OOM killer)."""
    lines: list[str] = []
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:", "VmPeak:")):
                    lines.append(line.strip())
    except OSError:
        pass
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        free_b, total_b = torch.cuda.mem_get_info(dev)
        lines.append(
            f"CUDA mem_get_info: free={free_b / 1e9:.2f} GB total={total_b / 1e9:.2f} GB"
        )
        lines.append(
            f"CUDA torch allocated={torch.cuda.memory_allocated(dev) / 1e9:.4f} GB "
            f"reserved={torch.cuda.memory_reserved(dev) / 1e9:.4f} GB"
        )
    return lines


def print_mem_diag(stage: str) -> None:
    print(f"[mem] {stage}", flush=True)
    for ln in _mem_status_lines():
        print(f"  {ln}", flush=True)


def apply_cpu_thread_limit(n: int) -> None:
    """Limit BLAS/OpenMP and PyTorch CPU threads to reduce parallel RAM spikes (OOM)."""
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(n)
    try:
        import mkl  # type: ignore

        mkl.set_num_threads(n)
    except Exception:
        pass
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(min(n, 4))
    except Exception:
        pass


CHECKPOINT_COLS = [
    "dataset",
    "category",
    "backbone",
    "backbone_short",
    "AUROC",
    "instability",
    "flip_rate_mean",
    "n_pairs",
    "n_test",
]


def _row_key(dataset: str, category: str, backbone: str) -> tuple[str, str, str]:
    return (str(dataset), str(category), str(backbone))


def _flip_rate_complete(rec: dict) -> bool:
    v = rec.get("flip_rate_mean", np.nan)
    return v == v and v is not None  # not NaN


def load_checkpoint_csv(path: str) -> tuple[list[dict], set[tuple[str, str, str]]]:
    """Load partial progress; dedupe by (dataset, category, backbone), last row wins."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return [], set()
    df = pd.read_csv(path)
    if len(df) == 0:
        return [], set()
    if "dataset" not in df.columns:
        df["dataset"] = "mvtec"
    by_key: dict[tuple[str, str, str], dict] = {}
    for _, row in df.iterrows():
        k = _row_key(row["dataset"], row["category"], row["backbone"])
        rec = {c: (row[c] if c in row.index else np.nan) for c in CHECKPOINT_COLS}
        by_key[k] = rec
    order = sorted(by_key.keys(), key=lambda t: (t[2], t[0], t[1]))
    rows = [by_key[k] for k in order]
    return rows, set(by_key.keys())


def _maybe_cap_train_ds(train_ds, max_images: int | None, salt: int) -> torch.utils.data.Dataset:
    """Reduce RAM in fit_fused_gaussian (esp. VisA × WR50); deterministic per salt."""
    if max_images is None:
        return train_ds
    n = len(train_ds)
    if n <= max_images:
        return train_ds
    rng = random.Random(salt)
    idx = sorted(rng.sample(range(n), max_images))
    return Subset(train_ds, idx)


def save_checkpoint_csv(rows: list[dict], path: str) -> None:
    """Atomic write so a kill mid-run does not corrupt the checkpoint file."""
    df = pd.DataFrame(rows)
    for c in CHECKPOINT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[CHECKPOINT_COLS]
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv", prefix=".padim_panel_a_", dir=d, text=True)
    try:
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            os.unlink(tmp)
        raise


def collect_rows(
    mvtec_path: str,
    visa_path: str,
    seed: int,
    batch_size: int | None,
    num_workers: int | None,
    include_mvtec: bool,
    include_visa: bool,
    checkpoint_path: str | None,
    resume: bool,
    max_train_images: int | None,
    arches: tuple[str, ...],
    single_dataset: str | None,
    single_category: str | None,
) -> pd.DataFrame:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    enable_fast_gpu()

    rows: list[dict] = []
    done: set[tuple[str, str, str]] = set()
    flip_complete: set[tuple[str, str, str]] = set()
    if resume and checkpoint_path and os.path.isfile(checkpoint_path):
        rows, done = load_checkpoint_csv(checkpoint_path)
        flip_complete = { _row_key(r["dataset"], r["category"], r["backbone"]) for r in rows if _flip_rate_complete(r) }
        n_skip = sum(1 for k in done if k in flip_complete)
        n_refill = len(done) - n_skip
        print(
            f"Resume: {len(done)} settings in {checkpoint_path}; "
            f"{n_skip} complete (incl. flip_rate_mean), {n_refill} need recompute/backfill.",
            flush=True,
        )

    outputs: list = []

    def hook(module, inp, out):
        outputs.append(out)

    dataset_specs: list[tuple[str, type, str, list]] = []
    if include_mvtec:
        dataset_specs.append(("mvtec", mvtec.MVTecDataset, mvtec_path, list(mvtec.CLASS_NAMES)))
    if include_visa:
        dataset_specs.append(("visa", visa.VISADataset, visa_path, list(visa.CLASS_NAMES)))
    if single_dataset is not None:
        dataset_specs = [s for s in dataset_specs if s[0] == single_dataset]
        if not dataset_specs:
            raise ValueError(f"No dataset selected for --single_dataset={single_dataset}")

    for arch in arches:
        pending_for_arch = []
        for ds_key, _, _, class_names in dataset_specs:
            for class_name in class_names:
                key = _row_key(ds_key, class_name, arch)
                if key not in done or (resume and key not in flip_complete):
                    pending_for_arch.append(key)
        if not pending_for_arch:
            print(f"  [resume] skip backbone {arch} (all categories done)", flush=True)
            continue

        if arch == "resnet18":
            model = resnet18(pretrained=True, progress=True)
            t_d, d_sub = 448, 100
        else:
            model = wide_resnet50_2(pretrained=True, progress=True)
            t_d, d_sub = 1792, 550

        model.to(device)
        model.eval()
        if os.environ.get("PADIM_MEM_DIAG", "").strip().lower() in ("1", "yes", "true"):
            print_mem_diag(f"after {arch} loaded -> {device}")
        random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

        idx_fused = torch.tensor(sample(range(t_d), d_sub))
        sort_ord = np.argsort(idx_fused.numpy())
        marginal_groups = [t.tolist() for t in np.array_split(sort_ord, 3)]

        for ds_key, DatasetCls, root, class_names in dataset_specs:
            for class_name in class_names:
                if single_category is not None and class_name != single_category:
                    continue
                key = _row_key(ds_key, class_name, arch)
                if resume and key in flip_complete:
                    print(f"  [resume] skip [{ds_key}/{arch}] {class_name}", flush=True)
                    continue
                train_ds = DatasetCls(root, class_name=class_name, is_train=True)
                cap_salt = (
                    zlib.crc32(f"{ds_key}\0{class_name}\0{arch}".encode()) & 0x7FFFFFFF
                ) ^ int(seed)
                train_ds = _maybe_cap_train_ds(train_ds, max_train_images, cap_salt)
                test_ds = DatasetCls(root, class_name=class_name, is_train=False)
                train_loader = make_feature_loader(
                    train_ds, arch, batch_size=batch_size, num_workers=num_workers
                )
                test_loader = make_feature_loader(
                    test_ds, arch, batch_size=batch_size, num_workers=num_workers
                )
                cov_f32 = ds_key == "visa" and arch == "wide_resnet50_2"
                compact, full_metrics = run_one_category(
                    class_name,
                    model,
                    device,
                    outputs,
                    idx_fused,
                    marginal_groups,
                    train_loader,
                    test_loader,
                    cov_float32=cov_f32,
                )
                row = {
                    "dataset": ds_key,
                    "category": class_name,
                    "backbone": arch,
                    "backbone_short": ARCH_SHORT[arch],
                    "AUROC": compact["fused_auroc"],
                    "instability": compact["mean_pairwise_I"],
                    "flip_rate_mean": compact["fraction_I_gt0"],
                    "n_pairs": compact["n_pairs"],
                    "n_test": compact["n_test"],
                }
                replaced = False
                for i, prev in enumerate(rows):
                    if _row_key(prev["dataset"], prev["category"], prev["backbone"]) == key:
                        rows[i] = row
                        replaced = True
                        break
                if not replaced:
                    rows.append(row)
                done.add(key)
                flip_complete.add(key)
                if checkpoint_path:
                    save_checkpoint_csv(rows, checkpoint_path)
                print(
                    f"  [{ds_key}/{arch}] {class_name}: AUROC={compact['fused_auroc']:.4f} "
                    f"I_mean={compact['mean_pairwise_I']:.6f}",
                    flush=True,
                )
                del train_loader, test_loader, train_ds, test_ds, compact, full_metrics
                gc.collect()
                if use_cuda:
                    if arch == "wide_resnet50_2":
                        model.to("cpu")
                        torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if arch == "wide_resnet50_2":
                        model.to(device)

        del model
        if use_cuda:
            torch.cuda.empty_cache()

    if not rows:
        return pd.DataFrame(columns=CHECKPOINT_COLS)
    return pd.DataFrame(rows)


def _point_label(row: pd.Series, max_len: int = 22) -> str:
    cat = str(row["category"]).replace("_", " ")
    bb = str(row.get("backbone_short", ""))
    tag = DATASET_TAG.get(str(row.get("dataset", "")), "")
    prefix = f"{tag}:" if tag else ""
    s = f"{prefix}{cat} ({bb})"
    if len(s) > max_len:
        return s[: max_len - 1] + "\u2026"
    return s


def draw_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    x_all = df["AUROC"].to_numpy(dtype=float)
    y_all = df["instability"].to_numpy(dtype=float)

    ax.scatter(
        x_all,
        y_all,
        alpha=0.45,
        s=28,
        edgecolors="none",
        color="0.35",
        zorder=1,
    )

    band = df[(df["AUROC"] >= 0.9) & (df["AUROC"] <= 0.95)].copy()
    if len(band) < 2:
        band = df[(df["AUROC"] >= 0.88) & (df["AUROC"] <= 0.97)].copy()
    if len(band) < 2:
        band = df[(df["AUROC"] >= 0.85) & (df["AUROC"] <= 0.99)].copy()

    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    ypad = max(1e-6, (ymax - ymin) * 0.08)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    if len(band) >= 2:
        band2 = band.sort_values("instability")
        low = band2.iloc[0]
        high = band2.iloc[-1]

        xl, yl = float(low["AUROC"]), float(low["instability"])
        xh, yh = float(high["AUROC"]), float(high["instability"])
        delta = yh - yl

        ax.scatter([xl], [yl], s=135, c="blue", edgecolors="black", linewidths=1.0, zorder=5)
        ax.scatter([xh], [yh], s=135, c="red", edgecolors="black", linewidths=1.0, zorder=5)

        afs = ANNOT_FS_PANEL_A
        ax.annotate(
            _point_label(low),
            (xl, yl),
            textcoords="offset points",
            xytext=(-32, -14),
            fontsize=afs,
            color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.5),
            zorder=6,
            clip_on=True,
        )
        ax.annotate(
            _point_label(high),
            (xh, yh),
            textcoords="offset points",
            xytext=(-26, -22),
            fontsize=afs,
            color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.5),
            zorder=6,
            clip_on=True,
        )

        ax.annotate(
            "",
            xy=(xh, yh),
            xytext=(xl, yl),
            arrowprops=dict(arrowstyle="->", color="0.25", lw=1.5, shrinkA=14, shrinkB=14),
            zorder=4,
            clip_on=True,
        )

        ax.text(
            0.97,
            0.97,
            f"Δ ≈ {delta:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=ANNOT_FS_PANEL_A,
            color="0.2",
            zorder=7,
        )

    ax.set_xlabel("AUROC", fontsize=AXIS_FS)
    ax.set_ylabel("Instability Score", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="MVTec root (alias for --mvtec_path), default ~/datasets/mvtec",
    )
    p.add_argument(
        "--mvtec_path",
        type=str,
        default=None,
        help="MVTec root, default ~/datasets/mvtec",
    )
    p.add_argument(
        "--visa_path",
        type=str,
        default=None,
        help="VisA root (pro_visa layout), default ~/datasets/pro_visa",
    )
    p.add_argument(
        "--no_visa",
        action="store_true",
        help="Only MVTec (same as previous single-dataset figure)",
    )
    p.add_argument(
        "--visa_only",
        action="store_true",
        help="Only VisA (use with full run split if the process is OOM-killed: "
        "run --no_visa then --visa_only, merge CSVs, then --skip_compute).",
    )
    p.add_argument("--out_dir", type=str, default=None, help="Output directory for CSV and figure")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers (try 0 if the OS kills the job with exit 137 / OOM).",
    )
    p.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        metavar="N",
        help="Cap CPU threads for NumPy/BLAS/OpenMP and torch (e.g. 1 when the OS kills the process). "
        "Also export OMP_NUM_THREADS=N before starting python for full BLAS compliance.",
    )
    p.add_argument(
        "--diag_wr50_warmup",
        action="store_true",
        help="Load wide_resnet50_2 on CUDA, run one dummy forward (batch=1), print host RSS + CUDA "
        "memory, then exit 0. Use to see whether 137 happens before full PaDiM (GPU/model) vs later (RAM fit).",
    )
    p.add_argument("--skip_compute", action="store_true", help="Only plot from existing CSV")
    p.add_argument("--csv_path", type=str, default=None, help="CSV to load when skip_compute")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing csv_path and recompute from scratch. "
        "Default: if csv_path already exists, resume (skip finished rows). "
        "Progress is saved to csv_path after every category.",
    )
    p.add_argument(
        "--bootstrap_from",
        type=str,
        default=None,
        help="If csv_path does not exist yet, copy this CSV to csv_path before computing "
        "(e.g. a partial run that never wrote a checkpoint). Requires columns "
        "category,backbone,AUROC,instability,flip_rate_mean,...; adds dataset=mvtec if missing.",
    )
    p.add_argument(
        "--max_train_images",
        type=int,
        default=None,
        help="Random subset cap on TRAINING images per (dataset,category,backbone). "
        "Strongly recommended on large VisA train sets to avoid OOM (e.g. 512). "
        "Does not affect test metrics path; Gaussian fit uses the subset only.",
    )
    p.add_argument(
        "--single_backbone",
        type=str,
        choices=list(ALL_ARCHES),
        default=None,
        help="Only run this backbone (useful with --single_category for one short process).",
    )
    p.add_argument(
        "--single_category",
        type=str,
        default=None,
        help="Only run this category name (e.g. candle). Combine with --visa_only --single_backbone wide_resnet50_2.",
    )
    p.add_argument(
        "--single_dataset",
        type=str,
        choices=["mvtec", "visa"],
        default=None,
        help="With --single_category, restrict to this dataset (optional if only one dataset is enabled).",
    )
    args = p.parse_args()

    if args.cpu_threads is not None and args.cpu_threads > 0:
        apply_cpu_thread_limit(args.cpu_threads)

    if args.diag_wr50_warmup:
        if not torch.cuda.is_available():
            raise SystemExit("diag_wr50_warmup needs CUDA.")
        enable_fast_gpu()
        print_mem_diag("before wide_resnet50_2(pretrained=True)")
        m = wide_resnet50_2(pretrained=True, progress=True)
        print_mem_diag("after wide_resnet50_2 weights loaded (default device CPU)")
        m.eval()
        m.to(torch.device("cuda:0"))
        print_mem_diag("after model.to(cuda:0)")
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224, device="cuda:0")
            _ = m(x)
        print_mem_diag("after one forward batch=1")
        print("diag_wr50_warmup OK", flush=True)
        return

    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(repo, "result_analysis", "figures")
    os.makedirs(out_dir, exist_ok=True)
    csv_default = os.path.join(out_dir, "padim_panel_a_mvtec_visa_r18_wr50.csv")
    csv_path = args.csv_path or csv_default

    if not args.skip_compute:
        if args.no_visa and args.visa_only:
            raise SystemExit("Choose at most one of --no_visa and --visa_only.")
        include_mvtec = not args.visa_only
        include_visa = not args.no_visa
        if not include_mvtec and not include_visa:
            raise SystemExit("Nothing to compute (use neither --no_visa nor --visa_only).")
        mvtec_path = os.path.expanduser(
            args.mvtec_path or args.data_path or "~/datasets/mvtec"
        )
        visa_path = os.path.expanduser(args.visa_path or "~/datasets/pro_visa")
        if include_mvtec and not os.path.isdir(mvtec_path):
            raise FileNotFoundError(f"MVTec root not found: {mvtec_path}")
        if include_visa and not os.path.isdir(visa_path):
            raise FileNotFoundError(
                f"VisA root not found: {visa_path} (use --visa_path or --no_visa)"
            )
        if (
            args.bootstrap_from
            and not args.overwrite
            and not os.path.isfile(csv_path)
        ):
            src = os.path.expanduser(args.bootstrap_from)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"--bootstrap_from not found: {src}")
            shutil.copy2(src, csv_path)
            bdf = pd.read_csv(csv_path)
            if "dataset" not in bdf.columns:
                bdf["dataset"] = "mvtec"
                bdf.to_csv(csv_path, index=False)
            print(f"Bootstrap checkpoint: copied {src} -> {csv_path}", flush=True)
        n_mv = len(mvtec.CLASS_NAMES) if include_mvtec else 0
        n_vis = len(visa.CLASS_NAMES) if include_visa else 0
        arches: tuple[str, ...] = (
            (args.single_backbone,) if args.single_backbone is not None else ALL_ARCHES
        )
        if args.single_category:
            names_ok = []
            if include_mvtec and args.single_category in mvtec.CLASS_NAMES:
                names_ok.append("mvtec")
            if include_visa and args.single_category in visa.CLASS_NAMES:
                names_ok.append("visa")
            if not names_ok:
                raise SystemExit(
                    f"--single_category={args.single_category!r} not in enabled dataset class lists."
                )
            if args.single_dataset and args.single_dataset not in names_ok:
                raise SystemExit(
                    f"--single_dataset={args.single_dataset} does not contain category "
                    f"{args.single_category!r}."
                )
        if args.single_dataset and not (
            (args.single_dataset == "mvtec" and include_mvtec)
            or (args.single_dataset == "visa" and include_visa)
        ):
            raise SystemExit("--single_dataset does not match enabled datasets.")
        n_cls = 1 if args.single_category else (n_mv + n_vis)
        n_arch = len(arches)
        print(
            (
                f"Single-category mode: {args.single_category!r} x {n_arch} backbone(s)..."
                if args.single_category
                else f"Collecting {n_cls} categories x {n_arch} backbones (mvtec={n_mv}, visa={n_vis})..."
            ),
            flush=True,
        )
        resume = not args.overwrite
        df = collect_rows(
            mvtec_path,
            visa_path,
            args.seed,
            args.batch_size,
            args.num_workers,
            include_mvtec,
            include_visa,
            checkpoint_path=csv_path,
            resume=resume,
            max_train_images=args.max_train_images,
            arches=arches,
            single_dataset=args.single_dataset,
            single_category=args.single_category,
        )
        save_checkpoint_csv(df.to_dict("records"), csv_path)
        print("Wrote", csv_path, flush=True)
    else:
        df = pd.read_csv(csv_path)
        if "dataset" not in df.columns:
            df["dataset"] = "mvtec"

    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    draw_panel_a(ax, df)
    ax.set_title("AUROC vs instability", fontsize=TITLE_FS)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "padim_panel_a_auroc_instability.png")
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Wrote", out_png, flush=True)


if __name__ == "__main__":
    main()
