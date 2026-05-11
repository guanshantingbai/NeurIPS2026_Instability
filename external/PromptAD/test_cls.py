from __future__ import annotations

import argparse
import os
import warnings

import cv2
import numpy as np
import torch
import torch.optim.lr_scheduler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from sklearn.metrics import roc_auc_score

from utils.instability_penalty import (
    metric_cal_img_with_h2_instability_correction,
    metric_cal_img_with_thresholded_instability_penalty,
    save_instability_csv_and_summary,
    save_strategy_a_instability_fusion_csv,
)
from torchvision import transforms
from tqdm import tqdm

TASK = 'CLS'


def compute_support_normal_s_final_mean(
    model,
    train_dataloader: DataLoader,
    device: str,
    resolution: int,
) -> float:
    """
    t = mean(s_final) over k-shot **normal** support (label == 0).
    Uses the same score path as test (max-pool visual map, harmonic fuse).
    """
    semantic_scores: list = []
    visual_score_maps: list = []
    model.eval()
    with torch.no_grad():
        for (data, mask, label, name, img_type) in train_dataloader:
            data = data.to(device, non_blocking=True)
            score_img, score_map = model(data, "cls")
            for i in range(len(label)):
                if int(label[i].item()) != 0:
                    continue
                semantic_scores.append(score_img[i])
                visual_score_maps.append(score_map[i])

    if not semantic_scores:
        warnings.warn(
            "H2: no label==0 in train loader; using all k-shot images to compute t.",
            RuntimeWarning,
            stacklevel=2,
        )
        with torch.no_grad():
            for (data, mask, label, name, img_type) in train_dataloader:
                data = data.to(device, non_blocking=True)
                score_img, score_map = model(data, "cls")
                for i in range(len(label)):
                    semantic_scores.append(score_img[i])
                    visual_score_maps.append(score_map[i])

    visual_score_maps = [
        cv2.resize(score, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        for score in visual_score_maps
    ]
    sem = np.array(semantic_scores)
    vis_maps = np.array(visual_score_maps)
    s_vis = vis_maps.reshape(sem.shape[0], -1).max(axis=1)
    s_final = harmonic_mean_fuse_scores(sem, vis_maps)
    return float(np.mean(s_final))


def test(model,
        args,
        dataloader: DataLoader,
        device: str,
        img_dir: str,
        check_path: str,
        csv_path: str,
        train_dataloader: DataLoader | None = None,
        ):

    # change the model into eval mode
    model.eval_mode()

    model.load_state_dict(torch.load(check_path), strict=False)

    t_support: float | None = None
    if (
        float(getattr(args, "instability_fusion_alpha", 0.0) or 0.0) <= 0.0
        and args.instability_correction == "h2"
    ):
        if train_dataloader is None:
            raise ValueError("instability_correction='h2' requires train_dataloader")
        t_support = compute_support_normal_s_final_mean(
            model, train_dataloader, device, int(args.resolution)
        )

    # semantic branch score (per-image scalar)
    semantic_scores = []
    # visual/memory-bank branch score (per-image 2D score map)
    visual_score_maps = []
    gt_list = []
    gt_mask_list = []
    names = []

    # Build a stable mapping from `img_name` -> `img_path`
    # (img_name format is defined in `datasets/dataset.py`)
    name_to_path = {}
    ds = getattr(dataloader, "dataset", None)
    if ds is not None and hasattr(ds, "img_paths") and hasattr(ds, "types") and hasattr(ds, "category"):
        for p, t in zip(ds.img_paths, ds.types):
            key = f"{ds.category}-{t}-{os.path.basename(p[:-4])}"
            name_to_path[key] = p

    for (data, mask, label, name, img_type) in dataloader:
        for n, l, m in zip(name, label, mask):
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

        data = data.to(device, non_blocking=True)
        score_img, score_map = model(data, 'cls')
        visual_score_maps += score_map
        semantic_scores += score_img

    visual_score_maps = [
        cv2.resize(score, (args.resolution, args.resolution), interpolation=cv2.INTER_CUBIC)
        for score in visual_score_maps
    ]
    gt_mask_list = [
        cv2.resize(mask, (args.resolution, args.resolution), interpolation=cv2.INTER_NEAREST)
        for mask in gt_mask_list
    ]
    sem = np.array(semantic_scores)
    vis_maps = np.array(visual_score_maps)
    image_paths = [name_to_path.get(n, n) for n in names]
    alpha_f = float(getattr(args, "instability_fusion_alpha", 0.0) or 0.0)

    if alpha_f > 0.0:
        s_new = instability_aware_harmonic_fuse_scores(sem, vis_maps, alpha_f)
        gt_arr = np.asarray(gt_list, dtype=int).reshape(-1)
        n0 = int((gt_arr == 0).sum())
        n1 = int((gt_arr == 1).sum())
        result_baseline = metric_cal_img(semantic_scores=sem, visual_scores=vis_maps, gt_list=gt_list)
        if n0 > 0 and n1 > 0:
            try:
                i_fus = float(roc_auc_score(gt_arr, s_new)) * 100.0
            except ValueError:
                i_fus = float("nan")
        else:
            i_fus = float("nan")
        result_dict = {
            "i_roc": i_fus,
            "p_roc": 0.0,
            "i_roc_baseline": result_baseline["i_roc"],
        }
        per_sample_path = os.path.join(
            os.path.dirname(csv_path),
            f"{TASK}-{args.dataset}-{args.class_name}-k{args.k_shot}-seed{args.seed}-per_sample.csv",
        )
        save_per_sample_score_table(
            semantic_scores=sem,
            visual_scores=vis_maps,
            image_paths=image_paths,
            labels=gt_list,
            save_path=per_sample_path,
            instability_fusion_scores=s_new,
        )
        fusion_path = os.path.join(
            os.path.dirname(csv_path),
            f"{TASK}-{args.dataset}-{args.class_name}-k{args.k_shot}-seed{args.seed}-per_sample_instability_fusion_a.csv",
        )
        save_strategy_a_instability_fusion_csv(
            semantic_scores=sem,
            visual_score_maps=vis_maps,
            image_paths=image_paths,
            labels=gt_list,
            category=args.class_name,
            save_path=fusion_path,
            alpha=alpha_f,
        )
        return result_dict

    result_baseline = metric_cal_img(semantic_scores=sem, visual_scores=vis_maps, gt_list=gt_list)
    if args.instability_correction == "h2":
        assert t_support is not None
        result_penalized = metric_cal_img_with_h2_instability_correction(
            sem,
            vis_maps,
            gt_list,
            t_support_normal=t_support,
            lambda_penalty=args.instability_penalty_lambda,
            quantile_q=args.instability_threshold_quantile,
        )
    else:
        result_penalized = metric_cal_img_with_thresholded_instability_penalty(
            sem,
            vis_maps,
            gt_list,
            lambda_penalty=args.instability_penalty_lambda,
            quantile_q=args.instability_threshold_quantile,
        )
    result_dict = {
        "i_roc": result_penalized["i_roc"],
        "p_roc": 0.0,
        "i_roc_baseline": result_baseline["i_roc"],
    }

    per_sample_path = os.path.join(
        os.path.dirname(csv_path),
        f"{TASK}-{args.dataset}-{args.class_name}-k{args.k_shot}-seed{args.seed}-per_sample.csv",
    )
    save_per_sample_score_table(
        semantic_scores=sem,
        visual_scores=vis_maps,
        image_paths=image_paths,
        labels=gt_list,
        save_path=per_sample_path,
    )

    instability_path = os.path.join(
        os.path.dirname(csv_path),
        f"{TASK}-{args.dataset}-{args.class_name}-k{args.k_shot}-seed{args.seed}-per_sample_instability.csv",
    )
    save_instability_csv_and_summary(
        semantic_scores=sem,
        visual_score_maps=vis_maps,
        image_paths=image_paths,
        labels=gt_list,
        category=args.class_name,
        save_path=instability_path,
        quantile_q=args.instability_threshold_quantile,
        correction_mode=args.instability_correction,
        t_support_normal=t_support,
    )

    return result_dict


def main(args):
    kwargs = vars(args)

    _nt = os.environ.get("OMP_NUM_THREADS", "").strip()
    if _nt.isdigit():
        _n = int(_nt)
        if _n > 0:
            torch.set_num_threads(_n)

    if kwargs['seed'] is None:
        kwargs['seed'] = 222

    oo = kwargs.get('output_root', '')
    kwargs['output_root'] = oo.strip() if isinstance(oo, str) and oo.strip() else None

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, transform=model.transform, **kwargs)

    train_dataloader = None
    if float(kwargs.get("instability_fusion_alpha", 0.0) or 0.0) <= 0.0 and kwargs.get(
        "instability_correction"
    ) == "h2":
        train_dataloader, _ = get_dataloader_from_args(
            phase="train", perturbed=False, transform=model.transform, **kwargs
        )

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = test(
        model,
        args,
        test_dataloader,
        device,
        img_dir=img_dir,
        check_path=check_path,
        csv_path=csv_path,
        train_dataloader=train_dataloader,
    )

    object = kwargs["class_name"]
    i_pen = metrics["i_roc"]
    i_base = metrics["i_roc_baseline"]
    delta = i_pen - i_base
    alpha_f = float(kwargs.get("instability_fusion_alpha", 0.0) or 0.0)
    if alpha_f > 0.0:
        print(
            f"Object:{object} =========================== Image-AUROC Strategy A fusion "
            f"(alpha={alpha_f}): {round(i_pen, 2)}"
        )
        print(
            f"Object:{object} =========================== Image-AUROC baseline (harmonic): {round(i_base, 2)}"
        )
        print(
            f"Object:{object} =========================== Delta_AUROC (fusion - baseline): {round(delta, 2)}\n"
        )
    else:
        _mode = kwargs["instability_correction"]
        _label = "H2" if _mode == "h2" else "thresholded"
        print(
            f'Object:{object} =========================== Image-AUROC {_label} '
            f'(lambda={kwargs["instability_penalty_lambda"]}, '
            f'quantile_q={kwargs["instability_threshold_quantile"]}): {round(i_pen, 2)}'
        )
        print(
            f'Object:{object} =========================== Image-AUROC baseline (s_final): {round(i_base, 2)}'
        )
        print(
            f'Object:{object} =========================== Delta ({_label} - baseline): {round(delta, 2)}\n'
        )

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result",
                        help="Root that contains checkpoints (…/checkpoint/*.pt).")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="If set, write csv/imgs under this root; checkpoints still loaded from --root-dir.",
    )
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", type=str2bool, default=True)
    parser.add_argument("--persistent-workers", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--use-cpu", type=int, default=0)

    parser.add_argument(
        "--instability-penalty-lambda",
        type=float,
        default=0.1,
        help="Inference: subtract lambda*u_abs on samples with u_abs > quantile(u_abs, q).",
    )
    parser.add_argument(
        "--instability-threshold-quantile",
        type=float,
        default=0.8,
        help="Threshold tau = quantile(u_abs, q); default 0.8 => penalize top ~20%% (strict u>tau).",
    )
    parser.add_argument(
        "--instability-correction",
        type=str,
        default="h2",
        choices=["h2", "thresholded"],
        help="h2: bidirectional sign(s-t) on high-u (t from k-shot normals). thresholded: s - lambda*u only.",
    )
    parser.add_argument(
        "--instability-fusion-alpha",
        type=float,
        default=0.0,
        help="Strategy A (inference): weighted harmonic fusion w=1/(1+α|s_sem-s_vis|). "
        "If >0, overrides H2/thresholded post-hoc correction for this run. 0=disabled. Typical: 1.0.",
    )

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
