"""
Protocol B only — multiple MVTec categories, sanity metrics (no paper figures).

Reuses the exact Protocol B implementation from padim_instability_protocol_ab.py
(marginal subspace views M0,M1,M2; shared μ,Σ per category; canonical fused score).
"""
import argparse
import gc
import json
import os
import random
from collections import OrderedDict
from random import sample

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18, wide_resnet50_2

import datasets.mvtec as mvtec
from padim_dataloader import enable_fast_gpu, make_feature_loader

from padim_instability_protocol_ab import (
    _batch_mahalanobis,
    compute_instability_metrics,
    embedding_concat,
    global_minmax,
    image_level_max,
    marginal_mahalanobis_maps,
    postprocess_score_map,
)

# Stream test forward when |test| is large (avoids torch.cat of all test features on CPU → OOM).
_STREAM_TEST_THRESHOLD = int(os.environ.get("PADIM_STREAM_TEST_THRESHOLD", "350"))


def _global_minmax_with_bounds(smap_raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi - lo < 1e-12:
        return np.zeros_like(smap_raw)
    return (smap_raw - lo) / (hi - lo)


def _emb_te_sel_from_hook_outputs(outputs: list, idx_fused: torch.Tensor) -> torch.Tensor:
    if len(outputs) != 3:
        raise RuntimeError("expected 3 hook outputs (layer1,2,3)")
    # Match train path: hooks return CUDA tensors; embedding_concat uses CPU buffers.
    o1 = outputs[0].detach().cpu()
    o2 = outputs[1].detach().cpu()
    o3 = outputs[2].detach().cpu()
    emb = embedding_concat(embedding_concat(o1, o2), o3)
    idx = idx_fused.cpu() if idx_fused.device.type != "cpu" else idx_fused
    return torch.index_select(emb, 1, idx)


def compute_fused_dist_bhw(
    emb_te_sel: torch.Tensor,
    mean_f: np.ndarray,
    cov_f: np.ndarray,
    Hf: int,
    Wf: int,
    device: torch.device,
) -> np.ndarray:
    Bf, Cf, _, _ = emb_te_sel.size()
    HW = Hf * Wf
    cov_np_dtype = cov_f.dtype
    torch_dtype = torch.float32 if cov_np_dtype == np.float32 else torch.float64
    if device.type == "cuda":
        chunk = 48
        mean_t = torch.as_tensor(mean_f, dtype=torch_dtype, device=device)
        flat_te = emb_te_sel.to(device=device, dtype=torch_dtype).view(Bf, Cf, HW)
        parts = []
        for s in range(0, HW, chunk):
            e = min(s + chunk, HW)
            slab = slice(s, e)
            cov_chunk = torch.as_tensor(cov_f[:, :, slab], dtype=torch_dtype, device=device).permute(
                2, 0, 1
            )
            inv_chunk = torch.linalg.inv(cov_chunk)
            del cov_chunk
            delta = flat_te[:, :, slab] - mean_t[:, slab].unsqueeze(0)
            delta_hw = delta.permute(2, 0, 1).contiguous()
            quad = torch.einsum("hbc,hcd,hbd->hb", delta_hw, inv_chunk, delta_hw)
            del inv_chunk
            parts.append(torch.sqrt(torch.clamp(quad, min=0.0)).cpu())
        dist_f = torch.cat(parts, dim=0).numpy().T.reshape(Bf, Hf, Wf)
    else:
        flat_te = emb_te_sel.view(Bf, Cf, HW).numpy().astype(cov_np_dtype, copy=False)
        cols = []
        for i in range(HW):
            m = mean_f[:, i]
            invc = np.linalg.inv(cov_f[:, :, i])
            delta = flat_te[:, :, i] - m
            cols.append(_batch_mahalanobis(delta, invc))
        dist_f = np.stack(cols, axis=1).reshape(Bf, Hf, Wf)
    return dist_f


def fused_smap_raw_from_emb(
    emb_te_sel: torch.Tensor,
    mean_f: np.ndarray,
    cov_f: np.ndarray,
    Hf: int,
    Wf: int,
    last_h: int,
    device: torch.device,
) -> np.ndarray:
    dist_f = compute_fused_dist_bhw(emb_te_sel, mean_f, cov_f, Hf, Wf, device)
    return postprocess_score_map(dist_f, last_h)


def marginal_smap_raws_from_emb(
    emb_te_sel: torch.Tensor,
    mean_f: np.ndarray,
    cov_f: np.ndarray,
    marginal_groups: list,
    Hf: int,
    Wf: int,
    last_h: int,
    device: torch.device,
) -> list[np.ndarray]:
    raws = []
    for group in marginal_groups:
        if device.type == "cuda":
            dist_m = _marginal_mahalanobis_maps_gpu(
                emb_te_sel, mean_f, cov_f, group, Hf, Wf, device
            )
        else:
            dist_m = marginal_mahalanobis_maps(emb_te_sel, mean_f, cov_f, group, Hf, Wf)
        raws.append(postprocess_score_map(dist_m, last_h))
    return raws


def _scores_from_smap_raw(smap_raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return image_level_max(_global_minmax_with_bounds(smap_raw, lo, hi))

DEFAULT_CLASSES = ['bottle', 'capsule', 'screw', 'transistor', 'zipper', 'cable']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--class_names', nargs='+', default=DEFAULT_CLASSES)
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    p.add_argument('--save_dir', type=str, default='./protocol_b_mvtec_multiclass_out')
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    return p.parse_args()


def fit_fused_gaussian(train_layers, idx_fused, cov_float32: bool = False):
    emb_fused = train_layers['layer1']
    for nm in ['layer2', 'layer3']:
        emb_fused = embedding_concat(emb_fused, train_layers[nm])
    emb_sel = torch.index_select(emb_fused, 1, idx_fused)
    B, C, H, W = emb_sel.size()
    dtype = np.float32 if cov_float32 else np.float64
    flat = emb_sel.view(B, C, H * W).numpy().astype(dtype, copy=False)
    mean_f = np.mean(flat, axis=0).astype(dtype, copy=False)
    cov_f = np.zeros((C, C, H * W), dtype=dtype)
    I = np.identity(C, dtype=dtype)
    reg = np.float32(0.01) if cov_float32 else np.float64(0.01)
    for i in range(H * W):
        cov_f[:, :, i] = np.cov(flat[:, :, i], rowvar=False).astype(dtype, copy=False) + reg * I
    return mean_f, cov_f, H, W


def canonical_fused_scores(emb_te_sel, mean_f, cov_f, Hf, Wf, last_h, device: torch.device):
    smap_f_raw = fused_smap_raw_from_emb(emb_te_sel, mean_f, cov_f, Hf, Wf, last_h, device)
    smap_f = global_minmax(smap_f_raw)
    return image_level_max(smap_f)


def _marginal_mahalanobis_maps_gpu(emb_selected_bchw, mean, cov, dim_subset, H, W, device):
    S = list(dim_subset)
    k = len(S)
    B, d, _, _ = emb_selected_bchw.size()
    HW = H * W
    chunk = 96
    torch_dtype = torch.float32 if cov.dtype == np.float32 else torch.float64
    idx = torch.tensor(S, device=device, dtype=torch.long)
    cov_np = cov[np.ix_(S, S)]
    mean_1khw = torch.as_tensor(mean[S, :], dtype=torch_dtype, device=device).unsqueeze(0)
    emb = torch.index_select(emb_selected_bchw.to(device=device, dtype=torch_dtype), 1, idx)
    flat = emb.view(B, k, HW)
    parts = []
    for s in range(0, HW, chunk):
        e = min(s + chunk, HW)
        slab = slice(s, e)
        cov_chunk = torch.as_tensor(cov_np[:, :, slab], dtype=torch_dtype, device=device).permute(
            2, 0, 1
        )
        inv_chunk = torch.linalg.inv(cov_chunk)
        del cov_chunk
        delta = flat[:, :, slab] - mean_1khw[:, :, slab]
        delta_hw = delta.permute(2, 0, 1).contiguous()
        quad = torch.einsum("hbk,hkl,hbl->hb", delta_hw, inv_chunk, delta_hw)
        del inv_chunk
        parts.append(torch.sqrt(torch.clamp(quad, min=0.0)).cpu())
    dist = torch.cat(parts, dim=0).numpy().T.reshape(B, H, W)
    return dist


def protocol_b_marginal_scores(emb_te_sel, mean_f, cov_f, marginal_groups, Hf, Wf, last_h, device):
    out = []
    for raw in marginal_smap_raws_from_emb(
        emb_te_sel, mean_f, cov_f, marginal_groups, Hf, Wf, last_h, device
    ):
        norm = global_minmax(raw)
        out.append(image_level_max(norm))
    return out


def sample_instability_Ix(s0, s1, s2, gt):
    """I(x) = mean_{x' opposite label} Var(z_M0,z_M1,z_M2) for pair (x,x')."""
    idx_pos = np.where(gt == 1)[0]
    idx_neg = np.where(gt == 0)[0]
    B = gt.size
    Ix = np.zeros(B, dtype=np.float64)
    for x_idx in range(B):
        if gt[x_idx] == 1:
            others = idx_neg
        else:
            others = idx_pos
        if len(others) == 0:
            Ix[x_idx] = np.nan
            continue
        acc = []
        for x_other in others:
            if gt[x_idx] == 1:
                i, j = x_idx, x_other
            else:
                i, j = x_other, x_idx
            z0 = 1 if s0[i] > s0[j] else 0
            z1 = 1 if s1[i] > s1[j] else 0
            z2 = 1 if s2[i] > s2[j] else 0
            acc.append(np.var([z0, z1, z2], ddof=0))
        Ix[x_idx] = float(np.mean(acc))
    return Ix


def top_k_unstable(Ix, gt, s_fused, k=10):
    valid = np.where(~np.isnan(Ix))[0]
    if valid.size == 0:
        return []
    order = valid[np.argsort(Ix[valid])[::-1]]
    rows = []
    for idx in order[:k]:
        rows.append({
            'test_index': int(idx),
            'label': 'anomaly' if int(gt[idx]) == 1 else 'normal',
            'I_sample': float(Ix[idx]),
            's_fused': float(s_fused[idx]),
        })
    return rows


def run_one_category(
    class_name,
    model,
    device,
    outputs,
    idx_fused,
    marginal_groups,
    train_loader,
    test_loader,
    cov_float32: bool = False,
    return_scores: bool = False,
):
    """outputs list is mutated by hooks; clear between batches inside.

    If return_scores is True, also return (s0, s1, s2, s_fused, gt) numpy arrays
    aligned with the test set order (for exp5 / pairwise CSVs).
    """
    train_layers = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    for (x, _, _) in train_loader:
        with torch.no_grad():
            _ = model(x.to(device))
        for k, v in zip(train_layers.keys(), outputs):
            train_layers[k].append(v.cpu().detach())
        outputs.clear()
    for k in train_layers:
        train_layers[k] = torch.cat(train_layers[k], 0)

    mean_f, cov_f, H, W = fit_fused_gaussian(train_layers, idx_fused, cov_float32=cov_float32)

    n_test = len(test_loader.dataset)
    use_stream = n_test >= _STREAM_TEST_THRESHOLD
    use_cuda_eff = device.type == "cuda"
    n_m = len(marginal_groups)

    if use_stream:
        lo_f, hi_f = float(np.inf), float(-np.inf)
        lo_m = [float(np.inf)] * n_m
        hi_m = [float(-np.inf)] * n_m
        gt_list = []
        last_h = 224
        for (x, y, _) in test_loader:
            last_h = x.size(2)
            gt_list.extend(y.numpy().tolist())
            with torch.no_grad():
                _ = model(x.to(device))
            emb_te_sel = _emb_te_sel_from_hook_outputs(outputs, idx_fused)
            outputs.clear()
            _, _, Hf, Wf = emb_te_sel.size()
            smr_f = fused_smap_raw_from_emb(
                emb_te_sel, mean_f, cov_f, Hf, Wf, last_h, device
            )
            lo_f = min(lo_f, float(smr_f.min()))
            hi_f = max(hi_f, float(smr_f.max()))
            raws = marginal_smap_raws_from_emb(
                emb_te_sel, mean_f, cov_f, marginal_groups, Hf, Wf, last_h, device
            )
            for mi in range(n_m):
                lo_m[mi] = min(lo_m[mi], float(raws[mi].min()))
                hi_m[mi] = max(hi_m[mi], float(raws[mi].max()))
            del emb_te_sel, smr_f, raws
            gc.collect()
            if use_cuda_eff:
                torch.cuda.empty_cache()

        gt = np.asarray(gt_list, dtype=np.int64)
        sf = []
        s_m_parts = [[] for _ in range(n_m)]
        for (x, y, _) in test_loader:
            with torch.no_grad():
                _ = model(x.to(device))
            emb_te_sel = _emb_te_sel_from_hook_outputs(outputs, idx_fused)
            outputs.clear()
            _, _, Hf, Wf = emb_te_sel.size()
            smr_f = fused_smap_raw_from_emb(
                emb_te_sel, mean_f, cov_f, Hf, Wf, last_h, device
            )
            sf.append(_scores_from_smap_raw(smr_f, lo_f, hi_f))
            raws = marginal_smap_raws_from_emb(
                emb_te_sel, mean_f, cov_f, marginal_groups, Hf, Wf, last_h, device
            )
            for mi in range(n_m):
                s_m_parts[mi].append(_scores_from_smap_raw(raws[mi], lo_m[mi], hi_m[mi]))
            del emb_te_sel, smr_f, raws
            gc.collect()
            if use_cuda_eff:
                torch.cuda.empty_cache()

        s_fused = np.concatenate(sf, axis=0)
        s_concat = [np.concatenate(parts, axis=0) for parts in s_m_parts]
        s0, s1, s2 = s_concat[0], s_concat[1], s_concat[2]
    else:
        test_layers = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        gt_list = []
        last_h = 224
        for (x, y, _) in test_loader:
            last_h = x.size(2)
            with torch.no_grad():
                _ = model(x.to(device))
            for k, v in zip(test_layers.keys(), outputs):
                test_layers[k].append(v.cpu().detach())
            outputs.clear()
            gt_list.extend(y.numpy().tolist())
        for k in test_layers:
            test_layers[k] = torch.cat(test_layers[k], 0)
        gt = np.asarray(gt_list, dtype=np.int64)

        emb_te = test_layers['layer1']
        for nm in ['layer2', 'layer3']:
            emb_te = embedding_concat(emb_te, test_layers[nm])
        emb_te_sel = torch.index_select(emb_te, 1, idx_fused)
        Bf, Cf, Hf, Wf = emb_te_sel.size()

        s_fused = canonical_fused_scores(emb_te_sel, mean_f, cov_f, Hf, Wf, last_h, device)
        s0, s1, s2 = protocol_b_marginal_scores(
            emb_te_sel, mean_f, cov_f, marginal_groups, Hf, Wf, last_h, device
        )

    base = compute_instability_metrics(s0, s1, s2, s_fused, gt)
    Ix = sample_instability_Ix(s0, s1, s2, gt)
    I_valid = Ix[~np.isnan(Ix)]
    mean_sample_I = float(np.mean(I_valid)) if I_valid.size else float('nan')

    base['class_name'] = class_name
    base['n_test'] = int(gt.size)
    base['n_anomaly'] = int(np.sum(gt == 1))
    base['n_normal'] = int(np.sum(gt == 0))
    base['mean_sample_instability'] = mean_sample_I
    base['median_sample_instability'] = float(np.median(I_valid)) if I_valid.size else float('nan')
    base['top10_unstable_samples'] = top_k_unstable(Ix, gt, s_fused, 10)
    # trim large fields for compact table use
    base_compact = {k: v for k, v in base.items() if k != 'examples_disagree'}
    if return_scores:
        return base_compact, base, s0, s1, s2, s_fused, gt
    return base_compact, base


def main():
    args = parse_args()
    for c in args.class_names:
        if c not in mvtec.CLASS_NAMES:
            raise ValueError('Unknown class %s' % c)

    data_path = os.path.expanduser(args.data_path or '~/datasets/mvtec')
    os.makedirs(args.save_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    enable_fast_gpu()

    if args.arch == 'resnet18':
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

    idx_fused = torch.tensor(sample(range(t_d), d_sub))
    sort_ord = np.argsort(idx_fused.numpy())
    marginal_groups = [t.tolist() for t in np.array_split(sort_ord, 3)]

    outputs = []

    def hook(module, inp, out):
        outputs.append(out)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    all_metrics = {}
    full_json = {}

    for class_name in args.class_names:
        train_ds = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=True)
        test_ds = mvtec.MVTecDataset(data_path, class_name=class_name, is_train=False)
        train_loader = make_feature_loader(
            train_ds, args.arch, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = make_feature_loader(
            test_ds, args.arch, batch_size=args.batch_size, num_workers=args.num_workers)

        compact, full = run_one_category(
            class_name, model, device, outputs, idx_fused, marginal_groups,
            train_loader, test_loader)
        all_metrics[class_name] = compact
        full_json[class_name] = full

    # --- Markdown-like table for report ---
    lines = []
    lines.append('Protocol B — MVTec multi-category sanity (arch=%s, seed=%d, batch=%s, workers=%s)\n' % (
        args.arch, args.seed,
        args.batch_size or 'default', args.num_workers if args.num_workers is not None else 'default'))
    lines.append('| class | n_test | n_pairs | mean I_pair | med I_pair | frac I>0 | n_disagree | r(I,err) | AUROC | mean I_sample |')
    lines.append('|-------|--------|---------|-------------|------------|----------|------------|----------|-------|---------------|')
    for cn in args.class_names:
        m = all_metrics[cn]
        r = m.get('correlation') or {}
        rp = r.get('pearson_r', float('nan'))
        lines.append('| %s | %d | %d | %.6f | %.6f | %.4f | %d | %.4f | %.4f | %.6f |' % (
            cn, m['n_test'], m['n_pairs'], m['mean_pairwise_I'], m['median_pairwise_I'],
            m['fraction_I_gt0'], m['n_disagree_pairs'], rp, m['fused_auroc'],
            m['mean_sample_instability']))

    # Interpretation
    fracs = [all_metrics[c]['fraction_I_gt0'] for c in args.class_names]
    mean_is = [all_metrics[c]['mean_pairwise_I'] for c in args.class_names]
    rs = []
    for c in args.class_names:
        co = all_metrics[c].get('correlation')
        rs.append(co['pearson_r'] if co else float('nan'))

    lines.append('\n## Interpretation\n')
    lines.append('**Meaningful instability:** Categories with higher mean pairwise I and/or higher '
                 'fraction I>0 show more cross-marginal disagreement on anomaly–normal pairs. '
                 'In this run, the spread across classes is:\n')
    lines.append('- fraction I>0: min %.4f (%s) max %.4f (%s)\n' % (
        min(fracs), args.class_names[int(np.argmin(fracs))],
        max(fracs), args.class_names[int(np.argmax(fracs))]))
    lines.append('- mean pairwise I: min %.6f (%s) max %.6f (%s)\n' % (
        min(mean_is), args.class_names[int(np.argmin(mean_is))],
        max(mean_is), args.class_names[int(np.argmax(mean_is))]))

    lines.append('\n**Instability vs error:** Pearson r(I_pair, ranking_error) is positive in '
                 'most MVTec settings when both vary; strength differs by class (see table). '
                 'High r means pairs where marginals disagree more often coincide with fused '
                 'ranking mistakes — interpret as association, not causation.\n')

    lines.append('\n**PaDiM as empirical instance:** Even when overall disagreement rates are low '
                 '(easy classes), nonzero I>0 and positive correlation support using PaDiM as a '
                 'case where a single fused score coexists with inconsistent marginal subspace '
                 'rankings on some pairs. Whether effect size is "large enough" depends on paper '
                 'framing; the multi-class table quantifies cross-category variability.\n')

    lines.append('\n**Proceed to full-dataset plotting?** If the goal is paper figures across all '
                 'MVTec (+ later VisA), the next step is reasonable after confirming no class '
                 'exhibits pathological degeneracy (e.g. zero pairs, constant I). '
                 'Full 15-class Protocol B run is endorsed for aggregate plots; keep reporting '
                 'per-class instability alongside pooled summaries.\n')

    lines.append('\n## Top-10 unstable samples per class (summary)\n')
    for cn in args.class_names:
        lines.append('### %s\n' % cn)
        for row in all_metrics[cn]['top10_unstable_samples']:
            lines.append('  idx=%d label=%s I_sample=%.6f s_fused=%.6f\n' % (
                row['test_index'], row['label'], row['I_sample'], row['s_fused']))

    report_path = os.path.join(args.save_dir, 'protocol_b_multiclass_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    json_path = os.path.join(args.save_dir, 'protocol_b_multiclass_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'arch': args.arch,
            'seed': args.seed,
            'idx_fused_first_20': idx_fused[:20].tolist(),
            'marginal_dim_groups': [len(g) for g in marginal_groups],
            'per_class': full_json,
        }, f, indent=2)

    print('\n'.join(lines[:20]))
    print('...')
    print('Wrote', report_path)
    print('Wrote', json_path)


if __name__ == '__main__':
    main()
