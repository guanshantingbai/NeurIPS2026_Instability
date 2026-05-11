"""
PaDiM layer-wise instability sanity check (PromptAD-style protocol).
No paper figures; prints/writes metrics and text summaries only.
"""
import argparse
import json
import os
import random
from collections import OrderedDict
from random import sample

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18, wide_resnet50_2

import datasets.mvtec as mvtec
from padim_dataloader import enable_fast_gpu, make_feature_loader


def parse_args():
    p = argparse.ArgumentParser('PaDiM instability sanity')
    p.add_argument('--class_name', type=str, required=True, help='Single MVTec category, e.g. bottle')
    p.add_argument('--data_path', type=str, default=None,
                     help='MVTec root; default ~/datasets/mvtec')
    p.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    p.add_argument('--save_dir', type=str, default='./instability_sanity_out')
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    return p.parse_args()


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


def fit_gaussian_per_spatial(embedding_vectors, d_subsample, seed_offset):
    """embedding_vectors: (B, C, H, W) tensor on CPU. Returns mean (C', HW), cov (C', C', HW), idx tensor."""
    B, C, H, W = embedding_vectors.size()
    d_use = min(d_subsample, C)
    rng = random.Random(seed_offset)
    idx = torch.tensor(rng.sample(range(C), d_use), dtype=torch.long)
    emb = torch.index_select(embedding_vectors, 1, idx)
    B2, C2, H2, W2 = emb.size()
    assert H2 == H and W2 == W
    emb = emb.view(B2, C2, H * W).numpy()
    mean = np.mean(emb, axis=0)
    cov = np.zeros((C2, C2, H * W), dtype=np.float64)
    I = np.identity(C2)
    for i in range(H * W):
        cov[:, :, i] = np.cov(emb[:, :, i], rowvar=False) + 0.01 * I
    return mean, cov, idx, H, W


def mahalanobis_maps(embedding_vectors, mean, cov, idx, H_spatial, W_spatial):
    """embedding_vectors: (B,C,H,W) full layer tensor; idx selects channels (same as fit)."""
    emb = torch.index_select(embedding_vectors, 1, idx)
    B, C2, H, W = emb.size()
    assert H == H_spatial and W == W_spatial
    emb = emb.view(B, C2, H * W).numpy()
    dist_list = []
    for i in range(H * W):
        m = mean[:, i]
        invc = np.linalg.inv(cov[:, :, i])
        dist_list.append([mahalanobis(emb[b, :, i], m, invc) for b in range(B)])
    dist = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
    return dist


def postprocess_score_map(dist_bhw, crop_size):
    """Upsample, gaussian sigma=4; returns map before global norm."""
    t = torch.as_tensor(dist_bhw, dtype=torch.float32)
    smap = F.interpolate(t.unsqueeze(1), size=crop_size, mode='bilinear', align_corners=False).squeeze(1).numpy()
    for i in range(smap.shape[0]):
        smap[i] = gaussian_filter(smap[i], sigma=4)
    return smap


def global_minmax(score_map):
    lo, hi = score_map.min(), score_map.max()
    if hi - lo < 1e-12:
        return np.zeros_like(score_map)
    return (score_map - lo) / (hi - lo)


def image_level_max(norm_map):
    return norm_map.reshape(norm_map.shape[0], -1).max(axis=1)


def main():
    args = parse_args()
    if args.class_name not in mvtec.CLASS_NAMES:
        raise ValueError('class_name must be in %s' % mvtec.CLASS_NAMES)
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

    outputs = []

    def hook(module, inp, out):
        outputs.append(out)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    train_ds = mvtec.MVTecDataset(data_path, class_name=args.class_name, is_train=True)
    test_ds = mvtec.MVTecDataset(data_path, class_name=args.class_name, is_train=False)
    train_loader = make_feature_loader(
        train_ds, args.arch, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = make_feature_loader(
        test_ds, args.arch, batch_size=args.batch_size, num_workers=args.num_workers)

    train_layers = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    for (x, _, _) in train_loader:
        with torch.no_grad():
            _ = model(x.to(device))
        for k, v in zip(train_layers.keys(), outputs):
            train_layers[k].append(v.cpu().detach())
        outputs = []
    for k in train_layers:
        train_layers[k] = torch.cat(train_layers[k], 0)

    # Per-layer Gaussian (same recipe as main.py; only embedding source differs)
    layer_stats = {}
    for li, name in enumerate(['layer1', 'layer2', 'layer3']):
        mean, cov, idx_l, H_l, W_l = fit_gaussian_per_spatial(train_layers[name], d_sub, args.seed + 10 * (li + 1))
        layer_stats[name] = {'mean': mean, 'cov': cov, 'idx': idx_l, 'H': H_l, 'W': W_l}

    # Fused Gaussian (identical to main.py)
    emb_fused = train_layers['layer1']
    for nm in ['layer2', 'layer3']:
        emb_fused = embedding_concat(emb_fused, train_layers[nm])
    emb_fused = torch.index_select(emb_fused, 1, idx_fused)
    B, C, H, W = emb_fused.size()
    flat = emb_fused.view(B, C, H * W).numpy()
    mean_f = np.mean(flat, axis=0)
    cov_f = np.zeros((C, C, H * W))
    I = np.identity(C)
    for i in range(H * W):
        cov_f[:, :, i] = np.cov(flat[:, :, i], rowvar=False) + 0.01 * I

    # Test features
    test_layers = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    gt_list = []
    last_h = 224
    for (x, y, _) in test_loader:
        last_h = x.size(2)
        with torch.no_grad():
            _ = model(x.to(device))
        for k, v in zip(test_layers.keys(), outputs):
            test_layers[k].append(v.cpu().detach())
        outputs = []
        gt_list.extend(y.numpy().tolist())
    for k in test_layers:
        test_layers[k] = torch.cat(test_layers[k], 0)
    gt = np.asarray(gt_list, dtype=np.int64)
    B = test_layers['layer1'].size(0)

    # Fused distance maps -> postprocess -> global norm -> s_fused
    emb_te = test_layers['layer1']
    for nm in ['layer2', 'layer3']:
        emb_te = embedding_concat(emb_te, test_layers[nm])
    emb_te = torch.index_select(emb_te, 1, idx_fused)
    Bf, Cf, Hf, Wf = emb_te.size()
    flat_te = emb_te.view(Bf, Cf, Hf * Wf).numpy()
    dist_f = []
    for i in range(Hf * Wf):
        m = mean_f[:, i]
        invc = np.linalg.inv(cov_f[:, :, i])
        dist_f.append([mahalanobis(flat_te[b, :, i], m, invc) for b in range(Bf)])
    dist_f = np.array(dist_f).transpose(1, 0).reshape(Bf, Hf, Wf)
    smap_f_raw = postprocess_score_map(dist_f, last_h)
    smap_f = global_minmax(smap_f_raw)
    s_fused = image_level_max(smap_f)

    # Per-layer scores (same postprocess + global minmax per layer)
    s_layers = {}
    raw_for_hist = {}
    for name in ['layer1', 'layer2', 'layer3']:
        st = layer_stats[name]
        dist = mahalanobis_maps(test_layers[name], st['mean'], st['cov'], st['idx'], st['H'], st['W'])
        raw = postprocess_score_map(dist, last_h)
        norm = global_minmax(raw)
        s_layers[name] = image_level_max(norm)
        raw_for_hist[name] = s_layers[name].copy()

    s_l1 = s_layers['layer1']
    s_l2 = s_layers['layer2']
    s_l3 = s_layers['layer3']

    idx_pos = np.where(gt == 1)[0]
    idx_neg = np.where(gt == 0)[0]
    pairs = []
    for i in idx_pos:
        for j in idx_neg:
            pairs.append((i, j))

    if len(pairs) == 0:
        raise RuntimeError('Need both anomaly and normal test images for pairing.')

    I_pairs = []
    errs = []
    disagree_pool = []

    for (i, j) in pairs:
        z1 = 1 if s_l1[i] > s_l1[j] else 0
        z2 = 1 if s_l2[i] > s_l2[j] else 0
        z3 = 1 if s_l3[i] > s_l3[j] else 0
        zs = np.array([z1, z2, z3], dtype=np.float64)
        I_pair = float(np.var(zs, ddof=0))
        I_pairs.append(I_pair)
        err = 1.0 if s_fused[i] <= s_fused[j] else 0.0
        errs.append(err)
        if I_pair > 1e-12:
            disagree_pool.append({
                'anomaly_idx': int(i),
                'normal_idx': int(j),
                'z_l1': int(z1),
                'z_l2': int(z2),
                'z_l3': int(z3),
                'I_pair': I_pair,
                's_l1_anom': float(s_l1[i]),
                's_l1_norm': float(s_l1[j]),
                's_l2_anom': float(s_l2[i]),
                's_l2_norm': float(s_l2[j]),
                's_l3_anom': float(s_l3[i]),
                's_l3_norm': float(s_l3[j]),
                's_fused_anom': float(s_fused[i]),
                's_fused_norm': float(s_fused[j]),
                'ranking_error_fused': int(err),
            })

    disagree_examples = disagree_pool[:20]

    I_pairs = np.array(I_pairs)
    errs = np.array(errs)

    # Sample instability I(x)
    I_sample = np.zeros(B, dtype=np.float64)
    for x_idx in range(B):
        if gt[x_idx] == 1:
            others = idx_neg
        else:
            others = idx_pos
        if len(others) == 0:
            I_sample[x_idx] = np.nan
            continue
        acc = []
        for x_other in others:
            if gt[x_idx] == 1:
                i, j = x_idx, x_other
            else:
                i, j = x_other, x_idx
            z1 = 1 if s_l1[i] > s_l1[j] else 0
            z2 = 1 if s_l2[i] > s_l2[j] else 0
            z3 = 1 if s_l3[i] > s_l3[j] else 0
            acc.append(np.var([z1, z2, z3], ddof=0))
        I_sample[x_idx] = float(np.mean(acc))

    I_valid = I_sample[~np.isnan(I_sample)]
    mean_I_sample = float(np.mean(I_valid)) if I_valid.size else float('nan')
    median_I_sample = float(np.median(I_valid)) if I_valid.size else float('nan')

    mean_I = float(np.mean(I_pairs))
    median_I = float(np.median(I_pairs))
    frac_gt0 = float(np.mean(I_pairs > 1e-12))

    if np.std(I_pairs) < 1e-15 or np.std(errs) < 1e-15:
        corr_note = 'undefined (constant I or constant error)'
        corr_val = None
    else:
        r, p = pearsonr(I_pairs, errs)
        if np.isnan(r):
            corr_note = 'undefined (pearson nan)'
            corr_val = None
        else:
            corr_val = {'pearson_r': float(r), 'p_value': float(p)}
            corr_note = 'pearson_r=%.4f p=%.4g' % (r, p)

    def hist_text_simple(name, arr):
        arr = np.asarray(arr, dtype=np.float64).ravel()
        hist, edges = np.histogram(arr, bins=10)
        lines = [
            '%s: n=%d mean=%.6f std=%.6f' % (name, arr.size, float(arr.mean()), float(arr.std())),
            '  percentiles p0,p25,p50,p75,p100: ' + ', '.join(
                '%.6f' % q for q in np.percentile(arr, [0, 25, 50, 75, 100])),
            '  histogram (10 equal-width bins):',
        ]
        for bi in range(len(hist)):
            lines.append('    [%.6f, %.6f): %d' % (edges[bi], edges[bi + 1], hist[bi]))
        return '\n'.join(lines)

    report_lines = [
        'PaDiM instability sanity check',
        'class=%s arch=%s seed=%d' % (args.class_name, args.arch, args.seed),
        'n_test=%d n_anomaly=%d n_normal=%d n_pairs=%d' % (B, len(idx_pos), len(idx_neg), len(pairs)),
        '',
        '--- Step 6 metrics ---',
        '1. mean pairwise instability: %.8f' % mean_I,
        '2. median pairwise instability: %.8f' % median_I,
        '3. fraction I>0 (pairwise): %.8f' % frac_gt0,
        '4. correlation(instability, ranking_error): %s' % corr_note,
        '',
        '(Step 5) sample instability I(x): mean=%.8f median=%.8f' % (mean_I_sample, median_I_sample),
        '',
        'Canonical fused image-level AUROC (sanity): %.6f' % roc_auc_score(gt, s_fused),
        '',
        '--- Example pairs with layer disagreement (I_pair>0), up to 20 ---',
        '(total disagreeing pairs in dataset: %d)' % len(disagree_pool),
    ]
    report_lines.append(json.dumps(disagree_examples, indent=2))
    report_lines.extend([
        '',
        '--- Distributions of s_l1, s_l2, s_l3 (image-level, after same norm as canonical per layer) ---',
        hist_text_simple('s_l1', s_l1),
        '',
        hist_text_simple('s_l2', s_l2),
        '',
        hist_text_simple('s_l3', s_l3),
        '',
        '--- s_fused (canonical) ---',
        hist_text_simple('s_fused', s_fused),
    ])

    report_path = os.path.join(args.save_dir, 'sanity_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    metrics_path = os.path.join(args.save_dir, 'sanity_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'class_name': args.class_name,
            'arch': args.arch,
            'mean_pairwise_instability': mean_I,
            'median_pairwise_instability': median_I,
            'fraction_I_gt_0': frac_gt0,
            'correlation': corr_val,
            'n_pairs': len(pairs),
            'n_disagree_pairs': len(disagree_pool),
            'mean_sample_instability_Ix': mean_I_sample,
            'median_sample_instability_Ix': median_I_sample,
            'image_auroc_fused': float(roc_auc_score(gt, s_fused)),
        }, f, indent=2)

    print('\n'.join(report_lines[:25]))
    print('...')
    print('Wrote', report_path)
    print('Wrote', metrics_path)


if __name__ == '__main__':
    main()
