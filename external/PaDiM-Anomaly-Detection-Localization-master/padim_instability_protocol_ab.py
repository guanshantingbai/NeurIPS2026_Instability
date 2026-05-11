"""
Protocol A vs B comparison for PaDiM instability (single category, no figures).

Protocol A (current): separate Gaussian per native backbone layer (l1,l2,l3).
Protocol B (shared reference): ONE fused Gaussian (same as canonical PaDiM);
three scores = marginal Mahalanobis on a partition of the SAME selected dims.

See protocol_audit section in output for why B cannot be literal l1/l2/l3 after fold.
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
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18, wide_resnet50_2

import datasets.mvtec as mvtec
from padim_dataloader import enable_fast_gpu, make_feature_loader


def _batch_mahalanobis(delta: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Mahalanobis norm for each row of delta; matches scipy per-row but vectorized."""
    quad = np.sum((delta @ inv_cov) * delta, axis=1)
    return np.sqrt(np.maximum(quad, 0.0))


PROTOCOL_AUDIT_TEXT = """
=== TASK 1 — PROTOCOL AUDIT (written) ===

Q1: Does fitting separate mean/cov per layer make the three scores behave like
three different detectors rather than three views of one detector?
A1: Yes. Each native layer has its own channel dimension and spatial layout before
the shared PaDiM pipeline; a separate Gaussian N(mu_l, Sigma_l) on normal training
data is a distinct statistical test in a distinct representation. They answer
"is this image extreme under layer-l's marginal normal model?" — not "what is
layer l's opinion within the single fused PaDiM patch model?". The canonical
PaDiM paper model is ONE joint Gaussian on the concatenated (fold-mixed) patch
vector (after multi-scale fusion), not three independent patch models.

Q2: If yes, why would this reduce disagreement?
A2: Separate fitting can align each score with the same downstream goal (anomaly
vs normal) so that all three often rank x+ above x- on easy categories (bottle).
Strong per-layer discriminability → correlated binary z_l → low Var(z). This is
not "artificial suppression" in a bug sense, but it is a different estimand than
"views of the fused detector": you are measuring agreement among three
co-trained but separately calibrated detectors.

Q3: What protocol better matches "multiple equivalent views of the same detector"?
A3: Use a SINGLE reference distribution — the fused PaDiM Gaussian on the
selected subspace — and define "views" as different functionals of the SAME
latent z (same mu, same Sigma). The mathematically clean split is marginal
Mahalanobis on disjoint subsets of coordinates: (z_S - mu_S)^T Sigma_{SS}^{-1}
(...), all derived from the joint (mu, Sigma) of the fused embedding.

Caveat (important): After PaDiM's embedding_concat + fold, output channels are
linear mixtures of spatial windows; you cannot map final 448 channels to pure
ResNet layer1/2/3 exclusively. Therefore Protocol B implements three marginals
of the SHARED fused (mu, Sigma) by partitioning the SELECTED dimension indices
into three disjoint groups (here: sort by original channel id, then split into
thirds). These are "coarse subspace views" of the SAME detector, not literal
layer1/2/3. Naming in JSON: view_M0, view_M1, view_M2 (marginal tertiles).

"""


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
    B, C, H, W = embedding_vectors.size()
    d_use = min(d_subsample, C)
    rng = random.Random(seed_offset)
    idx = torch.tensor(rng.sample(range(C), d_use), dtype=torch.long)
    emb = torch.index_select(embedding_vectors, 1, idx)
    B2, C2, H2, W2 = emb.size()
    emb = emb.view(B2, C2, H * W).numpy()
    mean = np.mean(emb, axis=0)
    cov = np.zeros((C2, C2, H * W), dtype=np.float64)
    I = np.identity(C2)
    for i in range(H * W):
        cov[:, :, i] = np.cov(emb[:, :, i], rowvar=False) + 0.01 * I
    return mean, cov, idx, H, W


def mahalanobis_maps(embedding_vectors, mean, cov, idx, H_spatial, W_spatial):
    emb = torch.index_select(embedding_vectors, 1, idx)
    B, C2, H, W = emb.size()
    emb = emb.view(B, C2, H * W).numpy()
    cols = []
    for i in range(H * W):
        m = mean[:, i]
        invc = np.linalg.inv(cov[:, :, i])
        delta = emb[:, :, i] - m
        cols.append(_batch_mahalanobis(delta, invc))
    dist = np.stack(cols, axis=1).reshape(B, H, W)
    return dist


def marginal_mahalanobis_maps(emb_selected_bchw, mean, cov, dim_subset, H, W):
    """
    emb_selected: (B, d, H, W) already index-selected fused embedding.
    mean: (d, H*W), cov: (d, d, H*W)
    dim_subset: list of indices in [0, d)
    """
    S = list(dim_subset)
    if len(S) == 0:
        raise ValueError('empty marginal subset')
    B, d, _, _ = emb_selected_bchw.size()
    emb = emb_selected_bchw.view(B, d, H * W).numpy()
    cols = []
    for p in range(H * W):
        m = mean[S, p]
        c = cov[S][:, S, p]
        invc = np.linalg.inv(c)
        delta = emb[:, S, p] - m
        cols.append(_batch_mahalanobis(delta, invc))
    dist = np.stack(cols, axis=1).reshape(B, H, W)
    return dist


def postprocess_score_map(dist_bhw, crop_size):
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


def compute_instability_metrics(s_a, s_b, s_c, s_fused, gt):
    """s_* : (n,) image-level scores; gt binary 0/1."""
    idx_pos = np.where(gt == 1)[0]
    idx_neg = np.where(gt == 0)[0]
    pairs = [(i, j) for i in idx_pos for j in idx_neg]
    I_pairs = []
    errs = []
    disagree = []
    for (i, j) in pairs:
        z1 = 1 if s_a[i] > s_a[j] else 0
        z2 = 1 if s_b[i] > s_b[j] else 0
        z3 = 1 if s_c[i] > s_c[j] else 0
        zs = np.array([z1, z2, z3], dtype=np.float64)
        Ip = float(np.var(zs, ddof=0))
        I_pairs.append(Ip)
        err = 1.0 if s_fused[i] <= s_fused[j] else 0.0
        errs.append(err)
        if Ip > 1e-12:
            disagree.append({
                'anomaly_idx': int(i), 'normal_idx': int(j),
                'z': [int(z1), int(z2), int(z3)], 'I_pair': Ip,
                'ranking_error_fused': int(err),
            })
    I_pairs = np.array(I_pairs)
    errs = np.array(errs)
    mean_I = float(np.mean(I_pairs))
    median_I = float(np.median(I_pairs))
    frac = float(np.mean(I_pairs > 1e-12))
    if np.std(I_pairs) < 1e-15 or np.std(errs) < 1e-15:
        corr = None
    else:
        r, p = pearsonr(I_pairs, errs)
        corr = {'pearson_r': float(r), 'p_value': float(p)} if not np.isnan(r) else None
    auroc = float(roc_auc_score(gt, s_fused))
    return {
        'mean_pairwise_I': mean_I,
        'median_pairwise_I': median_I,
        'fraction_I_gt0': frac,
        'correlation': corr,
        'fused_auroc': auroc,
        'n_disagree_pairs': len(disagree),
        'examples_disagree': disagree[:20],
        'n_pairs': len(pairs),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--class_name', type=str, required=True)
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    p.add_argument('--save_dir', type=str, default='./protocol_ab_out')
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.class_name not in mvtec.CLASS_NAMES:
        raise ValueError('invalid class')
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
    # Protocol B: partition selected dims by ascending original channel id (transparent)
    sort_ord = np.argsort(idx_fused.numpy())
    thirds = np.array_split(sort_ord, 3)
    marginal_groups = [t.tolist() for t in thirds]

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

    # --- Protocol A: per-layer fit ---
    layer_stats = {}
    for li, name in enumerate(['layer1', 'layer2', 'layer3']):
        mean, cov, idx_l, H_l, W_l = fit_gaussian_per_spatial(
            train_layers[name], d_sub, args.seed + 10 * (li + 1))
        layer_stats[name] = {'mean': mean, 'cov': cov, 'idx': idx_l, 'H': H_l, 'W': W_l}

    # --- Fused train (canonical + Protocol B reference) ---
    emb_fused = train_layers['layer1']
    for nm in ['layer2', 'layer3']:
        emb_fused = embedding_concat(emb_fused, train_layers[nm])
    emb_sel = torch.index_select(emb_fused, 1, idx_fused)
    B, C, H, W = emb_sel.size()
    flat = emb_sel.view(B, C, H * W).numpy()
    mean_f = np.mean(flat, axis=0)
    cov_f = np.zeros((C, C, H * W))
    I = np.identity(C)
    for i in range(H * W):
        cov_f[:, :, i] = np.cov(flat[:, :, i], rowvar=False) + 0.01 * I

    # --- Test ---
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

    # Fused test embedding (selected)
    emb_te = test_layers['layer1']
    for nm in ['layer2', 'layer3']:
        emb_te = embedding_concat(emb_te, test_layers[nm])
    emb_te_sel = torch.index_select(emb_te, 1, idx_fused)

    # Canonical s_fused (same as main.py)
    Bf, Cf, Hf, Wf = emb_te_sel.size()
    flat_te = emb_te_sel.view(Bf, Cf, Hf * Wf).numpy()
    cols_f = []
    for i in range(Hf * Wf):
        m = mean_f[:, i]
        invc = np.linalg.inv(cov_f[:, :, i])
        delta = flat_te[:, :, i] - m
        cols_f.append(_batch_mahalanobis(delta, invc))
    dist_f = np.stack(cols_f, axis=1).reshape(Bf, Hf, Wf)
    smap_f_raw = postprocess_score_map(dist_f, last_h)
    smap_f = global_minmax(smap_f_raw)
    s_fused = image_level_max(smap_f)

    # Protocol A layer scores
    sA = []
    for name in ['layer1', 'layer2', 'layer3']:
        st = layer_stats[name]
        dist = mahalanobis_maps(test_layers[name], st['mean'], st['cov'], st['idx'], st['H'], st['W'])
        raw = postprocess_score_map(dist, last_h)
        norm = global_minmax(raw)
        sA.append(image_level_max(norm))

    # Protocol B: three marginals of SAME mean_f, cov_f
    sB = []
    for gi, group in enumerate(marginal_groups):
        dist_m = marginal_mahalanobis_maps(emb_te_sel, mean_f, cov_f, group, Hf, Wf)
        raw = postprocess_score_map(dist_m, last_h)
        norm = global_minmax(raw)
        sB.append(image_level_max(norm))

    metrics_A = compute_instability_metrics(sA[0], sA[1], sA[2], s_fused, gt)
    metrics_B = compute_instability_metrics(sB[0], sB[1], sB[2], s_fused, gt)

    interpretation = """
=== TASK 3 — SHORT INTERPRETATION ===
Protocol A operationalizes "three native feature maps" with three independent
normal models — strong for ablation, weak for "views of the fused PaDiM detector".
Protocol B operationalizes "three subspace readings of the SAME fused Gaussian"
(marginal tertiles in channel-index order). It is the closest standard-object
approximation when literal l1/l2/l3 cannot be recovered after fold.

=== TASK 4 — RECOMMENDATION ===
- For claims aligned with the paper sentence about the fused PaDiM model, prefer
  Protocol B (or explicitly report both).
- Keep Protocol A as a secondary analysis: "agreement among separately calibrated
  layer detectors" — different estimand.
- If the paper text insists on naming l1,l2,l3, either (i) clarify that
  architectural layers are not identifiable in the fused 448-d after fold, and
  use Protocol B with renamed marginal views, or (ii) avoid fold-level views and
  instead define scores on pre-fold tensors (much heavier; changes spatial grid).

Do not scale to all categories until this estimand choice is fixed in prose.
"""

    out_path = os.path.join(args.save_dir, 'protocol_ab_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(PROTOCOL_AUDIT_TEXT)
        f.write('\n=== PROTOCOL B PARTITION (sorted channel idx, thirds) ===\n')
        f.write('idx_fused (first 20): %s\n' % idx_fused[:20].tolist())
        for i, g in enumerate(marginal_groups):
            orig_ch = idx_fused[g].numpy().tolist() if len(g) else []
            f.write('M%d size=%d orig_channels sample: %s\n' % (i, len(g), orig_ch[:15]))
        f.write('\n=== PROTOCOL A METRICS ===\n')
        f.write(json.dumps(metrics_A, indent=2, default=str))
        f.write('\n\n=== PROTOCOL B METRICS ===\n')
        f.write(json.dumps(metrics_B, indent=2, default=str))
        f.write(interpretation)

    json_path = os.path.join(args.save_dir, 'protocol_ab_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'protocol_A': metrics_A, 'protocol_B': metrics_B,
                   'marginal_dim_groups': [len(g) for g in marginal_groups]}, f, indent=2)

    print(PROTOCOL_AUDIT_TEXT[:800], '...\n')
    print('Protocol A', {k: metrics_A[k] for k in ['mean_pairwise_I', 'fraction_I_gt0', 'fused_auroc', 'n_disagree_pairs']})
    print('Protocol B', {k: metrics_B[k] for k in ['mean_pairwise_I', 'fraction_I_gt0', 'fused_auroc', 'n_disagree_pairs']})
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
