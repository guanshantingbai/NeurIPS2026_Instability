#!/usr/bin/env python3
"""
Verify: instability_aware_harmonic_fuse_scores(..., alpha=0) == harmonic_mean_fuse_scores(...).

Strategy A uses this identity; test_cls only enters Strategy A when alpha > 0,
so end-to-end parity with "pure harmonic i_roc" requires either alpha=0 on that path
(see --instability-fusion-alpha) or turning off Strategy A and using lambda=0 correction.

Run from PromptAD repo root:
  python utils/verify_instability_fusion_alpha0.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils.metrics import harmonic_mean_fuse_scores, instability_aware_harmonic_fuse_scores


def main() -> None:
    rng = np.random.default_rng(0)
    max_abs_err = 0.0
    for trial in range(500):
        s_sem = rng.uniform(0.02, 0.98, size=200).astype(np.float64)
        s_vis = rng.uniform(0.02, 0.98, size=200).astype(np.float64)
        h = harmonic_mean_fuse_scores(s_sem, s_vis)
        g = instability_aware_harmonic_fuse_scores(s_sem, s_vis, 0.0)
        d = float(np.max(np.abs(h - g)))
        max_abs_err = max(max_abs_err, d)
        if d > 1e-9:
            raise SystemExit(f"Mismatch trial {trial}: max_abs_diff={d}")

    # 2D visual maps (max-pool path)
    maps = rng.uniform(0.02, 0.98, size=(50, 17, 17)).astype(np.float64)
    sem = rng.uniform(0.02, 0.98, size=50).astype(np.float64)
    h2 = harmonic_mean_fuse_scores(sem, maps)
    g2 = instability_aware_harmonic_fuse_scores(sem, maps, 0.0)
    d2 = float(np.max(np.abs(h2 - g2)))
    if d2 > 1e-9:
        raise SystemExit(f"2D map path mismatch: max_abs_diff={d2}")

    print("OK: alpha=0 fusion matches PromptAD harmonic_mean_fuse_scores (vector + 2D maps).")
    print(f"    max_abs_error over 500 random trials (N=200): {max_abs_err:.3e}")
    print(f"    max_abs_error 2D maps trial: {d2:.3e}")


if __name__ == "__main__":
    main()
