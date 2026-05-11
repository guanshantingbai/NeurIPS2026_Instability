import argparse
import csv
import math
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score


def harmonic_mean_fuse_scores(semantic_scores, visual_scores, eps: float = 1e-12):
    semantic_scores = np.asarray(semantic_scores, dtype=float).reshape(-1)
    visual_scores = np.asarray(visual_scores, dtype=float).reshape(-1)
    return 2.0 / (1.0 / (visual_scores + eps) + 1.0 / (semantic_scores + eps))


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")


def parse_label(v: str) -> int:
    s = str(v).strip().lower()
    if s in ("1", "anomaly", "abnormal", "true"):
        return 1
    if s in ("0", "normal", "good", "false"):
        return 0
    raise ValueError(f"Unsupported label value: {v}")


def main():
    parser = argparse.ArgumentParser(description="Validate AUROC from per-sample CSV table")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--score-col", type=str, default="harmonic_score")
    parser.add_argument("--label-col", type=str, default="image_label")
    parser.add_argument("--check-fusion", type=str2bool, default=True)
    parser.add_argument("--semantic-col", type=str, default="semantic_score")
    parser.add_argument("--visual-col", type=str, default="visual_score")
    parser.add_argument("--tol", type=float, default=1e-8, help="Absolute tolerance for fusion consistency check")
    args = parser.parse_args()

    labels: List[int] = []
    scores: List[float] = []
    semantic_scores: List[float] = []
    visual_scores: List[float] = []

    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(parse_label(row[args.label_col]))
            scores.append(float(row[args.score_col]))
            if args.check_fusion:
                semantic_scores.append(float(row[args.semantic_col]))
                visual_scores.append(float(row[args.visual_col]))

    labels_np = np.asarray(labels, dtype=int)
    scores_np = np.asarray(scores, dtype=float)

    auroc = roc_auc_score(labels_np, scores_np) * 100.0
    print(f"Samples: {len(labels_np)}")
    print(f"AUROC ({args.score_col}): {auroc:.6f}")

    if args.check_fusion:
        recomputed = harmonic_mean_fuse_scores(semantic_scores, visual_scores)
        max_abs_err = np.max(np.abs(recomputed - scores_np))
        ok = max_abs_err <= args.tol or math.isclose(max_abs_err, 0.0, abs_tol=args.tol)
        print(f"Fusion consistency max_abs_err: {max_abs_err:.12g}")
        print(f"Fusion consistency: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()

