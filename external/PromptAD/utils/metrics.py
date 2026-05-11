import numpy as np
import os
import csv
from skimage import measure
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def calculate_max_f1(gt, scores):
    precision, recall, thresholds = precision_recall_curve(gt, scores)
    a = 2 * precision * recall
    b = precision + recall
    f1s = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    index = np.argmax(f1s)
    max_f1 = f1s[index]
    threshold = thresholds[index]
    return max_f1, threshold


def metric_cal_img(semantic_scores=None, gt_list=None, visual_scores=None):
    # calculate image-level ROC AUC score
    img_scores = harmonic_mean_fuse_scores(semantic_scores, visual_scores)
    # harmonic mean

    gt_list = np.asarray(gt_list, dtype=int)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)

    result_dict = {'i_roc': img_roc_auc * 100}

    return result_dict


def harmonic_mean_fuse_scores(semantic_scores, visual_scores, eps: float = 1e-12):
    """
    Fuse semantic and visual (memory-bank) anomaly scores using the same
    harmonic-mean rule as `metric_cal_img`:

      fused = 2 / (1/(visual+eps) + 1/(semantic+eps))
    """
    semantic_scores = np.asarray(semantic_scores, dtype=float)
    visual_scores = np.asarray(visual_scores, dtype=float)
    # Accept visual scores as either per-image scalars (N,) or score maps (N,H,W).
    if visual_scores.ndim > 1:
        visual_scores = visual_scores.reshape(visual_scores.shape[0], -1).max(axis=1)
    semantic_scores = semantic_scores.reshape(-1)
    visual_scores = visual_scores.reshape(-1)
    return 2.0 / (1.0 / (visual_scores + eps) + 1.0 / (semantic_scores + eps))


def instability_aware_harmonic_fuse_scores(
    semantic_scores,
    visual_scores,
    alpha: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Strategy A (inference): weighted harmonic fusion with instability-dependent weights.

      u = |s_sem - s_vis|
      w = 1 / (1 + α u)
      1/s_new = w/s_sem + (1-w)/s_vis
      s_new = 1 / ( w/s_sem + (1-w)/s_vis )

    PromptAD baseline is symmetric harmonic with equal weights::

      s_base = 2/(1/s_sem + 1/s_vis) = 1/(0.5/s_sem + 0.5/s_vis)

    So α=0 recovers the same fusion as ``harmonic_mean_fuse_scores``.
    """
    if float(alpha) == 0.0:
        return harmonic_mean_fuse_scores(semantic_scores, visual_scores, eps=eps)
    semantic_scores = np.asarray(semantic_scores, dtype=float)
    visual_scores = np.asarray(visual_scores, dtype=float)
    if visual_scores.ndim > 1:
        visual_scores = visual_scores.reshape(visual_scores.shape[0], -1).max(axis=1)
    semantic_scores = semantic_scores.reshape(-1)
    visual_scores = visual_scores.reshape(-1)
    u = np.abs(semantic_scores - visual_scores)
    w = 1.0 / (1.0 + float(alpha) * u)
    denom = w / (semantic_scores + eps) + (1.0 - w) / (visual_scores + eps)
    return (1.0 / denom).astype(np.float64, copy=False)


def save_per_sample_score_table(
    *,
    semantic_scores,
    visual_scores,
    fused_scores=None,
    instability_fusion_scores=None,
    image_paths=None,
    labels=None,
    save_path: str,
    label_names: tuple[str, str] = ("normal", "anomaly"),
):
    """
    Save a per-sample table with:
      1) image path
      2) image label (normal/anomaly)
      3) semantic branch anomaly score
      4) visual (memory-bank) branch anomaly score
      5) harmonic-mean fused anomaly score (``harmonic_score`` = PromptAD s_final baseline)
      6) optional ``s_final_instability_fusion`` when ``instability_fusion_scores`` is set

    Output format: CSV.
    """
    if save_path is None or str(save_path).strip() == "":
        raise ValueError("save_path must be a non-empty path")

    semantic_scores = np.asarray(semantic_scores, dtype=float).reshape(-1)
    visual_scores = np.asarray(visual_scores, dtype=float)
    # For per-sample table, "visual_score" should be a per-image scalar.
    if visual_scores.ndim > 1:
        visual_scores = visual_scores.reshape(visual_scores.shape[0], -1).max(axis=1)
    visual_scores = visual_scores.reshape(-1)

    if fused_scores is None:
        fused_scores = harmonic_mean_fuse_scores(semantic_scores, visual_scores)
    else:
        fused_scores = np.asarray(fused_scores, dtype=float).reshape(-1)

    if image_paths is None:
        image_paths = [""] * len(semantic_scores)
    if labels is None:
        labels = [-1] * len(semantic_scores)

    labels_arr = np.asarray(labels).reshape(-1)

    n = len(semantic_scores)
    if len(visual_scores) != n or len(fused_scores) != n:
        raise ValueError("scores length mismatch")
    if len(image_paths) != n:
        raise ValueError(f"image_paths ({len(image_paths)}) and scores ({n}) length mismatch")
    if len(labels_arr) != n:
        raise ValueError(f"labels ({len(labels_arr)}) and scores ({n}) length mismatch")

    fusion_extra = None
    if instability_fusion_scores is not None:
        fusion_extra = np.asarray(instability_fusion_scores, dtype=float).reshape(-1)
        if len(fusion_extra) != n:
            raise ValueError("instability_fusion_scores length mismatch")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    header = [
        "image_path",
        "image_label",
        "semantic_score",
        "visual_score",
        "harmonic_score",
    ]
    if fusion_extra is not None:
        header.extend(["s_final_baseline", "s_final_instability_fusion"])

    normal_name, anomaly_name = label_names
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (s_sem, s_vis, s_fus, p, y) in enumerate(
            zip(semantic_scores, visual_scores, fused_scores, image_paths, labels_arr)
        ):
            y_int = int(y) if y in (0, 1, True, False) else int(y) if str(y).isdigit() else -1
            y_name = anomaly_name if y_int == 1 else normal_name if y_int == 0 else "unknown"
            row = [str(p), y_name, float(s_sem), float(s_vis), float(s_fus)]
            if fusion_extra is not None:
                row.append(float(s_fus))
                row.append(float(fusion_extra[i]))
            writer.writerow(row)

    return save_path


def metric_cal_pix(map_scores, gt_mask_list):

    gt_mask = np.asarray(gt_mask_list, dtype=int)

    # calculate per-pixel level ROCAUC
    # fpr, tpr, _ = roc_curve(gt_mask.flatten(), map_scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), map_scores.flatten())

    # pro_auc_score = cal_pro_metric(gt_mask_list, map_scores, fpr_thresh=0.3)

    result_dict = {'p_roc': per_pixel_rocauc * 100}

    return result_dict


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score

def calculate_max_f1_region(labeled_imgs, score_imgs, pro_thresh=0.6, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    # labeled_imgs[labeled_imgs <= 0.1] = 0
    # labeled_imgs[labeled_imgs > 0.1] = 1
    labeled_imgs = labeled_imgs.astype(bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    f1_list = []
    recall_list = []
    precision_list = []

    binary_score_maps = np.zeros_like(score_imgs, dtype=bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap

        predict_region_number = 0
        gt_region_number = 0

        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)

            score_map = measure.label(binary_score_maps[i], connectivity=2)
            score_props = measure.regionprops(score_map)

            predict_region_number += len(score_props)
            gt_region_number += len(props)

            # if len(score_props) == 0 or len(props) == 0:
            #     pro.append(0)
            #     continue

            for score_prop in score_props:
                x_min_0, y_min_0, x_max_0, y_max_0 = score_prop.bbox
                cur_pros = [0]
                for prop in props:
                    x_min_1, y_min_1, x_max_1, y_max_1 = prop.bbox

                    x_min = min(x_min_0, x_min_1)
                    y_min = min(y_min_0, y_min_1)
                    x_max = max(x_max_0, x_max_1)
                    y_max = max(y_max_0, y_max_1)

                    cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                    cropped_gt_label = labeled_imgs[i][x_min:x_max, y_min:y_max]

                    # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                    # cropped_mask = prop.filled_image  # corrected!
                    intersection = np.logical_and(cropped_pred_label, cropped_gt_label).astype(np.float32).sum()
                    union = np.logical_or(cropped_pred_label, cropped_gt_label).astype(np.float32).sum()
                    cur_pros.append(intersection / union)

                pro.append(max(cur_pros))

        pro = np.array(pro)

        if gt_region_number == 0 or predict_region_number == 0:
            print(f'gt_number: {gt_region_number}, pred_number: {predict_region_number}')
            recall = 0
            precision = 0
            f1 = 0
        else:
            recall = np.array(pro >= pro_thresh).astype(np.float32).sum() / gt_region_number
            precision = np.array(pro >= pro_thresh).astype(np.float32).sum() / predict_region_number

            if recall == 0 or precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)


        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)

    # as array
    f1_list = np.array(f1_list)
    max_f1 = f1_list.max()
    cor_recall = recall_list[f1_list.argmax()]
    cor_precision = precision_list[f1_list.argmax()]
    print(f'cor recall: {cor_recall}, cor precision: {cor_precision}')
    return max_f1
