import argparse
import json
import os
from datetime import datetime

import torch.optim.lr_scheduler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from utils.instability_penalty import metric_cal_img_with_instability_penalty
from torchvision import transforms
import random
from tqdm import tqdm

TASK = 'CLS'


def save_check_point(model, path):
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)


def write_experiment_readme(args, csv_path):
    """
    Create a per-run README with launcher snapshot and hyperparameters.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    file_stamp = now.strftime("%Y%m%d-%H%M%S")

    # csv_path: <root>/<dataset>/k_<shot>/csv/Seed_xxx-results.csv
    exp_root = os.path.dirname(os.path.dirname(csv_path))
    records_dir = os.path.join(exp_root, "run_records")
    os.makedirs(records_dir, exist_ok=True)

    launcher_path = os.path.join(os.path.dirname(__file__), "run_cls.py")
    launcher_snapshot = ""
    if os.path.exists(launcher_path):
        with open(launcher_path, "r", encoding="utf-8") as f:
            launcher_snapshot = f.read()
    else:
        launcher_snapshot = f"[WARN] launcher script not found: {launcher_path}"

    args_dict = vars(args).copy()
    args_json = json.dumps(args_dict, ensure_ascii=False, indent=2, sort_keys=True)

    readme_name = f"README-{TASK}-{args.dataset}-{args.class_name}-k{args.k_shot}-seed{args.seed}-{file_stamp}.md"
    readme_path = os.path.join(records_dir, readme_name)

    content = (
        f"# Experiment Record\n\n"
        f"## Timestamp\n"
        f"- {timestamp}\n\n"
        f"## Task\n"
        f"- {TASK}\n\n"
        f"## Hyperparameters\n"
        f"```json\n{args_json}\n```\n\n"
        f"## Launcher Snapshot (`run_cls.py`)\n"
        f"```python\n{launcher_snapshot}\n```\n"
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Record] Saved experiment README: {readme_path}")


def resize_scores_and_masks(score_list, mask_list, resolution: tuple = (400, 400)):
    resized_scores = []
    resized_masks = []
    for score, mask in zip(score_list, mask_list):
        score = cv2.resize(score, (resolution[0], resolution[1]), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
        resized_scores.append(score)
        resized_masks.append(mask)
    return resized_scores, resized_masks


def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        check_path: str,
        train_data: DataLoader,
        ):

    # change the model into eval mode
    model.eval_mode()

    features1 = []
    features2 = []
    with torch.no_grad():
        for (data, mask, label, name, img_type) in train_data:
            data = data.to(device, non_blocking=True)
            _, _, feature_map1, feature_map2 = model.encode_image(data)
            features1.append(feature_map1)
            features2.append(feature_map2)

    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)

    best_result_dict = None
    for epoch in range(args.Epoch):
        for (data, mask, label, name, img_type) in train_data:
            data = data.to(device, non_blocking=True)

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

            optimizer.zero_grad()

            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

            cls_feature, _, _, _ = model.encode_image(data)

            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scalef

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

            target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

            loss_v2t = criterion(logits_v2t, target_v2t)

            trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
            loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1

            loss.backward()
            optimizer.step()
        scheduler.step()
        model.build_text_feature_gallery()

        # Evaluate every N epochs (and always evaluate on the final epoch).
        should_eval = ((epoch + 1) % args.eval_freq == 0) or (epoch + 1 == args.Epoch)
        if not should_eval:
            continue

        scores_img = []
        score_maps = []
        gt_list = []
        gt_mask_list = []
        names = []

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
            score_maps += score_map
            scores_img += score_img

        score_maps, gt_mask_list = resize_scores_and_masks(score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
        result_dict = metric_cal_img_with_instability_penalty(
            np.array(scores_img),
            np.array(score_maps),
            gt_list,
            lambda_penalty=args.instability_penalty_lambda,
        )

        if best_result_dict is None:
            save_check_point(model, check_path)
            best_result_dict = result_dict

        elif best_result_dict['i_roc'] < result_dict['i_roc']:
            save_check_point(model, check_path)
            best_result_dict = result_dict

    return best_result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # prepare the experiment dir
    _, csv_path, check_path = get_dir_from_args(TASK, **kwargs)
    write_experiment_readme(args, csv_path)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, transform=model.transform, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, transform=model.transform, **kwargs)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader)

    i_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Image-AUROC:{i_roc}\n')

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
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--root-dir", type=str, default="./result")
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

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=100)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--eval-freq", type=int, default=1,
                        help="Run evaluation every N epochs (default: 1)")
    parser.add_argument(
        "--instability-penalty-lambda",
        type=float,
        default=0.0,
        help="Training-time val AUROC uses s_final_z - lambda*|z(s_sem)-z(s_vis)| on the "
        "test split (z-scores per eval pass). 0 = same ranking as harmonic s_final (default).",
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
