import argparse
import csv
import glob
import os

import numpy as np
import pyiqa
import torch
from pyiqa.utils.img_util import imread2tensor


def load_test_img_batch(img_glob, ref_glob, all_metrics):
    img_list = sorted(glob.glob(img_glob))
    ref_list = sorted(glob.glob(ref_glob))

    if not img_list:
        raise FileNotFoundError(f'No predicted images matched: {img_glob}')
    if not ref_list:
        raise FileNotFoundError(f'No reference images matched: {ref_glob}')
    if len(img_list) != len(ref_list):
        raise ValueError(
            f'The number of predicted images ({len(img_list)}) does not match '
            f'the number of reference images ({len(ref_list)}).')

    all_metrics['input_path'] = img_list
    all_metrics['gt_path'] = ref_list
    img_batch = []
    ref_batch = []

    for img_path, ref_path in zip(img_list, ref_list):
        img_batch.append(imread2tensor(img_path).unsqueeze(0))
        ref_batch.append(imread2tensor(ref_path).unsqueeze(0))

    return img_batch, ref_batch, all_metrics, img_list


def dict2csv(dic, filename):
    with open(filename, 'w', encoding='utf-8', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
        csv_writer.writeheader()
        for i in range(len(dic[list(dic.keys())[0]])):
            row = {key: dic[key][i] for key in dic.keys()}
            csv_writer.writerow(row)


def run_test(img_glob, ref_glob, test_metric_names, device, output_csv):
    all_metrics = {}
    img_batch, ref_batch, all_metrics, img_paths = load_test_img_batch(
        img_glob, ref_glob, all_metrics)

    avg_scores = []
    print(f'============> Testing on {device}')

    for metric_name in test_metric_names:
        metric = pyiqa.create_metric(metric_name, as_loss=True, device=device)
        scores = []
        for img_path, pred_tensor, ref_tensor in zip(img_paths, img_batch,
                                                     ref_batch):
            _, _, h, w = pred_tensor.shape
            score = metric(pred_tensor[:, :, :h, :w].to(device),
                           ref_tensor[:, :, :h, :w].to(device))
            score = score.squeeze().data.cpu().numpy()
            scores.append(score)
            print(score, img_path)

        avg_score = np.mean(scores)
        print(f'============> {metric_name} average score: {avg_score}')
        avg_scores.append(avg_score)
        all_metrics[metric_name] = scores

    dict2csv(all_metrics, output_csv)
    print(test_metric_names, avg_scores)
    print(f'Saved metric details to {output_csv}')


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--metric_names',
        type=str,
        nargs='+',
        default=['psnr', 'ssim', 'lpips'],
        help='Metric name list.')
    parser.add_argument(
        '--pred_glob',
        type=str,
        required=True,
        help='Glob pattern for predicted images.')
    parser.add_argument(
        '--gt_glob',
        type=str,
        required=True,
        help='Glob pattern for reference images.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device for metric evaluation.')
    parser.add_argument(
        '--output_csv',
        type=str,
        default='metrics.csv',
        help='Path to save per-image metric results.')
    args = parser.parse_args()

    run_test(args.pred_glob, args.gt_glob, args.metric_names,
             torch.device(args.device), args.output_csv)


if __name__ == '__main__':
    main()
