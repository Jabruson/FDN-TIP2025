import argparse

import pyiqa
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred_dir',
        type=str,
        required=True,
        help='Directory containing predicted images.')
    parser.add_argument(
        '--gt_dir',
        type=str,
        required=True,
        help='Directory containing reference images.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Device for FID evaluation.')
    args = parser.parse_args()

    device = torch.device(args.device)
    fid_metric = pyiqa.create_metric('fid', device=device)
    fid_score = fid_metric(args.pred_dir, args.gt_dir)
    print(args.pred_dir, args.gt_dir, fid_score)


if __name__ == '__main__':
    main()
