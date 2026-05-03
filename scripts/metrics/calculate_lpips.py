import argparse
import glob

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_glob', type=str, required=True)
    parser.add_argument('--restored_glob', type=str, required=True)
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    gt_list = sorted(glob.glob(args.gt_glob))
    restored_list = sorted(glob.glob(args.restored_glob))
    if len(gt_list) != len(restored_list):
        raise ValueError(
            f'Mismatched image counts: {len(gt_list)} vs {len(restored_list)}')

    device = torch.device(args.device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    lpips_all = []
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    for i, (gt_path, restored_path) in enumerate(zip(gt_list, restored_list),
                                                  start=1):
        img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        img_restored = cv2.imread(restored_path,
                                  cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        img_gt, img_restored = img2tensor(
            [img_gt, img_restored], bgr2rgb=True, float32=True)
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).to(device),
                                img_gt.unsqueeze(0).to(device))
        lpips_all.append(lpips_val.item())
        print(f'{i:3d}: LPIPS {lpips_val.item():.6f} {restored_path}')

    print(f'Average LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()
