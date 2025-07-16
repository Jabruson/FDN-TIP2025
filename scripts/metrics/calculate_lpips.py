import cv2
import glob
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def main():
    # Configurations
    # -------------------------------------------------------------------------
    folder_gt = '/data/tuluwei/dataset/lolblur/test/high_sharp_scaled/*/*'
    folder_restored = '/data/tuluwei/dataset/lolblur/aaai/aaai_saved/*/*'

    # gt_path = [item.replace("low_blur_noise", "high_sharp_scaled") for item in img_path]

    # crop_border = 4
    suffix = ''
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    gt_list = sorted(glob.glob(folder_gt))
    restore_path =  sorted(glob.glob(folder_restored))
    # print(restore_path)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(gt_list):
        # basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(restore_path[i], cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        print(f'{i+1:3d}: . \tLPIPS: {lpips_val.item():.6f}.')
        lpips_all.append(lpips_val.item())

        print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')
    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


if __name__ == '__main__':
    main()
