import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from basicsr.models.archs.FDN_arch import FDN
from basicsr.models.archs.LPNet_arch import I_predict_net
from basicsr.utils import img2tensor, imwrite, tensor2img


def load_weights(network, load_path, device):
    state = torch.load(load_path, map_location=device)
    if 'params' in state:
        state = state['params']
    network.load_state_dict(state, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restorer_ckpt', type=str, required=True)
    parser.add_argument('--predictor_ckpt', type=str, required=True)
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='multi_r')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ratio_start', type=float, default=0.0)
    parser.add_argument('--ratio_end', type=float, default=1.0)
    parser.add_argument('--ratio_step', type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    restorer = FDN().to(device).eval()
    predictor = I_predict_net().to(device).eval()
    load_weights(restorer, args.restorer_ckpt, device)
    load_weights(predictor, args.predictor_ckpt, device)

    img_lq = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    if img_lq is None:
        raise FileNotFoundError(f'Failed to read image: {args.input_image}')

    img_lq = img_lq.astype(np.float32) / 255.0
    img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
    img_lq = img_lq.unsqueeze(0).to(device)
    _, _, h, w = img_lq.shape

    h_pad = (32 - h % 32) % 32
    w_pad = (32 - w % 32) % 32
    if h_pad or w_pad:
        img_lq = F.pad(img_lq, (0, w_pad, 0, h_pad), mode='reflect')

    gray_trans = transforms.Grayscale(num_output_channels=1)

    with torch.no_grad():
        base_ratio = predictor(img_lq)
        low_ratio = gray_trans(img_lq)
        low_ratio = torch.mean(low_ratio, dim=(2, 3)) / base_ratio

        for ratio_scale in np.arange(args.ratio_start, args.ratio_end,
                                     args.ratio_step):
            ratio = base_ratio / base_ratio * ratio_scale
            result = restorer(img_lq, ratio_i=ratio, device=device)[0]
            result = result[:, :, :h, :w]
            result = tensor2img(result, rgb2bgr=True)
            output_path = os.path.join(args.output_dir,
                                       f'{ratio_scale:.2f}.png')
            imwrite(result, output_path)
            print(ratio_scale, output_path, low_ratio.item())


if __name__ == '__main__':
    main()
