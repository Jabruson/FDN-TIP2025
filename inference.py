import argparse
import glob
import os
from copy import deepcopy
from os import path as osp

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from basicsr.models.archs import define_network
from basicsr.utils import img2tensor, imwrite, tensor2img
from basicsr.utils.options import parse


def _resolve_checkpoint_path(load_path):
    if osp.isdir(load_path):
        checkpoint_files = [
            osp.join(load_path, name) for name in os.listdir(load_path)
            if name.endswith('.pth')
        ]
        if not checkpoint_files:
            raise FileNotFoundError(
                f'No checkpoint file was found in directory: {load_path}')

        def _checkpoint_sort_key(path):
            stem = osp.splitext(osp.basename(path))[0]
            for token in reversed(stem.split('_')):
                if token.isdigit():
                    return (1, int(token))
            return (0, stem)

        return max(checkpoint_files, key=_checkpoint_sort_key)
    return load_path


def _load_network(network, load_path, device, param_key='params'):
    load_path = _resolve_checkpoint_path(load_path)
    load_net = torch.load(load_path, map_location=device)
    if param_key is not None and param_key in load_net:
        load_net = load_net[param_key]

    cleaned_state_dict = {}
    for key, value in load_net.items():
        if key.startswith('module.'):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    network.load_state_dict(cleaned_state_dict, strict=True)
    return load_path


def _build_output_path(img_path, input_root, output_root):
    if input_root:
        rel_path = osp.relpath(img_path, input_root)
    else:
        rel_path = osp.basename(img_path)
    return osp.join(output_root, rel_path)


def run(opt_path):
    opt = parse(opt_path, is_train=False)
    inference_opt = opt['inference']

    device_name = inference_opt.get('device')
    if not device_name:
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    restorer = define_network(deepcopy(opt['network_g'])).to(device)
    restorer.eval()
    restorer_path = _load_network(
        restorer,
        opt['path']['pretrain_network_g'],
        device,
        param_key=opt['path'].get('param_key', 'params'))

    predictor = None
    predictor_path = None
    if opt.get('network_predictor') is not None:
        predictor = define_network(deepcopy(opt['network_predictor'])).to(device)
        predictor.eval()
        predictor_path = _load_network(
            predictor,
            opt['path']['pretrain_network_predictor'],
            device,
            param_key=opt['path'].get('param_key_predictor', 'params'))

    input_glob = inference_opt['input_glob']
    input_root = inference_opt.get('input_root')
    output_root = inference_opt['output_root']
    os.makedirs(output_root, exist_ok=True)

    img_paths = sorted(glob.glob(input_glob))
    if not img_paths:
        raise FileNotFoundError(f'No images matched: {input_glob}')

    ratio_mode = inference_opt.get('ratio_mode', 'predicted_ratio')
    default_ratio = float(inference_opt.get('default_ratio', 1.0))
    gray_trans = transforms.Grayscale(num_output_channels=1)

    print(f'Loaded restorer weights from: {restorer_path}')
    if predictor_path is not None:
        print(f'Loaded predictor weights from: {predictor_path}')
    print(f'Found {len(img_paths)} images.')

    with torch.no_grad():
        for idx, img_path in enumerate(img_paths, start=1):
            img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_lq is None:
                raise FileNotFoundError(f'Failed to read image: {img_path}')

            img_lq = img_lq.astype('float32') / 255.0
            img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
            img_lq = img_lq.unsqueeze(0).to(device)

            _, _, h, w = img_lq.shape
            h_pad = (32 - h % 32) % 32
            w_pad = (32 - w % 32) % 32
            if h_pad or w_pad:
                img_lq = F.pad(img_lq, (0, w_pad, 0, h_pad), mode='reflect')

            if predictor is not None:
                ratio = predictor(img_lq)
                if ratio_mode == 'low_ratio':
                    low_ratio = gray_trans(img_lq)
                    ratio = torch.mean(low_ratio, dim=(2, 3)) / ratio
            else:
                ratio = torch.ones((1, 1), device=device) * default_ratio

            result = restorer(img_lq, ratio_i=ratio, device=device)
            if isinstance(result, (list, tuple)):
                result = result[0]
            result = result[:, :, :h, :w]
            result = tensor2img(result, rgb2bgr=True)

            output_path = _build_output_path(img_path, input_root, output_root)
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            imwrite(result, output_path)
            print(f'[{idx}/{len(img_paths)}] {img_path} -> {output_path}')


def main(default_opt=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',
        type=str,
        default=default_opt,
        help='Path to option YAML file.')
    args = parser.parse_args()

    if not args.opt:
        parser.error('Please provide -opt or set a default option path.')

    run(args.opt)


if __name__ == '__main__':
    main()
