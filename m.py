import os
import csv
import numpy as np
import torch
import pyiqa
import argparse
from pyiqa.utils.img_util import imread2tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS
import glob


def load_test_img_batch(img_dir, ref_dir, all_metrics):
    img_list = sorted(glob.glob(img_dir))

    ref_list = sorted(glob.glob(ref_dir))

    print(len(img_list), len(ref_list))
    all_metrics['input_path'] = img_list
    all_metrics['gt_path'] = ref_list
    img_batch = []
    ref_batch = []
    img_pths = []

    for img_name, ref_name in zip(img_list, ref_list):
        img_path = img_name
        ref_path = ref_name
        img_pths.append(img_name)
        print(img_name,ref_name)
        img_tensor = imread2tensor(img_name).unsqueeze(0)
        # print(img_tensor.shape,img_tensor.max(),img_tensor.min())
        ref_tensor = imread2tensor(ref_name).unsqueeze(0)
        img_batch.append(img_tensor)
        ref_batch.append(ref_tensor)


    return img_batch, ref_batch, all_metrics, img_pths


def dict2csv(dic, filename):
    """
    将字典写入csv文件，要求字典的值长度一致。
    :param dic: the dict to csv
    :param filename: the name of the csv file
    :return: None
    """
    file = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):  # 将字典逐行写入csv
        dic1 = {key: dic[key][i] for key in dic.keys()}
        csv_writer.writerow(dic1)
    file.close()


# python test_metric.py -m psnr ssim ssimc niqe lpips --use_cpu

def run_test(img_dir, ref_dir, test_metric_names):
    all1 = []

    device = torch.device('cuda:2')
    print(f'============> Testing on {device}')
    all_metrics = dict()
    img_batch, ref_batch, all_metrics, img_pthsx = load_test_img_batch(img_dir, ref_dir, all_metrics)

    
    for metric_name in test_metric_names:
        if metric_name == 'fid':
            continue
            fid_metric = pyiqa.create_metric('fid', device='cpu')
            FID = fid_metric("/data/tuluwei/dataset/lolblur/test/high_sharp_scaled","/data/tuluwei/dataset/lolblur_v48_wo_mar/test/low_blur_noise")
            print(f'============> {metric_name} Results score is {FID}')
        iqa_metric = pyiqa.create_metric(metric_name, as_loss=True, device=device)

        metric_mode = 'FR'
        if metric_mode == 'FR':
            score = []
            for i in range(len(img_batch)):
                print(img_pthsx[i])

                b, c, h, w = img_batch[i].shape

                score.append(iqa_metric(img_batch[i][:, :, :h, :w].to(device),
                                        ref_batch[i][:, :, :h, :w].to(device)).squeeze().data.cpu().numpy())
                print(score[i], i, img_pthsx[i])
        else:
            score = []
            for i in range(len(img_batch)):
                score.append(iqa_metric(img_batch[i]).squeeze().data.cpu().numpy())
                print(score[i], i)
        our_score = np.mean(score)
        # our_score_std = np.std(score)
        print(f'============> {metric_name} Results Avg score is {our_score}')

        all1.append(our_score)
        all_metrics[metric_name] = score

    dict2csv(all_metrics, './four_lolblur.csv')

    print(test_metric_names, all1)


if __name__ == '__main__':
    import sys
    import os
    import glob

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric_names', type=str, nargs='+', default=None, help='metric name list.')
    parser.add_argument('--use_cpu', action='store_true', help='use cpu for test')
    args = parser.parse_args()
   

    ref_dir="/data/tuluwei/dataset/lolblur/test/high_sharp_scaled/*/*"
    
    img_dir="/data/tuluwei/dataset/real_blur_result/realblur_FDN/*"


    if args.metric_names is not None:
        test_metric_names = args.metric_names
    else:
        test_metric_names = pyiqa.list_models()

    run_test(img_dir, ref_dir, test_metric_names)
    print(img_dir)

