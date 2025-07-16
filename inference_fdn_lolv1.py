from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.models.archs.LPNet_arch import *
from basicsr.models.archs.fdnlol24_arch import FDN_lolv1
from basicsr.utils import img2tensor
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob



load_path="/data/tuluwei/code/FDN/checkpoint/FDN_lolv1.pth"
load_path_pred="/data/tuluwei/code/FDN/checkpoint/LPNet_lolv1.pth"

file="/data/tuluwei/dataset/data/tulw/dataset/lol_v1/testlow/*"

device=torch.device("cuda:5")
net_ipred=I_predict_net()
load_net_i = torch.load(
            load_path_pred, map_location=lambda storage, loc: storage.cuda(device))
load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage.cuda(device))
net=FDN_lolv1().to(device)

net=net.eval()
net.load_state_dict(load_net["params"],strict=True)
net=net.to(device)

net_ipred=net_ipred.eval()
net_ipred.load_state_dict(load_net_i["params"],strict=True)
net_ipred=net_ipred.to(device)
gray_trans=transforms.Compose([transforms.Grayscale(num_output_channels=1)])

with torch.no_grad():

    imgs=sorted(glob.glob(file))
    i=0
    for img1 in imgs:
        i=i+1
        print(img1.replace("lol_v1","lol_v1_FDN"),i)
        img_lq=cv2.imread(img1)

        img_lq = img_lq.astype(np.float32) / 255.
        img_gt, img_lq = img2tensor([img_lq, img_lq],
                                            bgr2rgb=True,
                                            float32=True)

        img_lq=img_lq.unsqueeze(0)
        img_lq=img_lq.to(device)
        b, c, h, w = img_lq.shape
        # 让输入是32的倍数
        #

        h_n = (32 - h % 32) % 32
        w_n = (32 - w % 32) % 32
        img_lq = F.pad(img_lq, (0, w_n, 0, h_n), mode='reflect')
        low_ratio=gray_trans(img_lq)

        ratio =net_ipred(img_lq) #gt
        low_ratio=torch.mean(low_ratio, dim=(2, 3)) /ratio

        # ratio=ratio.unsqueeze(0)
        result, x_high1q, x_high2q, x_high3q=net(img_lq,ratio_i=low_ratio,device=device)
        result = result[:, :, :h, :w]
        result=tensor2img([result], rgb2bgr=True)

        imwrite(result,img1.replace("lol_v1","lol_v1_FDN"))
        # break