from basicsr.utils import get_root_logger, imwrite, tensor2img

from basicsr.models.archs.FDN_arch import *

from basicsr.models.archs.LPNet_arch import *
from basicsr.utils import img2tensor
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
def hist3d(img):
    b,g,r=cv2.split(img)
    b=cv2.equalizeHist(b)
    g=cv2.equalizeHist(g)
    r=cv2.equalizeHist(r)
    b=b[:,:,np.newaxis]
    g = g[:, :, np.newaxis]
    r = r[:, :, np.newaxis]
    return np.concatenate([b,g,r],axis=2)
load_path="/data/tuluwei/code/48v2_500000.pth"

load_path_pred="/data/tuluwei/code/FDN/net_g_172600.pth"
file="/data/tuluwei/dataset/lolblur/test/low_blur_noise/*/*"
gt_file="/data/tuluwei/dataset/lolblur/test/high_sharp_scaled/*/*"

device=torch.device("cuda:3")
net_ipred=I_predict_net()
load_net_i = torch.load(
            load_path_pred, map_location=lambda storage, loc: storage.cuda(device))
load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage.cuda(device))
# net=fft_high_light_with_mlp_new_gamma_pformer_v48_fftffn().to(device)
net=FDN().to(device)

net=net.eval()
net.load_state_dict(load_net["params"],strict=True)
net=net.to(device)

net_ipred=net_ipred.eval()
net_ipred.load_state_dict(load_net_i["params"],strict=True)
net_ipred=net_ipred.to(device)
gray_trans=transforms.Compose([transforms.Grayscale(num_output_channels=1)])

with torch.no_grad():

    imgs=sorted(glob.glob(file))
    imgs_gt=sorted(glob.glob(gt_file))
    i=0
    for img1,gt in zip(imgs,imgs_gt):
        i=i+1
        for i in np.arange(0, 1, 0.01):
            img1="/data/tuluwei/dataset/lolblur/test/low_blur_noise/0256/0089.png"
            print(img1.replace("lolblur","FDN_lolblur"),i)
            
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
            # ratio=torch.ones((1,1)).to(device)*0.85
            low_ratio=torch.mean(low_ratio, dim=(2, 3)) /ratio
            # low_ratio=low_ratio/low_ratio*0.0
            ratio=ratio/ratio*i
            print(ratio,i)
            # ratio=ratio.unsqueeze(0)
            result, x_high1q, x_high2q, x_high3q=net(img_lq,ratio_i=ratio,device=device)
            result = result[:, :, :h, :w]
            result=tensor2img([result], rgb2bgr=True)
            # imwrite(result,"./0089ll.png")
            imwrite(result,"./multi_r/{}.png".format(i))
        input()
        # break