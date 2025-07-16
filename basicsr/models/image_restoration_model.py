import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import numpy as np
import torchvision.utils
from tqdm import tqdm
import os
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from gpu_mem_track import MemTracker
from basicsr.models.losses.losses import *

from basicsr.models.archs.fdnlol24_arch import *
from torchvision import transforms


import torch.nn as nn

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self, device):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)

        self.kernel = self.kernel.to(device)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class L_exp(nn.Module):

    def __init__(self, patch_size=16):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, gt):
        # mean_val=mean_val.unsqueeze(2).unsqueeze(3).to(device)
        # mean_val = mean_val.to(device)
        # b,c,h,w = x.shape
        gt = torch.mean(gt, 1, keepdim=True)
        mean_val = self.pool(gt)

        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        # print(mean.shape,mean_val.shape)
        d = torch.mean(torch.pow(mean - mean_val, 2))
        return d


class cri_i_adjust(nn.Module):
    def __init__(self, device, patch_size=16):
        super(cri_i_adjust, self).__init__()
        self.exp = L_exp(patch_size=patch_size)
        # self.edge=EdgeLoss(device)
        self.per = PerceptualLoss(layer_weights={'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                                  use_input_norm=True, perceptual_weight=0.01, style_weight=0, range_norm=True,
                                  criterion='l1')

    def forward(self, i_adjust, i_gt):
        return torch.mean(self.exp(i_adjust, i_gt)) + self.per(i_adjust, i_gt)[0]


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt,
                 ori=False, is_deblur=False, with_ir=False, only_i=False, ir_deblur=False,
                 i_adjust=False, i_adjust_merge=False, img_only=True,img_only_finetune=True,
                 img_3stage=False, img_only_2stage=False, fft2stage=False, img_dir=False):
        super(ImageRestorationModel, self).__init__(opt)
        if is_deblur == True and with_ir == True:
            exit()
        # define network
        self.fft2stage = fft2stage
        self.is_ori = ori
        self.is_deblur = is_deblur
        self.with_ir = with_ir
        self.only_i = only_i
        self.ir_deblur = ir_deblur
        self.i_adjust = i_adjust
        self.i_adjust_merge = i_adjust_merge
        self.img_only = img_only
        self.img_only_finetune=img_only_finetune
        self.img_dir = img_dir
        self.img_3stage = img_3stage
        self.img_only_2stage = img_only_2stage
        self.sigmoid = nn.Sigmoid()
        self.use3stage=True

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # if True:
        # dict=torch.load
        # load_net = torch.load(
        #     load_path, map_location=lambda storage, loc: storage.cuda(device))
        # self.net_g = self.model_to_device(self.net_g)

        self.gray_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)

        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.cri_mse = nn.MSELoss()
        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_loss_opt'):
            fft_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(
                self.device)

        else:
            self.cri_fft = None
        # self.cri_fft=nn.L1Loss()
        self.cri_p = PerceptualLoss(layer_weights={'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                                    use_input_norm=True, perceptual_weight=0.01, style_weight=0, range_norm=True,
                                    criterion='l1').to(self.device)
        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if self.use3stage:
            self.cri_i1=MARLoss(scale=1).to(self.device)
            self.cri_i2 = MARLoss(scale=1/2).to(self.device)
            self.cri_i3 = MARLoss(scale=1 / 4).to(self.device)
        else:
            self.cri_i = MARLoss().to(self.device)
        self.cri_a_gai = nn.MSELoss()
        # self.cri_a_gai=a_loss(self.device).to(self.device)
        # self.cri_i = dce_loss(device=self.device).to(self.device)
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
                #             optim_params_lowlr.append(v)
                #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_data_ir(self, data, is_val=False):
        with torch.no_grad():
            if self.img_dir:
                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)

            elif self.is_ori or self.img_only or self.img_3stage or self.img_only_2stage or self.fft2stage:
                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)
                self.gray_lq = self.gray_trans(self.lq)
                self.gray_gt = self.gray_trans(self.gt)

                if self.img_only:
                    self.gray_lq = self.gray_trans(self.lq)
                    self.gray_gt = self.gray_trans(self.gt)


        self.lq = data['lq'].to(self.device)

        self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters_deblur(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        # gpu_tracker.track()
        # print("deblur.............==============")
        if self.img_dir:
            preds = self.net_g(self.lq)
        elif self.i_adjust:
            self.ratio = torch.mean(self.i_low, dim=(2, 3)) / torch.mean(self.i_high, dim=(2, 3))
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1), device=self.device, ratio_i=self.ratio)
        elif self.i_adjust_merge:
            self.ratio = torch.mean(self.i_low, dim=(2, 3)) / torch.mean(self.i_high, dim=(2, 3))
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1), device=self.device, ratio_i=self.ratio)
        elif self.img_only:
            if self.opt['use_ratio']:
                # print("use ratio")
                self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / torch.mean(self.gray_gt, dim=(2, 3))
            else:
                self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / 1.0
            preds = self.net_g(self.lq, device=self.device, ratio_i=self.ratio)

        elif self.img_3stage:
            preds = self.net_g(self.lq)
        # gpu_tracker.track()
        # if not isinstance(preds, list):
        #     preds = [preds]
        if self.img_dir:
            self.output = preds

        elif self.img_only:
            if len(preds) == 2:
                self.output, i_adjust = preds
                i_adjust2 = None
            elif len(preds) == 3: #大 小
                self.output, i_adjust, i_adjust2 = preds
            elif len(preds) == 4:
                self.output, i_adjust, i_adjust2,i_adjust3 = preds
        elif self.img_3stage:
            self.output, i_adjust2, i_adjust = preds
        else:
            # self.output = preds[-1]
            self.output = preds

        l_total = 0
        loss_dict = OrderedDict()
        if self.img_dir:

            if self.cri_pix:
                l_pix = 0.
                l_pix += self.cri_pix(self.output, self.gt)

                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # fft loss
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

                l_per = self.cri_p(self.output, self.gt)[0]
                loss_dict['l_per'] = l_per
                l_total += l_per
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

        elif self.img_only:
            if not self.use3stage:
                if current_iter > 2e5:
                    aa = 0.01
                else:
                    aa = 0.8
                if i_adjust2 != None:
                    l_g = self.cri_mse(i_adjust2, self.gt) * aa
                    loss_dict['l_iglobal'] = l_g
                    l_total += l_g
                if self.cri_pix:
                    l_pix = 0.
                    l_pix += self.cri_pix(self.output, self.gt)

                    # print('l pix ... ', l_pix)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # fft loss
                if self.cri_fft:
                    l_fft = self.cri_fft(self.output, self.gt)
                    l_total += l_fft
                    loss_dict['l_fft'] = l_fft

                    l_per = self.cri_p(self.output, self.gt)[0]
                    loss_dict['l_per'] = l_per
                    l_total += l_per

                l_adjust = self.cri_i(i_adjust, self.gt) * aa

                loss_dict['l_i_pred'] = l_adjust
                l_total += l_adjust

                l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

                l_total.backward()

                ######################################################

                use_grad_clip = self.opt['train'].get('use_grad_clip', True)
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                self.optimizer_g.step()

                self.log_dict = self.reduce_loss_dict(loss_dict)
            elif self.use3stage:

                # 固定参数训练
                # l_g1 = self.cri_i1(i_adjust, self.gt,self.cri_p) * aa
                # loss_dict['l_i1'] = l_g1
                # l_total += l_g1
                # l_g2 = self.cri_i2(i_adjust2, self.gt,self.cri_p) * aa
                # loss_dict['l_i2'] = l_g2
                # l_total += l_g2
                # l_g3 = self.cri_i3(i_adjust3, self.gt,self.cri_p) * aa
                # loss_dict['l_i3'] = l_g3
                # l_total += l_g3
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)

                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # fft loss
                if self.cri_fft:
                    l_fft = self.cri_fft(self.output, self.gt)
                    l_total += l_fft
                    loss_dict['l_fft'] = l_fft

                    l_per = self.cri_p(self.output, self.gt)[0]
                    loss_dict['l_per'] = l_per
                    l_total += l_per



                l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

                l_total.backward()

                ######################################################

                use_grad_clip = self.opt['train'].get('use_grad_clip', True)
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                self.optimizer_g.step()

                self.log_dict = self.reduce_loss_dict(loss_dict)
        elif self.img_3stage:
            if self.cri_pix:
                # 1
                l_pix = 0.
                l_pix += self.cri_pix(self.output, self.gt)
                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix1'] = l_pix
                # print(l_pix,"1")
                # 2
                l_pix = 0.
                l_pix += self.cri_pix(i_adjust2, self.gt)
                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix_adjust2'] = l_pix
                # print(l_pix,"2")
                # 3
                l_adjust1 = self.cri_i(i_adjust, self.gt)
                # l_pix += self.cri_pix(i_adjust, self.gt)*0.1
                # print('l pix ... ', l_pix)
                l_total += l_adjust1
                loss_dict['l_pix_adjust1'] = l_pix
                # print(l_pix,"3")
            # fft loss
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_fft += self.cri_fft(i_adjust2, self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

            l_per = self.cri_p(self.output, self.gt)[0]
            loss_dict['l_per'] = l_per
            l_total += l_per

            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

    def center(self, mag_image):
        N, C, H, W = mag_image.size()
        center_h = H // 2
        center_w = W // 2


        mag_image = torch.roll(mag_image, shifts=(center_h, center_w), dims=(2, 3))

        return mag_image

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            outs_mid = []
            outs_ilow = []
            outs_high = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                b, c, h, w = self.lq[i:j].shape
                # 让输入是32的倍数
                #
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                in_tensor = F.pad(self.lq[i:j], (0, w_n, 0, h_n), mode='reflect')

                if self.i_adjust or self.i_adjust_merge or self.img_only or self.fft2stage:
                    gt_tensor = F.pad(self.gt[i:j], (0, w_n, 0, h_n), mode='reflect')
                if self.img_dir:
                    pred = self.net_g(in_tensor)
                    i_high = None
                    i_low = None
                    pred_mid = None

                elif self.fft2stage:
                    _, _, H, W = in_tensor.shape
                    i_low = self.gray_trans(in_tensor)
                    i_high = self.gray_trans(gt_tensor)
                    # batch=i_high.shape[0]
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    # ratio = torch.mean(i_low, dim=(2, 3))
                    pred = self.net_g(in_tensor, ori=in_tensor, device=self.device, ratio_i=ratio)
                    # pred = self.net_g(in_tensor,device=self.device)
                    pred_mid = pred[2]  # zhenfu
                    pred_i = pred[1]  # i adjust
                    pred = pred[0]  # result

                    i_low = self.center(torch.log(pred_mid + 1.0))
                    i_low = (i_low - i_low.min()) / (i_low.max() - i_low.min())
                    # fft_img = torch.fft.fft2(gt_tensor)
                    i_high = pred_i
                    # recover _mid
                    image_fft = torch.fft.fft2(pred_i, norm='backward')
                    pha_image = torch.angle(image_fft)

                    real_image_enhanced = pred_mid * torch.cos(pha_image)
                    imag_image_enhanced = pred_mid * torch.sin(pha_image)
                    pred_mid = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                               norm='backward').real

                    #
                elif self.is_deblur:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                elif self.with_ir:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                elif self.only_i:
                    _, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([in_tensor, i_low], dim=1))
                elif self.ir_deblur:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                    pred = pred[-1] * pred[-2]
                elif self.i_adjust:
                    r_low, i_low = self.model_decome(in_tensor)
                    _, i_high = self.model_decome(gt_tensor)
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1), device=self.device, ratio_i=ratio)
                    pred_mid = pred[-1]
                    pred = pred[-3] * pred[-2]
                elif self.i_adjust_merge:
                    # print("mememememmemememem!!!!!")
                    r_low, i_low = self.model_decome(in_tensor)
                    _, i_high = self.model_decome(gt_tensor)
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1), ori=in_tensor, device=self.device,
                                      ratio_i=ratio)
                    pred_mid = pred[-1] * r_low
                    pred = pred[-2]
                elif self.img_only:
                    i_low = self.gray_trans(in_tensor)
                    i_high = self.gray_trans(gt_tensor)
                    # batch=i_high.shape[0]
                    if self.opt['use_ratio']:
                        ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    else:
                        ratio = torch.mean(i_low, dim=(2, 3)) / 1.0
                    # ratio = torch.mean(i_low, dim=(2, 3))
                    pred = self.net_g(in_tensor, ori=in_tensor, device=self.device, ratio_i=ratio)
                    if len(pred) == 3:
                        i_low = pred[2]
                        pred_mid = pred[1]
                        pred = pred[0]
                    elif len(pred)==4:
                        i_high=pred[3]
                        i_low = pred[2]
                        pred_mid = pred[1]
                        pred = pred[0]

                elif self.img_3stage:
                    pred = self.net_g(in_tensor)
                    pred_mid = pred[1]
                    i_low = pred[2]
                    pred = pred[0]

                    i_high = pred
                else:
                    pred = self.net_g(in_tensor)
                if self.img_dir:
                    pred = pred[:, :, :h, :w]
                    # pred_mid = pred_mid[:, :, :h, :w]
                    # i_low = i_low[:, :, :h, :w]
                    # i_high = i_high[:, :, :h, :w]
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    # outs_mid.append(pred_mid.detach().cpu())
                    # outs_ilow.append(i_low.detach().cpu())
                    # outs_high.append(i_high.detach().cpu())
                else:
                    pred = pred[:, :, :h, :w]
                    pred_mid = pred_mid[:, :, :h, :w]
                    i_low = i_low[:, :, :h, :w]
                    i_high = i_high[:, :, :h, :w]
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    outs_mid.append(pred_mid.detach().cpu())
                    outs_ilow.append(i_low.detach().cpu())
                    outs_high.append(i_high.detach().cpu())
                i = j
            if self.img_dir:
                self.mid = None
                self.output = torch.cat(outs, dim=0)
                self.i_low = None
                self.i_high = None
            else:
                self.mid = torch.cat(outs_mid, dim=1)
                self.output = torch.cat(outs, dim=0)
                self.i_low = torch.cat(outs_ilow)
                self.i_high = torch.cat(outs_high)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            # if idx>10:
            #     break
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_ir(val_data, is_val=True)

            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            if self.img_dir:
                mid_img = None
                i_low = None
                i_high = None
            else:
                mid_img = tensor2img([visuals['mid']], rgb2bgr=rgb2bgr)
                i_low = tensor2img([visuals['i_low']], rgb2bgr=rgb2bgr)
                i_high = tensor2img([visuals['i_high']], rgb2bgr=rgb2bgr)
            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.mid
            del self.i_low
            del self.i_high
            torch.cuda.empty_cache()
            if save_img:
                if self.img_dir:
                    if self.opt['is_train']:
                        dataset_name = os.path.dirname(
                            val_data['lq_path'][0].replace("low_blur", ("low_blur_result")).replace("lolblur",
                                                                                                    "lol_result"))
                        # print(dataset_name)
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                    else:

                        dataset_name = os.path.dirname(val_data['lq_path'][0].replace("low_blur", ("low_blur_result")))
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                else:
                    if self.opt['is_train']:
                        dataset_name = os.path.dirname(
                            val_data['lq_path'][0].replace("low_blur", ("low_blur_result")).replace("lolblur",
                                                                                                    "lol_result"))
                        # print(dataset_name)
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')
                        save_mid_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_mid.png')
                        save_i_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_i_low.png')
                        save_i_img_path2 = osp.join(
                            dataset_name,
                            f'{img_name}_i_high.png')

                    else:

                        dataset_name = os.path.dirname(val_data['lq_path'][0].replace("low_blur", ("low_blur_result")))
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                if self.img_dir:
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    # print(mid_img.shape,save_mid_img_path)

                else:
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    # print(mid_img.shape,save_mid_img_path)
                    imwrite(mid_img, save_mid_img_path)
                    imwrite(i_low, save_i_img_path)
                    imwrite(i_high, save_i_img_path2)
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.img_dir:
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
        else:
            if hasattr(self, 'lq'):
                out_dict['lq'] = self.lq.detach().cpu()
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'i_low'):
                out_dict['i_low'] = self.i_low.detach().cpu()

            # out_dict['mid']=self.mid.cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
            if hasattr(self, 'i_high'):
                out_dict['i_high'] = self.i_high.detach().cpu()
            if hasattr(self, 'mid'):
                out_dict['mid'] = self.mid.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)



class ImageRestorationModel_ipretrain(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel_ipretrain, self).__init__(opt)
        # if is_deblur == True and with_ir == True:
        #     exit()
        # define network
        
        self.sigmoid = nn.Sigmoid()
        self.use3stage = True
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.gray_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.cri_mse = nn.MSELoss()
        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_loss_opt'):
            fft_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(
                self.device)

        else:
            self.cri_fft = None
        # self.cri_fft=nn.L1Loss()
        self.cri_p = PerceptualLoss(layer_weights={'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                                    use_input_norm=True, perceptual_weight=0.01, style_weight=0, range_norm=True,
                                    criterion='l1').to(self.device)
        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if self.use3stage:
            self.cri_i1 = MARLoss(scale=1).to(self.device)
            self.cri_i2 = MARLoss(scale=1 / 2).to(self.device)
            self.cri_i3 = MARLoss(scale=1 / 4).to(self.device)
        else:
            self.cri_i = MARLoss().to(self.device)
        self.cri_a_gai = nn.MSELoss()
        # self.cri_a_gai=a_loss(self.device).to(self.device)
        # self.cri_i = dce_loss(device=self.device).to(self.device)
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
                #             optim_params_lowlr.append(v)
                #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_data_ir(self, data, is_val=False):
        with torch.no_grad():
           

                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)
                self.gray_lq = self.gray_trans(self.lq)
                self.gray_gt = self.gray_trans(self.gt)

        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        _,_,h,w=self.gt.shape





    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)
 
        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters_deblur(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        # gpu_tracker.track()
        # print("deblur.............==============")
       

        if self.opt['use_ratio']:
            # print("use ratio")
            self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / torch.mean(self.gray_gt, dim=(2, 3))
            self.ratio=self.ratio.unsqueeze(-1).unsqueeze(-1)
        else:
            self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / 1.0
            self.ratio = self.ratio.unsqueeze(-1).unsqueeze(-1)
        preds = self.net_g(self.lq,self.ratio)


        # gpu_tracker.track()
        # if not isinstance(preds, list):
        #     preds = [preds]
       
        i_adjust3, i_adjust2, i_adjust = preds
        self.output=i_adjust
        

        l_total = 0
        loss_dict = OrderedDict()
        

            # print(i_adjust.shape,i_adjust2.shape)
        l_g1 = self.cri_i1(i_adjust, self.gt, self.cri_p)
        loss_dict['l_i1'] = l_g1
        l_total += l_g1
        l_g2 = self.cri_i2(i_adjust2, self.gt, self.cri_p)
        loss_dict['l_i2'] = l_g2
        l_total += l_g2
        l_g3 = self.cri_i3(i_adjust3, self.gt, self.cri_p)
        loss_dict['l_i3'] = l_g3
        l_total += l_g3
        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()

        ######################################################

        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def center(self, mag_image):
        N, C, H, W = mag_image.size()
        center_h = H // 2
        center_w = W // 2


        mag_image = torch.roll(mag_image, shifts=(center_h, center_w), dims=(2, 3))

        return mag_image

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            outs_mid = []
            outs_ilow = []
            outs_high = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                b, c, h, w = self.lq[i:j].shape
                # 让输入是32的倍数
                #
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                in_tensor = F.pad(self.lq[i:j], (0, w_n, 0, h_n), mode='reflect')

               
                gt_tensor = F.pad(self.gt[i:j], (0, w_n, 0, h_n), mode='reflect')
                
                
                i_low = self.gray_trans(in_tensor)
                i_high = self.gray_trans(gt_tensor)
                # batch=i_high.shape[0]
                if self.opt['use_ratio']:
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                else:
                    ratio = torch.mean(i_low, dim=(2, 3)) / 1.0
                ratio=ratio.unsqueeze(-1).unsqueeze(-1)
                # ratio = torch.mean(i_low, dim=(2, 3))
                pred = self.net_g(in_tensor,ratio)

                i_low = pred[0]
                pred_mid = pred[1]
                pred = pred[2]


               
                pred = pred[:, :, :h, :w]
                pred_mid = pred_mid[:, :, :h, :w]
                i_low = i_low[:, :, :h, :w]
                i_high = i_high[:, :, :h, :w]
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                outs_mid.append(pred_mid.detach().cpu())
                outs_ilow.append(i_low.detach().cpu())
                outs_high.append(i_high.detach().cpu())
                i = j
           
            self.mid = torch.cat(outs_mid, dim=1)
            self.output = torch.cat(outs, dim=0)
            self.i_low = torch.cat(outs_ilow)
            self.i_high = torch.cat(outs_high)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            # if idx>10:
            #     break
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_ir(val_data, is_val=True)

            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            
            mid_img = tensor2img([visuals['mid']], rgb2bgr=rgb2bgr)
            i_low = tensor2img([visuals['i_low']], rgb2bgr=rgb2bgr)
            i_high = tensor2img([visuals['i_high']], rgb2bgr=rgb2bgr)
            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.mid
            del self.i_low
            del self.i_high
            torch.cuda.empty_cache()
            if save_img:
                
                if self.opt['is_train']:
                    dataset_name = os.path.dirname(
                        val_data['lq_path'][0].replace("low_blur", ("low_blur_result")).replace("lolblur",
                                                                                                "lol_result"))
                    # print(dataset_name)
                    if not os.path.exists(dataset_name):
                        os.makedirs(dataset_name, exist_ok=True)
                    save_img_path = osp.join(
                        dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        dataset_name,
                        f'{img_name}_gt.png')
                    save_mid_img_path = osp.join(
                        dataset_name,
                        f'{img_name}_mid.png')
                    save_i_img_path = osp.join(
                        dataset_name,
                        f'{img_name}_i_low.png')
                    save_i_img_path2 = osp.join(
                        dataset_name,
                        f'{img_name}_i_high.png')

                else:

                    dataset_name = os.path.dirname(val_data['lq_path'][0].replace("low_blur", ("low_blur_result")))
                    if not os.path.exists(dataset_name):
                        os.makedirs(dataset_name, exist_ok=True)
                    save_img_path = osp.join(
                        dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        dataset_name,
                        f'{img_name}_gt.png')

               
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)
                # print(mid_img.shape,save_mid_img_path)
                imwrite(mid_img, save_mid_img_path)
                imwrite(i_low, save_i_img_path)
                imwrite(i_high, save_i_img_path2)
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        
        if hasattr(self, 'lq'):
            out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'i_low'):
            out_dict['i_low'] = self.i_low.detach().cpu()

        # out_dict['mid']=self.mid.cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'i_high'):
            out_dict['i_high'] = self.i_high.detach().cpu()
        if hasattr(self, 'mid'):
            out_dict['mid'] = self.mid.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)



class ImageRestorationModel_ipred(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt,
                 ori=False, is_deblur=False, with_ir=False, only_i=False, ir_deblur=False,
                 i_adjust=False, i_adjust_merge=False, img_only=False,
                 img_3stage=False, img_i_pred=True, img_i_pred2=False):
        super(ImageRestorationModel_ipred, self).__init__(opt)
        if is_deblur == True and with_ir == True:
            exit()
        # define network
        self.img_i_pred2 = img_i_pred2
        self.img_i_pred = img_i_pred
        self.is_ori = ori
        self.is_deblur = is_deblur
        self.with_ir = with_ir
        self.only_i = only_i
        self.ir_deblur = ir_deblur
        self.i_adjust = i_adjust
        self.i_adjust_merge = i_adjust_merge
        self.img_only = img_only
        self.img_3stage = img_3stage
        if self.img_i_pred or self.img_i_pred2:
            self.model_fft = FDN()
            state = torch.load(
                '/data/tuluwei/code/48v2_500000.pth',
                map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            self.model_fft.load_state_dict(state["params"], strict=True)
            for param in self.model_fft.parameters():
                param.requires_grad = False
            self.model_fft.eval()
            device_id = torch.cuda.current_device()
            device = torch.device("cuda", device_id)
            self.model_fft = self.model_fft.to(device)
            self.net_g = define_network(deepcopy(opt['network_g']))
        elif is_deblur or with_ir or only_i or ir_deblur or i_adjust or i_adjust_merge:
            self.net_g = define_network(deepcopy(opt['network_g']))
        else:
            self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        self.gray_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        # self.pth_files=os.listdir("")
        # self.pth_index=0
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])
        # for param in self.net_g.parameters():
        #     param.requires_grad = False
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_loss_opt'):
            fft_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(
                self.device)

        else:
            self.cri_fft = None
        self.cri_p = PerceptualLoss(layer_weights={'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                                    use_input_norm=True, perceptual_weight=0.01, style_weight=0, range_norm=True,
                                    criterion='l1').to(self.device)
        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        self.cri_i = gamma_loss(self.device).to(self.device)
        # self.cri_i = dce_loss(device=self.device).to(self.device)
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_data_deblur(self, data, is_val=False):
        with torch.no_grad():
            self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
            self.r_high, _ = self.model_decome(data["gt"].to(self.device))
        self.lq = self.r_low

        self.gt = self.r_high


    def feed_data_ir(self, data, is_val=False):
        with torch.no_grad():
            if self.img_i_pred or self.img_i_pred2:
                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)

                self.gray_lq = self.gray_trans(self.lq)
                self.gray_gt = self.gray_trans(self.gt)
            elif self.with_ir:
                self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
            elif self.only_i:
                self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
            elif self.ir_deblur:
                self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
                self.r_high, self.i_high = self.model_decome(data["gt"].to(self.device))

            elif self.i_adjust or self.i_adjust_merge:
                self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
                self.r_high, self.i_high = self.model_decome(data["gt"].to(self.device))
            elif self.is_deblur:
                with torch.no_grad():
                    self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
                    self.r_high, _ = self.model_decome(data["gt"].to(self.device))
                self.lq = self.r_low
                # self.i_low=self.i_low.to(self.device)
                # if r_low_gt!=None:
                self.gt = self.r_high
                return
            elif self.is_ori or self.img_only or self.img_3stage:
                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)
                if self.img_only:
                    self.gray_lq = self.gray_trans(self.lq)
                    # self.gray_gt=self.gray_trans(self.gt)

            # torchvision.utils.save_image(self.r_low.to(torch.device("cpu")),"./r.png")
            # torchvision.utils.save_image(torch.cat([self.i_low,self.i_low,self.i_low],dim=1).to(torch.device("cpu")),"./i.png")
            # self.r_high, _ = self.model_decome(data["gt"].to(self.device))
        self.lq = data['lq'].to(self.device)
        # self.i_low=self.i_low.to(self.device)
        # if self.with_ir or self.ir_deblur or self.i_adjust:
        #     self.r_low=self.r_low.to(self.device)
        # if r_low_gt!=None:
        self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters_deblur(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        # gpu_tracker.track()
        # print("deblur.............==============")
        if self.is_deblur:
            preds = self.net_g(torch.cat([self.lq, self.i_low], dim=1))
        elif self.with_ir:
            preds = self.net_g(torch.cat([self.lq, self.r_low, self.i_low], dim=1))
        elif self.only_i:
            preds = self.net_g(torch.cat([self.lq, self.i_low], dim=1))
        elif self.ir_deblur:
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1))
        elif self.i_adjust:
            self.ratio = torch.mean(self.i_low, dim=(2, 3)) / torch.mean(self.i_high, dim=(2, 3))
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1), device=self.device, ratio_i=self.ratio)
        elif self.i_adjust_merge:
            self.ratio = torch.mean(self.i_low, dim=(2, 3)) / torch.mean(self.i_high, dim=(2, 3))
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1), device=self.device, ratio_i=self.ratio)
        elif self.img_only:
            # batch=self.gray_lq.shape[0]
            # print(torch.mean(self.gray_lq, dim=(2, 3)).shape)
            self.ratio = torch.mean(self.gray_lq, dim=(2, 3))
            # self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / torch.mean(self.gray_gt, dim=(2, 3))
            preds = self.net_g(self.lq, device=self.device, ratio_i=self.ratio)
        elif self.img_3stage:
            preds = self.net_g(self.lq)
        elif self.img_i_pred:
            # self.ratio_lq = torch.mean(self.gray_lq, dim=(2, 3))
            # self.ratio_gt=torch.mean(self.gray_gt, dim=(2, 3))
            preds = self.net_g(self.lq)
        elif self.img_i_pred2:
            ratio = self.net_g(self.lq)
            preds = self.model_fft(self.lq, device=self.device, ratio_i=ratio)
        # gpu_tracker.track()
        # if not isinstance(preds, list):
        #     preds = [preds]
        # if self.img_i_pred:
        # ratio_t=torch.mean(self.lq, dim=(2, 3)) / preds
        # self.output=self.model_fft(self.lq, device=self.device, ratio_i=ratio_t)
        if self.ir_deblur and not self.i_adjust:
            self.output = preds[-1] * preds[-2]
        elif self.i_adjust:
            # print(len(preds))
            # self.mid = torch.cat([preds[-1], preds[-1], preds[-1]], dim=1)
            self.output = preds[-3] * preds[-2]
        elif self.i_adjust_merge:
            # print(type(preds),len(preds))
            # print("right!!!!!!!!!")
            self.output, i_adjust = preds
            # print(i_adjust.shape)
        elif self.img_only:
            self.output, i_adjust = preds
        elif self.img_i_pred2:
            self.output, i_adjust = preds
        elif self.img_3stage:
            self.output, i_adjust2, i_adjust = preds
        else:
            # self.output = preds[-1]
            self.output = preds

        l_total = 0
        loss_dict = OrderedDict()
        if self.img_i_pred:
            gt_ratio = torch.mean(self.gray_gt, dim=(2, 3))
            # print(gt_ratio.shape)
            # lq_ratio = torch.mean(self.gray_lq, dim=(2, 3))
            # ratiot = gt_ratio
            l_total = self.cri_pix(preds, gt_ratio)
            loss_dict['l_ipred'] = l_total
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)
        elif self.img_i_pred2:
            l_pix = 0.
            l_pix += self.cri_pix(self.output, self.gt)
            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix1'] = l_pix
            l_fft = self.cri_fft(self.output, self.gt)
            # l_fft += self.cri_fft(i_adjust2, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)
        elif self.img_3stage:
            if self.cri_pix:
                # 1
                l_pix = 0.
                l_pix += self.cri_pix(self.output, self.gt)
                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix1'] = l_pix
                # print(l_pix,"1")
                # 2
                l_pix = 0.
                l_pix += self.cri_pix(i_adjust2, self.gt)
                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix_adjust2'] = l_pix
                # print(l_pix,"2")
                # 3
                l_adjust1 = self.cri_i(i_adjust, self.gt)
                # l_pix += self.cri_pix(i_adjust, self.gt)*0.1
                # print('l pix ... ', l_pix)
                l_total += l_adjust1
                loss_dict['l_pix_adjust1'] = l_pix
                # print(l_pix,"3")
            # fft loss
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_fft += self.cri_fft(i_adjust2, self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

            l_per = self.cri_p(self.output, self.gt)[0]
            loss_dict['l_per'] = l_per
            l_total += l_per

            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            outs_mid = []
            outs_ilow = []
            outs_high = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                b, c, h, w = self.lq[i:j].shape
                # 让输入是32的倍数
                #

                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                in_tensor = F.pad(self.lq[i:j], (0, w_n, 0, h_n), mode='reflect')
                gt_tensor = F.pad(self.gt[i:j], (0, w_n, 0, h_n), mode='reflect')

                if self.is_deblur:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                elif self.with_ir:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                elif self.only_i:
                    _, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([in_tensor, i_low], dim=1))
                elif self.ir_deblur:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                    pred = pred[-1] * pred[-2]
                elif self.i_adjust:
                    r_low, i_low = self.model_decome(in_tensor)
                    _, i_high = self.model_decome(gt_tensor)
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1), device=self.device, ratio_i=ratio)
                    pred_mid = pred[-1]
                    pred = pred[-3] * pred[-2]
                elif self.i_adjust_merge:
                    # print("mememememmemememem!!!!!")
                    r_low, i_low = self.model_decome(in_tensor)
                    _, i_high = self.model_decome(gt_tensor)
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1), ori=in_tensor, device=self.device,
                                      ratio_i=ratio)
                    pred_mid = pred[-1] * r_low
                    pred = pred[-2]
                elif self.img_only:
                    i_low = self.gray_trans(in_tensor)
                    i_high = self.gray_trans(gt_tensor)
                    # batch=i_high.shape[0]
                    # ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high      , dim=(2, 3))
                    ratio = torch.mean(i_low, dim=(2, 3))
                    pred = self.net_g(in_tensor, ori=in_tensor, device=self.device, ratio_i=ratio)
                    pred_mid = pred[-1]
                    pred = pred[-2]
                elif self.img_3stage:
                    pred = self.net_g(in_tensor)
                    pred_mid = pred[1]
                    i_low = pred[2]
                    pred = pred[0]

                    i_high = pred
                elif self.img_i_pred or self.img_i_pred2:
                    i_low = self.gray_trans(in_tensor)
                    i_high = self.gray_trans(gt_tensor)
                    ratio_pred = self.net_g(in_tensor) #gtratio
                    ratio = torch.mean(i_low, dim=(2, 3))/ratio_pred
                    pred = self.model_fft(in_tensor, ori=in_tensor, device=self.device, ratio_i=ratio)
                    pred_mid = pred[1]
                    i_low=pred[2]
                    pred = pred[0]
                else:
                    pred = self.net_g(in_tensor)
                pred = pred[:, :, :h, :w]
                pred_mid = pred_mid[:, :, :h, :w]
                i_low = i_low[:, :, :h, :w]
                i_high = i_high[:, :, :h, :w]
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                outs_mid.append(pred_mid.detach().cpu())
                outs_ilow.append(i_low.detach().cpu())
                outs_high.append(i_high.detach().cpu())
                i = j
            self.mid = torch.cat(outs_mid, dim=1)
            self.output = torch.cat(outs, dim=0)
            self.i_low = torch.cat(outs_ilow)
            self.i_high = torch.cat(outs_high)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        pth_path=None
        if pth_path!=None:
            self.load_network(self.net_g, pth_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            # if idx>10:
            #     break
            if idx % world_size != rank:
                continue
            # save_name = name[0]
            #
            # print("mkdir ", data_root[0])

            # print(val_data['lq_path'][0],"==========")
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_ir(val_data, is_val=True)
            # if self.is_deblur:
            #     self.feed_data_deblur(val_data, is_val=True)
            # elif self.with_ir or self.only_i:
            #     self.feed_data_ir(val_data, is_val=True)
            # else:
            #     self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            mid_img = tensor2img([visuals['mid']], rgb2bgr=rgb2bgr)
            i_low = tensor2img([visuals['i_low']], rgb2bgr=rgb2bgr)
            i_high = tensor2img([visuals['i_high']], rgb2bgr=rgb2bgr)
            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.mid
            del self.i_low
            del self.i_high
            torch.cuda.empty_cache()
            if save_img:
                if sr_img.shape[2] == 6 and False:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    # print(visual_dir)
                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:
                        dataset_name = os.path.dirname(
                            val_data['lq_path'][0].replace("dataset","dataset_result"))
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        print(save_img_path)
                        
                    else:

                        dataset_name = os.path.dirname(
                            val_data['lq_path'][0].replace("dataset","dataset_result"))
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')
   

                    imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['i_low'] = self.i_low.detach().cpu()

        # out_dict['mid']=self.mid.cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
            out_dict['i_high'] = self.i_high.detach().cpu()
        if hasattr(self, 'mid'):
            out_dict['mid'] = self.mid.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
class ImageRestorationModel_3stage(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt,
                 ori=False, is_deblur=False, with_ir=False, only_i=False, ir_deblur=False,
                 i_adjust=False, i_adjust_merge=False, img_only=True, img_only_finetune=False,
                 img_3stage=False, img_only_2stage=False, fft2stage=False, img_dir=False):
        super(ImageRestorationModel_3stage, self).__init__(opt)
        if is_deblur == True and with_ir == True:
            exit()
        # define network
        self.fft2stage = fft2stage
        self.is_ori = ori
        self.is_deblur = is_deblur
        self.with_ir = with_ir
        self.only_i = only_i
        self.ir_deblur = ir_deblur
        self.i_adjust = i_adjust
        self.i_adjust_merge = i_adjust_merge
        self.img_only = img_only
        self.img_only_finetune = img_only_finetune
        self.img_dir = img_dir
        self.img_3stage = img_3stage
        self.img_only_2stage = img_only_2stage
        self.sigmoid = nn.Sigmoid()
        self.use3stage = True

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.gray_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.cri_mse = nn.MSELoss()
        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_loss_opt'):
            fft_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(
                self.device)

        else:
            self.cri_fft = None
        # self.cri_fft=nn.L1Loss()
        self.cri_p = PerceptualLoss(layer_weights={'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                                    use_input_norm=True, perceptual_weight=0.01, style_weight=0, range_norm=True,
                                    criterion='l1').to(self.device)
        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        # if self.use3stage:
        #     self.cri_i1 = loss_down(scale=1).to(self.device)
        #     self.cri_i2 = loss_down(scale=1 / 2).to(self.device)
        #     self.cri_i3 = loss_down(scale=1 / 4).to(self.device)
        # else:
        #     self.cri_i = loss_down().to(self.device)
        self.cri_a_gai = nn.MSELoss()
        # self.cri_a_gai=a_loss(self.device).to(self.device)
        # self.cri_i = dce_loss(device=self.device).to(self.device)
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
                #             optim_params_lowlr.append(v)
                #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_data_ir(self, data, is_val=False):
        with torch.no_grad():
            if self.img_dir:
                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)

            elif self.is_ori or self.img_only or self.img_3stage or self.img_only_2stage or self.fft2stage:
                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)
                self.gray_lq = self.gray_trans(self.lq)
                self.gray_gt = self.gray_trans(self.gt)

                if self.img_only:
                    self.gray_lq = self.gray_trans(self.lq)
                    self.gray_gt = self.gray_trans(self.gt)

        self.lq = data['lq'].to(self.device)

        self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters_deblur(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        # gpu_tracker.track()
        # print("deblur.............==============")
        if self.img_dir:
            preds = self.net_g(self.lq)
        elif self.i_adjust:
            self.ratio = torch.mean(self.i_low, dim=(2, 3)) / torch.mean(self.i_high, dim=(2, 3))
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1), device=self.device, ratio_i=self.ratio)
        elif self.i_adjust_merge:
            self.ratio = torch.mean(self.i_low, dim=(2, 3)) / torch.mean(self.i_high, dim=(2, 3))
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1), device=self.device, ratio_i=self.ratio)
        elif self.img_only:
            if self.opt['use_ratio']:
                # print("use ratio")
                self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / torch.mean(self.gray_gt, dim=(2, 3))
            else:
                self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / 1.0
            preds = self.net_g(self.lq, device=self.device, ratio_i=self.ratio)

        elif self.img_3stage:
            preds = self.net_g(self.lq)
        # gpu_tracker.track()
        # if not isinstance(preds, list):
        #     preds = [preds]
        if self.img_dir:
            self.output = preds

        elif self.img_only:
            if len(preds) == 2:
                self.output, i_adjust = preds
                i_adjust2 = None
            elif len(preds) == 3:  # 大 小
                self.output, i_adjust, i_adjust2 = preds
            elif len(preds) == 4:
                self.output, i_adjust, i_adjust2, i_adjust3 = preds
        elif self.img_3stage:
            self.output, i_adjust2, i_adjust = preds
        else:
            # self.output = preds[-1]
            self.output = preds

        l_total = 0
        loss_dict = OrderedDict()
        if self.img_dir:

            if self.cri_pix:
                l_pix = 0.
                l_pix += self.cri_pix(self.output, self.gt)

                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # fft loss
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

                l_per = self.cri_p(self.output, self.gt)[0]
                loss_dict['l_per'] = l_per
                l_total += l_per
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

        elif self.img_only:
            if not self.use3stage:
                if current_iter > 2e5:
                    aa = 0.01
                else:
                    aa = 0.8
                if i_adjust2 != None:
                    l_g = self.cri_mse(i_adjust2, self.gt) * aa
                    loss_dict['l_iglobal'] = l_g
                    l_total += l_g
                if self.cri_pix:
                    l_pix = 0.
                    l_pix += self.cri_pix(self.output, self.gt)

                    # print('l pix ... ', l_pix)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # fft loss
                if self.cri_fft:
                    l_fft = self.cri_fft(self.output, self.gt)
                    l_total += l_fft
                    loss_dict['l_fft'] = l_fft

                    l_per = self.cri_p(self.output, self.gt)[0]
                    loss_dict['l_per'] = l_per
                    l_total += l_per

                l_adjust = self.cri_i(i_adjust, self.gt) * aa

                loss_dict['l_i_pred'] = l_adjust
                l_total += l_adjust

                l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

                l_total.backward()

                ######################################################

                use_grad_clip = self.opt['train'].get('use_grad_clip', True)
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                self.optimizer_g.step()

                self.log_dict = self.reduce_loss_dict(loss_dict)
            elif self.use3stage:

                # 固定参数训练
                # l_g1 = self.cri_i1(i_adjust, self.gt,self.cri_p) * aa
                # loss_dict['l_i1'] = l_g1
                # l_total += l_g1
                # l_g2 = self.cri_i2(i_adjust2, self.gt,self.cri_p) * aa
                # loss_dict['l_i2'] = l_g2
                # l_total += l_g2
                # l_g3 = self.cri_i3(i_adjust3, self.gt,self.cri_p) * aa
                # loss_dict['l_i3'] = l_g3
                # l_total += l_g3
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)

                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # fft loss
                if self.cri_fft:
                    l_fft = self.cri_fft(self.output, self.gt)
                    l_total += l_fft
                    loss_dict['l_fft'] = l_fft

                    l_per = self.cri_p(self.output, self.gt)[0]
                    loss_dict['l_per'] = l_per
                    l_total += l_per

                l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

                l_total.backward()

                ######################################################

                use_grad_clip = self.opt['train'].get('use_grad_clip', True)
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                self.optimizer_g.step()

                self.log_dict = self.reduce_loss_dict(loss_dict)
        elif self.img_3stage:
            if self.cri_pix:
                # 1
                l_pix = 0.
                l_pix += self.cri_pix(self.output, self.gt)
                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix1'] = l_pix
                # print(l_pix,"1")
                # 2
                l_pix = 0.
                l_pix += self.cri_pix(i_adjust2, self.gt)
                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix_adjust2'] = l_pix
                # print(l_pix,"2")
                # 3
                l_adjust1 = self.cri_i(i_adjust, self.gt)
                # l_pix += self.cri_pix(i_adjust, self.gt)*0.1
                # print('l pix ... ', l_pix)
                l_total += l_adjust1
                loss_dict['l_pix_adjust1'] = l_pix
                # print(l_pix,"3")
            # fft loss
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_fft += self.cri_fft(i_adjust2, self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

            l_per = self.cri_p(self.output, self.gt)[0]
            loss_dict['l_per'] = l_per
            l_total += l_per

            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

    def center(self, mag_image):
        N, C, H, W = mag_image.size()
        center_h = H // 2
        center_w = W // 2

        # 2. 使用 torch.roll 函数将振幅图的四个象限重排
        # 重排的目标是将中心点移到图像的中心
        mag_image = torch.roll(mag_image, shifts=(center_h, center_w), dims=(2, 3))

        return mag_image

    def test(self):
        self.net_g.eval()


        with torch.no_grad():
            n = len(self.lq)
            outs = []
            outs_mid = []
            outs_ilow = []
            outs_high = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                b, c, h, w = self.lq[i:j].shape
                # 让输入是32的倍数
                #
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                in_tensor = F.pad(self.lq[i:j], (0, w_n, 0, h_n), mode='reflect')

                if self.i_adjust or self.i_adjust_merge or self.img_only or self.fft2stage:
                    gt_tensor = F.pad(self.gt[i:j], (0, w_n, 0, h_n), mode='reflect')
                if self.img_dir:
                    pred = self.net_g(in_tensor)
                    i_high = None
                    i_low = None
                    pred_mid = None

                elif self.fft2stage:
                    _, _, H, W = in_tensor.shape
                    i_low = self.gray_trans(in_tensor)
                    i_high = self.gray_trans(gt_tensor)
                    # batch=i_high.shape[0]
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    # ratio = torch.mean(i_low, dim=(2, 3))
                    pred = self.net_g(in_tensor, ori=in_tensor, device=self.device, ratio_i=ratio)
                    # pred = self.net_g(in_tensor,device=self.device)
                    pred_mid = pred[2]  # zhenfu
                    pred_i = pred[1]  # i adjust
                    pred = pred[0]  # result

                    i_low = self.center(torch.log(pred_mid + 1.0))
                    i_low = (i_low - i_low.min()) / (i_low.max() - i_low.min())
                    # fft_img = torch.fft.fft2(gt_tensor)
                    i_high = pred_i
                    # recover _mid
                    image_fft = torch.fft.fft2(pred_i, norm='backward')
                    pha_image = torch.angle(image_fft)

                    real_image_enhanced = pred_mid * torch.cos(pha_image)
                    imag_image_enhanced = pred_mid * torch.sin(pha_image)
                    pred_mid = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                               norm='backward').real

                    #
                elif self.is_deblur:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                elif self.with_ir:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                elif self.only_i:
                    _, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([in_tensor, i_low], dim=1))
                elif self.ir_deblur:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                    pred = pred[-1] * pred[-2]
                elif self.i_adjust:
                    r_low, i_low = self.model_decome(in_tensor)
                    _, i_high = self.model_decome(gt_tensor)
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1), device=self.device, ratio_i=ratio)
                    pred_mid = pred[-1]
                    pred = pred[-3] * pred[-2]
                elif self.i_adjust_merge:
                    # print("mememememmemememem!!!!!")
                    r_low, i_low = self.model_decome(in_tensor)
                    _, i_high = self.model_decome(gt_tensor)
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1), ori=in_tensor, device=self.device,
                                      ratio_i=ratio)
                    pred_mid = pred[-1] * r_low
                    pred = pred[-2]
                elif self.img_only:
                    i_low = self.gray_trans(in_tensor)
                    i_high = self.gray_trans(gt_tensor)
                    # batch=i_high.shape[0]
                    if self.opt['use_ratio']:
                        ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    else:
                        ratio = torch.mean(i_low, dim=(2, 3)) / 1.0
                    # ratio = torch.mean(i_low, dim=(2, 3))
                    pred = self.net_g(in_tensor, ori=in_tensor, device=self.device, ratio_i=ratio)
                    if len(pred) == 3:
                        i_low = pred[2]
                        pred_mid = pred[1]
                        pred = pred[0]
                    elif len(pred) == 4:
                        i_high = pred[3]
                        i_low = pred[2]
                        pred_mid = pred[1]
                        pred = pred[0]

                elif self.img_3stage:
                    pred = self.net_g(in_tensor)
                    pred_mid = pred[1]
                    i_low = pred[2]
                    pred = pred[0]

                    i_high = pred
                else:
                    pred = self.net_g(in_tensor)
                if self.img_dir:
                    pred = pred[:, :, :h, :w]
                    # pred_mid = pred_mid[:, :, :h, :w]
                    # i_low = i_low[:, :, :h, :w]
                    # i_high = i_high[:, :, :h, :w]
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    # outs_mid.append(pred_mid.detach().cpu())
                    # outs_ilow.append(i_low.detach().cpu())
                    # outs_high.append(i_high.detach().cpu())
                else:
                    pred = pred[:, :, :h, :w]
                    pred_mid = pred_mid[:, :, :h, :w]
                    i_low = i_low[:, :, :h, :w]
                    i_high = i_high[:, :, :h, :w]
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    outs_mid.append(pred_mid.detach().cpu())
                    outs_ilow.append(i_low.detach().cpu())
                    outs_high.append(i_high.detach().cpu())
                i = j
            if self.img_dir:
                self.mid = None
                self.output = torch.cat(outs, dim=0)
                self.i_low = None
                self.i_high = None
            else:
                self.mid = torch.cat(outs_mid, dim=1)
                self.output = torch.cat(outs, dim=0)
                self.i_low = torch.cat(outs_ilow)
                self.i_high = torch.cat(outs_high)
        self.net_g.train()
    def occupy(self):
        self.x_occupied = []
        block_mem=2200
        for i in range(4):

            self.x_occupied.append(torch.zeros(
                (256, 1024, block_mem),
                dtype=torch.float32,
                device='cuda:{}'.format(i)
            ))
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        # if rank == 0:
        #     self.occupy()
        for idx, val_data in enumerate(dataloader):
            # if idx>10:
            #     break
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_ir(val_data, is_val=True)

            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            if self.img_dir:
                mid_img = None
                i_low = None
                i_high = None
            else:
                mid_img = tensor2img([visuals['mid']], rgb2bgr=rgb2bgr)
                i_low = tensor2img([visuals['i_low']], rgb2bgr=rgb2bgr)
                i_high = tensor2img([visuals['i_high']], rgb2bgr=rgb2bgr)
            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.mid
            del self.i_low
            del self.i_high
            # torch.cuda.empty_cache()
            if save_img:
                if self.img_dir:
                    if self.opt['is_train']:
                        dataset_name = os.path.dirname(
                            val_data['lq_path'][0].replace("low_blur", ("low_blur_result_{}".format(str(current_iter)))).replace("lolblur",
                                                                                                    "lol_result"))
                        # print(dataset_name)
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                    else:

                        dataset_name = os.path.dirname(val_data['lq_path'][0].replace("low_blur", ("low_blur_result_{}".format(str(current_iter)))))
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                else:
                    if self.opt['is_train']:
                        dataset_name = os.path.dirname(
                            val_data['lq_path'][0].replace("low_blur", ("low_blur_result_{}".format(str(current_iter)))).replace("lolblur",
                                                                                                    "lol_result"))
                        # print(dataset_name)
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')
                        save_mid_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_mid.png')
                        save_i_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_i_low.png')
                        save_i_img_path2 = osp.join(
                            dataset_name,
                            f'{img_name}_i_high.png')

                    else:

                        dataset_name = os.path.dirname(val_data['lq_path'][0].replace("low_blur", ("low_blur_result_{}".format(str(current_iter)))))
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                if self.img_dir:
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    # print(mid_img.shape,save_mid_img_path)

                else:
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    # print(mid_img.shape,save_mid_img_path)
                    imwrite(mid_img, save_mid_img_path)
                    imwrite(i_low, save_i_img_path)
                    imwrite(i_high, save_i_img_path2)
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        # torch.cuda.empty_cache()
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        # self.x_occupied=[]
        torch.cuda.empty_cache()
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.img_dir:
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
        else:
            if hasattr(self, 'lq'):
                out_dict['lq'] = self.lq.detach().cpu()
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'i_low'):
                out_dict['i_low'] = self.i_low.detach().cpu()

            # out_dict['mid']=self.mid.cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
            if hasattr(self, 'i_high'):
                out_dict['i_high'] = self.i_high.detach().cpu()
            if hasattr(self, 'mid'):
                out_dict['mid'] = self.mid.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


class ImageRestorationModel_dire(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt,
                 ori=False, is_deblur=False, with_ir=False, only_i=False, ir_deblur=False,
                 i_adjust=False, i_adjust_merge=False, img_only=False,img_only_finetune=True,
                 img_3stage=False, img_only_2stage=False, fft2stage=False, img_dir=True,sharp=False):
        super(ImageRestorationModel_dire, self).__init__(opt)
        if is_deblur == True and with_ir == True:
            exit()

        # define network
        self.sharp=sharp
        self.fft2stage = fft2stage
        self.is_ori = ori
        self.is_deblur = is_deblur
        self.with_ir = with_ir
        self.only_i = only_i
        self.ir_deblur = ir_deblur
        self.i_adjust = i_adjust
        self.i_adjust_merge = i_adjust_merge
        self.img_only = img_only
        self.img_only_finetune=img_only_finetune
        self.img_dir = img_dir
        self.img_3stage = img_3stage
        self.img_only_2stage = img_only_2stage
        self.sigmoid = nn.Sigmoid()
        self.use3stage=False
        if is_deblur or with_ir or only_i or ir_deblur or i_adjust or i_adjust_merge:
            # self.model_decome = DecomNet()
            # state = torch.load('/nas2/dldata/tulw/pretrain/decom_200.pkl',
            #                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            # self.model_decome.load_state_dict(state)
            for param in self.model_decome.parameters():
                param.requires_grad = False
            self.model_decome.eval()
            device_id = torch.cuda.current_device()
            device = torch.device("cuda", device_id)
            self.model_decome = self.model_decome.to(device)
            self.net_g = define_network(deepcopy(opt['network_g']))
        else:
            self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.gray_trans = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.cri_mse = nn.MSELoss()
        self.cri_pix_psnr=PSNRLoss(loss_weight=0.5,reduction= 'mean')
        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_loss_opt'):
            fft_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(
                self.device)

        else:
            self.cri_fft = None
        # self.cri_fft=nn.L1Loss()
        self.cri_p = PerceptualLoss(layer_weights={'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1},
                                    use_input_norm=True, perceptual_weight=0.01, style_weight=0, range_norm=True,
                                    criterion='l1').to(self.device)
        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        if self.use3stage:
            self.cri_i1=loss_down(scale=1/2).to(self.device)
            self.cri_i2 = loss_down(scale=1/4).to(self.device)
        else:
            self.cri_i = loss_down(scale=1/8).to(self.device)
        self.cri_a_gai = nn.MSELoss()
        # self.cri_a_gai=a_loss(self.device).to(self.device)
        # self.cri_i = dce_loss(device=self.device).to(self.device)
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
                #             optim_params_lowlr.append(v)
                #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                 **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def feed_data_deblur(self, data, is_val=False):
        with torch.no_grad():
            self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
            self.r_high, _ = self.model_decome(data["gt"].to(self.device))
        self.lq = self.r_low
        # self.i_low=self.i_low.to(self.device)
        # if r_low_gt!=None:
        self.gt = self.r_high

    # def feed_data_for_ir(self,data,is_val=False):
    #     with torch.no_grad():
    #         if self.with_ir:
    #             self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
    #         elif self.only_i:
    #             self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
    def feed_data_ir(self, data, is_val=False):
        with torch.no_grad():
            if self.img_dir:
                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)
            elif self.with_ir:
                self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
            elif self.only_i:
                self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
            elif self.ir_deblur:
                self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
                self.r_high, self.i_high = self.model_decome(data["gt"].to(self.device))

            elif self.i_adjust or self.i_adjust_merge:
                self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
                self.r_high, self.i_high = self.model_decome(data["gt"].to(self.device))
            elif self.is_deblur:
                with torch.no_grad():
                    self.r_low, self.i_low = self.model_decome(data["lq"].to(self.device))
                    self.r_high, _ = self.model_decome(data["gt"].to(self.device))
                self.lq = self.r_low
                # self.i_low=self.i_low.to(self.device)
                # if r_low_gt!=None:
                self.gt = self.r_high
                return
            elif self.is_ori or self.img_only or self.img_3stage or self.img_only_2stage or self.fft2stage:
                self.lq = data['lq'].to(self.device)
                if 'gt' in data:
                    self.gt = data['gt'].to(self.device)
                self.gray_lq = self.gray_trans(self.lq)
                self.gray_gt = self.gray_trans(self.gt)

                if self.img_only:
                    self.gray_lq = self.gray_trans(self.lq)
                    self.gray_gt = self.gray_trans(self.gt)

            # torchvision.utils.save_image(self.r_low.to(torch.device("cpu")),"./r.png")
            # torchvision.utils.save_image(torch.cat([self.i_low,self.i_low,self.i_low],dim=1).to(torch.device("cpu")),"./i.png")
            # self.r_high, _ = self.model_decome(data["gt"].to(self.device))
        self.lq = data['lq'].to(self.device)
        # self.i_low=self.i_low.to(self.device)
        # if self.with_ir or self.ir_deblur or self.i_adjust:
        #     self.r_low=self.r_low.to(self.device)
        # if r_low_gt!=None:
        self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        # adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i // scale * scale
        step_j = step_j // scale * scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(
                    self.lq[:, :, i // scale:(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters_deblur(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        # gpu_tracker.track()
        # print("deblur.............==============")
        if self.img_dir:
            preds = self.net_g(self.lq)
        elif self.fft2stage:
            self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / torch.mean(self.gray_gt, dim=(2, 3))
            preds = self.net_g(self.lq, ratio_i=self.ratio, device=self.device)
        elif self.img_only_2stage:
            self.ratio = torch.mean(self.gray_lq, dim=(2, 3))
            # self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / torch.mean(self.gray_gt, dim=(2, 3))
            preds = self.net_g(self.lq, device=self.device, ratio_i=self.ratio)

        elif self.is_deblur:
            preds = self.net_g(torch.cat([self.lq, self.i_low], dim=1))
        elif self.with_ir:
            preds = self.net_g(torch.cat([self.lq, self.r_low, self.i_low], dim=1))
        elif self.only_i:
            preds = self.net_g(torch.cat([self.lq, self.i_low], dim=1))
        elif self.ir_deblur:
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1))
        elif self.i_adjust:
            self.ratio = torch.mean(self.i_low, dim=(2, 3)) / torch.mean(self.i_high, dim=(2, 3))
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1), device=self.device, ratio_i=self.ratio)
        elif self.i_adjust_merge:
            self.ratio = torch.mean(self.i_low, dim=(2, 3)) / torch.mean(self.i_high, dim=(2, 3))
            preds = self.net_g(torch.cat([self.r_low, self.i_low], dim=1), device=self.device, ratio_i=self.ratio)
        elif self.img_only:
            if self.opt['use_ratio']:
                # print("use ratio")
                self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / torch.mean(self.gray_gt, dim=(2, 3))
            else:
                self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / 1.0
            # batch=self.gray_lq.shape[0]
            # print(torch.mean(self.gray_lq, dim=(2, 3)).shape)
            # self.ratio = torch.mean(self.gray_lq, dim=(2, 3))
            # self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / torch.mean(self.gray_gt, dim=(2, 3))
            # self.ratio = torch.mean(self.gray_lq, dim=(2, 3)) / 1.0
            preds = self.net_g(self.lq, device=self.device, ratio_i=self.ratio)

        elif self.img_3stage:
            preds = self.net_g(self.lq)
        # gpu_tracker.track()
        # if not isinstance(preds, list):
        #     preds = [preds]
        if self.img_dir:
            self.output = preds
        elif self.fft2stage:
            self.output, i_adjust_img, i_adjust = preds
        elif self.ir_deblur and not self.i_adjust:
            self.output = preds[-1] * preds[-2]
        elif self.i_adjust:
            # print(len(preds))
            # self.mid = torch.cat([preds[-1], preds[-1], preds[-1]], dim=1)
            self.output = preds[-3] * preds[-2]
        elif self.i_adjust_merge:
            # print(type(preds),len(preds))
            # print("right!!!!!!!!!")
            self.output, i_adjust = preds
            # print(i_adjust.shape)
        elif self.img_only:
            if len(preds) == 2:
                self.output, i_adjust = preds
                i_adjust2 = None
            elif len(preds) == 3: #xiao da
                self.output, i_adjust2,i_adjust = preds
            elif len(preds) == 4:
                self.output, i_adjust, i_adjust2,i_adjust3 = preds
        elif self.img_3stage:
            self.output, i_adjust2, i_adjust = preds
        else:
            # self.output = preds[-1]
            self.output = preds

        l_total = 0
        loss_dict = OrderedDict()
        if self.sharp:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix_psnr(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)
        elif self.img_dir:

            if self.cri_pix:
                l_pix = 0.
                l_pix += self.cri_pix(self.output, self.gt)

                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix'] = l_pix

            # fft loss
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

                l_per = self.cri_p(self.output, self.gt)[0]
                loss_dict['l_per'] = l_per
                l_total += l_per
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)
        # elif self.img_only_finetune:
        #     if current_iter > 2e5:
        #         aa = 0.01
        #     else:
        #         aa = 0.8
        #     if i_adjust2 != None:
        #         l_g = self.cri_mse(i_adjust2, self.gt) * aa
        #         loss_dict['l_iglobal'] = l_g
        #         l_total += l_g
        #
        #
        #     l_adjust = self.cri_i(i_adjust, self.gt) * aa
        #
        #     loss_dict['l_i_pred'] = l_adjust
        #     l_total += l_adjust
        #     l_total += l_adjust
        #
        #     l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
        #
        #     l_total.backward()
        #
        #     ######################################################
        #
        #     use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        #     if use_grad_clip:
        #         torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        #     self.optimizer_g.step()
        #
        #     self.log_dict = self.reduce_loss_dict(loss_dict)
        elif self.img_only:
            if not self.use3stage:
                if current_iter > 2e5:
                    aa = 0.01
                else:
                    aa = 0.8
                if i_adjust2 != None:
                    l_g = self.cri_mse(i_adjust, self.gt) * aa
                    loss_dict['l_iglobal'] = l_g
                    l_total += l_g
                if self.cri_pix:
                    l_pix = 0.
                    l_pix += self.cri_pix(self.output, self.gt)

                    # print('l pix ... ', l_pix)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # fft loss
                if self.cri_fft:
                    l_fft = self.cri_fft(self.output, self.gt)
                    l_total += l_fft
                    loss_dict['l_fft'] = l_fft
                    #
                    # l_per = self.cri_p(self.output, self.gt)[0]
                    # loss_dict['l_per'] = l_per
                    # l_total += l_per

                l_adjust = self.cri_i(i_adjust2, self.gt) * aa*2

                loss_dict['l_i_pred'] = l_adjust
                l_total += l_adjust

                l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

                l_total.backward()

                ######################################################

                use_grad_clip = self.opt['train'].get('use_grad_clip', True)
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                self.optimizer_g.step()

                self.log_dict = self.reduce_loss_dict(loss_dict)
            elif self.use3stage:
                if current_iter > 2e5:
                    aa = 0.01
                else:
                    aa = 0.8

                l_g1 = self.cri_pix(i_adjust, self.gt) * aa*0.5
                loss_dict['l_i1'] = l_g1
                l_total += l_g1
                l_g2 = self.cri_i1(i_adjust2, self.gt) * aa
                loss_dict['l_i2'] = l_g2
                l_total += l_g2
                l_g3 = self.cri_i2(i_adjust3, self.gt) * aa*2.
                loss_dict['l_i3'] = l_g3
                l_total += l_g3
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)

                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # fft loss
                if self.cri_fft:
                    l_fft = self.cri_fft(self.output, self.gt)
                    l_total += l_fft
                    loss_dict['l_fft'] = l_fft

                    l_per = self.cri_p(self.output, self.gt)[0]
                    loss_dict['l_per'] = l_per
                    l_total += l_per



                l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

                l_total.backward()

                ######################################################

                use_grad_clip = self.opt['train'].get('use_grad_clip', True)
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
                self.optimizer_g.step()

                self.log_dict = self.reduce_loss_dict(loss_dict)
        elif self.img_3stage:
            if self.cri_pix:
                # 1
                l_pix = 0.
                l_pix += self.cri_pix(self.output, self.gt)
                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix1'] = l_pix
                # print(l_pix,"1")
                # 2
                l_pix = 0.
                l_pix += self.cri_pix(i_adjust2, self.gt)
                # print('l pix ... ', l_pix)
                l_total += l_pix
                loss_dict['l_pix_adjust2'] = l_pix
                # print(l_pix,"2")
                # 3
                l_adjust1 = self.cri_i(i_adjust, self.gt)
                # l_pix += self.cri_pix(i_adjust, self.gt)*0.1
                # print('l pix ... ', l_pix)
                l_total += l_adjust1
                loss_dict['l_pix_adjust1'] = l_pix
                # print(l_pix,"3")
            # fft loss
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_fft += self.cri_fft(i_adjust2, self.gt)
                l_total += l_fft
                loss_dict['l_fft'] = l_fft

            l_per = self.cri_p(self.output, self.gt)[0]
            loss_dict['l_per'] = l_per
            l_total += l_per

            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

            l_total.backward()

            ######################################################

            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()

            self.log_dict = self.reduce_loss_dict(loss_dict)

    def center(self, mag_image):
        N, C, H, W = mag_image.size()
        center_h = H // 2
        center_w = W // 2

        # 2. 使用 torch.roll 函数将振幅图的四个象限重排
        # 重排的目标是将中心点移到图像的中心
        mag_image = torch.roll(mag_image, shifts=(center_h, center_w), dims=(2, 3))

        return mag_image

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            outs_mid = []
            outs_ilow = []
            outs_high = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                b, c, h, w = self.lq[i:j].shape
                # 让输入是32的倍数
                #
                h_n = (32 - h % 32) % 32
                w_n = (32 - w % 32) % 32
                in_tensor = F.pad(self.lq[i:j], (0, w_n, 0, h_n), mode='reflect')

                if self.i_adjust or self.i_adjust_merge or self.img_only or self.fft2stage:
                    gt_tensor = F.pad(self.gt[i:j], (0, w_n, 0, h_n), mode='reflect')
                if self.img_dir:
                    pred = self.net_g(in_tensor)
                    i_high = None
                    i_low = None
                    pred_mid = None

                elif self.fft2stage:
                    _, _, H, W = in_tensor.shape
                    i_low = self.gray_trans(in_tensor)
                    i_high = self.gray_trans(gt_tensor)
                    # batch=i_high.shape[0]
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    # ratio = torch.mean(i_low, dim=(2, 3))
                    pred = self.net_g(in_tensor, ori=in_tensor, device=self.device, ratio_i=ratio)
                    # pred = self.net_g(in_tensor,device=self.device)
                    pred_mid = pred[2]  # zhenfu
                    pred_i = pred[1]  # i adjust
                    pred = pred[0]  # result

                    i_low = self.center(torch.log(pred_mid + 1.0))
                    i_low = (i_low - i_low.min()) / (i_low.max() - i_low.min())
                    # fft_img = torch.fft.fft2(gt_tensor)
                    i_high = pred_i
                    # recover _mid
                    image_fft = torch.fft.fft2(pred_i, norm='backward')
                    pha_image = torch.angle(image_fft)

                    real_image_enhanced = pred_mid * torch.cos(pha_image)
                    imag_image_enhanced = pred_mid * torch.sin(pha_image)
                    pred_mid = torch.fft.ifft2(torch.complex(real_image_enhanced, imag_image_enhanced), s=(H, W),
                                               norm='backward').real

                    #
                elif self.is_deblur:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                elif self.with_ir:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                elif self.only_i:
                    _, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([in_tensor, i_low], dim=1))
                elif self.ir_deblur:
                    r_low, i_low = self.model_decome(in_tensor)
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1))
                    pred = pred[-1] * pred[-2]
                elif self.i_adjust:
                    r_low, i_low = self.model_decome(in_tensor)
                    _, i_high = self.model_decome(gt_tensor)
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1), device=self.device, ratio_i=ratio)
                    pred_mid = pred[-1]
                    pred = pred[-3] * pred[-2]
                elif self.i_adjust_merge:
                    # print("mememememmemememem!!!!!")
                    r_low, i_low = self.model_decome(in_tensor)
                    _, i_high = self.model_decome(gt_tensor)
                    ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    pred = self.net_g(torch.cat([r_low, i_low], dim=1), ori=in_tensor, device=self.device,
                                      ratio_i=ratio)
                    pred_mid = pred[-1] * r_low
                    pred = pred[-2]
                elif self.img_only:
                    i_low = self.gray_trans(in_tensor)
                    i_high = self.gray_trans(gt_tensor)
                    # batch=i_high.shape[0]
                    if self.opt['use_ratio']:
                        ratio = torch.mean(i_low, dim=(2, 3)) / torch.mean(i_high, dim=(2, 3))
                    else:
                        ratio = torch.mean(i_low, dim=(2, 3)) / 1.0
                    # ratio = torch.mean(i_low, dim=(2, 3))
                    pred = self.net_g(in_tensor, ori=in_tensor, device=self.device, ratio_i=ratio)
                    if len(pred) == 3:
                        i_low = pred[2]
                        pred_mid = pred[1]
                        pred = pred[0]
                    elif len(pred)==4:
                        i_high=pred[3]
                        i_low = pred[2]
                        pred_mid = pred[1]
                        pred = pred[0]

                elif self.img_3stage:
                    pred = self.net_g(in_tensor)
                    pred_mid = pred[1]
                    i_low = pred[2]
                    pred = pred[0]

                    i_high = pred
                else:
                    pred = self.net_g(in_tensor)
                if self.img_dir:
                    pred = pred[:, :, :h, :w]
                    # pred_mid = pred_mid[:, :, :h, :w]
                    # i_low = i_low[:, :, :h, :w]
                    # i_high = i_high[:, :, :h, :w]
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    # outs_mid.append(pred_mid.detach().cpu())
                    # outs_ilow.append(i_low.detach().cpu())
                    # outs_high.append(i_high.detach().cpu())
                else:
                    pred = pred[:, :, :h, :w]
                    pred_mid = pred_mid[:, :, :h, :w]
                    i_low = i_low[:, :, :h, :w]
                    i_high = i_high[:, :, :h, :w]
                    if isinstance(pred, list):
                        pred = pred[-1]
                    outs.append(pred.detach().cpu())
                    outs_mid.append(pred_mid.detach().cpu())
                    outs_ilow.append(i_low.detach().cpu())
                    outs_high.append(i_high.detach().cpu())
                i = j
            if self.img_dir:
                self.mid = None
                self.output = torch.cat(outs, dim=0)
                self.i_low = None
                self.i_high = None
            else:
                self.mid = torch.cat(outs_mid, dim=1)
                self.output = torch.cat(outs, dim=0)
                self.i_low = torch.cat(outs_ilow)
                self.i_high = torch.cat(outs_high)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            # if idx>10:
            #     break
            if idx % world_size != rank:
                continue
            # save_name = name[0]
            #
            # print("mkdir ", data_root[0])

            # print(val_data['lq_path'][0],"==========")
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_ir(val_data, is_val=True)
            # if self.is_deblur:
            #     self.feed_data_deblur(val_data, is_val=True)
            # elif self.with_ir or self.only_i:
            #     self.feed_data_ir(val_data, is_val=True)
            # else:
            #     self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            if self.img_dir:
                mid_img = None
                i_low = None
                i_high = None
            else:
                mid_img = tensor2img([visuals['mid']], rgb2bgr=rgb2bgr)
                i_low = tensor2img([visuals['i_low']], rgb2bgr=rgb2bgr)
                i_high = tensor2img([visuals['i_high']], rgb2bgr=rgb2bgr)
            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.mid
            del self.i_low
            del self.i_high
            torch.cuda.empty_cache()
            # print(osp.join(self.opt['path']['visualization'],"===", dataset_name))
            if save_img:
                if self.img_dir:
                    if self.opt['is_train']:
                        dataset_name = os.path.dirname(
                            val_data['lq_path'][0].replace("low_blur", ("low_blur_result_{}".format(str(current_iter)))).replace("lolblur",
                                                                                                    "lol_result"))
                        # print(dataset_name)
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                    else:

                        dataset_name = os.path.dirname(val_data['lq_path'][0].replace("low_blur", ("low_blur_result_{}".format(str(current_iter)))))
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                else:
                    if self.opt['is_train']:
                        dataset_name = os.path.dirname(
                            val_data['lq_path'][0].replace("low_blur", ("low_blur_result_{}".format(str(current_iter)))).replace("lolblur",
                                                                                                    "lol_result"))
                        # print(dataset_name)
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')
                        save_mid_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_mid.png')
                        save_i_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_i_low.png')
                        save_i_img_path2 = osp.join(
                            dataset_name,
                            f'{img_name}_i_high.png')

                    else:

                        dataset_name = os.path.dirname(val_data['lq_path'][0].replace("low_blur", ("low_blur_result_{}".format(str(current_iter)))))
                        if not os.path.exists(dataset_name):
                            os.makedirs(dataset_name, exist_ok=True)
                        save_img_path = osp.join(
                            dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            dataset_name,
                            f'{img_name}_gt.png')

                if self.img_dir:
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    # print(mid_img.shape,save_mid_img_path)

                else:
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    # print(mid_img.shape,save_mid_img_path)
                    imwrite(mid_img, save_mid_img_path)
                    imwrite(i_low, save_i_img_path)
                    imwrite(i_high, save_i_img_path2)
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics

        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.img_dir:
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
        else:
            if hasattr(self, 'lq'):
                out_dict['lq'] = self.lq.detach().cpu()
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'i_low'):
                out_dict['i_low'] = self.i_low.detach().cpu()

            # out_dict['mid']=self.mid.cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
            if hasattr(self, 'i_high'):
                out_dict['i_high'] = self.i_high.detach().cpu()
            if hasattr(self, 'mid'):
                out_dict['mid'] = self.mid.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)