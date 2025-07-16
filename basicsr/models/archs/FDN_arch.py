import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

# from basicsr.models.archs.mymimo2_arch import *
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True,relu=True, transpose=False):
        super(BasicConv, self).__init__()
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if relu:
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)




class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)




class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge1 = nn.Conv2d(channel*2,channel,kernel_size=1)

        self.merge2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1)
    def forward(self, x1, x2):
        out=torch.cat([x1,x2],dim=1)
        out=self.merge2(self.merge1(out))
        return out
class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return x + self.block(x)


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out + x
class ProcessBlock(nn.Module):
    def __init__(self, in_nc, spatial=False):
        super(ProcessBlock, self).__init__()
        self.spatial = spatial
        self.spatial_process = SpaBlock(in_nc) if spatial else nn.Identity()
        self.frequency_process = FreBlock(in_nc)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0) if spatial else nn.Conv2d(in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        if self.spatial:
            xcat = torch.cat([x_spatial, x_freq], 1)
            x_out = self.cat(xcat)
            return x_out + xori
        else:
            return x_freq + xori


class fourier_fuse(nn.Module):
    def __init__(self,in_nc,out_nc):
        super(fourier_fuse, self).__init__()
        # self.fpre = nn.Conv2d(in_nc, out_nc, 3, 1,1)
        self.fpre=nn.Sequential(nn.Conv2d(in_nc, out_nc, 1,1),
                                nn.Conv2d(out_nc, out_nc, 1, 1,1,groups=out_nc))
        self.process1 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_nc, out_nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_nc, out_nc, 1, 1, 0))
        self.fourier_out=nn.Conv2d(out_nc, out_nc, 3,1,1)
    def forward(self, x1,x2,x4):
        x = torch.cat([x1, x2, x4], dim=1)
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return self.fourier_out(x_out)
class MAR_archa(nn.Module):
    def __init__(self,use_ratio):
        super(MAR_archa, self).__init__()
        self.use_ratio=use_ratio
        base_channel = 12

        self.Encoder = nn.ModuleList([
            ProcessBlock(in_nc=base_channel),
            ProcessBlock(in_nc=base_channel * 2),
            ProcessBlock(in_nc=base_channel * 4),
        ])

        self.Decoder = nn.ModuleList([
            ProcessBlock(in_nc=base_channel * 4),
            ProcessBlock(in_nc=base_channel * 2),
            ProcessBlock(in_nc=base_channel)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            fourier_fuse(base_channel * 7, base_channel * 1),
            fourier_fuse(base_channel * 7, base_channel * 2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        # self.f1=nn.Conv2d(3,base_channel*4,kernel_size=3,stride=1)
        self.f1 = nn.Sequential(*[nn.Conv2d(3 * 16, base_channel * 4, 1, 1, 0),
                                  ProcessBlock(base_channel * 4)])
        self.f2 = nn.Sequential(*[nn.Conv2d(3 * 4, base_channel * 2, 1, 1, 0),
                                  ProcessBlock(base_channel * 2)])
        self.f3 = nn.Sequential(*[nn.Conv2d(3, base_channel, 1, 1, 0),
                                  ProcessBlock(base_channel)])
        self.f3_down = BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2)
        self.f2_down = BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2)
        self.f2_up = BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True)
        self.f3_up = BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True)
        self.out = BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        self.FAM2 = FAM(base_channel * 2)
        self.sigmoid = nn.Sigmoid()
        self.downsample1 = nn.PixelUnshuffle(2)
        self.downsample2 = nn.PixelUnshuffle(4)
        self.e = 0.00000001

    def forward(self, x, ratio=None):
        # print(ratio,"???")
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_2_p = self.downsample1(x)
        x_4_p = self.downsample2(x)
        # print(x_2_p.shape,x_4_p.shape)
        z2 = self.f2(x_2_p)
        print(ratio.shape)
        
        z2 = z2 * ratio
        z4 = self.f1(x_4_p)
        z4 = z4 * ratio
        outputs = list()

        x_ = self.f3(x)
        x_ = x_ * ratio
        res1 = self.Encoder[0](x_)

        z = self.f3_down(res1)  # 4->2 c->2c
        z = self.FAM2(z, z2)  # 融合尺度
        res2 = self.Encoder[1](z)

        z = self.f2_down(res2)  # 2->1 2c->4c
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.f2_up(z)  # 1->2 4c->2c
        outputs.append(self.sigmoid(z_ + x_4) + self.e)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.f3_up(z)
        outputs.append(self.sigmoid(z_ + x_2) + self.e)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.out(z)

        outputs.append(self.sigmoid(z + x) + self.e)

        return outputs


import numpy as np
class MAR(nn.Module):
    def __init__(self,use_ratio=True):
        super(MAR, self).__init__()
        self.net=MAR_archa(use_ratio=True)
        self.down1 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=False)
        self.scale=40.0
        self.use_ratio=use_ratio

    def forward(self, x, ratio=None):

        B, _, h, w = x.shape
        x_high1 = x
        x_high2 = self.down1(x_high1)
        x_high3 = self.down1(x_high2)



        i_high3m, i_high2m, i_high1m = self.net(x,ratio)  # 8 downsample small to large


        # print(i_high1m.max(),ratio)
        x_high1 = 1.0 - torch.pow(1.0 - x_high1, i_high1m*self.scale)
        x_high2 = 1.0 - torch.pow(1.0 - x_high2, i_high2m*self.scale)
        x_high3 = 1.0 - torch.pow(1.0 - x_high3, i_high3m*self.scale)
      
        return x_high3, x_high2, x_high1

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)





class FCAFFN(nn.Module):
    def __init__(self, dim, bias, r=1.0,use_light=True,use_img=True):
        super(FCAFFN, self).__init__()
        # self.ps = nn.PixelShuffle(r)
        hidden=int(r*dim)
        self.use_light=use_light
        self.use_img=use_img
        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        if use_light:
            self.conv1_xa= nn.Conv2d(3, hidden, kernel_size=1, bias=bias)
            self.conv1_xp = nn.Conv2d(3, hidden, kernel_size=1, bias=bias)
        if use_img:
            self.conv1_add = nn.Conv2d(3, hidden, kernel_size=1, bias=bias)
            self.conv1_mul = nn.Conv2d(3, hidden, kernel_size=1, bias=bias)
            self.conv3_add = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1,
                                       groups=hidden, bias=bias)
            self.conv3_mul = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1,
                                       groups=hidden, bias=bias)
            self.norm = LayerNorm(hidden, LayerNorm_type='WithBias')
            self.dwconv = nn.Conv2d(hidden, hidden*2, kernel_size=3, stride=1, padding=1,
                                     groups=hidden, bias=bias)


    def forward(self, x, x_high,xp2,x_img=None):
        # print(x.shape,x_high.shape,xp2.shape)
        _, _, h, w = x.shape

        if self.use_light:
            x1=x
            x = torch.fft.rfft2(x.float(), norm='backward')
            x = replace_denormals(x)
            x_p = torch.angle(x) - self.conv1_xp(xp2)
            x_a = torch.abs(x)
            x_a = self.conv1_xa(x_high) * x_a

            x = torch.complex(x_a * torch.cos(x_p), x_a * torch.sin(x_p))
            x = torch.fft.irfft2(x, s=(h, w), norm='backward')

            x = self.norm(x)*x1+x1
        x = self.project_in(x)
        if self.use_light:
            x = x * self.conv3_mul(self.conv1_mul(x_img)) + self.conv3_add(self.conv1_add(x_img))

        # print("++", torch.max(x), torch.min(x))
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x=self.project_out(x)
        return x
class FDFFN(nn.Module):
    def __init__(self, dim, bias, r=2.7,use_light=True,use_img=True):
        super(FDFFN, self).__init__()
        # self.ps = nn.PixelShuffle(r)
        hidden=int(r*dim)
        self.space = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1,
                      groups=hidden, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1,
                      groups=hidden, bias=bias)
        )
        self.patch_size=8
        self.use_light=use_light
        self.use_img=use_img
        self.ffta = nn.Parameter(torch.ones((hidden, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.fftp = nn.Parameter(torch.zeros((hidden, 1, 1, self.patch_size, self.patch_size // 2 + 1))*torch.pi)
        self.gelu = nn.GELU()
        self.dwconv = nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden, bias=bias)

        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
    def forward(self, x, x_high=None,xp2=None,x_img=None):
        # print(x.shape,x_high.shape,xp2.shape)
        _, _, h, w = x.shape
        x=self.project_in(x)
        x1=self.space(x)
        x = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                      patch2=self.patch_size)
        x = torch.fft.rfft2(x.float())
        x = replace_denormals(x)
        x_p = torch.angle(x)
        x_a = torch.abs(x)
        x_p = x_p - self.fftp
        x_a = x_a * self.ffta
        real = x_a * torch.cos(x_p)
        imag = x_a * torch.sin(x_p)
        x = torch.complex(real, imag)
        x = torch.fft.irfft2(x, s=(self.patch_size, self.patch_size))
        x = rearrange(x, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)')+x1
        # print("++", torch.max(x), torch.min(x))
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x=self.project_out(x)
        return x


class PatchEmbedding(nn.Module): 
    def __init__(self, patch_size=4, in_channels=3, emb_dim=96):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size) 
    def forward(self, x):
        # (B,C,H,W)
        x = self.conv(x)
        _, _, H, W = x.shape
        x = rearrange(x, "B C H W -> B (H W) C")  # Linear Embedding
        return x, H, W


class Mlp(nn.Module):
    def __init__(self, hidden_size, expand_factor=4):
        super(Mlp, self).__init__()
        mid_dim = int(expand_factor * hidden_size)
        self.fc1 = nn.Linear(hidden_size, mid_dim)
        self.fc2 = nn.Linear(mid_dim, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = x
        # x = self.ffn_norm(x)
        # mlp start
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        # mlp end
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        return x


class get_p(nn.Module):
    def __init__(self):
        super(get_p, self).__init__()

    def forward(self, x):
        B, _, H, W = x.shape
        # _, _, H, W = inp_img.shape
        y = torch.fft.rfft2(x, norm='backward')
        # mag = torch.abs(y)  # 振幅图
        pha = torch.angle(y)  # 相位图

        real = torch.cos(pha)
        imag = torch.sin(pha)
        # # 纯相位图
        y_pha = torch.complex(real, imag)

        y = torch.fft.irfft2(y_pha, s=(H, W), norm='backward')


def replace_denormals_angle(x: torch.tensor, threshold=1e-10):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y


def replace_denormals(x, threshold=1e-10):
    y_real = x.real.clone()
    y_imag = x.imag.clone()
    y_real[(x.real < threshold) & (x.real > -1.0 * threshold)] = threshold
    y_imag[(x.imag < threshold) & (x.imag > -1.0 * threshold)] = threshold
    return torch.complex(y_real, y_imag)

import numpy as np
class FDSA(nn.Module):
    def __init__(self, dim, bias):
        super(FDSA, self).__init__()
        self.inner = 4

        self.expand_dim = int(dim * 1.2)
        self.to_hidden = nn.Conv2d(dim, self.expand_dim * self.inner, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(self.expand_dim * self.inner, self.expand_dim * self.inner, kernel_size=3,
                                      stride=1, padding=1, groups=self.expand_dim * self.inner, bias=bias)

        self.project_out = nn.Conv2d(self.expand_dim * 3, dim, kernel_size=1, bias=bias)

        self.norm1 = LayerNorm(self.expand_dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(self.expand_dim, LayerNorm_type='WithBias')
        self.norm3 = LayerNorm(self.expand_dim, LayerNorm_type='WithBias')
        self.patch_size = 8
        self.fft = nn.Parameter(torch.ones((self.expand_dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.e = 1e-10

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v, v_value = self.to_hidden_dw(hidden).chunk(self.inner, dim=1)
        v = rearrange(v, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                      patch2=self.patch_size)
        q = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                      patch2=self.patch_size)
        k = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                      patch2=self.patch_size)
        q = torch.fft.rfft2(q.float(), norm='backward')

        k = torch.fft.rfft2(k.float(), norm='backward')

        v = torch.fft.rfft2(v.float(), norm='backward')
        # q=replace_denormals(q)
        v = v * self.fft
        # k=replace_denormals(k)
        v = replace_denormals(v)
       
        qk = q * k
       
        qk = replace_denormals(qk)
      
        qka = torch.abs(qk)
        # qkp=torch.angle(qk)
        v_a = torch.abs(v)
        v_p = torch.angle(v)
        q = replace_denormals(q)
        k = replace_denormals(k)
        qp = torch.angle(q)
        kp = torch.angle(k)
        qkp = qp - kp
        # print(v_a.shape,out_p.shape)
        real = v_a * torch.cos(qkp)
        imag = v_a * torch.sin(qkp)

        out = torch.complex(real, imag)
        # out2=torch.complex(real2, imag2)
        out1 = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size), norm='backward')

        # qka=qa*ka
        real = qka * torch.cos(v_p)
        imag = qka * torch.sin(v_p)
        out = torch.complex(real, imag)
        out2 = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size), norm='backward')
        out1 = rearrange(out1, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                         patch2=self.patch_size)
        out2 = rearrange(out2, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                         patch2=self.patch_size)
        # out3=rearrange(out2, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                 patch2=self.patch_size)
        real = qka * torch.cos(qkp)
        imag = qka * torch.sin(qkp)
        out = torch.complex(real, imag)
        out3 = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size), norm='backward')
        out3 = rearrange(out3, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                         patch2=self.patch_size)
        out1 = self.norm1(out1)
        out2 = self.norm2(out2)
        out3 = self.norm3(out3)
        output1 = v_value * out1
        output2 = v_value * out2
        output3 = v_value * out3
        output = self.project_out(torch.cat([output1, output2, output3], dim=1))

        return output



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, mode=1, bias=False, LayerNorm_type='WithBias', att=False,use_light=True,use_img=True):
        super(TransformerBlock, self).__init__()
        self.use_light=use_light
        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FDSA(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        if mode == 1:
            self.ffn = FDFFN(dim, bias,use_light=use_light,use_img=use_img)
        elif mode == 2:
            self.ffn = FDFFN(dim, bias,use_light=use_light,use_img=use_img)
        if use_light==True:
            self.norm3 = LayerNorm(dim, LayerNorm_type)
            self.ffn2 = FCAFFN(dim, bias, use_light=use_light, use_img=use_img)
        # self.ffn = MLP(dim)

    def forward(self, x):

        x, x_high ,x_p,x_img = x
        if self.att:
            # print(x.shape)
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x), x_high,x_p,x_img)
        if self.use_light == True:
            x=x + self.ffn2(self.norm3(x),x_high,x_p,x_img)

        return x, x_high,x_p,x_img

class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.att_channel = TransformerBlock(dim=n_feat * 2,use_light=False,use_img=False)

        self.conv = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0)

    def forward(self, enc, dnc, x_high,x_high_p,x_img):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        x = self.att_channel((x, x_high,x_high_p,x_img))[0]
        x = self.conv2(x)
        e, d = torch.split(x, [self.n_feat, self.n_feat], dim=1)
        output = e + d

        return output


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                     out_channels=out_c,
                     kernel_size=k,
                     stride=s,
                     padding=p, dilation=dilation, groups=groups)










class FDformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12, 8],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(FDformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), mode=2, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True,use_light=False,use_img=False) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), mode=2, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True,use_light=False,use_img=False) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), mode=2, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True,use_light=False,use_img=False) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), mode=2, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, att=True,use_light=False,use_img=False) for i in range(num_refinement_blocks)])

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.down1 = nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=False)
        self.norm=LayerNorm(3, LayerNorm_type='WithBias')

    def forward(self, inp_img, ori_img=None, x_high1=None,x_high2=None,x_high3=None,x_high12=None,x_high22=None,x_high32=None,x1=None,x2=None,x3=None):

        _, _, h, w = inp_img.shape
        out_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1((out_enc_level1, x_high1,x_high12,x1))
        out_enc_level1 = out_enc_level1[0]
        out_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2((out_enc_level2, x_high2,x_high22,x2))
        out_enc_level2 = out_enc_level2[0]
        out_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3((out_enc_level3, x_high3,x_high32,x3))
        out_enc_level3 = out_enc_level3[0]
        out_enc_level3 = self.decoder_level3((out_enc_level3, x_high3,x_high32,x3))
        out_enc_level3 = out_enc_level3[0]
        inp_dec_level2 = self.up3_2(out_enc_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2, x_high2,x_high22,x2)
        inp_dec_level2 = self.decoder_level2((inp_dec_level2, x_high2,x_high22,x2))
        inp_dec_level2 = inp_dec_level2[0]
        inp_dec_level2 = self.up2_1(inp_dec_level2)

        inp_dec_level2 = self.fuse1(inp_dec_level2, out_enc_level1, x_high1,x_high12,x1)
        inp_dec_level2 = self.decoder_level1((inp_dec_level2, x_high1,x_high12,x1))
        inp_dec_level2 = inp_dec_level2[0]
        inp_dec_level2 = self.refinement((inp_dec_level2, x_high1,x_high12,x1))
        inp_dec_level2 = inp_dec_level2[0]
        inp_dec_level2 = self.output(inp_dec_level2)
        if ori_img == None:

            result = inp_dec_level2 + inp_img
        else:
            result = inp_dec_level2 + ori_img
        return result




class FDN(nn.Module):
    def __init__(self):
        super(FDN, self).__init__()
        self.net_a = MAR(use_ratio=True)
        self.net_p = FDformer(inp_channels=3,
                            out_channels=3,
                            dim=32,
                            num_blocks=[6, 6, 10],
                            num_refinement_blocks=4,
                            ffn_expansion_factor=3,
                            bias=False)
        for param in self.net_a.parameters():
            param.requires_grad = False
        state = torch.load(
            '/data/tuluwei/code/fourier_gamma.pth')
        self.net_a.load_state_dict(state['params'], strict=True)
        self.norm1 = LayerNorm(3, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(3, LayerNorm_type='WithBias')
        self.norm3 = LayerNorm(3, LayerNorm_type='WithBias')
        self.down1 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=False)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, inp_img, ori=None, device=None, ratio_i=None, mode=1):
        B, _, h, w = inp_img.shape

        ratio_i = ratio_i.unsqueeze(-1).unsqueeze(-1)
        # phase
        x_high1 = inp_img
        x_high2 = self.down1(x_high1)
        x_high3 = self.down1(x_high2)

        x_high3 = self.norm3(x_high3)
        x_high2 = self.norm2(x_high2)
        x_high1 = self.norm1(x_high1)

        x_high12 = torch.fft.rfft2(x_high1.float(), norm='backward')
        x_high12 = replace_denormals(x_high12)
        x_high12 = torch.angle(x_high12)

        x_high22 = torch.fft.rfft2(x_high2.float(), norm='backward')
        x_high22 = replace_denormals(x_high22)
        x_high22 = torch.angle(x_high22)

        x_high32 = torch.fft.rfft2(x_high3.float(), norm='backward')
        x_high32 = replace_denormals(x_high32)
        x_high32 = torch.angle(x_high32)


        x_high3q, x_high2q, x_high1q = self.net_a(inp_img, ratio_i)  # 8 downsample small to large
        x_high3 = self.norm3(x_high3q)
        x_high2 = self.norm2(x_high2q)
        x_high1 = self.norm1(x_high1q)

        # ################
        x_high1 = torch.fft.rfft2(x_high1.float(), norm='backward')
        # x_high1 = replace_denormals(x_high1)
        # x_high12 = torch.angle(x_high1)
        x_high1 = torch.abs(x_high1)

        x_high2 = torch.fft.rfft2(x_high2.float(), norm='backward')
        # x_high2 = replace_denormals(x_high2)
        # x_high22 = torch.angle(x_high2)
        x_high2 = torch.abs(x_high2)

        x_high3 = torch.fft.rfft2(x_high3.float(), norm='backward')
        # x_high3 = replace_denormals(x_high3)
        # x_high32 = torch.angle(x_high3)
        x_high3 = torch.abs(x_high3)

        inp_img = self.net_p(inp_img, ori_img=inp_img,
                             x_high1=x_high1, x_high2=x_high2, x_high3=x_high3,
                             x_high12=x_high12, x_high22=x_high22, x_high32=x_high32,
                             x1=x_high1q, x2=x_high2q, x3=x_high3q)
        # global global_npy
        return inp_img, x_high1q, x_high2q, x_high3q


