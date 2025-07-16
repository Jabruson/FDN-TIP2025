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
class MAR_ARCH(nn.Module):
    def __init__(self,use_ratio):
        super(MAR_ARCH, self).__init__()
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

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        x_2_p = self.downsample1(x)
        x_4_p = self.downsample2(x)
        # print(x_2_p.shape,x_4_p.shape)
        z2 = self.f2(x_2_p)
        # print(ratio.shape)
        if self.use_ratio:
            z2 = z2 * ratio
        z4 = self.f1(x_4_p)
        if self.use_ratio:
            z4 = z4 * ratio
        outputs = list()

        x_ = self.f3(x)
        if self.use_ratio:
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
        self.net=MAR_ARCH(use_ratio=use_ratio)
        self.down1 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=False)
        self.scale=40.0
        self.use_ratio=use_ratio

    def forward(self, x, ratio=None):
        # print(self.use_ratio)
        B, _, h, w = x.shape
        x_high1 = x
        x_high2 = self.down1(x_high1)
        x_high3 = self.down1(x_high2)
       



        i_high3m, i_high2m, i_high1m = self.net(x,ratio)  # 8 downsample small to large


            # global_npy = np.fft.fftshift(tensor_cpu.numpy())

        x_high1 = 1.0 - torch.pow(1.0 - x_high1, i_high1m*self.scale)
        x_high2 = 1.0 - torch.pow(1.0 - x_high2, i_high2m*self.scale)
        x_high3 = 1.0 - torch.pow(1.0 - x_high3, i_high3m*self.scale)
        
            # 将 CPU 上的张量转换为 numpy 数组
        return x_high3, x_high2, x_high1

        # x_high3 = torch.fft.rfft2(x_high3.float(), norm='backward')
        # # x_high32 = torch.angle(x_high3)
        # x_high3 = torch.abs(x_high3)
        # ###################
        # x_high12 = torch.complex(x_high1 * torch.cos(x_high12), x_high1 * torch.sin(x_high12))
        # x_high1 = torch.fft.irfft2(x_high12, s=(h, w), norm='backward')
        #
        # x_high22 = torch.complex(x_high2 * torch.cos(x_high22), x_high2 * torch.sin(x_high22))
        # x_high2 = torch.fft.irfft2(x_high22, s=(h // 2, w // 2), norm='backward')
        #
        # x_high32 = torch.complex(x_high3 * torch.cos(x_high32), x_high3 * torch.sin(x_high32))
        # x_high3 = torch.fft.irfft2(x_high32, s=(h // 4, w // 4), norm='backward')
        # return x_high3,x_high2,x_high1