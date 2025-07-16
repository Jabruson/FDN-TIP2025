import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange



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



class SEBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(SEBlock, self).__init__()
        filter1, filter2, filter3 = filters
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3))
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(filter3, filter3 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filter3 // 16, filter3, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.se(x1)
        x1 = x1 * x2
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x1 = x1 + x_shortcut
        x1 = self.relu(x1)
        return x1


from torchvision import transforms

class I_predict_net(nn.Module):
    def __init__(self,c=16):
        super(I_predict_net, self).__init__()
        # N,C,H//4,W//4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        self.conv2 = self._make_layer(c, (c, c, 2*c), 3, 1)
        self.conv3 = self._make_layer(2*c, (2*c, 2*c, 4*c), 3, 2)
        self.conv4 = self._make_layer(4*c, (4*c, 4*c, 8*c), 6, 6)
        # self.conv5=nn.Conv2d(8*c, 8*c, kernel_size=3, stride=1, padding=1, bias=False)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(8*c, 8*c))
        self.fc2 = nn.Sequential(nn.Linear(8*c, 1))
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.trans_gray = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

    def _make_layer(self, in_channels, filters, num, stride=1):
        layers = []
        block_1 = SEBlock(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(SEBlock(filters[2], filters, stride=1, is_1x1conv=False))
        return nn.Sequential(*layers)

    def forward(self, x,use_ori_i=False):
        gray_value = self.trans_gray(x)

        gray_value=torch.mean(gray_value, dim=(2, 3))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = self.global_average_pool(x)
        x = rearrange(x, "B C H W -> B (H W C)")  # Linear Embedding

        x = self.fc(x)
        x = self.fc2(x)
        x = self.sigmoid(x) #hq

        if use_ori_i:
            x=gray_value/x

        return x

