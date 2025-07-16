

import os
import csv
import numpy as np
import torch
import pyiqa
import argparse
from pyiqa.utils.img_util import imread2tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS
import glob
device = torch.device('cuda:2')
fid_metric = pyiqa.create_metric('fid', device=device)
f1='/data/tuluwei/dataset/lolblur_v48/test/low_blur_noise'
f2='/data/tuluwei/dataset/lolblur/test/high_sharp_scaled'

FID = fid_metric(f1,f2)
print(f1,f2,FID)