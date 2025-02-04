import torch
from utils_vq import vq_get_sample_size
from einops import rearrange
import wandb
from utils_vq import array2row_pixel, array2grid_pixel
import imageio
import os
import uuid
from datasets_wds.cityscapes_helper import (
    cityscapes_only_categories_indices_segmentation_to_img,
)
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import random


def all_gather_my(tensor_in):
    return tensor_in


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def resize_img_as(img, target, mode="bilinear"):
    return F.interpolate(img.float(), size=target.shape[-2:], mode=mode).byte()


def btchw_resize_img_as(source, target, mode="bilinear"):
    b, t, c, h, w = source.shape
    img = rearrange(source, "b t c h w -> (b t) c h w")
    img = resize_img_as(img, target, mode)
    img = rearrange(img, "(b t) c h w -> b t c h w", b=b)
    return img

