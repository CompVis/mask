"""This file contains the definition of data loader using webdataset.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
from typing import List, Union, Text
from torch.utils.data import Dataset
import webdataset as wds
from torch.utils.data import default_collate
from torchvision import transforms
import torchvision
from einops import rearrange
import torch
import h5py
from torch.utils.data import DataLoader


import numpy as np
import io


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


class H5ImageDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int = 12,
        image_size: int = 256,
        video_frames: int = 16,
        frame_interval: int = 1,
        name: str = "ucf101",
        **kwargs,
    ):
        """Initializes the WebDatasetReader class.

        Args:
            train_shards_path: A string or list of string, path to the training data shards in webdataset format.
            eval_shards_path: A string or list of string, path to the evaluation data shards in webdataset format.
            num_train_examples: An integer, total number of training examples.
            per_gpu_batch_size: An integer, number of examples per GPU batch.
            global_batch_size: An integer, total number of examples in a batch across all GPUs.
            num_workers_per_gpu: An integer, number of workers per GPU.
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        """

        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.image_len = len(f["indice"])
            print("image_len", self.image_len)

    def __len__(self):
        return self.image_len

    def __getitem__(self, index):
        with h5py.File(self.h5_path, "r") as f:
            indice = torch.from_numpy(f["indice"][index])  #
            cls_id = int(f["cls_id"][index])  #
        return dict(indices=indice, cls_id=cls_id)


def get_inf_h5_dataloader(**kwargs):
    per_gpu_batch_size = kwargs["per_gpu_batch_size"]
    num_workers_per_gpu = kwargs["num_workers_per_gpu"]
    dataloader = H5ImageDataset(**kwargs)

    def _gen():
        while True:
            dl = DataLoader(
                dataloader,
                batch_size=per_gpu_batch_size,
                num_workers=num_workers_per_gpu,
            )
            for batch in dl:
                yield batch

    return _gen()


if __name__ == "__main__":

    dataloader = H5ImageDataset(
        h5_path="./data/imagenet100_size256.h5",
        num_train_examples=13320,
        per_gpu_batch_size=3,
        global_batch_size=3,
        num_workers_per_gpu=1,
    )

    dl = DataLoader(dataloader, batch_size=40, num_workers=1, shuffle=True)

    for batch in dl:
        print(batch.keys())
        print(batch["indices"].shape, batch["indices"].max(), batch["indices"].min())
        print(batch["cls_id"].shape, batch["cls_id"].max(), batch["cls_id"].min())
        break
