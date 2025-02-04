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

try:
    from datasets_wds.video_utils import (
        TemporalRandomCrop,
        RandomHorizontalFlipVideo,
        ToTensorVideo,
        UCFCenterCropVideo,
    )
except ImportError:
    from video_utils import (
        TemporalRandomCrop,
        RandomHorizontalFlipVideo,
        ToTensorVideo,
        UCFCenterCropVideo,
    )
import numpy as np
import io


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


class TWHC_To_TCWH:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, x):
        x = rearrange(x, "t  h w  c-> t c h w")
        return x

    def __repr__(self) -> str:
        return self.__class__.__name__


class H5VideoDataset(Dataset):
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
        self.transform = transforms.Compose(
            [
                ToTensorVideo(),  # TCHW
                RandomHorizontalFlipVideo(),
                UCFCenterCropVideo(image_size),
            ]
        )
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.video_len = len(f["start_index_list"])
            self.start_index_list = f["start_index_list"][()]
            print("video_len", self.video_len)

        self.target_video_len = video_frames
        self.temporal_sample = TemporalRandomCrop(video_frames * frame_interval)  # 16 1

    def __len__(self):
        return self.video_len

    def __getitem__(self, index):
        start_index, end_index = self.start_index_list[index]
        total_frames = end_index - start_index
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int
        )
        with h5py.File(self.h5_path, "r") as f:
            video = torch.from_numpy(f["video"][frame_indice + start_index])  #
        return dict(indices=video)


def get_inf_h5_dataloader(**kwargs):
    per_gpu_batch_size = kwargs["per_gpu_batch_size"]
    num_workers_per_gpu = kwargs["num_workers_per_gpu"]
    dataloader = H5VideoDataset(**kwargs)

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

    dataloader = H5VideoDataset(
        h5_path="./data/faceforensics_train.h5",
        num_train_examples=13320,
        per_gpu_batch_size=3,
        global_batch_size=3,
        num_workers_per_gpu=1,
    )

    dl = DataLoader(dataloader, batch_size=4, num_workers=1)

    for batch in dl:
        print(batch.keys())
        print(batch["image"].shape, batch["image"].max(), batch["image"].min())
        # print(batch["class_id"].shape, batch["class_id"].max(), batch["class_id"].min())
        break
