"""This file contains the definition of data loader using webdataset.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
from typing import List, Union, Text
import webdataset as wds
from torch.utils.data import default_collate
from torchvision import transforms
import random
from functools import partial

try:
    from utils_vq import print_rank_0
except ImportError:

    def print_rank_0(msg):
        print(msg)


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


class ImageTransform:
    def __init__(
        self,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: List[float] = [0.0, 0.0, 0.0],
        normalize_std: List[float] = [1.0, 1.0, 1.0],
    ):
        """Initializes the WebDatasetReader with specified augmentation parameters.

        Args:
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.

        Raises:
            NotImplementedError: If the interpolation mode is not one of ["bicubic", "bilinear"].
        """
        train_transform = []
        interpolation = transforms.InterpolationMode.BICUBIC

        train_transform.append(
            transforms.Resize(
                resize_shorter_edge, interpolation=interpolation, antialias=True
            )
        )
        if random_crop:
            train_transform.append(transforms.RandomCrop(crop_size))
        else:
            train_transform.append(transforms.CenterCrop(crop_size))
        if random_flip:
            train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.append(transforms.ToTensor())
        # normalize_mean = [0, 0, 0] and normalize_std = [1, 1, 1] will normalize images into [0, 1],
        # normalize_mean = [0.5, 0.5, 0.5] and normalize_std = [0.5, 0.5, 0.5] will normalize images into [-1, 1].
        train_transform.append(transforms.Normalize(normalize_mean, normalize_std))

        self.train_transform = transforms.Compose(train_transform)
        self.eval_transform = transforms.Compose(
            [
                # Note that we always resize to crop_size during eval to ensure the results
                # can be compared against reference numbers on ImageNet etc.
                transforms.Resize(
                    crop_size, interpolation=interpolation, antialias=True
                ),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        print_rank_0(f"self.train_transform: {self.train_transform}")
        print_rank_0(f"self.eval_transform: {self.eval_transform}")


class SimpleImageDataset_T2I:
    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int = 12,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop=True,
        random_flip=True,
        normalize_mean: List[float] = [0.0, 0.0, 0.0],
        normalize_std: List[float] = [1.0, 1.0, 1.0],
        **kwargs,  #
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

        def random_caption_fn(key_set):
            # print("caption_feat.shape: ", key_set["caption_feat"].shape)
            # print("caption.shape: ", key_set["caption"].shape)
            cap_num = len(key_set["caption_feat"])
            # print("cap_num: ", cap_num)
            random_caption_ind = random.randint(0, cap_num - 1)
            # print("random_caption_ind: ", random_caption_ind)
            key_set["caption_feat"] = key_set["caption_feat"][random_caption_ind]
            key_set["caption"] = key_set["caption"][random_caption_ind]

            return key_set

        train_processing_pipeline = [
            wds.decode(),
            wds.rename(
                indices="jpg;png;jpeg;webp;indices.npy",
                caption_feat="cls;caption_feat.npy",
                caption="json;caption.json",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys((set(["indices", "caption_feat", "caption"])))),
            wds.map(random_caption_fn),
        ]

        # Create train dataset and loader.
        pipeline = [
            wds.ResampledShards(train_shards_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=5000, initial=1000),
            *train_processing_pipeline,
            wds.batched(
                per_gpu_batch_size, partial=False, collation_fn=default_collate
            ),
        ]

        # Each worker is iterating over the complete dataset.
        self._train_dataset = wds.DataPipeline(*pipeline)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=False,  # I use yield to achieve infinite loop, no concept of epoch any more, set it to False
        )
        ############################################################
        # Create train dataset and loader.
        eval_pipeline = [
            wds.ResampledShards(eval_shards_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=5000, initial=1000),
            *train_processing_pipeline,
            wds.batched(
                per_gpu_batch_size, partial=False, collation_fn=default_collate
            ),
        ]

        # Each worker is iterating over the complete dataset.

        self._eval_dataset = wds.DataPipeline(*eval_pipeline)
        self._eval_dataloader = wds.WebLoader(
            self._eval_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=False,  # I use yield to achieve infinite loop, no concept of epoch any more, set it to False
        )

    def train_dataset(self):
        return self._train_dataset

    def train_dataloader(self):
        return self._train_dataloader

    def eval_dataset(self):
        return self._eval_dataset

    def eval_dataloader(self):
        return self._eval_dataloader


if __name__ == "__main__":
    dataloader = SimpleImageDataset_T2I(
        train_shards_path="./data/coco_raw_varysize_wds_indices/train-{000000..000002}.tar",
        eval_shards_path="./data/coco_raw_varysize_wds_indices/train-{000000..000002}.tar",
        num_train_examples=1000,
        per_gpu_batch_size=128,
        global_batch_size=128,
        num_workers_per_gpu=12,
        crop_size=256,
        random_crop=True,
        random_flip=True,
    )

    for batch in dataloader.train_dataloader():
        print(batch.keys())
        print(batch["indices"].shape, batch["indices"].max(), batch["indices"].min())
        print(
            batch["caption_feat"].shape,
            batch["caption_feat"].max(),
            batch["caption_feat"].min(),
        )
        break