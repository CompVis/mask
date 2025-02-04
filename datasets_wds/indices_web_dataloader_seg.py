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
import numpy as np
from functools import partial
from utils.train_utils import print_rank_0


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


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

        train_processing_pipeline = [
            wds.decode(),
            wds.rename(
                indices="indices.npy",
                cls_id="segmasks.npy",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys((set(["indices", "cls_id"])))),
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
            # wds.shuffle(bufsize=5000, initial=1000),
            wds.shuffle(1000, rng=np.random.RandomState(42)),  # Use a fixed RNG
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
        train_shards_path="./data/coco_raw_varysize_wds/train-{000000..000002}.tar",
        eval_shards_path="./data/coco_raw_varysize_wds/train-{000000..000002}.tar",
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
        print(batch["image"].shape, batch["image"].max(), batch["image"].min())
        print(
            batch["caption_feat"].shape,
            batch["caption_feat"].max(),
            batch["caption_feat"].min(),
        )
        break
