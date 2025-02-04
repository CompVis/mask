import sys

import PIL

sys.path.append("..")
import os
import argparse
import numpy as np
import webdataset as wds
from tqdm import tqdm
import torch
from PIL import Image
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from torchvision.transforms import Resize, ToTensor
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import torch.nn as nn

import h5py

image_size = 256
is_debug = False

num_workers = 16
is_shuffle = True
dataset_name = "imagenet1k"
is_main_process = True

try:
    DATASET_ROOT = "/export/compvis-nfs/group/datasets/ILSVRC/ILSVRC2012_train/data"
    assert os.path.exists(DATASET_ROOT)
except:
    DATASET_ROOT = "/export/group/datasets/ILSVRC/ILSVRC2012_train/data/"
    assert os.path.exists(DATASET_ROOT)
print(f"DATASET_ROOT: {DATASET_ROOT}")
wds_target_dir = f"./data/{dataset_name}_size{image_size}.h5"

if image_size == 256:
    batch_size = 32
    latent_size = 32
    size_per_wds = 0.03  # size_per_wds is 0.03GB
elif image_size == 512:
    batch_size = 32
    latent_size = 64
    size_per_wds = 0.1  # size_per_wds is 0.1GB
else:
    raise ValueError(f"Unsupported image size: {image_size}")
if is_debug:
    wds_target_dir = wds_target_dir.replace(".h5", "_debug.h5")
    batch_size = 2
wds_target_dir = os.path.expanduser(wds_target_dir)
if dataset_name == "imagenet1k":
    imagenet_classes = open("./datasets_wds/imagenet1k.txt").readlines()
elif dataset_name == "imagenet100":
    imagenet_classes = open("./datasets_wds/imagenet100.txt").readlines()
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")
imagenet_classes = [cls.strip() for cls in imagenet_classes]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def np_crop_and_resize_flip(image, size, interpolation=PIL.Image.BICUBIC, flip_p=0.5):
    # default to score-sde preprocessing
    img = np.array(image).astype(np.uint8)
    crop = min(img.shape[0], img.shape[1])
    (
        h,
        w,
    ) = (
        img.shape[0],
        img.shape[1],
    )
    img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

    image = Image.fromarray(img)

    image = image.resize((size, size), resample=interpolation)
    flip = transforms.RandomHorizontalFlip(p=flip_p)
    image = flip(image)
    # image = transforms.PILToTensor()(image)
    # image = (image / 127.5 - 1.0).float()
    return image


class FolderDataset(Dataset):
    def __init__(
        self,
    ):
        self.path = []
        for root, dirs, files in os.walk(os.path.join(opt.path)):
            # Iterate over the files in each subdirectory
            for _filename in tqdm(files, total=len(files)):
                try:
                    cls_name = root.split("/")[-1]
                    cls_id = imagenet_classes.index(cls_name)
                    self.path.append((root, _filename))
                except:
                    pass

        print("total files: ", len(self.path))

    def __len__(self):
        if is_debug:
            return 106
        else:
            return len(self.path)

    def __getitem__(self, idx):
        root, _filename = self.path[idx]
        cls_name = root.split("/")[-1]
        cls_id = imagenet_classes.index(cls_name)
        with open(os.path.join(root, _filename), "rb") as stream:
            image = stream.read()
            image = Image.open(stream).convert("RGB")
            image_resized0 = np_crop_and_resize_flip(
                image=np.array(image), size=image_size, flip_p=0
            )
            image_resized1 = np_crop_and_resize_flip(
                image=np.array(image), size=image_size, flip_p=1
            )
            img = np.stack([image_resized0, image_resized1], axis=0)
            cls_id = np.array([cls_id, cls_id])
            return img, cls_id, cls_name


def SD_VQ_tokenizer():
    sys.path.insert(0, os.path.abspath("./ldm"))
    from ldm.util import instantiate_from_config

    ckpt_path = "./pretrained_ckpt/ldm/vq-f8.ckpt"
    config_path = "./ldm/models/first_stage_models/vq-f8/config.yaml"

    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]
    _tokenizer = instantiate_from_config(config.model)
    _tokenizer.load_state_dict(sd, strict=False)
    _tokenizer.eval()
    _tokenizer.requires_grad_(False)
    _tokenizer = _tokenizer.to(device)

    @torch.no_grad()
    def tokenizer_encode_fn(img, mini_bs=25):
        img = img / 255.0
        img = (img - 0.5) * 2
        # somelogic about video
        img_shape = img.shape
        if len(img_shape) == 5:
            b, t, c, h, w = img.shape
            img = rearrange(img, "b t c h w -> (b t) c h w")
        ############################################################
        for i in range(0, len(img), mini_bs):
            _img = img[i : i + mini_bs]
            encode_res = _tokenizer.encode(_img)
            quant = encode_res[0]
            diff = encode_res[1]
            _indices = encode_res[2][-1]
            if i == 0:
                indices = _indices
            else:
                indices = torch.cat([indices, _indices], dim=0)
        ############################################################
        if len(img_shape) == 5:
            indices = rearrange(
                indices,
                "(b t h w) -> b t h w",
                b=b,
                t=t,
                h=latent_size,
                w=latent_size,
            )
        elif len(img_shape) == 4:
            indices = rearrange(
                indices,
                "(b h w) -> b h w",
                b=img_shape[0],
                h=latent_size,
                w=latent_size,
            )
        else:
            raise ValueError(f"Unsupported batch dimensions: {len(img_shape)}")

        return indices
        ############################################################

    @torch.no_grad()
    def tokenizer_decode_fn(indices, mini_bs=25):

        indices_shape = indices.shape
        if len(indices_shape) == 4:
            b, t, h, w = indices.shape
            indices = rearrange(indices, "b t h w -> (b t) (h w)")
        elif len(indices_shape) == 3:
            indices = rearrange(indices, "b h w -> b (h w)")
        else:
            raise ValueError(f"Unsupported batch dimensions: {len(indices_shape)}")
        # somelogic about video

        for i in range(0, len(indices), mini_bs):
            _indices = indices[i : i + mini_bs]
            _img = _tokenizer.decode_tokens(_indices)
            if i == 0:
                img = _img
            else:
                img = torch.cat([img, _img], dim=0)
        # somelogic about video
        if len(indices_shape) == 4:
            img = rearrange(img, "(b t) c h w -> b t c h w", b=b, t=t)

        img = img.clamp(-1, 1)
        img = ((img + 1) * 0.5 * 255.0).to(dtype=torch.uint8)
        return img

    return tokenizer_encode_fn, tokenizer_decode_fn


tokenizer_encode_fn, tokenizer_decode_fn = SD_VQ_tokenizer()
if __name__ == "__main__":

    class OPT:
        path = DATASET_ROOT
        wds_target_dir = wds_target_dir
        split = "train"
        category_name = "cake"

    opt = OPT()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    to_tensor = ToTensor()

    print(opt.path)
    file_index = 0
    dataset = FolderDataset()
    if is_main_process:
        len_dataset = len(dataset) * 2
        h5_file = h5py.File(wds_target_dir, "w")
        h5_file.create_dataset(
            "indice", (len_dataset, latent_size, latent_size), dtype=np.int32
        )
        h5_file.create_dataset("cls_id", (len_dataset,), dtype=np.int32)
        print("h5_file created, len(dataset): ", len_dataset)
    dataset = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    for _iddd, (img_pairs, cls_id, cls_name) in tqdm(
        enumerate(dataset), total=len(dataset)
    ):
        b, t, h, w, c = img_pairs.shape
        img_pairs = rearrange(img_pairs, "b t h w c -> (b t) c h w")
        cls_id = rearrange(cls_id, "b t -> (b t)")
        indices = tokenizer_encode_fn(img_pairs.to(device))
        if is_debug:
            for _id, (_indices, _cls_id) in enumerate(zip(indices, cls_id)):
                print(
                    f"rank: , _id: {_id}, indices[3,2]: {_indices[3,2].item()}, cls_id: {_cls_id.item()}"
                )

        if is_main_process:
            if is_debug:
                for _indices, _cls_id in zip(indices, cls_id):
                    print(
                        f" indices[3,2]: {_indices[3,2].item()}, _cls_id: {_cls_id.item()}"
                    )

            for _indices, _cls_id in zip(indices, cls_id):
                h5_file["indice"][file_index] = _indices.cpu().numpy().astype(np.int32)
                h5_file["cls_id"][file_index] = int(_cls_id)
                file_index += 1
            print("image index: ", file_index, opt.wds_target_dir)
    if is_main_process:
        h5_file.close()
    print("done")

    # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch  --mixed_precision no  --num_processes 4 --num_machines 1 --multi_gpu --main_process_ip 127.0.0.1 --main_process_port 8878  datasets_wds/prepare_ddp_h5_in.py

# CUDA_VISIBLE_DEVICES=1 python  datasets_wds/prepare_ddp_h5_in.py
