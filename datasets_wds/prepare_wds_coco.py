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
from torch.utils.data import DataLoader
import clip
from einops import rearrange
from omegaconf import OmegaConf


wds_target_dir = f"./data/coco_raw_varysize_wds_indices"
wds_target_dir = os.path.expanduser(wds_target_dir)
subset = "val"
if subset == "train":
    coco_root = os.path.expanduser("~/data/coco2014/train2014")
    coco_annFile = os.path.join(coco_root, "../annotations/captions_train2014.json")
elif subset == "val":
    coco_root = os.path.expanduser("~/data/coco2014/val2014")
    coco_annFile = os.path.join(coco_root, "../annotations/captions_val2014.json")
else:
    raise ValueError(f"subset {subset} not supported")

img_resize_dim = 256
latent_size = 32


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile):
        from pycocotools.coco import COCO

        self.root = root

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _get_path(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return os.path.join(self.root, path)

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann["caption"])
        return key, target


def pil_to_tensor(image):
    # Convert PIL Image to tensor
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image)

    # Scale to [0, 255] range
    tensor = tensor * 255

    # Round to nearest integer and convert to uint8
    tensor = tensor.round().byte()

    return tensor


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p2",
        "--wds_target_dir",
        type=str,
        default=wds_target_dir,
        help="path to dataset",
    )
    parser.add_argument(
        "-s", "--split", type=str, default="train", help="split to convert"
    )
    parser.add_argument(
        "-c", "--category_name", type=str, default="cake", help="category name"
    )
    parser.add_argument("--max_size", type=float, default=1, help="gb per shard")
    opt = parser.parse_args()
    os.makedirs(opt.wds_target_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if subset == "train":
        pass
    elif subset == "val":
        coco_root = coco_root.replace("train2014", "val2014")
        coco_annFile = coco_annFile.replace("train2014", "val2014")
        print(f"coco_root: {coco_root}")
        print(f"coco_annFile: {coco_annFile}")
        print(f"split: {subset}")

    else:
        raise ValueError(f"split {subset} not supported")

    to_tensor = ToTensor()

    writer = wds.ShardWriter(
        os.path.join(opt.wds_target_dir, "{}-%06d.tar".format(subset)),
        maxcount=1e6,
        maxsize=opt.max_size * 1e9 * 0.4,
    )  # -> each shard will be 0.4GB
    # Iterate over the files in the dataset directory

    ############################################################
    ldm_path = os.path.expanduser("~/lab/discretediffusion/ldm")
    sys.path.insert(0, ldm_path)
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
        indices = rearrange(
            indices,
            "(b h w) -> b h w",
            b=img_shape[0],
            h=latent_size,
            w=latent_size,
        )
        return indices
        ############################################################

    clip = clip.FrozenCLIPEmbedder()
    clip.eval()
    clip.to(device)

    file_index = 0

    dataset = MSCOCODatabase(coco_root, coco_annFile)
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for _key, _captions in tqdm(dl, total=len(dl)):
        _captions = [_caption[0] for _caption in _captions]
        wds_dict = {}
        wds_dict["__key__"] = f"{file_index}".zfill(10)
        _img_path = dataset._get_path(int(_key))
        x = Image.open(_img_path).convert("RGB")
        x = center_crop_arr(x, img_resize_dim)
        x = pil_to_tensor(x).to(device)
        indices = tokenizer_encode_fn(x.unsqueeze(0))[0]
        wds_dict["indices.npy"] = indices.cpu().numpy()

        caption_feat = clip.encode(_captions)
        wds_dict["caption_feat.npy"] = caption_feat.cpu().numpy()
        wds_dict["caption.json"] = _captions
        writer.write(wds_dict)
        file_index += 1
        # print("image index: ", file_index)
    writer.close()
    print("done")
