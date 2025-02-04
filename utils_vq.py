import torch
from einops import repeat
import sys
import os
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from diffusers import AutoencoderKL
import wandb
import numpy as np
import random
import torch
import imageio
import uuid
import torch.nn.functional as F
from datetime import datetime

from einops import repeat
from torchvision.utils import draw_bounding_boxes
import torch.distributed as dist

import torch
import torchvision.transforms as transforms

import numpy as np
import wandb
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math
from omegaconf import OmegaConf
import os
import torch.distributed as dist

import logging


import re


cityscapes_big8 = [
    "Flat",
    "Human",
    "Vehicle",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Void",
]


def get_dataloader(cfg):
    args_data = cfg.data
    if (
        hasattr(args_data, "video_frames")
        and args_data.video_frames > 0
        and "indices" in args_data.name
    ):
        from datasets_wds.indices_h5_dataloader_video import (
            get_inf_h5_dataloader,
        )

        loader = get_inf_h5_dataloader(**args_data)
        return loader
    elif "h5" in args_data.name:
        from datasets_wds.indices_h5_dataloader_image import (
            get_inf_h5_dataloader,
        )

        loader = get_inf_h5_dataloader(**args_data)
        return loader

    elif "indices" in args_data.name and args_data.name.startswith("coco"):
        from datasets_wds.indices_web_dataloader_t2i import SimpleImageDataset_T2I

        datamod = SimpleImageDataset_T2I(**args_data)
        if args_data.subset == "train":
            loader = datamod.train_dataloader()
        elif args_data.subset == "val":
            loader = datamod.eval_dataloader()
        else:
            raise ValueError(f"subset {args_data.subset} not supported")
        return loader
    elif "indices" in args_data.name and "imagenet" in args_data.name:
        from datasets_wds.indices_web_dataloader_imagenet import SimpleImageDataset

        datamod = SimpleImageDataset(**args_data)
        if args_data.subset == "train":
            loader = datamod.train_dataloader()
        elif args_data.subset == "val":
            loader = datamod.eval_dataloader()
        else:
            raise ValueError(f"subset {args_data.subset} not supported")

        return loader

    elif args_data.name.startswith("cs_wds_indices"):
        from datasets_wds.indices_web_dataloader_seg import SimpleImageDataset_T2I

        datamod = SimpleImageDataset_T2I(**args_data)
        if args_data.subset == "train":
            loader = datamod.train_dataloader()
        elif args_data.subset == "val":
            loader = datamod.eval_dataloader()
        else:
            raise ValueError(f"subset {args_data.subset} not supported")
        return loader

    else:
        raise NotImplementedError(f"data {args_data.name} not supported")


def array2grid_pixel(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=False)
    x = x.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


def array2row_pixel(x, pad_value=0.5):  # default 0
    nrow = len(x)
    x = make_grid(x, nrow=nrow, normalize=False, pad_value=pad_value)
    x = x.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


def get_max_ckpt_from_dir(dir_path):
    dir_path = os.path.join(dir_path, "checkpoints")
    # Define the pattern to match
    pattern = r"(\d+)\.pt"

    # Initialize the maximum step number and corresponding file name
    max_step = -1
    max_step_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(dir_path):
        # If the filename matches the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the step number from the filename
            step = int(match.group(1))
            # If this step number is larger than the current maximum
            if step > max_step:
                # Update the maximum step number and corresponding file name
                max_step = step
                max_step_file = filename

    if max_step_file is None:
        raise ValueError(f"No checkpoint files found in {dir_path}")
    else:
        print(
            f"Found checkpoint file {max_step_file} with step {max_step} from {dir_path}"
        )
        return os.path.join(dir_path, max_step_file)


def print_rank_0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            # print(*args, **kwargs)
            try:
                logging.info(*args, **kwargs)
            except:
                print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def wandb_visual_dict(wandb_key, visual_tensor, is_video, num=16, captions=None):
    if captions is None:
        captions = ["null caption" for _ in range(num)]
    if is_video:
        b, t, c, w, h = visual_tensor.shape
        visual_tensor = visual_tensor.cpu().numpy()
        return {
            wandb_key: wandb.Video(visual_tensor[:num]),
        }
    else:
        b, c, w, h = visual_tensor.shape
        return {
            wandb_key: wandb.Image(array2grid_pixel(visual_tensor[:num])),
        }


def get_version_number():
    # return "v1.1"  # add t_sample_eps
    # return "v1.2" #previous accelerator's accum is only 1 actually.
    return "v1.3"  # loss is divided by accum now.


def has_label(dataset_name):
    if dataset_name.startswith("ffs"):
        return False
    else:
        return True


def get_dataset_id2label(dataset_name):
    if "imagenet" in dataset_name:
        imagenet_id2realname = open("./datasets_wds/imagenet1k_name.txt").readlines()
        imagenet_id2realname = [
            _cls.strip().split()[-1] for _cls in imagenet_id2realname
        ]
        return imagenet_id2realname

    elif "cs" in dataset_name:
        return cityscapes_big8
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def torch_distributed_barrier():
    try:
        torch.distributed.barrier()
    except:
        print("torch_distributed_barrier failed in torch.distributed")


def out2img(samples):
    return torch.clamp(127.5 * samples + 128.00, 0, 255).to(
        dtype=torch.uint8, device="cuda"
    )


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def vq_get_sample_size(bs, cfg):
    if cfg.input_tensor_type == "bt":
        return (bs, cfg.tokenizer.token_len)
    elif cfg.input_tensor_type == "bwh":
        return (bs, cfg.tokenizer.latent_size, cfg.tokenizer.latent_size)
    elif cfg.input_tensor_type == "bcwh":
        return (
            bs,
            cfg.tokenizer.in_channels,
            cfg.tokenizer.latent_size,
            cfg.tokenizer.latent_size,
        )
    elif cfg.input_tensor_type == "btwh":
        assert cfg.data.video_frames > 0, "video_frames must be > 0"
        return (
            bs,
            cfg.data.video_frames,
            cfg.tokenizer.latent_size,
            cfg.tokenizer.latent_size,
        )
    else:
        raise ValueError(f"Unknown tensor type: {cfg.input_tensor_type}")


def vq_get_vae(cfg, device):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(device)
    # stabilityai/sd-vae-ft-ema
    vae.eval()
    return vae


def vq_get_dynamic(cfg, device, is_train=True):
    if cfg.dynamic.name.startswith("uni_tc0"):
        from dynamics.dynamic_di_uni_tc0 import (
            DiscreteInterpolants,
            Ucoupling,
            Ccoupling,
            SimpleSampler,
            MaskgitSampler,
            get_scheduler
        )

        input_tensor_type = cfg.input_tensor_type
        _scheduler = cfg.dynamic.disint.scheduler
        _coupling = cfg.dynamic.disint.coupling
        _sampler_name = cfg.dynamic.disint.sampler
        smoothing_factor = cfg.dynamic.disint.smooth

        
        _kappa = get_scheduler(_scheduler)
        if _coupling == "ucoupling":
            _coupling = Ucoupling()
        elif _coupling == "ccoupling":
            _coupling = Ccoupling(
                msk_prop=cfg.dynamic.disint.ccoupling_prob,
            )
        else:
            raise ValueError(f"coupling={_coupling} not supported")

        type_y = "bt" if cfg.model.params.second_modal_type == "label" else "bwh"
        disint = DiscreteInterpolants(
            vocab_size_x=cfg.tokenizer.vocab_size,
            coupling=_coupling,
            kappa=_kappa,
            device=device,
            vocab_size_y=cfg.data.num_classes + 1,
            type_x=input_tensor_type,
            type_y=type_y,
            smoothing_factor=smoothing_factor,
            mask_ce=cfg.dynamic.mask_ce,
        )
        training_losses_fn = disint.training_losses

        if _sampler_name == "simple":
            sampler = SimpleSampler(
                mask_token_id=cfg.tokenizer.mask_token_id,
                input_tensor_type=input_tensor_type,
            )
        elif _sampler_name == "maskgit":
            sampler = MaskgitSampler(
                mask_token_id=cfg.tokenizer.mask_token_id,
                input_tensor_type=input_tensor_type,
            )
        else:
            raise ValueError(f"sampler name={_sampler_name} not supported")

        def sample_fn(sample_size, model, **model_kwargs):

            if _sampler_name == "maskgit":
                model_kwargs["maskgit_mode"] = cfg.maskgit_mode
                model_kwargs["maskgit_randomize"] = cfg.maskgit_randomize
            r = sampler.sample(
                sample_size,
                disint,
                model=model,
                kappa=_kappa,
                n_steps=cfg.dynamic.disint.step_num,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                anneal_noise=cfg.anneal_noise,
                **model_kwargs,
            )

            return r[0]  # r is a tuple

    else:
        from dynamics.dynamic_di import (
            DiscreteInterpolants,
            Ucoupling,
            Ccoupling,
            SimpleSampler,
            MaskgitSampler,
            get_scheduler
        )

        input_tensor_type = cfg.input_tensor_type
        _scheduler = cfg.dynamic.disint.scheduler
        _coupling = cfg.dynamic.disint.coupling
        _sampler_name = cfg.dynamic.disint.sampler
        smoothing_factor = cfg.dynamic.disint.smooth
        
        _kappa = get_scheduler(_scheduler)
        if _coupling == "ucoupling":
            _coupling = Ucoupling(mask_token_id=cfg.tokenizer.mask_token_id)
        elif _coupling == "ccoupling":
            _coupling = Ccoupling(
                mask_token_id=cfg.tokenizer.mask_token_id,
                msk_prop=cfg.dynamic.disint.ccoupling_prob,
            )
        else:
            raise ValueError(f"coupling={_coupling} not supported")

        disint = DiscreteInterpolants(
            vocab_size=cfg.tokenizer.vocab_size,
            coupling=_coupling,
            kappa=_kappa,
            device=device,
            input_tensor_type=input_tensor_type,
            smoothing_factor=smoothing_factor,
            mask_ce=cfg.dynamic.mask_ce,
            elbo=cfg.dynamic.elbo,
        )
        training_losses_fn = disint.training_losses

        if _sampler_name == "simple":
            sampler = SimpleSampler(
                mask_token_id=cfg.tokenizer.mask_token_id,
                input_tensor_type=input_tensor_type,
            )
        elif _sampler_name == "maskgit":
            sampler = MaskgitSampler(
                mask_token_id=cfg.tokenizer.mask_token_id,
                input_tensor_type=input_tensor_type,
            )
        else:
            raise ValueError(f"sampler name={_sampler_name} not supported")

        def sample_fn(sample_size, model, **model_kwargs):

            if _sampler_name == "maskgit":
                model_kwargs["maskgit_mode"] = cfg.maskgit_mode
                model_kwargs["maskgit_randomize"] = cfg.maskgit_randomize
            r = sampler.sample(
                sample_size,
                disint,
                model=model,
                kappa=_kappa,
                n_steps=cfg.dynamic.disint.step_num,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                anneal_noise=cfg.anneal_noise,
                **model_kwargs,
            )
            return r[0]  # r is a tuple

    return training_losses_fn, sample_fn


def vq_get_encoder_decoder(cfg, device):
    if cfg.tokenizer.name in ["sd_vq_f8", "sd_vq_f8_res512"]:
        use_id = cfg.input_tensor_type == "bt"
        vocab_size = cfg.tokenizer.vocab_size
        latent_size = cfg.tokenizer.latent_size
        config_path = cfg.tokenizer.config_path
        ckpt_path = cfg.tokenizer.ckpt_path

        sys.path.insert(0, os.path.abspath("./ldm"))
        from ldm.ldm.util import instantiate_from_config

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
            if use_id:
                raise NotImplementedError
            return indices
            ############################################################

        @torch.no_grad()
        def tokenizer_decode_fn(indices, mini_bs=25):
            indices[indices == cfg.tokenizer.mask_token_id] = (
                cfg.tokenizer.mask_token_reindex
            )
            indices_shape = indices.shape
            if len(indices_shape) == 4:  # video
                b, t, h, w = indices.shape
                indices = rearrange(indices, "b t h w -> (b t) (h w)")
            elif len(indices_shape) == 3:  # image
                indices = rearrange(indices, "b h w -> b (h w)")
            else:
                raise ValueError(f"Unsupported batch dimensions: {len(indices_shape)}")

            for i in range(0, len(indices), mini_bs):
                _indices = indices[i : i + mini_bs]
                _img = _tokenizer.decode_tokens(_indices.long())
                if i == 0:
                    img = _img
                else:
                    img = torch.cat([img, _img], dim=0)

            if len(indices_shape) == 4:  # if video
                img = rearrange(img, "(b t) c h w -> b t c h w", b=b, t=t)

            img = img.clamp(-1, 1)
            img = ((img + 1) * 0.5 * 255.0).to(dtype=torch.uint8)
            return img

    elif cfg.tokenizer.name in ["pixelimage"]:
        vocab_size = cfg.tokenizer.vocab_size
        mask_token_id = cfg.tokenizer.mask_token_id

        @torch.no_grad()
        def tokenizer_encode_fn(img):
            return img

        @torch.no_grad()
        def tokenizer_decode_fn(indices):
            return indices

    else:
        raise ValueError(f"tokenizer={cfg.tokenizer.name} not supported")

    if "indice" in cfg.data.name:
        tokenizer_encode_fn = lambda x: x
    return tokenizer_encode_fn, tokenizer_decode_fn


def vq_get_generator(cfg, device, loader, rank_id, train_steps, vae=None):

    def get_data_generator(return_cls_id=True):
        _init = train_steps
        while True:
            for data in tqdm(
                loader,
                disable=rank_id > 0,
                initial=_init,
                desc="data fetching",
            ):
                x = data["image"].to(device)
                try:
                    y = data["cls_id"].to(device)
                except:
                    try:
                        y = data["caption_feat"].to(device)
                    except:
                        y = None
                x = out2img(x)

                if return_cls_id:
                    yield x, y
                else:
                    yield x

    def get_caption_generator():
        while True:
            for data in tqdm(
                loader,
                disable=rank_id > 0,
                desc="gen caption",
            ):
                captiopn_feat = data["caption_feat"].to(device)
                caption = data["caption"]

                yield captiopn_feat, caption

    def get_indices_generator(return_cls_id=True):
        _init = train_steps
        while True:
            for data in tqdm(
                loader,
                disable=rank_id > 0,
                initial=_init,
                desc="data fetching",
            ):

                x = data["indices"].to(device)
                try:
                    y = data["cls_id"].to(device)
                except:
                    try:
                        y = data["caption_feat"].to(device)
                    except:
                        y = None

                if return_cls_id:
                    yield x, y
                else:
                    yield x

    if "indices" in cfg.data.name:
        data_gen = get_indices_generator(return_cls_id=True)
        realimg_gen = get_indices_generator(return_cls_id=False)
    else:
        raise NotImplementedError
    cap_gen = get_caption_generator()
    return data_gen, realimg_gen, cap_gen


if __name__ == "__main__":
    pass
