from collections import OrderedDict
import os
import glob
import numpy as np
import torch
from PIL import Image
import logging
import importlib
import logging
import torch

import torch.distributed as dist


import logging

def rankzero_logging_info(rank, log):
    if rank == 0:
        logging.info(log)  


def print_rank_0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def wandb_runid_from_checkpoint(checkpoint_path):
    # Python
    import os
    import re

    # Define the directory to search
    dir_path = os.path.join(checkpoint_path, "wandb/latest-run")
    # Define the pattern to match
    pattern = r"run-(\w+)\.wandb"

    # Iterate over all files in the directory
    for filename in os.listdir(dir_path):
        # If the filename matches the pattern
        if re.match(pattern, filename):
            # Extract the desired part of the filename
            extracted_part = re.match(pattern, filename).group(1)
            logging.info(f"induced run id from ckpt: {extracted_part}")
            return extracted_part
    raise ValueError("No file found that matches the pattern")


def instantiate_from_config(config):
    module_name, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**config.get("params", {}))
    return instance


def get_model(args):

    model = instantiate_from_config(args.model)
   
    return model


def create_logger(rank, logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


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


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def update_ema_for_accelerator( model_ema, model, accelerator, decay=0.9999):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        msd = accelerator.get_state_dict(model)
        for k, ema_v in model_ema.state_dict().items():
            if k in msd:
                model_v = msd[k].detach().to(ema_v.device, dtype=ema_v.dtype)
                ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def grad_clip(opt, model, max_grad_norm=2.0):
    if hasattr(opt, "clip_grad_norm"):
        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
        opt.clip_grad_norm(max_grad_norm)
    else:
        # Revert to normal clipping otherwise, handling Apex or full precision
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
            max_grad_norm,
        )


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR

    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == "customized":
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def get_latest_checkpoint(checkpoint_dir):

    # Get a list of all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*"))
    # Check if there are any checkpoints
    if not checkpoint_files:
        # print("No checkpoints found")
        # raise FileNotFoundError
        return "No_checkpoints_found"
    else:
        # Get the checkpoint file with the latest modification time
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        # print(f"Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
