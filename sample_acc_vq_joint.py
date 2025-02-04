# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
For a simple single-GPU/CPU sampling script, see sample.py.
"""
import random
from einops import rearrange, repeat
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import os
import numpy as np
import math
import hydra
import shutil
import uuid
from utils.train_utils import get_model, requires_grad
import accelerate
import wandb
from utils_vq import print_rank_0
from utils_vq import (
    vq_get_encoder_decoder,
    vq_get_generator,
    vq_get_dynamic,
    vq_get_vae,
    vq_get_sample_size,
)


from utils_vq import get_dataset_id2label
from torchvision.utils import save_image
from utils_vq import wandb_visual_dict, get_dataloader
from torchmetrics.segmentation import MeanIoU
from utils.eval_tools.fid_score import calculate_fid_given_path_fake


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class AccCalculator:
    def __init__(self):
        self.num_samples = 0
        self.num_correct = 0

    def update(self, samples, gts):
        self.num_samples += len(samples)
        self.num_correct += (samples == gts).sum().item()

    def compute(self):
        return self.num_correct * 1.0 / self.num_samples


class mIoUCalculator:
    def __init__(self, num_classes=8, per_class=False, include_background=True):
        self.gts = []
        self.preds = []
        self.miou = MeanIoU(
            num_classes=num_classes,
            per_class=per_class,
            include_background=include_background,
        )

    def update(self, preds, gts):
        assert len(preds.shape) == len(
            gts.shape
        ), f"preds.shape={preds.shape}, gts.shape={gts.shape}"
        assert len(preds.shape) == 3
        self.preds.extend(preds)
        self.gts.extend(gts)

    def compute(self):
        _preds = torch.stack(self.preds)
        _gts = torch.stack(self.gts)
        assert _preds.shape == _gts.shape
        self.miou.to(_preds.device)
        assert (
            _preds.device == _gts.device
        ), f"preds.device={_preds.device}, gts.device={_gts.device}"
        result = self.miou(_preds.clone(), _gts.long().clone())
        return result


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    if cfg.debug:
        cfg.data.batch_size = 4
        cfg.ckpt_every = 10
        cfg.data.sample_fid_n = 1_00
        cfg.data.sample_fid_bs = 4
        cfg.data.sample_fid_every = 5
        cfg.data.sample_vis_every = 3
        cfg.data.sample_vis_n = 2
        #####
        cfg.num_fid_samples = 100
        cfg.offline.lbs = 4

    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = (
        True  # True: fast but may lead to some small numerical differences
    )
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)
    if cfg.data.num_classes > 0:
        _id2label = get_dataset_id2label(cfg.data.name)

    from accelerate.utils import AutocastKwargs

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        slurm_job_id = "local"
    print_rank_0(f"slurm_job_id: {slurm_job_id}")

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(cfg.global_seed, device_specific=True)
    set_seed(cfg.global_seed)
    rank = accelerator.state.process_index
    print_rank_0(
        f"Starting rank={rank}, world_size={accelerator.state.num_processes}, device={device}."
    )

    assert cfg.ckpt is not None, "Must specify a checkpoint to sample from"
    model = get_model(cfg)

    if True:
        state_dict = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)
        _model_dict = state_dict["ema"]
        _model_dict = {k.replace("module.", ""): v for k, v in _model_dict.items()}
        model.load_state_dict(_model_dict)
        model = model.to(device)
        requires_grad(model, False)
        print_rank_0(f"Loaded checkpoint from {cfg.ckpt}")

    model.eval()

    local_bs = cfg.offline.lbs
    cfg.data.batch_size = local_bs  # used for generating captions,etc.

    print_rank_0("local_bs", local_bs)

    loader = get_dataloader(cfg)

    loader, model = accelerator.prepare(loader, model)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    global_bs = local_bs * accelerator.state.num_processes
    total_samples = int(math.ceil(cfg.num_fid_samples / global_bs) * global_bs)
    assert (
        total_samples % accelerator.state.num_processes == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.state.num_processes)
    assert (
        samples_needed_this_gpu % local_bs == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // local_bs)

    print_rank_0(
        f"Total number of images that will be sampled: {total_samples} with global_batch_size={global_bs}"
    )

    tokenizer_encode_fn, tokenizer_decode_fn = vq_get_encoder_decoder(
        cfg=cfg, device=device
    )
    training_losses_fn, sample_fn = vq_get_dynamic(cfg=cfg, device=device)

    data_generator, _generator, caption_generator = vq_get_generator(
        cfg, device, loader, rank, 0
    )

    wandb_name = "_".join(
        [
            "v1",
            cfg.data.name,
            cfg.model.name,
            f"bs{cfg.offline.lbs}fid{cfg.num_fid_samples}cfg{cfg.cfg_scale}smt{cfg.sm_t}",
            f"{slurm_job_id}",
        ]
    )
    sample_folder_dir = f"{cfg.sample_dir}/{wandb_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if cfg.use_wandb:
            entity = cfg.wandb.entity
            project = cfg.wandb.project + "_vis"
            print_rank_0(
                f"Logging to wandb entity={entity}, project={project},rank={rank}"
            )
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project=project,
                name=wandb_name,
                config=config_dict,
                dir=sample_folder_dir,
                resume="allow",
                mode="online",
            )

            wandb_project_url = (
                f"https://wandb.ai/dpose-team/{wandb.run.project}/runs/{wandb.run.id}"
            )
            wandb_sync_command = (
                f"wandb sync {sample_folder_dir}/wandb/latest-run --append"
            )
            wandb_desc = "\n".join(
                [
                    "*" * 24,
                    # str(config_dict),
                    wandb_name,
                    wandb_project_url,
                    wandb_sync_command,
                    "*" * 24,
                ]
            )
        else:
            wandb_project_url = "wandb_project_url_null"
            wandb_sync_command = "wandb_sync_command_null"
            wandb_desc = "wandb_desc_null"

        print_rank_0(f"Saving .png samples at {sample_folder_dir}")

    sample_strategy = cfg.ss
    assert sample_strategy in ["img2cls", "cls2img", "mask2img", "img2mask"]
    sample_img_dir = f"{sample_folder_dir}/samples"
    gt_img_dir = f"{sample_folder_dir}/gts"
    if sample_strategy == "img2cls":
        acc_calculator = AccCalculator()
    elif sample_strategy == "img2mask":
        miou_calculator = mIoUCalculator(
            num_classes=8, per_class=False, include_background=True
        )

    if rank == 0:
        shutil.rmtree(sample_img_dir, ignore_errors=True)
        shutil.rmtree(gt_img_dir, ignore_errors=True)
        os.makedirs(sample_img_dir, exist_ok=True)
        os.makedirs(gt_img_dir, exist_ok=True)
    is_video = cfg.data.video_frames > 0
    assert is_video == False, "video is not supported yet"

    accelerator.wait_for_everyone()
    pbar = range(iterations)
    pbar = tqdm(pbar, total=iterations, desc="sampling") if rank == 0 else pbar

    if rank == 0:
        print(wandb_desc)
    assert cfg.use_cfg
    print("cfg_scale", cfg.cfg_scale)
    assert cfg.cfg_scale >= 0

    for bs_index in pbar:
        if sample_strategy in ["cls2img", "mask2img"]:
            img_gts, y_gts = next(data_generator)
            img_gts = tokenizer_decode_fn(img_gts)

            model_kwargs = dict(y=y_gts, cfg_scale=cfg.cfg_scale)
            model_fn = model.forward_with_cfg

        elif sample_strategy in ["img2cls", "img2mask"]:
            img_gts, y_gts = next(data_generator)
            img_gts = tokenizer_encode_fn(img_gts)

            model_kwargs = dict(x=img_gts, cfg_scale=cfg.cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            raise ValueError(f"Unknown sample strategy: {sample_strategy}")
        model_kwargs["sample_mode"] = sample_strategy

        _sample_size = vq_get_sample_size(len(img_gts), cfg)

        with torch.no_grad():
            sample_dict = sample_fn(
                _sample_size, model_fn, return_dict=True, **model_kwargs
            )[-1]
            # as it returns a chain

            if sample_strategy in ["cls2img", "mask2img"]:
                samples = tokenizer_decode_fn(sample_dict)
            elif sample_strategy in ["img2cls", "img2mask"]:
                samples = sample_dict
            else:
                raise ValueError(f"Unknown sample strategy: {sample_strategy}")

        if sample_strategy in ["cls2img", "mask2img"]:
            img_gts = img_gts[: len(samples)]
            sam_4fid, gts_4fid = samples, img_gts
            gts_4fid = accelerator.gather(gts_4fid.to(device))
            sam_4fid = accelerator.gather(sam_4fid.to(device))
            accelerator.wait_for_everyone()
            if rank == 0 and cfg.use_wandb and bs_index <= 1:
                if cfg.data.num_classes > 0:
                    if "imagenet" in cfg.data.name:
                        captions_sample = [
                            _id2label[y_gts[_].item()]
                            for _ in range(min(16, len(sam_4fid)))
                        ]
                    else:
                        captions_sample = [
                            "null caption" for _ in range(min(16, len(sam_4fid)))
                        ]
                elif cfg.data.num_classes == -666:
                    captions_sample = [
                        cap_str[_] for _ in range(min(16, len(sam_4fid)))
                    ]
                else:
                    captions_sample = [
                        "null caption" for _ in range(min(16, len(sam_4fid)))
                    ]
                wandb_dict = {}
                wandb_dict.update(
                    wandb_visual_dict(
                        "vis/samples_single",
                        sam_4fid,
                        is_video=is_video,
                        num=16,
                        captions=captions_sample,
                    )
                )
                wandb_dict.update(
                    wandb_visual_dict(
                        "vis/gts_single",
                        gts_4fid,
                        is_video=is_video,
                        num=16,
                        captions=None,
                    )
                )
                wandb.log(
                    wandb_dict,
                    step=bs_index,
                )
                print_rank_0("log_image into wandb")
                if not is_video:
                    wandb.log(
                        {
                            f"vis/samples": wandb.Image(sam_4fid[:16]),
                            f"vis/gts": wandb.Image(gts_4fid[:16]),
                        },
                        step=bs_index,
                    )
                print_rank_0("log_image into wandb")
            accelerator.wait_for_everyone()

            if rank == 0:
                print_rank_0(f"saving sample images, in {sample_img_dir}")
                if (not is_video) or cfg.offline.save_samples_to_disk:
                    # Save samples to disk as individual .png files
                    for _iii, sample in enumerate(sam_4fid):
                        unique_id = uuid.uuid4().hex[:6]
                        save_image(sample / 255.0, f"{sample_img_dir}/{unique_id}.png")

                    for _iii, sample in enumerate(gts_4fid):
                        unique_id = uuid.uuid4().hex[:6]
                        save_image(sample / 255.0, f"{gt_img_dir}/{unique_id}.png")

        elif sample_strategy == "img2cls":
            samples = samples.unsqueeze(-1)
            y_gts = y_gts
            acc_calculator.update(samples=samples, gts=y_gts)
            print_rank_0(
                f"cfg, {cfg.cfg_scale}, acc_till_now: {acc_calculator.compute()}"
            )
            print_rank_0(f"samples: {samples}")
            print_rank_0(f"gts: {y_gts}")

        elif sample_strategy == "img2mask":
            samples = samples
            y_gts = y_gts
            miou_calculator.update(preds=samples, gts=y_gts)
            print_rank_0(f"miou_till_now: {miou_calculator.compute()}")
            # print_rank_0(f"samples: {samples}")
            # print_rank_0(f"gts: {y_gts}")

        else:
            raise ValueError(f"Unknown sample strategy: {sample_strategy}")

    if sample_strategy == "img2cls":
        acc = acc_calculator.compute()
        print_rank_0(f"acc: {acc}")
        if rank == 0:
            wandb.log({"acc": acc})
    elif sample_strategy == "img2mask":
        miou = miou_calculator.compute()
        print_rank_0(f"miou: {miou}")
        if rank == 0:
            wandb.log({"miou": miou})
    elif sample_strategy in ["cls2img", "mask2img"]:
        if rank == 0:
            if not is_video:
                metric_dict_all = {}
                fid = calculate_fid_given_path_fake(
                    path_fake=sample_img_dir, npy_real=cfg.data.npz_real
                )
                metric_dict_all.update({"fid": fid})  #
                print_rank_0("fid", fid)

                wandb.log(metric_dict_all)
            if False:
                try:
                    shutil.rmtree(sample_img_dir)
                    shutil.rmtree(gt_img_dir)
                    print_rank_0(
                        "removed sample_img_dir and gt_img_dir\n",
                        sample_img_dir,
                        "\n",
                        gt_img_dir,
                    )
                except Exception as e:
                    print_rank_0(
                        f"Error removing directory {sample_img_dir},{gt_img_dir}: {e}"
                    )

        print_rank_0("done sampling")
        # accelerator.wait_for_everyone(), important! remove this.


if __name__ == "__main__":

    main()
