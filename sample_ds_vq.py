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
import wandb
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from utils_deepspeed import all_gather_my
from utils_vq import print_rank_0
from utils_vq import (
    vq_get_encoder_decoder,
    vq_get_generator,
    vq_get_dynamic,
    vq_get_vae,
    vq_get_sample_size,
)
from utils.train_utils import (
    create_logger,
    get_latest_checkpoint,
    get_model,
    requires_grad,
    update_ema,
    wandb_runid_from_checkpoint,
)
from utils.openai_eval import _eval_by_npz

from utils.my_metrics_offline import MyMetric_Offline as MyMetric
from utils_vq import get_dataset_id2label
from torchvision.utils import save_image
from utils_vq import wandb_visual_dict,get_dataloader
from fvd_external import calculate_fvd_github

from torchvision.io import write_video


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    return _main(cfg)


def _main(cfg):
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
    torch.backends.cuda.matmul.allow_tf32 = True # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)
    if cfg.data.num_classes > 0:
        imagenet_id2label = get_dataset_id2label(cfg.data.name)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        slurm_job_id = "local"
    print_rank_0(f"slurm_job_id: {slurm_job_id}")
    skip_data_loader = True  # we didn't save real image in webdataset, so we need to load it from disk offline by preprocess
    if "coco14" in cfg.data.name:
        skip_data_loader = False
        print_rank_0(f"coco dataset,skip_data_loader force to be False: {skip_data_loader}")

    deepspeed.init_distributed()
    num_processes = dist.get_world_size()
    rank = rank_id = dist.get_rank()
    get_accelerator().set_device(int(rank))
    # logger = create_logger(rank_id, experiment_dir)

    print_rank_0(f"Starting rank={rank}, world_size={num_processes}")
    if rank == 0:
        metric_fid_with_npz = MyMetric(npz_real=cfg.data.npz_real)

    assert cfg.ckpt is not None, "Must specify a checkpoint to sample from"
    model = get_model(cfg)
    requires_grad(model, False)
    #########################################################
    _dtype = torch.float32
    if cfg.mixed_precision == "fp16":
        _dtype = torch.float16
    elif cfg.mixed_precision == "bf16":
        _dtype = torch.bfloat16
    elif cfg.mixed_precision == "no":
        _dtype = torch.float32
    else:
        raise ValueError(f"mixed_precision={cfg.mixed_precision} is not supported")
    print(f"using dtype {_dtype}")
    
    if os.path.isfile(cfg.ckpt):#pt checkpoint
        _cckpt = cfg.ckpt
        assert os.path.exists(_cckpt), f"ckpt file {_cckpt} does not exist"
        state_dict = torch.load(_cckpt, map_location=lambda storage, loc: storage)
        try:  # if by accelerator
            _model_dict = state_dict["model"]
            _model_dict = {k.replace("module.", ""): v for k, v in _model_dict.items()}
        except:  # if by deepspeed
            try:
                _model_dict = state_dict["module"]
            except:
                try:
                    _model_dict = state_dict #pytorch_model.bin from deepspeed
                except Exception as e:
                    print("model dict keys:", state_dict.keys())
                    print(f"Error loading ckpt from {_cckpt}: {e}")
                    raise e
        model.load_state_dict(_model_dict)
        model_engine = deepspeed.init_inference(
            model=model,
            checkpoint=None,
            dtype=_dtype,
            replace_with_kernel_inject=False,
        )
    elif os.path.isdir(cfg.ckpt):#deepspeed checkpoint
        raise NotImplementedError("deepspeed checkpoint is not supported yet, use zero_to_fp32.py to convert to pt, cd your ckpt dir and run: python zero_to_fp32.py ./ YOUR_CKPT_DIR/ ")
        model_engine = deepspeed.init_inference(
            model=model,
            checkpoint=cfg.ckpt,
            dtype=_dtype,
            replace_with_kernel_inject=False,
        )
    else:
        raise ValueError(f"ckpt={cfg.ckpt} is not supported")
    print(f"loading ckpt from {cfg.ckpt}")
    model = model_engine.module



    device = get_accelerator().device_name(rank)
    model.eval()  

    local_bs = cfg.offline.lbs
    cfg.data.batch_size = local_bs  # used for generating captions,etc.

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    global_bs = local_bs * num_processes
    total_samples = int(math.ceil(cfg.num_fid_samples / global_bs) * global_bs)
    assert (
        total_samples % num_processes == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // num_processes)
    assert (
        samples_needed_this_gpu % local_bs == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // local_bs)

    print_rank_0(
        f"Total number of images that will be sampled: {total_samples} with global_batch_size={global_bs},local_batch_size={local_bs}"
    )

    tokenizer_encode_fn, tokenizer_decode_fn = vq_get_encoder_decoder(
        cfg=cfg, device=device
    )
    training_losses_fn, sample_fn = vq_get_dynamic(
        cfg=cfg, device=device
    )
    vae = vq_get_vae(cfg, device)

    if not skip_data_loader:
        loader = get_dataloader(cfg)
        data_generator, _generator, caption_generator = vq_get_generator(
            cfg, device, loader, rank, 0, vae
        )

    assert cfg.cfg_scale >= 0.0#, "In almost all cases, cfg_scale be >= 1.0"
    if cfg.cfg_scale > 0.0:
        model_fn = model.forward_with_cfg
    elif cfg.cfg_scale == 0.0:
        model_fn = model.forward_without_cfg
    else:
        raise ValueError(f"cfg_scale={cfg.cfg_scale} is not supported")

    

    if cfg.dynamic.name.startswith("disint"):
        dynamic_desc = "_".join(
            [
                f"sampler={cfg.dynamic.disint.sampler}",
                f"scheduler={cfg.dynamic.disint.scheduler}",
            ]
        )
    else:
        dynamic_desc = cfg.dynamic.name

    wandb_name = "_".join(
        [
            "dsv0",  # sample ground truth images uniformly over the dataset
            cfg.note,
            cfg.data.name,
            cfg.model.name,
            dynamic_desc,
            f"mp{cfg.mixed_precision if cfg.mixed_precision else 'null'}",
            f"bs{cfg.offline.lbs}fid{cfg.num_fid_samples}cfg{cfg.cfg_scale}softmaxtem{cfg.sm_t}topk{cfg.top_k}topp{cfg.top_p}randomize{cfg.maskgit_randomize}"
            f"dstep{cfg.dstep_num}",
            f"maxlast{int(cfg.offline.max_last)}",
            f"{slurm_job_id}",
        ]
    )
    sample_sample_dict_4wandb = dict(
        cfg_scale=cfg.cfg_scale,
        sm_t=cfg.sm_t,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        maskgit_randomize=cfg.maskgit_randomize,
        dstep_num=cfg.dstep_num,
        max_last=int(cfg.offline.max_last),
        fid_samples=cfg.num_fid_samples,
        offline_sample_local_bs=cfg.offline.lbs,
    )

    sample_folder_dir = f"{cfg.sample_dir}/{wandb_name}"
    logger = create_logger(rank_id, sample_folder_dir)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if cfg.use_wandb:
            entity = cfg.wandb.entity
            project = cfg.wandb.project + "_vis"
            print_rank_0(
                f"Logging to wandb entity={entity}, project={project},rank={rank}"
            )
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb_run = wandb.init(
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

    sample_img_dir = f"{sample_folder_dir}/samples"  # visible to all ranks
    gt_img_dir = f"{sample_folder_dir}/gts"  # visible to all ranks
    shutil.rmtree(sample_img_dir, ignore_errors=True)
    shutil.rmtree(gt_img_dir, ignore_errors=True)
    os.makedirs(sample_img_dir, exist_ok=True)
    os.makedirs(gt_img_dir, exist_ok=True)
    
        

    is_video = cfg.data.video_frames > 0

    dist.barrier()
    pbar = range(iterations)
    pbar = tqdm(pbar, total=iterations, desc="sampling") if rank == 0 else pbar

    if rank == 0:
        logger.info(wandb_desc)

    for bs_index in pbar:
        if rank == 0:
            logger.info(
                f"dataset.subset:{ cfg.data.subset},skip_data_loader:{skip_data_loader}"
            )
            logger.info(f"progress: {bs_index}/{iterations}")
            logger.info(f"sample_img_dir: {sample_img_dir}")
            logger.info(wandb_desc)
        if cfg.data.num_classes > 0:
            y = torch.randint(0, cfg.data.num_classes, (local_bs,), device=device)
        elif cfg.data.num_classes == -666:  # a special value for caption generation
            cap_feat, cap_str = next(caption_generator)
            y = cap_feat[:local_bs]
            assert len(cap_str) == local_bs
        else:
            y = None

        model_kwargs = dict(
            y=y,
            temperature=cfg.sm_t,
            max_last=cfg.offline.max_last,
        )

        if cfg.cfg_scale != 0.0:
            model_kwargs.update(dict(cfg_scale=cfg.cfg_scale))

        if not skip_data_loader:
            gts = next(_generator)
            _sample_size = vq_get_sample_size(len(gts), cfg)

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=_dtype):
                    indices_chains = sample_fn(_sample_size, model_fn, **model_kwargs)
                _indices = indices_chains[-1]
                samples = tokenizer_decode_fn(_indices)

            # gts = gts[: len(samples)]
            gts = samples.clone()
        else:
            _sample_size = vq_get_sample_size(local_bs, cfg)

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=_dtype):
                    indices_chains = sample_fn(_sample_size, model_fn, **model_kwargs)
                _indices = indices_chains[-1]
                samples = tokenizer_decode_fn(_indices)

            gts = samples.clone()

        sam_4fid, gts_4fid = samples, gts
        sam_4fid_all = all_gather_my(sam_4fid.to(device))
        gts_4fid_all = all_gather_my(gts_4fid.to(device))
        dist.barrier()
        if rank == 0:
            metric_fid_with_npz.update_fake(sam_4fid_all)

        if rank == 0:
            if cfg.use_wandb and bs_index <= 1:
                if cfg.wandb_tag:
                    wandb_run.tags = wandb_run.tags + (cfg.wandb_tag,)
                if cfg.data.num_classes > 0:
                    captions_sample = [
                        imagenet_id2label[y[_].item()]
                        for _ in range(min(16, len(sam_4fid)))
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
                wandb_dict.update(sample_sample_dict_4wandb)
                wandb.log(
                    wandb_dict,
                    step=bs_index,
                )
                logger.info("log_image into wandb")
                if not is_video:
                    wandb.log(
                        {
                            f"vis/samples": wandb.Image(sam_4fid[:16]),
                            f"vis/gts": wandb.Image(gts_4fid[:16]),
                        },
                        step=bs_index,
                    )
                logger.info("log_image into wandb")
                logger.info(f"saving sample images, in {sample_img_dir}")

        dist.barrier()

        if is_video or cfg.offline.save_samples_to_disk:
            # not need to save samples to disk, only for video it is needed to save to disk
            if not is_video:
                # Save samples to disk as individual .png files
                for _iii, sample in tqdm(
                    enumerate(sam_4fid), total=len(sam_4fid), desc="saving png samples"
                ):
                    unique_id = uuid.uuid4().hex[:6]
                    save_image(
                        sample / 255.0, f"{sample_img_dir}/rank{rank}_{unique_id}.png"
                    )

                # for _iii, sample in enumerate(gts_4fid):
                #   unique_id = uuid.uuid4().hex[:6]
                #   save_image(sample / 255.0, f"{gt_img_dir}/rank{rank}_{unique_id}.png")


            else:

                for _video_id, (samples, gts) in enumerate(zip(sam_4fid, gts_4fid)):
                    samples = rearrange(samples, "t c h w -> t h w c")  # [0,255]
                    gts = rearrange(gts, "t c h w -> t h w c")  # [0,255]
                    samples = samples.cpu().numpy()
                    gts = gts.cpu().numpy()
                    unique_id = uuid.uuid4().hex[:6]
                    write_video(
                        os.path.join(sample_img_dir, f"rank{rank}_{unique_id}.mp4"),
                        samples,
                        fps=cfg.data.fps,  
                    )
                    write_video(
                        os.path.join(gt_img_dir, f"rank{rank}_{unique_id}.mp4"),
                        gts,
                        fps=cfg.data.fps,  
                    )

            dist.barrier()

    if rank == 0:
        if not is_video:
            mymetric_dict = {}
            _metric_dict = metric_fid_with_npz.compute()
            fid_with_npz = _metric_dict["fid"]
            print_rank_0(f"fid_with_npz: {fid_with_npz}")
            mymetric_dict.update(fid_with_npz=fid_with_npz)
            wandb.log(mymetric_dict)

        else:
            fake_root = os.path.expanduser(sample_img_dir)
            real_root = os.path.expanduser(cfg.data.fvd_real_video_dir)
            # real_root = os.path.expanduser(gt_img_dir)
            print_rank_0(real_root)
            print_rank_0(fake_root)

            mymetric_dict = calculate_fvd_github(
                gen_dir=fake_root,
                gt_dir=real_root,
                frames=16,
                resolution=128,  # 128,64
            )

            mymetric_dict = {f"video/{k}": v for k, v in mymetric_dict.items()}
            wandb.log(mymetric_dict)
        if False:
            try:
                shutil.rmtree(sample_img_dir)
                shutil.rmtree(gt_img_dir)
                print_rank_0(
                    f"removed sample_img_dir and gt_img_dir\n{sample_img_dir}\n{gt_img_dir}",
                )
            except Exception as e:
                print_rank_0(
                    f"Error removing directory {sample_img_dir},{gt_img_dir}: {e}"
                )

    print_rank_0("done sampling")
    # dist.barrier(), important! remove this.


if __name__ == "__main__":

    main()
