# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
"""
import random
import shutil
from einops import rearrange, repeat
from omegaconf import OmegaConf
from utils_vq import get_dataloader
import torch, math
import sys
from utils_vq import (
    vq_get_dynamic,
    vq_get_encoder_decoder,
    vq_get_generator,
    vq_get_vae,
    vq_get_sample_size,
)
from datetime import datetime
import socket

from utils.my_metrics_offline import MyMetric_Offline as MyMetric
from utils.train_utils import (
    create_logger,
    get_latest_checkpoint,
    get_model,
    requires_grad,
    update_ema,
    wandb_runid_from_checkpoint,
    get_lr_scheduler,
)
from utils_vq import get_version_number, print_rank_0

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
import torch.distributed as dist
from copy import deepcopy
from time import time
import logging
import os
from tqdm import tqdm
import wandb
from utils.train_utils import rankzero_logging_info
import hydra
from hydra.core.hydra_config import HydraConfig
import accelerate
import socket
from utils_vq import wandb_visual_dict, print_rank_0, has_label, get_max_ckpt_from_dir
from datasets_wds.cityscapes_helper import (
    cityscapes_only_categories_indices_segmentation_to_img,
)
import warnings
warnings.filterwarnings("ignore")  # ignore warning

def update_note(cfg, accelerator, slurm_job_id):

    cfg.note = "_".join(
        [
            f"{get_version_number()}",
            f"vqacc",
            str(cfg.note),
            f"{cfg.mixed_precision}",
            f"{cfg.data.name}",
            f"{cfg.model.name}",
            f"{cfg.dynamic.name}",
            f"{cfg.tokenizer.name}",
            f"bs{cfg.data.batch_size}acc{cfg.accum}",
            f"wd{cfg.optim.wd}",
            f"gc{float(cfg.max_grad_norm)}",
            f"{accelerator.state.num_processes}g",
            f"{socket.gethostname()}",
            f"{slurm_job_id}",
        ]
    )

    print_rank_0(f"note: {cfg.note}")

    return cfg.note


#################################################################################
#                                  Training Loop                                #
#################################################################################


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    return _main(cfg)


def _main(cfg):
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    print(f"slurm_job_id: {slurm_job_id}")
    try:
        slurm_job_id = str(slurm_job_id)
    except:
        slurm_job_id = "000"
    if cfg.accum > 1:
        cfg.data.train_steps = cfg.data.train_steps * cfg.accum
        cfg.log_every = cfg.log_every * cfg.accum
        cfg.ckpt_every = cfg.ckpt_every * cfg.accum
        cfg.data.sample_vis_every = cfg.data.sample_vis_every * cfg.accum
        cfg.data.sample_fid_every = cfg.data.sample_fid_every * cfg.accum

        print_rank_0(f"update accum to several params")

    if cfg.debug:
        cfg.data.batch_size = 4
        cfg.ckpt_every = 10
        cfg.data.sample_fid_n = 1_00
        cfg.data.sample_fid_bs = 4
        cfg.data.sample_fid_every = 5
        cfg.data.sample_vis_every = 3
        cfg.data.sample_vis_n = 2
        print_rank_0("debug mode, using smaller batch size and sample size")

    cfg.data.sample_fid_bs = cfg.data.batch_size//2
    print_rank_0(f"cfg.data.sample_fid_bs: {cfg.data.sample_fid_bs}")
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision, gradient_accumulation_steps=cfg.accum
    )
    if cfg.accum > 1:
        print_rank_0(f"accumulate gradients over {cfg.accum} steps")
    else:
        print_rank_0("not accumulate gradients")
    ############################################################

    print_rank_0(f"accelerator.mixed_precision:{accelerator.mixed_precision}")

    cfg.data.global_batch_size = (
        cfg.data.per_gpu_batch_size * accelerator.state.num_processes
    )
    print(f"update the webdataset's global_batch_size: {cfg.data.global_batch_size}")
    device = accelerator.device
    accelerate.utils.set_seed(cfg.global_seed, device_specific=True)
    rank = accelerator.state.process_index
    print(
        f"Starting rank={rank}, world_size={accelerator.state.num_processes}, accelerator.mixed_precision={accelerator.mixed_precision},device={device}."
    )
    is_multiprocess = True if accelerator.state.num_processes > 1 else False

    train_steps = 0

    accelerator.wait_for_everyone()
    wandb_name = cfg.note = update_note(
        cfg=cfg, accelerator=accelerator, slurm_job_id=slurm_job_id
    )
    now = datetime.now()
    cfg.run_dir = f"./outputs/{wandb_name}/{now:%Y-%m-%d_%H-%M-%S}"
    logger = create_logger(rank, cfg.run_dir)
    if accelerator.is_main_process:
        logging.info(cfg)
        experiment_dir = cfg.run_dir
        logging.info(f"Experiment directory created at {experiment_dir}")
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        print_rank_0(f"Experiment directory created at {experiment_dir}")
        if cfg.use_wandb:
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            config_dict = {
                **config_dict,
                "experiment_dir": experiment_dir,
                "world_size": accelerator.state.num_processes,
                "local_batch_size": cfg.data.batch_size
                * accelerator.state.num_processes,
                "job_id": slurm_job_id,
            }
            extra_wb_kwargs = dict()
            if cfg.resume is not None:
                runid = wandb_runid_from_checkpoint(cfg.resume)
                extra_wb_kwargs["resume"] = "must"
                extra_wb_kwargs["id"] = runid

            wandb_run = wandb.init(
                project=cfg.wandb.project,
                name=cfg.note,
                config=config_dict,
                dir=experiment_dir,
                **extra_wb_kwargs,
            )
            wandb_project_url = (
                f"https://wandb.ai/dpose-team/{wandb.run.project}/runs/{wandb.run.id}"
            )
            wandb_sync_command = (
                f"wandb sync {experiment_dir}/wandb/latest-run --append"
            )
            print(wandb_project_url + "\n" + wandb_sync_command)

    best_fid = 666
    best_ckpt = None

    model = get_model(cfg)
    model = model.to(device)

    print_rank_0(f"sample_fid_n: {cfg.data.sample_fid_n}")
    print_rank_0(f"sample_fid_bs: {cfg.data.sample_fid_bs}")
    print_rank_0(f"accelerator.state.num_processes: {accelerator.state.num_processes}")
    _fid_eval_batch_nums = cfg.data.sample_fid_n // (
        cfg.data.sample_fid_bs * accelerator.state.num_processes
    )
    assert _fid_eval_batch_nums > 0, f"{_fid_eval_batch_nums} <= 0"

    ema_model = deepcopy(model).to(device)

    if cfg.optim.name == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.wd,
        )
    elif cfg.optim.name == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd
        )
    else:
        raise ValueError(f"optimizer={cfg.optim.name} not supported")

    lr_scheduler = get_lr_scheduler(opt, **cfg.lrschedule)

    update_ema(
        ema_model, model, decay=0
    )  # Ensure EMA is initialized with synced weights

    training_losses_fn, sample_fn = vq_get_dynamic(cfg, device)

    encode_fn, decode_fn = vq_get_encoder_decoder(cfg, device)

    _param_amount = sum(p.numel() for p in model.parameters())
    param_num_embed_table = model.param_num_embed_table
    pre_logits_param_num = model.param_num_pre_logits

    if accelerator.is_main_process:
        print_rank_0(f"#parameters: {_param_amount}")
        wandb_summary = dict(
            dstep_num=cfg.dstep_num,
            param_amount=_param_amount,
            param_num_embed_table=param_num_embed_table,
            pre_logits_param_num=pre_logits_param_num,
            mixed_precision=accelerator.mixed_precision,
        )
        wandb.run.summary.update(wandb_summary)
        wandb.log(wandb_summary)

    train_loader = get_dataloader(cfg)
    train_loader, opt, model, ema_model = accelerator.prepare(
        train_loader, opt, model, ema_model
    )

    if cfg.ckpt_latte is not None:
        assert cfg.model.name.startswith("dlattte") and "xl2" in cfg.model.name
        state_dict = torch.load(
            cfg.ckpt_latte, map_location=lambda storage, loc: storage
        )
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("final_layer.")
        }
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("x_embedder.")
        }
        # We didn't use the position embedding of DiT in the pre-training of latte as well,
        try:
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)
            ema_model.load_state_dict(state_dict, strict=True)
            ema_model = ema_model.to(device)
        except Exception as e:
            if accelerator.is_main_process:
                print("load_state_dict error", e)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            ema_model.load_state_dict(state_dict, strict=False)
            ema_model = ema_model.to(device)

    if cfg.resume is not None:
        if os.path.isdir(cfg.resume):
            cfg.resume = get_max_ckpt_from_dir(cfg.resume)
        ckpt_path = cfg.resume
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"])
        model = model.to(device)
        ema_model.load_state_dict(state_dict["ema"])
        ema_model = ema_model.to(device)
        opt.load_state_dict(state_dict["opt"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        logging.info("overriding args with checkpoint args")
        logging.info(cfg)
        train_steps = state_dict["train_steps"]
        best_fid = state_dict["best_fid"]

        logging.info(f"Loaded checkpoint from {ckpt_path}, train_steps={train_steps}")
        requires_grad(ema_model, False)
        if rank == 0:
            shutil.copy(ckpt_path, checkpoint_dir)

    elif cfg.ckpt is not None:
        if os.path.isdir(cfg.ckpt):
            cfg.ckpt = get_max_ckpt_from_dir(cfg.ckpt)
        ckpt_path = cfg.ckpt
        logging.info(f"ckpt(no resume), Loaded checkpoint from {ckpt_path}, ")
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if accelerator.state.num_processes == 1:
            state_dict["model"] = {
                k.replace("module.", ""): v for k, v in state_dict["model"].items()
            }
            state_dict["ema"] = {
                k.replace("module.", ""): v for k, v in state_dict["ema"].items()
            }
        model.load_state_dict(state_dict["model"])
        model = model.to(device)
        ema_model.load_state_dict(state_dict["ema"])
        ema_model = ema_model.to(device)
        requires_grad(ema_model, False)
        if rank == 0:
            shutil.copy(ckpt_path, checkpoint_dir)

    model.train()
    ema_model.eval()

    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.mixed_precision == "fp16":
        target_dtype = torch.float16
        print_rank_0("using fp16 mixed precision")
    elif accelerator.mixed_precision == "bf16":
        target_dtype = torch.bfloat16
        print_rank_0("using bfloat16 mixed precision")
    elif accelerator.mixed_precision == "no":
        target_dtype = torch.float32
        print_rank_0("using no mixed precision")

    if cfg.use_ema:
        print_rank_0("using ema model for sampling...")

        if cfg.use_cfg:
            print_rank_0("using cfg for sampling...")
            model_sample_fn = accelerator.unwrap_model(ema_model).forward_with_cfg
        else:
            print_rank_0("using non-cfg for sampling...")
            if has_label(cfg.data.name):
                model_sample_fn = accelerator.unwrap_model(
                    ema_model
                ).forward_without_cfg
            else:
                model_sample_fn = accelerator.unwrap_model(ema_model).forward

    else:
        raise ValueError("args.use_ema must be True")

    @torch.no_grad()
    def sample_img(bs, cfg, _sample_size=None):
        model.eval()
        if _sample_size is None:
            _sample_size = vq_get_sample_size(bs, cfg)
        else:
            _sample_size = _sample_size
        print_rank_0(f"sampling with sample_size: {_sample_size}")
        vis_config, sample_kwargs = dict(), dict(mp_type=target_dtype)
        if "imagenet" in cfg.data.name and cfg.data.num_classes > 0:
            ys = torch.randint(0, cfg.data.num_classes, (bs,)).to(device)
            sample_kwargs["y"] = ys
        elif cfg.data.name.startswith("ucf101") and cfg.data.num_classes > 0:
            ys = torch.randint(0, cfg.data.num_classes, (bs,)).to(device)
            sample_kwargs["y"] = ys
        elif cfg.data.name.startswith("coco"):
            cap_feat, cap = next(cap_dg)
            cap_feat = cap_feat[:bs].to(device)
            assert len(cap_feat) == bs
            sample_kwargs["y"] = cap_feat
        elif cfg.data.name.startswith("cs"):
            x, y = next(train_dg)
            y = y[:bs].to(device)
            assert len(y) == bs
            vis_config.update(
                wandb_visual_dict(
                    "cityscapes/mask",
                    cityscapes_only_categories_indices_segmentation_to_img(y),
                    is_video=is_video,
                )
            )
            vis_config.update(
                wandb_visual_dict(
                    "cityscapes/img",
                    x,
                    is_video=is_video,
                )
            )
            x = encode_fn(x)
            sample_kwargs["x"] = x
            sample_kwargs["y"] = y
        else:
            # FaceForensics
            sample_kwargs["y"] = None
        #############
        print_rank_0("using non-cfg for sampling...")
        ##############
        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=target_dtype):
                    samples_chains = sample_fn(
                        _sample_size, model_sample_fn, **sample_kwargs
                    )
                samples = samples_chains[-1]

        except Exception as e:
            logging.info("sample_fn error", exc_info=True)
            logging.info(e)
            if accelerator.is_main_process:
                if "sampling_error" not in wandb_run.tags:
                    wandb_run.tags = wandb_run.tags + ("sampling_error",)
                    print_rank_0("sampling_error, wandb_run.tags:", wandb_run.tags)
            samples = (torch.randn(_sample_size) * 0).long().to(device)

        samples = decode_fn(samples)

        accelerator.wait_for_everyone()
        out_sample_global = accelerator.gather(samples.contiguous().to(device))
        model.train()
        return out_sample_global, samples, vis_config

    vae = vq_get_vae(cfg, device)
    train_dg, real_img_dg, cap_dg = vq_get_generator(
        cfg=cfg,
        device=device,
        loader=train_loader,
        rank_id=accelerator.state.process_index,
        train_steps=train_steps,
        vae=vae,
    )

    my_metric = MyMetric(npz_real=cfg.data.npz_real)
    is_video = cfg.data.video_frames > 0

    if "indices" in cfg.data.name:
        gtimg = next(real_img_dg)
        gtimg = accelerator.gather(gtimg.contiguous())
        if accelerator.is_main_process and cfg.use_wandb:
            gtimg = gtimg[: min(9, cfg.data.sample_fid_bs)]
            _indices = gtimg
            gtimg_recon = decode_fn(_indices)
            wandb_dict = {}
            wandb_dict.update(
                wandb_visual_dict(
                    "vis/gttest_recovered_from_indices", gtimg_recon, is_video=is_video
                )
            )
            wandb.log(wandb_dict)
            logging.info(wandb_project_url + "\n" + wandb_sync_command)
    else:
        gtimg = next(real_img_dg)
        gtimg = accelerator.gather(gtimg.contiguous())
        if accelerator.is_main_process and cfg.use_wandb:
            gtimg = gtimg[: min(9, cfg.data.sample_fid_bs)]
            _indices = encode_fn(gtimg)
            gtimg_recon = decode_fn(_indices)
            wandb_dict = {}
            wandb_dict.update(wandb_visual_dict("vis/gttest", gtimg, is_video=is_video))
            wandb_dict.update(
                wandb_visual_dict(
                    "vis/gttest_recovered", gtimg_recon, is_video=is_video
                )
            )
            wandb.log(wandb_dict)
            logging.info(wandb_project_url + "\n" + wandb_sync_command)

    progress_bar = tqdm(
        range(cfg.data.train_steps),
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    grad_norm = None
    while train_steps < cfg.data.train_steps:
        x, y = next(train_dg)
        x = encode_fn(x)
        model_kwargs = dict(y=y, mp_type=target_dtype) if y is not None else dict()

        with accelerator.accumulate(model):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=target_dtype):
                loss_dict = training_losses_fn(model, x, **model_kwargs)
                loss = loss_dict["loss"].mean()
            # Check if the loss is nan
            loss_value = loss.item()
            if not math.isfinite(loss_value):#don't let bad grad pollute the model weights
                print("Loss is {}, stopping training".format(loss_value), force=True)
                sys.exit(1)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.max_grad_norm
                )
            opt.step()
            if accelerator.sync_gradients:
                lr_scheduler.step()
                update_ema(ema_model, model)

        running_loss += loss.item()
        log_steps += 1
        train_steps += 1
        progress_bar.update(1)
        if train_steps % cfg.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            if is_multiprocess:
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            if grad_norm is not None:
                grad_norm = accelerator.gather(grad_norm).mean().item()
            avg_loss = avg_loss.item() / accelerator.state.num_processes
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logging.info(
                    f"(step={train_steps:07d}/{cfg.data.train_steps}), Best_FID: {best_fid}, Train Loss: {avg_loss:.4f}, BS-1GPU: {len(x)} Train Steps/Sec: {steps_per_sec:.2f}, slurm_job_id: {slurm_job_id}, {experiment_dir}"
                )
                logging.info(wandb_sync_command)
                latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
                logging.info(latest_checkpoint)
                logging.info(wandb_project_url)
                logging.info(wandb_name)

                if cfg.use_wandb:
                    wandb_dict = {
                        "train_loss": avg_loss,
                        "train_steps_per_sec": steps_per_sec,
                        "best_fid": best_fid,
                        "bs_1gpu": len(x) * cfg.accum,
                        "train_steps": train_steps,
                        "grad_norm": grad_norm,
                        "lr": opt.param_groups[0]["lr"],
                    }

                    for k, v in loss_dict.items():
                        if "log/" in k:
                            if isinstance(v, torch.Tensor):
                                wandb_dict[k] = v.mean().item()
                            else:
                                wandb_dict[k] = v

                    wandb.log(
                        wandb_dict,
                        step=train_steps//cfg.accum,
                    )

            running_loss = 0
            log_steps = 0
            start_time = time()

        if train_steps % cfg.data.sample_vis_every == 0 and train_steps > 0:

            _sample_size = vq_get_sample_size(cfg.data.sample_vis_n, cfg)
            out_sample_global_random, samples, vis_config = sample_img(
                bs=cfg.data.sample_vis_n, cfg=cfg, _sample_size=_sample_size
            )

            if accelerator.is_main_process and cfg.use_wandb:
                wandb_dict = {}
                wandb_dict.update(vis_config)

                wandb_dict.update(
                    wandb_visual_dict(
                        "vis/sample_random", out_sample_global_random, is_video=is_video
                    )
                )

                wandb.log(
                    wandb_dict,
                    step=train_steps//cfg.accum,
                )
                rankzero_logging_info(rank, "Generating samples done.")
            torch.cuda.empty_cache()

        if train_steps % cfg.data.sample_fid_every == 0 and train_steps > 0:
            with torch.no_grad():  # very important
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    my_metric.reset()
                ########
                print_rank_0(
                    f"Generating EMA samples, batch size_gpu = {cfg.data.sample_fid_bs}..."
                )

                vis_wandb_sample = None
                start_time_samplingfid = time()
                _desc_tqdm = f"({accelerator.state.num_processes} GPUs),local BS{cfg.data.sample_fid_bs}xIter{_fid_eval_batch_nums}_FID{cfg.data.sample_fid_n}"
                for _b_id in tqdm(
                    range(_fid_eval_batch_nums),
                    desc=f"sampling FID on the fly {_desc_tqdm}",
                    total=_fid_eval_batch_nums,
                ):
                    out_sample_global, samples, vis_config = sample_img(
                        bs=cfg.data.sample_fid_bs, cfg=cfg
                    )
                    if _b_id == 0:
                        vis_wandb_sample = out_sample_global
                    if accelerator.is_main_process:
                        my_metric.update_fake(out_sample_global)
                    del out_sample_global, samples
                    torch.cuda.empty_cache()

                ###
                sample_time_min = (time() - start_time_samplingfid) / 60

                if accelerator.is_main_process and cfg.use_wandb:
                    _metric_dict = my_metric.compute()
                    my_metric.reset()
                    fid = _metric_dict["fid"]
                    best_fid = min(fid, best_fid)
                    print_rank_0(f"FID: {fid}, best_fid: {best_fid}")
                    wandb_dict = {
                        "best_fid": best_fid,
                        "sample_time_min": sample_time_min,
                    }
                    wandb_dict.update({f"eval/{k}": v for k, v in _metric_dict.items()})
                    wandb_dict.update(
                        wandb_visual_dict(
                            "vis/sample", vis_wandb_sample, is_video=is_video
                        ),
                    )
                    wandb.log(
                        wandb_dict,
                        step=train_steps//cfg.accum,
                    )
                rankzero_logging_info(rank, "Generating EMA samples done.")

                torch.cuda.empty_cache()
        if train_steps % cfg.ckpt_every == 0 and train_steps > 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "opt": opt.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": cfg,
                    "train_steps": train_steps,
                    "best_fid": best_fid,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                try:
                    os.umask(0o000)
                    torch.save(checkpoint, checkpoint_path)
                except Exception as e:
                    logging.info(f"save_checkpoint error: {e}")
                    if rank == 0:
                        if "checkpoint_error" not in wandb_run.tags:
                            wandb_run.tags = wandb_run.tags + ("checkpoint_error",)
                            print_rank_0(
                                "checkpoint_error, wandb_run.tags:", wandb_run.tags
                            )
                wandb.run.summary["latest_checkpoint_path"] = checkpoint_path
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            accelerator.wait_for_everyone()

    progress_bar.close()
    #########
    model.eval()
    state_dict = torch.load(best_ckpt, map_location=lambda storage, loc: storage)
    _model_dict = state_dict["ema"]
    print_rank_0(f"loading best ckpt: {best_ckpt}, and use ema to eval final fid")

    model.load_state_dict(_model_dict)

    eval_last_fid_num = cfg.data.eval_last_fid_num
    _fid_eval_batch_nums = math.ceil(
        eval_last_fid_num / (cfg.data.sample_fid_bs * accelerator.state.num_processes)
    )
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        my_metric.reset()
    ########
    print_rank_0(
        f"Generating EMA samples, batch size_gpu = {cfg.data.sample_fid_bs}..."
    )

    for _b_id in tqdm(
        range(_fid_eval_batch_nums),
        desc="sampling FID on the fly",
        total=_fid_eval_batch_nums,
    ):
        out_sample_global, samples, vis_config = sample_img(
            bs=cfg.data.sample_fid_bs, cfg=cfg
        )
        if accelerator.is_main_process:
            my_metric.update_fake(out_sample_global)
        del out_sample_global, samples
        torch.cuda.empty_cache()

    ###
    if accelerator.is_main_process:
        _metric_dict = my_metric.compute()
        print_rank_0("final_eval")
        print_rank_0(_metric_dict)
        wandb_run.tags = wandb_run.tags + ("final_eval",)
        wandb.log({"eval_final/" + k: v for k, v in _metric_dict.items()})
        #####
        print_rank_0("Done!")
        wandb.finish()


if __name__ == "__main__":
    main()
