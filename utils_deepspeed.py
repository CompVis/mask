import torch
import torch.distributed as dist
import time
import json
import os


def all_gather_my(tensor_in):
    """Gathers arbitrary data from all ranks into a list."""
    try:
        world_size = torch.distributed.get_world_size()
        tensor_out = [torch.zeros_like(tensor_in) for _ in range(world_size)]
        tensor_out = torch.cat(tensor_out, dim=0)
        dist.all_gather_into_tensor(tensor_out, tensor_in.contiguous())
    except:
        print("all_gather_my failed in torch.distributed")
        return tensor_in
    return tensor_out


def get_ds_config(cfg):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        # "train_batch_size": cfg.data.batch_size,
        "train_micro_batch_size_per_gpu": cfg.data.batch_size,
        "gradient_accumulation_steps": cfg.accum,
        "gradient_accumulation_dtype": 'fp32',#https://github.com/microsoft/DeepSpeed/pull/2847
        "steps_per_print": cfg.log_every*cfg.accum,
        "optimizer": {
            "type": cfg.optim.name,
            "params": {
                "lr": cfg.optim.lr,
                "weight_decay": cfg.optim.wd,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": cfg.optim.lr,
                "warmup_type": "linear",
                "warmup_num_steps": cfg.lrschedule.warmup_steps,#already include accum
            },
        },
        "gradient_clipping": cfg.max_grad_norm,
        "prescale_gradients": False,
        "bf16": {"enabled": cfg.mixed_precision == "bf16"},
        "fp16": {
            "enabled": cfg.mixed_precision == "fp16",
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 48, # Huggingface documents: This means the DeepSpeed loss scaler is unable to find a scaling coefficient to overcome loss overflow. To fix it, try a higher initial_scale_power value (32 usually works).
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1,
           
        },
        "wall_clock_breakdown": False,
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": True,
        },
    }

    if cfg.mixed_precision == "bf16" and cfg.accum > 1 and False:
        raise ValueError("Huggingface documents: However, if you use gradient accumulation with bf16, gradients are accumulated in bf16 which may not be desired because this format’s low precision can lead to lossy accumulation.")
    
    if cfg.mixed_precision == "bf16" and False:
        raise ValueError("bf16 is not supported, as it is not stable in DeepSpeed, and doesnt be supported in flash attention")
    if cfg.ds.zero_stage > 0 and cfg.mixed_precision == "bf16" and False:
        raise ValueError("bf16 is not supported in zero stage > 0, as it is not stable in DeepSpeed")

    if cfg.ds.zero_stage == 1:
        ds_config["zero_optimization"] = {
            "stage": 1
        }
    elif cfg.ds.zero_stage == 0:
        ds_config["zero_optimization"] = {
            "stage": 0
        }
    elif cfg.ds.zero_stage == 2:
        ds_config["zero_optimization"] = {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        }
    elif cfg.ds.zero_stage == 3:
        ds_config["zero_optimization"] = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
        }
    else:
        raise ValueError(f"zero_stage {cfg.ds.zero_stage} not supported")

    return ds_config

