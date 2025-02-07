# [MASK] is All You Need


This repository represents the official implementation of the paper titled "[MASK] is All You Need".

[![Website](doc/badges/badge-website.svg)](https://compvis.github.io/mask)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2412.06787)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/collections/taohu/mask-is-all-you-need-6749a2ca0be7c4c5c055c122)
[![GitHub](https://img.shields.io/github/stars/CompVis/mask?style=social)](https://github.com/CompVis/mask)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/CompVis/mask?color=success&label=Issues)](https://github.com/CompVis/mask/issues?q=is%3Aissue+is%3Aclosed) 
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=CompVis/mask)

[Vincent Tao Hu](http://taohu.me),
[Björn Ommer](https://ommer-lab.com/people/ommer/ )

## TLDR

We present Discrete Interpolants, to bridge the Diffusion Models and Maskged Generative Models in discrete-state, and scale it up in vision domain.

![teaser](./doc/method.jpg)



## 🎓 Citation

Please cite our paper:

```bibtex
@InProceedings{hu2024mask,
      title={[MASK] is All You Need},
      author={Vincent Tao Hu and Björn Ommer},
      booktitle = {Arxiv},
      year={2024}
}
```

## :white_check_mark: Updates
* **` Feb. 4th, 2025`**: Training code released.
* **` Dec. 10th, 2024`**: Arxiv released.

## 📦 Training


#### COCO training(Deepspeed)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --num_processes 4 --num_machines 1 --main_process_ip 127.0.0.1 --main_process_port 8868    train_ds_vq.py   model=uvit_s2deep_it data=coco14_cond_indices dynamic=linear dynamic.mask_ce=1  input_tensor_type=bwh tokenizer=sd_vq_f8 optim.wd=0.00 "optim.betas=[0.9, 0.9]" data.train_steps=1_000_000 ckpt_every=20_000 data.sample_fid_every=100_000 data.sample_fid_n=20_000   data.batch_size=64 optim.name=adam optim.lr=2e-4 lrschedule.warmup_steps=5000 dstep_num=500  mixed_precision=bf16 accum=4
```

#### ImageNet training(accelerator,bs256)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch   --num_processes 4 --num_machines 1 --main_process_ip 127.0.0.1 --main_process_port 8868    train_acc_vq.py  model=uvit_h2_it dynamic=linear   input_tensor_type=bwh tokenizer=sd_vq_f8 data=imagenet256_cond_indices data.batch_size=64 data.sample_vis_n=16 data.sample_fid_every=50_000 ckpt_every=20_000 data.train_steps=1500_000  data.sample_fid_n=5_000 optim.name=adamw optim.lr=1e-4 optim.wd=0.0 lrschedule.warmup_steps=1     mixed_precision=bf16 accum=1
```

#### FaceForensics training(accelerator,bs64)


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch   --num_processes 4 --num_machines 1 --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_vq.py      model=dlattte_xl2_uncond_it  dynamic=linear  input_tensor_type=btwh tokenizer=sd_vq_f8  data=ffs_indices data.sample_fid_every=10_000  data.batch_size=2  data.sample_fid_bs=1   data.sample_fid_n=10_00 data.train_steps=400_000  data.sample_vis_n=1 ckpt_latte=pretrained_ckpt/dit/DiT-XL-2-256x256.pt  accum=8 mixed_precision=bf16 
```


## Evaluation

#### ImageNet 

```bash 
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --num_machines 1 --main_process_ip 127.0.0.1 --main_process_port 8868 sample_ds_vq.py model=dit_xl2_it dynamic=linear input_tensor_type=bwh tokenizer=sd_vq_f8 data=imagenet256_cond_indices data.batch_size=64 data.sample_vis_n=16 data.sample_fid_every=40_000 data.sample_fid_n=5_000 optim.name=adamw optim.lr=1e-4 optim.wd=0.0 lrschedule.warmup_steps=0 data.train_steps=1_400_000 ckpt_every=20_000 mixed_precision=bf16 accum=1  num_fid_samples=50000 offline.lbs=100 dynamic.disint.scheduler=linear dynamic.disint.sampler=maskgit maskgit_randomize=linear top_k=0 top_p=0  offline.save_samples_to_disk=1 sm_t=1.3  use_cfg=1 cfg_scale=2 dstep_num=20 ckpt="in256_ditxl2_it_1220000.pt"
```

#### COCO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --num_machines 1 --main_process_ip 127.0.0.1 --main_process_port 8868  sample_acc_vq.py model=uvit_s2deep_it data=coco14_cond_indices dynamic=linear dynamic.mask_ce=1 input_tensor_type=bwh tokenizer=sd_vq_f8 optim.wd=0.00 "optim.betas=[0.9, 0.9]" data.train_steps=1_000_000 ckpt_every=20_000 data.sample_fid_every=100_000 data.sample_fid_n=20_000 data.batch_size=64 optim.name=adam optim.lr=2e-4 lrschedule.warmup_steps=5000 dstep_num=500 mixed_precision=bf16   num_fid_samples=50000 offline.lbs=100 dynamic.disint.scheduler=linear dynamic.disint.sampler=maskgit maskgit_randomize=linear top_k=0 top_p=0  offline.save_samples_to_disk=1 sm_t=1.3  use_cfg=1 cfg_scale=2 dstep_num=20  ckpt="coco14_uvit_s2deep_it_1600000.pt"
```

#### FaceForensics 

```bash 
TODO
```

## Weights

|   Dataset   | Model | FID $\downarrow$ | HF weights🤗                                                                        |
|:----------:|:-----:|:-------:|:------------------------------------------------------------------------------------|
|  ImageNet $256\times 256$, latents: $32\times 32$| DiT_XL2_IT   |  8.26   | [weight.pth](https://huggingface.co/CompVis/discrete_interpolants/blob/main/in256_ditxl2_it_1220000.pt) |
|  COCO $256\times 256$, latents: $32\times 32$| DiT_S2Deep_IT   |  -   | [weight.pth](https://huggingface.co/CompVis/discrete_interpolants/blob/main/coco14_uvit_s2deep_it_1600000.pt) |



## Dataset Preparation

TODO

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=CompVis/discrete-interpolants&type=Date)](https://star-history.com/#CompVis/discrete-interpolants&Date)

## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)





