# Imports
import torch
from torch import nn
from torch.nn import functional as F
import os
from typing import Tuple, List
from einops import rearrange
from utils_vq import print_rank_0

import numpy as np
import math
from tqdm import tqdm


import torch.distributed as dist


class Multi_Modal_Vocab:
    def __init__(
        self,
        vocab_size_x: int,
        vocab_size_y: int,
    ):
        self.vs_x = vocab_size_x  # [VOCAB+1]:VOCAB_X, MASK_X
        self.vs_y = vocab_size_y  # [VOCAB+1]:VOCAB_Y, MASK_Y

    def y_add_delta(self, y):
        return y

    def y_minus_delta(self, y):
        return y

    def out_of_range_x_reindex(self, input_x, reindex_id):
        out_of_range = (input_x >= self.vs_x) | (input_x < 0)
        return input_x * (~out_of_range) + reindex_id * out_of_range

    def out_of_range_y_reindex(self, input_y, reindex_id):
        out_of_range = (input_y >= self.vs_y) | (input_y < 0)
        return input_y * (~out_of_range) + reindex_id * out_of_range


def print_rank_0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
            # logging.info(*args, **kwargs)
    else:
        print(*args, **kwargs)


def adap_scheduler(step, token_num, mode="arccos", leave=False):
    """Create a sampling scheduler
    :param
     step  -> int:  number of prediction during inference
     mode  -> str:  the rate of value to unmask
     leave -> bool: tqdm arg on either to keep the bar or not
    :return
     scheduler -> torch.LongTensor(): the list of token to predict at each step
    """
    r = torch.linspace(1, 0, step)
    if mode == "root":  # root scheduler
        val_to_mask = 1 - (r**0.5)
    elif mode == "linear":  # linear scheduler
        val_to_mask = 1 - r
    elif mode == "square":  # square scheduler
        val_to_mask = 1 - (r**2)
    elif mode == "cosine":  # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":  # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    else:
        return

    # fill the scheduler by the ratio of tokens to predict at each step
    sche = (val_to_mask / val_to_mask.sum()) * token_num
    sche = sche.round()
    sche[sche == 0] = 1  # add 1 to predict a least 1 token / step
    sche[-1] += (token_num) - sche.sum()  # need to sum up nb of code
    return tqdm(sche.int(), leave=leave)


def argmax_except_mask(pt, mask_token_id: int):
    pt[:, mask_token_id] = -torch.inf
    return pt.argmax(dim=1)


def logits_with_top_k_top_p_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    type_data: str = None,
) -> torch.Tensor:  # return idx, shaped (B, l)
    if type_data == "bwh":
        b, k, w, h = logits_BlV.shape
        logits_BlV = rearrange(logits_BlV, "b k w h -> b (w h) k")
    elif type_data == "bcwh":
        b, k, c, w, h = logits_BlV.shape
        logits_BlV = rearrange(logits_BlV, "b k c w h -> b (c w h) k")
    elif type_data == "bt":
        b, k, t = logits_BlV.shape
        logits_BlV = rearrange(logits_BlV, "b k t -> b t k")
    else:
        raise ValueError(f"type_data={type_data} not supported")
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (
            1 - top_p
        )
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(
                sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
            ),
            -torch.inf,
        )
    if type_data == "bwh":
        logits_BlV = rearrange(logits_BlV, "b (w h) k -> b k w h", w=w, h=h)
    elif type_data == "bcwh":
        logits_BlV = rearrange(logits_BlV, "b (c w h) k -> b k c w h", c=c, w=w, h=h)
    elif type_data == "bt":
        logits_BlV = rearrange(logits_BlV, "b t k -> b k t")
    else:
        raise ValueError(f"type_data={type_data} not supported")
    return logits_BlV


def get_uniform_n_samples(_input, chain_num):
    chain_indices = (np.linspace(0, 0.9999, chain_num) * len(_input)).astype(np.int32)
    # use 0.9999 instead of 1 for corner case
    _input = [_input[i] for i in chain_indices]
    return _input


def pop_keys(model_kwargs, key_list: List[str]):
    for key in key_list:
        if key in model_kwargs:
            model_kwargs.pop(key)


def pad_like_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *(1 for _ in range(y.ndim - x.ndim)))


def indices_to_diracp(x, vocab_size: int, input_tensor_type: str = "bt"):
    if input_tensor_type == "bt":
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b t k -> b k t")
    elif input_tensor_type == "bwh":
        b, w, h = x.shape
        x = rearrange(x, "b w h -> b (w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (w h) k -> b k w h", w=w, h=h)
    elif input_tensor_type == "bcwh":
        b, c, w, h = x.shape
        x = rearrange(x, "b c w h -> b (c w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (c w h) k -> b k c w h", c=c, w=w, h=h)
    else:
        raise ValueError(f"input_tensor_type {input_tensor_type} not supported")


def sample_p(pt, input_tensor_type: str, mask_token_id=None):
    if torch.isnan(pt).any() or torch.isinf(pt).any() or (pt < 0).any():
        raise ValueError("pt contains NaN, Inf, or negative values")

    if input_tensor_type == "bt":
        b, k, t = pt.shape
        pt = rearrange(pt, "b k t -> (b t) k")
        if mask_token_id is not None:
            pt[:, mask_token_id] = 0
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, t)
    elif input_tensor_type == "bwh":
        b, k, h, w = pt.shape
        pt = rearrange(pt, "b k h w -> (b h w) k")
        if mask_token_id is not None:
            pt[:, mask_token_id] = 0
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, h, w)
    elif input_tensor_type == "bcwh":
        b, k, c, h, w = pt.shape
        pt = rearrange(pt, "b  k c h w -> (b c h w) k")
        if mask_token_id is not None:
            pt[:, mask_token_id] = 0
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, c, h, w)
    else:
        raise ValueError(f"input_tensor_type {input_tensor_type} not supported")


class KappaScheduler:
    def __init__(self) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError


class Coupling:
    def __init__(self) -> None:
        pass

    def sample(self, x1):
        raise NotImplementedError


class Ucoupling(Coupling):
    def __init__(self) -> None:
        pass

    def sample(self, x1, mask_id: int):
        return torch.ones_like(x1) * mask_id, x1


class Ccoupling(Coupling):
    def __init__(self, msk_prop: float = 0.8) -> None:
        if msk_prop is None:
            print("Ccoupling, msk_prop is None, using coupling by random prob")
        elif msk_prop > 0:
            print("Ccoupling, msk_prop: ", msk_prop, "data_prob", 1 - msk_prop)
        else:
            raise ValueError("msk_prop must be non-negative")

        self.msk_prob = msk_prop

    def sample(self, x1, mask_id: int):
        if self.msk_prob is None:
            _msk_prob = torch.rand_like(x1.float())
        else:
            _msk_prob = self.msk_prob
        _mask = torch.rand_like(x1.float()) > _msk_prob
        _mask_id = torch.ones_like(x1) * mask_id
        x0 = x1 * _mask + _mask_id * (~_mask)
        return x0, x1


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return (
            -2 * (t**3)
            + 3 * (t**2)
            + self.a * (t**3 - 2 * t**2 + t)
            + self.b * (t**3 - t**2)
        )

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return (
            -6 * (t**2)
            + 6 * t
            + self.a * (3 * t**2 - 4 * t + 1)
            + self.b * (3 * t**2 - 2 * t)
        )


class LinearScheduler(KappaScheduler):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1.0


class QuadraticScheduler(KappaScheduler):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t**2

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 2 * t


class RootScheduler(KappaScheduler):
    def __init__(self) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t**0.5

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 0.5 * t ** (-0.5)


class CosineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = torch.pi * 0.5

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1 - torch.cos(self.coeff * t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.sin(self.coeff * t) * self.coeff


class SineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = torch.pi * 0.5

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.sin(self.coeff * t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.cos(self.coeff * t) * self.coeff


class ArcCosineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = 2 / torch.pi
        self.eps = 1e-6

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1 - self.coeff * torch.acos(t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff / torch.sqrt(1 - t**2 + self.eps)


class ArcSinScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = 2 / torch.pi
        self.eps = 1e-6

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff * torch.asin(t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff / torch.sqrt(1 - t**2 + self.eps)



def get_scheduler(scheduler_name: str):
    if scheduler_name == "linear":
        return LinearScheduler()
    elif scheduler_name == "cubic":
        return CubicScheduler()
    elif scheduler_name == "cosine":
        return CosineScheduler()
    elif scheduler_name == "sine":
        return SineScheduler()
    elif scheduler_name == "arc_cosine":
        return ArcCosineScheduler()
    elif scheduler_name == "arc_sin":
        return ArcSinScheduler()
    elif scheduler_name == "root":
        return RootScheduler()
    elif scheduler_name == "quadratic":
        return QuadraticScheduler()
    else:
        raise ValueError(f"scheduler={scheduler_name} not supported")
        
class DiscreteInterpolants:
    def __init__(
        self,
        vocab_size_x: int,
        coupling: Coupling,
        kappa: KappaScheduler,
        device: torch.device,
        vocab_size_y: int,
        type_x: str = "bt",
        type_y: str = "bt",
        mask_ce: bool = True,
        smoothing_factor: float = 0.0,
    ) -> None:

        self.mm_modal = Multi_Modal_Vocab(vocab_size_x, vocab_size_y)
        self.logitsize_x = self.mm_modal.vs_x
        self.logitsize_y = self.mm_modal.vs_y
        self.maskid_x = self.logitsize_x - 1
        self.maskid_y = self.logitsize_y - 1
        self.coupling = coupling
        self.kappa = kappa
        self.device = device
        self.type_x = type_x
        self.type_y = type_y
        self.mask_ce = mask_ce
        self.smoothing_factor = smoothing_factor

        print_rank_0(f"smoothing_factor: {smoothing_factor}")

    def model_forward(self, model, x, y, sample_mode, **model_kwargs):
        logits_x, logits_y = model(x, y, sample_mode=sample_mode, **model_kwargs)
        if sample_mode in ["cls2img", "mask2img"]:
            return logits_x
        elif sample_mode in ["img2cls", "img2mask"]:
            return logits_y
        else:
            raise ValueError(f"Unknown mode {sample_mode}")

    def forward_u(
        self,
        t: float | torch.Tensor,
        xt,
        model: nn.Module,
        sample_mode,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        type_data: str = None,
        **model_kwargs,
    ):
        if sample_mode in ["cls2img", "mask2img"]:
            dirac_xt = indices_to_diracp(xt, self.logitsize_x, self.type_x)
            xxx = xt
            yyy = model_kwargs.pop("y")
            yyy = self.mm_modal.y_add_delta(yyy)
        elif sample_mode in ["img2cls", "img2mask"]:
            dirac_xt = indices_to_diracp(xt, self.logitsize_y, self.type_y)
            xxx = model_kwargs.pop("x")
            yyy = xt
        else:
            raise ValueError(f"Unknown mode {sample_mode}")

        pop_keys(model_kwargs, ["x", "y", "return_dict", "reindex_final"])
        p1t = (
            self.model_forward(model, xxx, yyy, sample_mode, **model_kwargs)
            / temperature
        )
        p1t = logits_with_top_k_top_p_(
            p1t, top_k=top_k, top_p=top_p, type_data=type_data
        )
        p1t = torch.softmax(p1t, dim=1)
        kappa_coeff = self.kappa.derivative(t) / (1 - self.kappa(t))
        return kappa_coeff * (p1t - dirac_xt)

    def forward_u_maskgit(
        self,
        xt,
        model: nn.Module,
        sample_mode,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        type_data: str = None,
        **model_kwargs,
    ):
        if sample_mode in ["cls2img", "mask2img"]:
            xxx = xt
            yyy = model_kwargs.pop("y")
            yyy = self.mm_modal.y_add_delta(yyy)
        elif sample_mode in ["img2cls", "img2mask"]:
            xxx = model_kwargs.pop("x")
            yyy = xt
        else:
            raise ValueError(f"Unknown mode {sample_mode}")

        pop_keys(model_kwargs, ["x", "y", "return_dict", "reindex_final"])
        p1t = (
            self.model_forward(model, xxx, yyy, sample_mode, **model_kwargs)
            / temperature
        )
        p1t = logits_with_top_k_top_p_(
            p1t, top_k=top_k, top_p=top_p, type_data=type_data
        )
        p1t = torch.softmax(p1t, dim=1)
        return p1t

    def sample_cond_pt(
        self,
        p0,
        p1,
        t: torch.Tensor | float,
        kappa: KappaScheduler,
        input_tensor_type: str,
    ):
        p0_shape = p0.shape
        assert len(t.shape) == 1
        t = t.view(-1, *([1] * (len(p0_shape) - 1)))  # automaticaly broadcast
        pt = (1 - kappa(t)) * p0 + kappa(t) * p1
        if self.smoothing_factor > 0.0:
            pt = pt + torch.sqrt(self.smoothing_factor * (1 - t) * t)

        return sample_p(pt, input_tensor_type)

    def training_losses(
        self, model, x, sampling_eps=1e-3, **model_kwargs
    ) -> torch.Tensor:
        y = model_kwargs.pop("y")
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
        y = self.mm_modal.y_add_delta(y)
        if True:
            ttt = (1 - sampling_eps) * torch.rand(
                len(x), device=x.device
            ) + sampling_eps
            x0_target, x1_target = self.coupling.sample(x, mask_id=self.maskid_x)
            dirac_x0 = indices_to_diracp(
                x0_target.long(), self.logitsize_x, self.type_x
            )
            dirac_x1 = indices_to_diracp(
                x1_target.long(), self.logitsize_x, self.type_x
            )
            xt = self.sample_cond_pt(
                dirac_x0, dirac_x1, ttt, self.kappa, self.type_x
            )  # [B,T]

        if True:
            ttt_y = (1 - sampling_eps) * torch.rand(
                len(y), device=x.device
            ) + sampling_eps
            y0_target, y1_target = self.coupling.sample(y, mask_id=self.maskid_y)
            dirac_y0 = indices_to_diracp(
                y0_target.long(), self.logitsize_y, self.type_y
            )
            dirac_y1 = indices_to_diracp(
                y1_target.long(), self.logitsize_y, self.type_y
            )
            yt = self.sample_cond_pt(
                dirac_y0, dirac_y1, ttt_y, self.kappa, self.type_y
            )  # [B,T]

        logits_x, logits_y = model(xt, yt, **model_kwargs)

        loss_x = F.cross_entropy(
            logits_x, x1_target.long(), ignore_index=-1, reduction="none"
        )
        loss_y = F.cross_entropy(
            logits_y, y1_target.long(), ignore_index=-1, reduction="none"
        )
        target_mask_x = xt != x1_target
        if self.mask_ce:
            loss_x = torch.sum(loss_x * target_mask_x) / (
                torch.sum(target_mask_x) + 1e-7
            )
        else:
            loss_x = loss_x.mean()

        target_mask_y = yt != y1_target
        if self.mask_ce:
            loss_y = torch.sum(loss_y * target_mask_y) / (
                torch.sum(target_mask_y) + 1e-7
            )
        else:
            loss_y = loss_y.mean()

        loss = loss_x + loss_y
        acc_x = ((logits_x.argmax(dim=1) == x1_target) * target_mask_x).sum() / (
            torch.sum(target_mask_x) + 1e-7
        )
        acc_y = ((logits_y.argmax(dim=1) == y1_target) * target_mask_y).sum() / (
            torch.sum(target_mask_y) + 1e-7
        )
        ret_dict = {
            "loss": loss,
            "log/loss_x": loss_x.clone(),
            "log/loss_y": loss_y.clone(),
            "log/mask_ce": int(self.mask_ce),
            "log/acc_x": acc_x.clone(),
            "log/acc_y": acc_y.clone(),
            "logits_x": logits_x.clone(),
            "logits_y": logits_y.clone(),
            "x_corrupt": xt.clone(),
            "y_corrupt": yt.clone(),
        }
        return ret_dict


class SimpleSampler:
    def __init__(
        self,
        mask_token_id: int,
        input_tensor_type: str = "bt",
    ):
        super().__init__()
        self.input_tensor_type = input_tensor_type

    def u(
        self,
        t,
        xt,
        disint: DiscreteInterpolants,
        model: nn.Module,
        **model_kwargs,
    ):
        return disint.forward_u(t, xt, model, **model_kwargs)

    def construct_x0(
        self,
        sample_size: Tuple[int],
        device: torch.device,
        vocab_size: int,
        mask_id: int,
    ):
        x0 = torch.ones(sample_size, device=device, dtype=torch.long) * mask_id
        dirac_x0 = indices_to_diracp(x0, vocab_size, self.input_tensor_type)
        return x0, dirac_x0

    def sample(
        self,
        sample_size: Tuple[int],
        disint: DiscreteInterpolants,
        model: nn.Module,
        n_steps: int,
        t_min: float = 1e-4,
        chain_num: int = 7,  # hardcode here
        **model_kwargs,
    ):
        temperature = model_kwargs.pop("temperature", 1.0)
        return_chains = model_kwargs.pop("return_chains", 1)
        reindex_final = model_kwargs.pop("reindex_final", False)
        max_last = model_kwargs.pop("max_last", False)
        top_p = model_kwargs.pop("top_p", 0.0)
        top_k = model_kwargs.pop("top_k", 0)
        sample_mode = model_kwargs["sample_mode"]
        mask_back_for_oov = model_kwargs.pop("mask_back_for_oov", False)
        model_kwargs.pop("kappa", None)
        model_kwargs.pop("anneal_noise", None)
        print_rank_0(
            f"temperature: {temperature}, n_steps: {n_steps}, return_chains: {return_chains}, reindex_final: {reindex_final}, max_last: {max_last}, sample_mode: {sample_mode}, mask_back_for_oov: {mask_back_for_oov}"
        )
        t = t_min * torch.ones(sample_size[0], device=disint.device)
        default_h = 1 / n_steps
        if sample_mode in ["cls2img", "mask2img"]:
            type_data = disint.type_x
            vocab_size = disint.logitsize_x
            mask_id = disint.maskid_x
        elif sample_mode in ["img2cls", "img2mask"]:
            type_data = disint.type_y
            vocab_size = disint.logitsize_y
            mask_id = disint.maskid_y
        else:
            raise ValueError(f"Unknown mode {model_kwargs['sample_mode']}")

        xt, dirac_xt = self.construct_x0(
            sample_size, disint.device, vocab_size, mask_id
        )

        list_xt = [xt]
        max_chains = [xt]
        t = pad_like_x(t, dirac_xt)
        t_dummy = torch.zeros_like(t)

        for _ in tqdm(range(n_steps), desc="sampling", total=n_steps):
            h = default_h
            pt = dirac_xt + h * self.u(
                t_dummy,
                xt,
                disint,
                model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                type_data=type_data,
                **model_kwargs,
            )
            if return_chains == 2:
                max_chains.append(argmax_except_mask(pt.clone(), mask_id))
            xt = sample_p(pt, type_data)
            if mask_back_for_oov:
                pass
            dirac_xt = indices_to_diracp(xt, vocab_size, type_data)
            t += h
            list_xt.append(xt)

        if max_last:
            list_xt.append(argmax_except_mask(pt.clone(), mask_id))

        if True:  # minus delta y
            if sample_mode in ["cls2img", "mask2img"]:
                pass
            elif sample_mode in ["img2cls", "img2mask"]:
                list_xt = [disint.mm_modal.y_minus_delta(xt) for xt in list_xt]

        if True:  # reindex x
            if sample_mode in ["cls2img", "mask2img"]:
                list_xt = [
                    disint.mm_modal.out_of_range_x_reindex(xt, disint.mm_modal.vs_x - 1)
                    for xt in list_xt
                ]
            elif sample_mode in ["img2cls", "img2mask"]:
                list_xt = [
                    disint.mm_modal.out_of_range_y_reindex(xt, disint.mm_modal.vs_y - 1)
                    for xt in list_xt
                ]

        if return_chains == 1:
            list_xt = get_uniform_n_samples(list_xt, chain_num)
            list_xt = torch.stack(list_xt, dim=0)
            list_mask = torch.zeros_like(list_xt)
            list_mask[list_xt == mask_id] = 1
            return list_xt, list_mask
        elif return_chains == 2:
            list_maxchain = get_uniform_n_samples(max_chains, chain_num)
            list_maxchain = torch.stack(list_maxchain, dim=0)
            list_maxchain_mask = torch.zeros_like(list_maxchain)
            list_maxchain_mask[list_maxchain == mask_id] = 1
            return list_maxchain, list_maxchain_mask
        else:
            raise ValueError(f"return_chains={return_chains} not supported")


class MaskgitSampler:
    def __init__(self, mask_token_id: int, input_tensor_type: str = "bt") -> None:
        self.mask_token_id = mask_token_id
        self.data_type = input_tensor_type

    def logits_squeeze(self, x, data_type: str):
        if data_type == "bt":
            b, k, t = x.shape
            return x.view(b, t, k)
        elif data_type == "bcwh":
            b, k, c, h, w = x.shape
            return rearrange(x, "b k c h w -> b (c w h) k")
        elif data_type == "btwh":
            b, k, t, h, w = x.shape
            return rearrange(x, "b k t h w -> b (t h w) k")
        elif data_type == "bwh":
            b, k, h, w = x.shape
            return rearrange(x, "b k h w -> b (h w) k")
        else:
            raise ValueError(f"data_type={data_type} not supported")

    @torch.no_grad()
    def sample(
        self,
        sample_size,
        disint: DiscreteInterpolants,
        model: nn.Module,
        kappa: KappaScheduler,
        n_steps: int,
        init_code=None,
        r_temp=4.5,  # temperature for random perturb
        chain_num: int = 7,  # hardcode here
        **model_kwargs,
    ):

        temperature = model_kwargs.pop("temperature", 1.0)
        top_p = model_kwargs.pop("top_p", 0.0)
        top_k = model_kwargs.pop("top_k", 0)
        return_chains = model_kwargs.pop("return_chains", 1)
        max_last = model_kwargs.pop("max_last", False)
        maskgit_mode = model_kwargs.pop("maskgit_mode", "arccos")
        randomize = model_kwargs.pop("maskgit_randomize", "none")
        anneal_noise = model_kwargs.pop("anneal_noise", "none")
        sample_mode = model_kwargs["sample_mode"]
        # never used this variable,for backward compatibility

        if sample_mode in ["cls2img", "mask2img"]:
            type_data = disint.type_x
        elif sample_mode in ["img2cls", "img2mask"]:
            type_data = disint.type_y
        else:
            raise ValueError(f"Unknown mode {model_kwargs['sample_mode']}")
        try:
            cfg_scale = model_kwargs["cfg_scale"]
        except:
            cfg_scale = -1
        print_rank_0(
            f"n_steps={n_steps},kappa: {kappa},temperature: {temperature},chain_num={chain_num},return_chains={return_chains},r_temp={r_temp}, maskgit_mode={maskgit_mode}, randomize={randomize},cfg_scale={cfg_scale},init_code_is_None={init_code is None}"
        )
        token_num = math.prod(sample_size[1:])
        bs = sample_size[0]
        _device = disint.device

        l_codes = []  # Save the intermediate codes predicted
        l_mask = []  # Save the intermediate masks

        if init_code is not None:  # Start with a pre-define code
            code = init_code.long().to(_device)
        else:
            code = torch.full(
                sample_size,
                self.mask_token_id,
            ).to(_device)
        mask = torch.ones((bs, token_num)).to(_device)

        scheduler = adap_scheduler(n_steps, mode=maskgit_mode, token_num=token_num)
        code_list = []

        # Beginning of sampling, t = number of token to predict a step "indice"
        for indice, t in tqdm(
            enumerate(scheduler), total=len(scheduler), desc="maskgit sampling"
        ):
            if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                t = int(mask.sum().item())

            if mask.sum() == 0:  # Break if code is fully predicted
                break

            _prob = disint.forward_u_maskgit(
                xt=code,
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                type_data=type_data,
                **model_kwargs,
            )
            prob = self.logits_squeeze(_prob, type_data)
            distri = torch.distributions.Categorical(probs=prob)
            pred_code = distri.sample()

            conf = torch.gather(
                prob,
                2,
                pred_code.view(bs, token_num, 1),
            )  # [B,T,1]

            if (
                randomize == "linear"
            ):  # add gumbel noise decreasing over the sampling process
                ratio = indice / (len(scheduler) - 1)
                rand = r_temp * np.random.gumbel(size=(bs, token_num)) * (1 - ratio)
                conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(_device)
                conf = rearrange(conf, "b t -> b t 1")
            elif randomize == "warm_up":  # chose random sample for the 2 first steps
                conf = torch.rand_like(conf) if indice < 2 else conf
            elif randomize == "random":  # chose random prediction at each step
                conf = torch.rand_like(conf)
            elif randomize == "none":
                pass
            else:
                raise ValueError(f"randomize={randomize} not supported")

            # do not predict on already predicted tokens
            conf[~mask.bool()] = -math.inf

            # chose the predicted token with the highest confidence
            tresh_conf, indice_mask = torch.topk(conf.view(bs, -1), k=t, dim=-1)
            tresh_conf = tresh_conf[:, -1]

            # replace the chosen tokens
            conf = (conf >= tresh_conf.unsqueeze(-1).unsqueeze(-1)).view(*sample_size)
            f_mask = (
                mask.view(*sample_size).float() * conf.view(*sample_size).float()
            ).bool()
            code[f_mask] = pred_code.view(*sample_size)[f_mask]

            # update the mask
            for i_mask, ind_mask in enumerate(indice_mask):
                mask[i_mask, ind_mask] = 0
            l_codes.append(pred_code.view(*sample_size).clone())
            l_mask.append(mask.view(*sample_size).clone())
            code_list.append(code.clone())

        if True:  # minus delta y
            if sample_mode in ["cls2img", "mask2img"]:
                pass
            elif sample_mode in ["img2cls", "img2mask"]:
                code_list = [disint.mm_modal.y_minus_delta(xt) for xt in code_list]

        if True:  # reindex x
            if sample_mode in ["cls2img", "mask2img"]:
                code_list = [
                    disint.mm_modal.out_of_range_x_reindex(xt, disint.mm_modal.vs_x - 1)
                    for xt in code_list
                ]
            elif sample_mode in ["img2cls", "img2mask"]:
                code_list = [
                    disint.mm_modal.out_of_range_y_reindex(xt, disint.mm_modal.vs_y - 1)
                    for xt in code_list
                ]

        code_list = get_uniform_n_samples(code_list, chain_num)
        code_list = torch.stack(code_list, dim=0).to(disint.device)
        mask_list = torch.zeros_like(code_list, dtype=torch.uint8)
        mask_list[code_list == self.mask_token_id] = 1
        return code_list, mask_list


if __name__ == "__main__":
    pass
