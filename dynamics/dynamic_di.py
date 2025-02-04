# from https://github.com/valeoai/Maskgit-pytorch/blob/main/Trainer/vit.py
# another reference can be from https://github.com/bytedance/1d-tokenizer/blob/a31ad314d29818af7d3ebd13fe0b845adc894728/modeling/maskgit.py#L130

import torch
from torch import nn
from torch.nn import functional as F
import os
from typing import Tuple, List
from einops import rearrange
import numpy as np
from einops import rearrange
import torch.distributed as dist
import math, random
from tqdm import tqdm


def extend_as(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *(1 for _ in range(y.ndim - x.ndim)))


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
    return tqdm(sche.int(), leave=leave, desc="adap_scheduler in maskgit")


@torch.compile()
def logits_with_top_k_top_p_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    type_data: str = None,
) -> torch.Tensor:  # return idx, shaped (B, l)
    if type_data == "bwh":
        b, k, w, h = logits_BlV.shape
        logits_BlV = rearrange(logits_BlV, "b k w h -> b (w h) k")
    elif type_data in ["bcwh", "btwh"]:
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
    elif type_data in ["bcwh", "btwh"]:
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


def pad_like_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *(1 for _ in range(y.ndim - x.ndim)))


def indices_to_diracp(x, vocab_size: int, type_data: str = "bt"):
    if type_data == "bt":
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b t k -> b k t")
    elif type_data == "bwh":
        b, w, h = x.shape
        x = rearrange(x, "b w h -> b (w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (w h) k -> b k w h", w=w, h=h)
    elif type_data == "bcwh":
        b, c, w, h = x.shape
        x = rearrange(x, "b c w h -> b (c w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (c w h) k -> b k c w h", c=c, w=w, h=h)
    elif type_data == "btwh":
        b, t, w, h = x.shape
        x = rearrange(x, "b t w h -> b (t w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (t w h) k -> b k t w h", t=t, w=w, h=h)
    else:
        raise ValueError(f"input_tensor_type {type_data} not supported")


def sample_p(pt, type_data: str, mask_token_id=None):
    if type_data == "bt":
        b, k, t = pt.shape
        pt = rearrange(pt, "b k t -> (b t) k")
        if mask_token_id is not None:
            pt[:, mask_token_id] = 0
        xt = torch.multinomial(pt, 1)

        return xt.reshape(b, t)
    elif type_data == "bwh":
        b, k, h, w = pt.shape
        pt = rearrange(pt, "b k h w -> (b h w) k")
        if mask_token_id is not None:
            pt[:, mask_token_id] = 0
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, h, w)
    elif type_data in ["bcwh", "btwh"]:
        b, k, c, h, w = pt.shape
        pt = rearrange(pt, "b  k c h w -> (b c h w) k")
        if mask_token_id is not None:
            pt[:, mask_token_id] = 0
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, c, h, w)
    else:
        raise ValueError(f"input_tensor_type {type_data} not supported")


def argmax_p(pt, xt, mask_token_id):
    """
    pt: (B, K, T)
    xt: (B, T)
    """

    pt[:, mask_token_id] = 0  # make mask_token_id never be the max
    max_xt = pt.argmax(dim=1)
    is_mask = xt == mask_token_id
    xt[is_mask] = max_xt[is_mask]
    _ratio = (is_mask.sum() / is_mask.numel()).item()
    print(f"finish argmax_p, max_last ratio: {_ratio}")
    return xt


class KappaScheduler:
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return str(self.__class__.__name__)

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError

    def sigmoid_weight(
        self, t: float | torch.Tensor, shift: str = "none"
    ) -> float | torch.Tensor:
        if shift == "none":
            return torch.ones_like(t)
        else:
            shift = shift.replace("shift", "")
            shift = float(shift)
        snr = self._snr(t)
        _lambda = snr - shift
        _sigmoid = torch.sigmoid(_lambda)
        return _sigmoid


class Coupling:
    def __init__(self) -> None:
        pass

    def sample(self, x1):
        raise NotImplementedError


class Ucoupling(Coupling):
    def __init__(self, mask_token_id) -> None:
        self.mask_token_id = mask_token_id

    def sample(self, x1):
        return torch.ones_like(x1) * self.mask_token_id, x1


class Ccoupling(Coupling):
    def __init__(self, mask_token_id: int, msk_prop: float = 0.8) -> None:
        if msk_prop is None:
            print("Ccoupling, msk_prop is None, using coupling by random prob")
        elif msk_prop > 0:
            print("Ccoupling, msk_prop: ", msk_prop, "data_prob", 1 - msk_prop)
        else:
            raise ValueError("msk_prop must be non-negative")
        self.mask_token_id = mask_token_id
        self.msk_prob = msk_prop

    def sample(self, x1):
        if self.msk_prob is None:
            _msk_prob = torch.rand_like(x1.float())
        else:
            _msk_prob = self.msk_prob
        _mask20 = torch.rand_like(x1.float()) > _msk_prob
        _mask_id = torch.ones_like(x1) * self.mask_token_id
        x0 = x1 * _mask20 + _mask_id * (~_mask20)
        return x0, x1


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __str__(self) -> str:
        return f"cubic_a{self.a}_b{self.b}"

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

    def __str__(self) -> str:
        return "linear"

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1.0


class QuadraticScheduler(KappaScheduler):
    def __init__(
        self,
    ) -> None:
        pass

    def __str__(self) -> str:
        return "quadratic"

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t**2

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 2 * t


class RootScheduler(KappaScheduler):
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "root"

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t**0.5

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 0.5 * t ** (-0.5)


class CosineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = torch.pi * 0.5

    def __str__(self) -> str:
        return "cosine"

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1 - torch.cos(self.coeff * t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.sin(self.coeff * t) * self.coeff


class SineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = torch.pi * 0.5

    def __str__(self) -> str:
        return "sine"

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.sin(self.coeff * t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.cos(self.coeff * t) * self.coeff


class ArcCosineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = 2 / torch.pi
        self.eps = 1e-6

    def __str__(self) -> str:
        return "arc_cosine"

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1 - self.coeff * torch.acos(t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff / torch.sqrt(1 - t**2 + self.eps)


class ArcSinScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = 2 / torch.pi
        self.eps = 1e-6

    def __str__(self) -> str:
        return "arc_sin"

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
        vocab_size: int,
        coupling: Coupling,
        kappa: KappaScheduler,
        device: torch.device,
        input_tensor_type: str = "bt",
        smoothing_factor: float = 0.0,
        mask_ce=False,
        elbo=False,
        sigmoid_shift: str = "none",
    ) -> None:
        self.vocab_size = vocab_size
        self.coupling = coupling
        self.kappa = kappa
        self.device = device
        self.type_data = input_tensor_type
        self.smoothing_factor = smoothing_factor
        self.mask_ce = mask_ce
        self.elbo = elbo
        self.sigmoid_shift = sigmoid_shift
        print_rank_0(
            f"smoothing_factor: {smoothing_factor}, sigmoid_shift: {sigmoid_shift},mask_ce: {mask_ce},elbo: {elbo}"
        )

    def forward_u(
        self,
        t: float | torch.Tensor,
        xt,
        model,
        kappa: KappaScheduler,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        type_data: str = None,
        **model_kwargs,
    ):
        dirac_xt = indices_to_diracp(xt, self.vocab_size, self.type_data)
        p1t = model(xt, t.flatten(), **model_kwargs) / temperature
        p1t = logits_with_top_k_top_p_(
            p1t, top_k=top_k, top_p=top_p, type_data=type_data
        )
        p1t = torch.softmax(p1t, dim=1)
        kappa_coeff = kappa.derivative(t) / (1 - kappa(t))
        return kappa_coeff * (p1t - dirac_xt)

    def forward_u_maskgit(
        self,
        t: float | torch.Tensor,
        xt,
        model: nn.Module,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        type_data: str = None,
        **model_kwargs,
    ):
        p1t = model(xt, t.flatten(), **model_kwargs) / temperature
        p1t = logits_with_top_k_top_p_(
            p1t, top_k=top_k, top_p=top_p, type_data=type_data
        )
        p1t = torch.softmax(p1t, dim=1)
        return p1t

    def corrupt_data(
        self,
        p0,
        p1,
        t: torch.Tensor | float,
        kappa: KappaScheduler,
        type_data: str,
    ):
        p0_shape = p0.shape
        assert len(t.shape) == 1
        t = t.view(-1, *([1] * (len(p0_shape) - 1)))  # automaticaly broadcast
        pt = (1 - kappa(t)) * p0 + kappa(t) * p1

        if self.smoothing_factor > 0.0:
            pt = pt + torch.sqrt(self.smoothing_factor * (1 - t) * t)

        return sample_p(pt, type_data)

    # @torch.compile(fullgraph=True, mode="max-autotune")
    def loss_fn(self, xt, t, logits_x, x1_target):
        target_mask = xt == self.coupling.mask_token_id
        loss = F.cross_entropy(
            logits_x,
            x1_target.long(),
            reduction="none",
        )
        #loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

        if self.mask_ce:
            if self.elbo:
                dsigma = self.kappa.derivative(t) / (1 - self.kappa(t))
                _extra_weight = self.kappa.sigmoid_weight(t, self.sigmoid_shift)
                loss_wsigma = loss * extend_as(dsigma, loss)
                loss_wsigma = loss_wsigma * extend_as(_extra_weight, loss)
                loss = torch.sum(loss_wsigma * target_mask) / torch.sum(
                    target_mask
                ).clamp(min=1.0)
            else:
                loss = torch.sum(loss * target_mask) / torch.sum(target_mask).clamp(
                    min=1.0
                )
        else:
            assert self.elbo is False
            loss = loss.mean()
        return loss, target_mask

    def training_losses(
        self, model, x, sampling_eps=1e-3, **model_kwargs
    ) -> torch.Tensor:
        t = (1 - sampling_eps) * torch.rand(len(x), device=x.device) + sampling_eps
        x0, x1_target = self.coupling.sample(x)
        dirac_x0 = indices_to_diracp(x0.long(), self.vocab_size, self.type_data)
        dirac_x1 = indices_to_diracp(
            x1_target.long(), self.vocab_size, self.type_data
        )  # x1, real data; x0, mask token
        xt = self.corrupt_data(
            dirac_x0, dirac_x1, t, self.kappa, self.type_data
        )  # [B,T]
        logits_x = model(xt, t, **model_kwargs)

        loss, target_mask = self.loss_fn(xt, t, logits_x, x1_target)

        ret_dict = {
            "loss": loss,
            "logits": logits_x.clone(),
            "logits_x": logits_x.clone(),
            "x1_target": x1_target.clone(),
            "target_mask": target_mask.clone(),
            "x_corrupt": xt.clone(),
        }
        return ret_dict


class SimpleSampler:
    def __init__(
        self,
        mask_token_id: int,
        input_tensor_type: str = "bt",
    ) -> None:
        self.h = self.constant_h
        self.mask_token_id = mask_token_id
        self.data_type = input_tensor_type

    def u(
        self,
        t: float | torch.Tensor,
        xt,
        disint: DiscreteInterpolants,
        model: nn.Module,
        kappa: KappaScheduler,
        top_p: float = None,
        top_k: int = None,
        **model_kwargs,
    ):
        return disint.forward_u(
            t,
            xt,
            model,
            kappa=kappa,
            top_p=top_p,
            top_k=top_k,
            type_data=disint.type_data,
            **model_kwargs,
        )

    def constant_h(
        self,
        h: float | torch.Tensor,
        t: float | torch.Tensor,
        disint: DiscreteInterpolants,
    ) -> float | torch.Tensor:
        return h

    def construct_x0(
        self, sample_size: Tuple[int], device: torch.device, vocab_size: int
    ):
        x0 = (
            torch.ones(sample_size, device=device, dtype=torch.long)
            * self.mask_token_id
        )
        dirac_x0 = indices_to_diracp(x0, vocab_size, self.data_type)
        return x0, dirac_x0

    def sample(
        self,
        sample_size: Tuple[int],
        disint: DiscreteInterpolants,
        model: nn.Module,
        n_steps: int,
        kappa: KappaScheduler,
        chain_num=7,  # hardcode here
        t_min: float = 1e-4,
        **model_kwargs,
    ):
        temperature = model_kwargs.pop("temperature", 1.0)
        top_p = model_kwargs.pop("top_p", 0.0)
        top_k = model_kwargs.pop("top_k", 0)
        return_chains = model_kwargs.pop("return_chains", 1)
        max_last = model_kwargs.pop("max_last", False)
        anneal_noise = model_kwargs.pop("anneal_noise", "none")
        try:
            cfg_scale = model_kwargs["cfg_scale"]
        except:
            cfg_scale = -1
        device = disint.device
        data_type = disint.type_data
        vocab_size = disint.vocab_size

        print_rank_0(
            f"n_steps={n_steps},kappa: {kappa},temperature: {temperature}, top_p: {top_p}, top_k: {top_k},max_last: {max_last},chain_num={chain_num},t_min={t_min},cfg_scale={cfg_scale},anneal_noise={anneal_noise}"
        )
        t = t_min * torch.ones(sample_size[0], device=device)
        default_h = 1 / n_steps
        xt, dirac_xt = self.construct_x0(sample_size, device, vocab_size)
        list_xt = [xt.clone()]
        if return_chains == 2:
            list_maxchain = [xt.clone()]

        t = pad_like_x(t, dirac_xt)
        tqdm_bar = tqdm(range(n_steps))
        while t.max() <= 1 - default_h:
            h = self.h(default_h, t, disint)
            _u = self.u(
                t,
                xt,
                disint,
                model,
                kappa=kappa,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **model_kwargs,
            )
            pt = dirac_xt + h * _u  # [B,K,T]
            if return_chains == 2:
                list_maxchain.append(pt.clone().argmax(dim=1))
            if disint.mask_ce:
                will_unmask = xt == self.mask_token_id
            else:
                will_unmask = torch.ones_like(xt, device=device, dtype=torch.bool)

            if anneal_noise != "none":
                noise = 1
                if anneal_noise == "linear":
                    will_mask = torch.rand(xt.shape, device=device) < noise * (
                        1 - t.max() - default_h
                    )  # (B, D)
                    will_mask = will_mask * (xt != self.mask_token_id)  # (B, D)
                    xt[will_mask] = self.mask_token_id
                elif anneal_noise == "warmup":
                    if t.max() < 0.6:
                        will_mask = torch.rand(xt.shape, device=device) < noise * (
                            1 - t.max() - default_h
                        )  # (B, D)
                        will_mask = will_mask * (xt != self.mask_token_id)  # (B, D)
                        xt[will_mask] = self.mask_token_id
                if anneal_noise == "campbell":
                    will_mask = torch.rand(xt.shape, device=device) < 4 * h  # (B, D)
                    will_mask = will_mask * (xt != self.mask_token_id)  # (B, D)
                    xt[will_mask] = self.mask_token_id
                elif anneal_noise == "none":
                    will_mask = torch.zeros_like(xt, device=device, dtype=torch.bool)
                else:
                    raise ValueError(f"anneal_noise={anneal_noise} not supported")
            else:
                will_mask = torch.zeros_like(xt, device=device, dtype=torch.bool)
            _xt = sample_p(pt, data_type)
            xt[will_unmask] = _xt[will_unmask]  # unmask first

            t += h
            if t.max() < 1 - default_h:
                xt[will_mask] = self.mask_token_id  # mask later
            if max_last and t.max() >= 1 - default_h:
                xt = argmax_p(pt=pt.clone(), xt=xt, mask_token_id=self.mask_token_id)

            list_xt.append(xt.clone())
            dirac_xt = indices_to_diracp(xt, vocab_size, data_type)
            tqdm_bar.update(1)

        list_xt = get_uniform_n_samples(list_xt, chain_num)
        list_xt = torch.stack(list_xt, dim=0).to(device)

        list_xt_mask = torch.zeros_like(list_xt, dtype=torch.uint8)
        list_xt_mask[list_xt == self.mask_token_id] = 1

        if return_chains == 1:
            return list_xt, list_xt_mask
        elif return_chains == 2:
            list_maxchain = get_uniform_n_samples(list_maxchain, chain_num)
            list_maxchain = torch.stack(list_maxchain, dim=0).to(device)
            list_maxchain_mask = torch.zeros_like(list_maxchain, dtype=torch.uint8)
            list_maxchain_mask[list_maxchain == self.mask_token_id] = 1
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
        # never used this variable,for backward compatibility

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
                t=torch.zeros(bs, device=_device),  # t is not used here
                xt=code,
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                type_data=disint.type_data,
                **model_kwargs,
            )
            prob = self.logits_squeeze(_prob, disint.type_data)
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

        code_list = get_uniform_n_samples(code_list, chain_num)
        code_list = torch.stack(code_list, dim=0).to(disint.device)
        mask_list = torch.zeros_like(code_list, dtype=torch.uint8)
        mask_list[code_list == self.mask_token_id] = 1
        return code_list, mask_list


if __name__ == "__main__":
    sche = adap_scheduler(step=10, token_num=1024, mode="linear", leave=False)
    for i in sche:
        print(i)
