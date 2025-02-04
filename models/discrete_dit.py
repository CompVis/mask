# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange
from typing import Union, Any

try:
    import flash_attn

    if hasattr(flash_attn, "__version__") and int(flash_attn.__version__[0]) == 2:
        from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention
    else:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention
except Exception as e:
    print(f"flash_attn import failed: {e}")

try:
    import os

    TC_OPEN = int(os.environ.get("TC_OPEN"))
except:
    TC_OPEN = True
print(f"Torch Compile Open: {TC_OPEN}")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()
        use_cfg_embedding = True
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes

    def token_drop(self, labels, cond_drop_prob, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < cond_drop_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, cond_drop_prob, force_drop_ids=None):
        use_dropout = cond_drop_prob > 0
        if use_dropout or (force_drop_ids is not None):
            labels = self.token_drop(labels, cond_drop_prob, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                              Flash attention Layer.                           #
#################################################################################


class FlashSelfMHAModified(nn.Module):
    """
    self-attention with flashattention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        device=None,
        dtype=None,
        norm_layer=nn.LayerNorm,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, **factory_kwargs)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.inner_attn = FlashSelfAttention(attention_dropout=attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x,
    ):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        """
        b, s, d = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, h, d]
        q, k, v = qkv.unbind(dim=2)  # [b, s, h, d]
        q = self.q_norm(q).half()  # [b, s, h, d]
        k = self.k_norm(k).half()

        qkv = torch.stack([q, k, v], dim=2)  # [b, s, 3, h, d]
        context = self.inner_attn(qkv)
        out = self.out_proj(context.view(b, s, d))
        out = self.proj_drop(out)

        return out


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_checkpoint=False,
        use_flash_attn=False,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.attn = FlashSelfMHAModified(
                hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True
            )
        else:
            self.attn = Attention(
                hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
            )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x, c):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, c)
        else:
            return self._forward(x, c)

    def _forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    #

    def __init__(
        self,
        vocab_size,
        img_dim=32,
        patch_size=2,
        embed_dim=1152,
        in_chans=1,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=-1,
        learn_sigma=False,
        use_pe=-1,
        time_cond=True,
        use_flash_attn=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.vocab_size = vocab_size
        self.out_channels = in_chans * self.vocab_size

        self.patch_size = patch_size
        self.num_heads = num_heads
        num_classes = -1 if num_classes is None else num_classes
        self.num_classes = num_classes
        self.time_cond = time_cond

        print("*" * 30)
        print(
            f"vocab_size: {vocab_size}, time_cond: {time_cond},patch_size: {patch_size}, use_flash_attn: {use_flash_attn}"
        )
        self.x_pre_embed = nn.Embedding(vocab_size, embed_dim)
        

        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size=img_dim,
            patch_size=patch_size,
            in_chans=embed_dim * self.in_chans,
            embed_dim=embed_dim,
            bias=True,
            # img_dim, patch_size, embed_dim, bias=True
        )
        print(
            "x_embedder parameters:",
            sum(p.numel() for p in self.patch_embed.parameters()),
        )
        # following https://github.com/ML-GSAI/RADD/blob/92a23fba50279ee875e5dcfd08894326982066a3/model/transformer.py#L151C18-L151C29
        # and https://github.com/andrew-cr/discrete_flow_models/blob/800395d172be6b950d2ab87bcf154d752bd2cf76/flow_model.py#L226
        self.t_embedder = TimestepEmbedder(embed_dim)
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, embed_dim)
        num_patches = self.patch_embed.num_patches
        print(f"num_patches: {num_patches}", "use_checkpoint:", use_checkpoint)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
             torch.zeros(1, num_patches, embed_dim), requires_grad=False
         )#
        #self.register_buffer("pos_embed", torch.zeros(1, num_patches, embed_dim))
        

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_checkpoint=use_checkpoint,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(embed_dim, patch_size, self.out_channels)
        print(
            "final_layer parameters:",
            sum(p.numel() for p in self.final_layer.parameters()),
        )

        self.initialize_weights()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def param_num_embed_table(self):
        return sum(p.numel() for p in self.x_pre_embed.parameters())

    @property
    def param_num_pre_logits(self):
        return sum(p.numel() for p in self.final_layer.parameters())

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.x_pre_embed.weight, std=0.02)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        if self.num_classes > 0:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _forward_with_cfg(
        self,
        x: Union[torch.LongTensor, torch.IntTensor],
        t: Union[torch.FloatTensor, torch.DoubleTensor],
        cfg_scale: float = None,
        **kwargs,
    ):
        assert cfg_scale is not None
        cond_logits = self._forward(x.clone(), t.clone(), cond_drop_prob=0.0, **kwargs)
        uncond_logits = self._forward(
            x.clone(), t.clone(), cond_drop_prob=1.0, **kwargs
        )
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        return logits

    @torch.compile(mode="max-autotune", fullgraph=True)
    def _forward_with_cfg_compile_wrapper(self, *args, **kwargs):
        return self._forward_with_cfg(*args, **kwargs)

    def forward_with_cfg(self, *args, **kwargs):
        if TC_OPEN:
            return self._forward_with_cfg_compile_wrapper(*args, **kwargs)
        else:
            return self._forward_with_cfg(*args, **kwargs)

    def forward_without_cfg(
        self,
        x: Union[torch.LongTensor, torch.IntTensor],
        t: Union[torch.FloatTensor, torch.DoubleTensor],
        **kwargs,
    ):
        if TC_OPEN:
            return self._forward_compile_wrapper(x, t, cond_drop_prob=0.0, **kwargs)
        else:
            return self._forward(x, t, cond_drop_prob=0.0, **kwargs)

    def forward(self, *args, **kwargs):
        if TC_OPEN:
            return self._forward_compile_wrapper(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    @torch.compile(mode="max-autotune", fullgraph=True)
    def _forward_compile_wrapper(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x: Union[torch.LongTensor, torch.IntTensor],
        t: Union[torch.FloatTensor, torch.DoubleTensor],
        y: Any = None,
        cond_drop_prob=0.1,
        mp_type=None,
    ):
        """
        Forward pass of DiT.
        x: (N, H, W) is the indices of the tokens
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        assert x.dtype in [torch.int64, torch.int32]
        x_shape = x.shape

        x = self.x_pre_embed(x)  # [B,H,W,embed_dim]
        x = x.to(dtype=mp_type)

        if len(x_shape) == 4:
            x = rearrange(x, "b c w h k -> b (c k) w h")
        elif len(x_shape) == 3:
            x = rearrange(x, "b h w k -> b k h w")
        else:
            raise ValueError(f"Invalid input shape: {x_shape}")

        x = (
            self.patch_embed(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        if not self.time_cond:
            t = torch.zeros(len(x), device=x.device, dtype=t.dtype)
        t = self.t_embedder(t).to(dtype=mp_type)  # (N, D)
        if self.num_classes > 0:
            y = self.y_embedder(y, self.training, cond_drop_prob)  # (N, D)
            c = t + y  # (N, D)
        else:
            # assert y is None # y should be None
            c = t
        x = x.to(dtype=mp_type)
        c = c.to(dtype=mp_type)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if len(x_shape) == 4:
            x = rearrange(x, "b (c k) h w -> b k c h w", k=self.vocab_size)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, embed_dim=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, embed_dim=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, embed_dim=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, embed_dim=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, embed_dim=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, embed_dim=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, embed_dim=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, embed_dim=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, embed_dim=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, embed_dim=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, embed_dim=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, embed_dim=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}


if __name__ == "__main__":
    # Test DiT forward pass:
    bs = 2
    vocab_size = 4096
    img_dim = 32
    in_chans = 1
    model = DiT_B_2(
        vocab_size=vocab_size,
        img_dim=img_dim,
        in_chans=in_chans,
        num_classes=-1,
        learn_sigma=False,
        use_flash_attn=True,
    )
    print("params:", sum(p.numel() for p in model.parameters()))
    x = torch.randint(0, vocab_size, (bs, img_dim, img_dim))
    t = torch.randint(0, vocab_size, (bs,))
    y = None
    out = model(x, t, y)
    print(out.shape)
