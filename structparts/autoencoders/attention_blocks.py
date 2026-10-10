# Modified for the StructParts inference-only release: removed unused paths and dependencies.
# ==============================================================================
# Original work Copyright (c) 2025 Tencent.
# Modified work Copyright (c) 2025 UltraShape Team.
# 
# Modified by UltraShape on 2025.12.25
# ==============================================================================
# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.
# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.
# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Optional, Union, List
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from .attention_processors import CrossAttentionProcessor
scaled_dot_product_attention = nn.functional.scaled_dot_product_attention

class FourierEmbedder(nn.Module):

    def __init__(self, num_freqs: int=6, logspace: bool=True, input_dim: int=3, include_input: bool=True, include_pi: bool=True) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

class DropPath(nn.Module):

    def __init__(self, drop_prob: float=0.0, scale_by_keep: bool=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return x

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'

class MLP(nn.Module):

    def __init__(self, *, width: int, expand_ratio: int=4, output_width: int=None, drop_path_rate: float=0.0):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))

class QKVMultiheadCrossAttention(nn.Module):

    def __init__(self, *, heads: int, n_data: Optional[int]=None, width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-06) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-06) if qk_norm else nn.Identity()
        self.attn_processor = CrossAttentionProcessor()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = self.attn_processor(self, q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out

class MultiheadCrossAttention(nn.Module):

    def __init__(self, *, width: int, heads: int, qkv_bias: bool=True, n_data: Optional[int]=None, data_width: Optional[int]=None, norm_layer=nn.LayerNorm, qk_norm: bool=False, kv_cache: bool=False):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(heads=heads, n_data=n_data, width=width, norm_layer=norm_layer, qk_norm=qk_norm)
        self.kv_cache = kv_cache
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)
        if self.kv_cache:
            if self.data is None:
                self.data = self.c_kv(data)
                logger.info('Save kv cache,this should be called only once for one mesh')
            data = self.data
        else:
            data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x

class ResidualCrossAttentionBlock(nn.Module):

    def __init__(self, *, n_data: Optional[int]=None, width: int, heads: int, mlp_expand_ratio: int=4, data_width: Optional[int]=None, qkv_bias: bool=True, norm_layer=nn.LayerNorm, qk_norm: bool=False):
        super().__init__()
        if data_width is None:
            data_width = width
        self.attn = MultiheadCrossAttention(n_data=n_data, width=width, heads=heads, data_width=data_width, qkv_bias=qkv_bias, norm_layer=norm_layer, qk_norm=qk_norm)
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-06)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-06)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-06)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x

class QKVMultiheadAttention(nn.Module):

    def __init__(self, *, heads: int, n_ctx: int, width=None, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-06) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-06) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out

class MultiheadAttention(nn.Module):

    def __init__(self, *, n_ctx: int, width: int, heads: int, qkv_bias: bool, norm_layer=nn.LayerNorm, qk_norm: bool=False, drop_path_rate: float=0.0):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, width=width, norm_layer=norm_layer, qk_norm=qk_norm)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x

class ResidualAttentionBlock(nn.Module):

    def __init__(self, *, n_ctx: int, width: int, heads: int, qkv_bias: bool=True, norm_layer=nn.LayerNorm, qk_norm: bool=False, drop_path_rate: float=0.0):
        super().__init__()
        self.attn = MultiheadAttention(n_ctx=n_ctx, width=width, heads=heads, qkv_bias=qkv_bias, norm_layer=norm_layer, qk_norm=qk_norm, drop_path_rate=drop_path_rate)
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-06)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-06)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):

    def __init__(self, *, n_ctx: int, width: int, layers: int, heads: int, qkv_bias: bool=True, norm_layer=nn.LayerNorm, qk_norm: bool=False, drop_path_rate: float=0.0):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(n_ctx=n_ctx, width=width, heads=heads, qkv_bias=qkv_bias, norm_layer=norm_layer, qk_norm=qk_norm, drop_path_rate=drop_path_rate) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

class CrossAttentionDecoder(nn.Module):

    def __init__(self, *, num_latents: int, out_channels: int, fourier_embedder: FourierEmbedder, width: int, heads: int, mlp_expand_ratio: int=4, downsample_ratio: int=1, enable_ln_post: bool=True, qkv_bias: bool=True, qk_norm: bool=False, label_type: str='binary', use_varlen: bool=False):
        super().__init__()
        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.use_varlen = use_varlen
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        if self.downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        if self.enable_ln_post == False:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(n_data=num_latents, width=width, mlp_expand_ratio=mlp_expand_ratio, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)
        self.label_type = label_type
        self.count = 0

    def set_cross_attention_processor(self, processor):
        self.cross_attn_decoder.attn.attention.attn_processor = processor

    def set_default_cross_attention_processor(self):
        self.cross_attn_decoder.attn.attention.attn_processor = CrossAttentionProcessor

    def forward(self, queries=None, query_embeddings=None, latents=None, cu_seqlens_k=None, max_seqlen_k=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        self.count += query_embeddings.shape[1]
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        occ = self.output_proj(x)
        return occ
