# Modified for the StructParts inference-only release: removed unused paths and dependencies.
# ==============================================================================
# Original work Copyright (c) 2025 Tencent.
# Modified work Copyright (c) 2025 UltraShape Team.
# 
# Modified by UltraShape on 2025.12.25
# Part DIT: renamed from RefineDiT in dit_mask.py for part-based training.
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

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .moe_layers import MoEBlock
from .attention import flash_attn_varlen_func

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FourierMLPBboxEmbedder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, num_freqs: int=8, hidden: int=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_freqs = num_freqs
        feat_dim = in_dim * 2 * num_freqs
        hidden = hidden if hidden is not None else out_dim
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        freqs = torch.arange(num_freqs, dtype=torch.float32) / max(num_freqs, 1)
        freqs = 1.0 / 10000 ** freqs
        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_dim)
        scaled = x_flat.unsqueeze(-1) * self.freqs.to(x_flat.device).view(1, 1, -1)
        sincos = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        feat = sincos.reshape(x_flat.shape[0], -1)
        out = self.proj(feat)
        return out.reshape(*orig_shape, self.out_dim)

class AbsolutePositionEmbedder(nn.Module):

    def __init__(self, channels: int, in_channels: int=3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / max(self.freq_dim, 1)
        freqs = 1.0 / 10000 ** freqs
        self.register_buffer('freqs', freqs, persistent=False)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.outer(x, self.freqs.to(x.device))
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, D = x.shape
        assert D == self.in_channels, f'Got D={D}, expected {self.in_channels}'
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            pad = torch.zeros(N, self.channels - embed.shape[1], device=embed.device, dtype=embed.dtype)
            embed = torch.cat([embed, pad], dim=-1)
        return embed

class Timesteps(nn.Module):

    def __init__(self, num_channels: int, downscale_freq_shift: float=0.0, scale: int=1, max_period: int=10000):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1, 'Timesteps should be a 1d-array'
        embedding_dim = self.num_channels
        half_dim = embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256, cond_proj_dim=None, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(nn.Linear(hidden_size, frequency_embedding_size, bias=True), nn.GELU(), nn.Linear(frequency_embedding_size, out_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, frequency_embedding_size, bias=False)
        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):
        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)
        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t

class MLP(nn.Module):

    def __init__(self, *, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class CrossAttention(nn.Module):

    def __init__(self, qdim, kdim, num_heads, qkv_bias=True, qk_norm=False, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, 'self.qdim must be divisible by num_heads'
        self.head_dim = self.qdim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, 'Only support head_dim <= 128 and divisible by 8'
        self.scale = self.head_dim ** (-0.5)
        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-06) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-06) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(qdim, qdim, bias=True)

    def forward(self, x, y):
        b, s1, c = x.shape
        y_mask = (y != -1).any(dim=-1)
        has_padding = not y_mask.all()
        _, s2, c = y.shape
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)
        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)
        q = q.view(b, s1, self.num_heads, self.head_dim)
        k = k.view(b, s2, self.num_heads, self.head_dim)
        v = v.view(b, s2, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if has_padding:
            seqlens_k = y_mask.sum(dim=1).int()
            q_flat = q.reshape(-1, self.num_heads, self.head_dim)
            valid_indices = []
            cu_seqlens_k = [0]
            for i in range(b):
                valid_len = seqlens_k[i].item()
                batch_indices = torch.arange(valid_len, device=y.device) + i * s2
                valid_indices.append(batch_indices)
                cu_seqlens_k.append(cu_seqlens_k[-1] + valid_len)
            valid_indices = torch.cat(valid_indices)
            k_flat = k.reshape(b * s2, self.num_heads, self.head_dim)[valid_indices]
            v_flat = v.reshape(b * s2, self.num_heads, self.head_dim)[valid_indices]
            cu_seqlens_q = torch.arange(0, (b + 1) * s1, s1, dtype=torch.int32, device=x.device)
            cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=x.device)
            q_flat = q_flat.to(torch.bfloat16)
            k_flat = k_flat.to(torch.bfloat16)
            v_flat = v_flat.to(torch.bfloat16)
            context = flash_attn_varlen_func(q_flat, k_flat, v_flat, cu_seqlens_q, cu_seqlens_k, s1, seqlens_k.max().item(), dropout_p=0.0, softmax_scale=None, causal=False)
            context = context.reshape(b, s1, -1)
        else:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.num_heads), (q, k, v))
                attn_mask = None
                context = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2).reshape(b, s1, -1)
        out = self.out_proj(context)
        return out

class LocalGlobalProcessor:

    def __init__(self, use_global=False, apply_rope: bool=True):
        self.use_global = use_global
        self.apply_rope = apply_rope

    def __call__(self, attn: 'Attention', hidden_states: torch.Tensor, num_parts=None, rotary_cos=None, rotary_sin=None, **kwargs):
        if not self.use_global:
            return self._attend(attn, hidden_states, rotary_cos, rotary_sin)
        if num_parts is None:
            num_parts = [hidden_states.shape[0]]
        elif isinstance(num_parts, torch.Tensor):
            num_parts = num_parts.tolist()
        outputs = []
        idx = 0
        for n_p in num_parts:
            hs_i = hidden_states[idx:idx + n_p]
            B_, N_, C_ = hs_i.shape
            hs_merged = hs_i.reshape(1, n_p * N_, C_)
            rc = rs = None
            if self.apply_rope and rotary_cos is not None:
                rc_i = rotary_cos[idx:idx + n_p]
                rs_i = rotary_sin[idx:idx + n_p]
                rc = rc_i.reshape(1, n_p * N_, rc_i.shape[-1])
                rs = rs_i.reshape(1, n_p * N_, rs_i.shape[-1])
            out_merged = self._attend(attn, hs_merged, rc, rs)
            outputs.append(out_merged.reshape(n_p, N_, -1))
            idx += n_p
        return torch.cat(outputs, dim=0)

    def _attend(self, attn, hidden_states, rotary_cos=None, rotary_sin=None):
        B, N, C = hidden_states.shape
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // attn.num_heads // 3
        qkv = qkv.view(B, -1, attn.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)
        q = q.reshape(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = k.reshape(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = v.reshape(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if rotary_cos is not None:
            q = apply_rotary_emb(q, rotary_cos, rotary_sin)
            k = apply_rotary_emb(k, rotary_cos, rotary_sin)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(B, N, -1)
        return attn.out_proj(out)

class Attention(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, 'Only support head_dim <= 128 and divisible by 8'
        self.scale = self.head_dim ** (-0.5)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-06) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-06) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim)
        self.processor = None

    def forward(self, x, rotary_cos=None, rotary_sin=None, **kwargs):
        if self.processor is not None:
            return self.processor(self, x, rotary_cos=rotary_cos, rotary_sin=rotary_sin, **kwargs)
        B, N, C = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // self.num_heads // 3
        qkv = qkv.view(1, -1, self.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rotary_cos is not None:
            q = apply_rotary_emb(q, rotary_cos, rotary_sin)
            k = apply_rotary_emb(k, rotary_cos, rotary_sin)
        _attn_bias = kwargs.get('attn_bias')
        with torch.backends.cuda.sdp_kernel(enable_flash=_attn_bias is None, enable_math=True, enable_mem_efficient=_attn_bias is None):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=_attn_bias)
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.out_proj(x)
        return x

class DiTBlock(nn.Module):

    def __init__(self, hidden_size, c_emb_size, num_heads, text_states_dim=1024, use_flash_attn=False, qk_norm=False, norm_layer=nn.LayerNorm, qk_norm_layer=nn.RMSNorm, init_scale=1.0, qkv_bias=True, skip_connection=True, timested_modulate=False, use_moe: bool=False, num_experts: int=8, moe_top_k: int=2, **kwargs):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-06)
        self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=qk_norm_layer)
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-06)
        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_emb_size, hidden_size, bias=True))
        self.attn2 = CrossAttention(hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=qk_norm_layer, init_scale=init_scale)
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-06)
        if skip_connection:
            self.skip_norm = norm_layer(hidden_size, elementwise_affine=True, eps=1e-06)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None
        self.use_moe = use_moe
        if self.use_moe:
            self.moe = MoEBlock(hidden_size, num_experts=num_experts, moe_top_k=moe_top_k, dropout=0.0, activation_fn='gelu', final_dropout=False, ff_inner_dim=int(hidden_size * 4.0), ff_bias=True)
        else:
            self.mlp = MLP(width=hidden_size)

    def forward(self, x, c=None, text_states=None, skip_value=None, rotary_cos=None, rotary_sin=None, num_parts=None, **kwargs):
        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)
        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa
        attn_out = self.attn1(self.norm1(x), rotary_cos=rotary_cos, rotary_sin=rotary_sin, num_parts=num_parts, attn_bias=kwargs.get('attn_bias'))
        x = x + attn_out
        x = x + self.attn2(self.norm2(x), text_states)
        mlp_inputs = self.norm3(x)
        if self.use_moe:
            x = x + self.moe(mlp_inputs)
        else:
            x = x + self.mlp(mlp_inputs)
        return x

class AttentionPool(nn.Module):

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)
        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(query=x[:1], key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x.squeeze(0)

class FinalLayer(nn.Module):

    def __init__(self, final_hidden_size, out_channels):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-06)
        self.linear = nn.Linear(final_hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x

class PartDIT(nn.Module):

    def __init__(self, input_size=1024, in_channels=4, hidden_size=1024, context_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, norm_type='layer', qk_norm_type='rms', qk_norm=False, text_len=257, guidance_cond_proj_dim=None, qkv_bias=True, num_moe_layers: int=6, num_experts: int=8, moe_top_k: int=2, voxel_query_res: int=128, use_part_embed: bool=False, part_embed_valid_num: int=50, use_part_loc_pe: bool=False, part_loc_pe_res: int=128, part_loc_pe_mode: str='per_part', part_loc_pe_with_id: bool=True, part_loc_pe_num_freqs: int=8, use_local_global: bool=False, rope_in_global: bool=True, rope_in_local: bool=True, use_global_rope_shape: bool=False, global_rope_shape_res: int=512, global_rope_shape_theta: float=40000.0, use_global_rope_shape_in_local: bool=False, **kwargs):
        super().__init__()
        self.use_local_global = use_local_global
        self.rope_in_global = rope_in_global
        self.rope_in_local = rope_in_local
        self.use_global_rope_shape = use_global_rope_shape
        self.global_rope_shape_res = global_rope_shape_res
        self.global_rope_shape_theta = global_rope_shape_theta
        self.use_global_rope_shape_in_local = use_global_rope_shape_in_local
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm if norm_type == 'layer' else nn.RMSNorm
        self.qk_norm = nn.RMSNorm if qk_norm_type == 'rms' else nn.LayerNorm
        self.context_dim = context_dim
        self.voxel_query_res = voxel_query_res
        self.guidance_cond_proj_dim = guidance_cond_proj_dim
        self.text_len = text_len
        self.use_part_embed = use_part_embed
        if self.use_part_embed:
            self.part_embed_valid_num = part_embed_valid_num
            self.part_embed = nn.Parameter(torch.randn(part_embed_valid_num, hidden_size))
            self.part_embed.data.zero_()
        self.use_part_loc_pe = use_part_loc_pe
        self.part_loc_pe_res = part_loc_pe_res
        self.part_loc_pe_mode = part_loc_pe_mode
        self.part_loc_pe_with_id = part_loc_pe_with_id
        self.part_loc_pe_num_freqs = part_loc_pe_num_freqs
        if self.use_part_loc_pe:
            if self.part_loc_pe_mode == 'per_part':
                self.bbox_embedder = FourierMLPBboxEmbedder(in_dim=3, out_dim=hidden_size, num_freqs=part_loc_pe_num_freqs)
            elif self.part_loc_pe_mode == 'per_token':
                in_ch = 32 if self.part_loc_pe_with_id else 24
                self.abs_pos_embedder = AbsolutePositionEmbedder(hidden_size, in_channels=in_ch)
            else:
                raise ValueError(f'Unknown part_loc_pe_mode: {self.part_loc_pe_mode}')
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4, cond_proj_dim=guidance_cond_proj_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size=hidden_size, c_emb_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, text_states_dim=context_dim, qk_norm=qk_norm, norm_layer=self.norm, qk_norm_layer=self.qk_norm, skip_connection=layer > depth // 2, qkv_bias=qkv_bias, use_moe=True if depth - layer <= num_moe_layers else False, num_experts=num_experts, moe_top_k=moe_top_k) for layer in range(depth)])
        if self.use_local_global:
            for layer, block in enumerate(self.blocks):
                if hasattr(block, 'attn1') and (layer + 1) % 2 == 0:
                    block.attn1.processor = LocalGlobalProcessor(use_global=True, apply_rope=self.rope_in_global)
        self.depth = depth
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def _compute_part_loc_pe(self, bbox_minmax: torch.Tensor) -> torch.Tensor:
        mn = bbox_minmax[:, 0]
        mx = bbox_minmax[:, 1]
        center = 0.5 * (mn + mx)
        corners = []
        for mask in range(8):
            x = mx[:, 0:1] if mask & 1 else mn[:, 0:1]
            y = mx[:, 1:2] if mask & 2 else mn[:, 1:2]
            z = mx[:, 2:3] if mask & 4 else mn[:, 2:3]
            corners.append(torch.cat([x, y, z], dim=-1))
        corners = torch.stack(corners, dim=1)
        pts = torch.cat([center.unsqueeze(1), corners], dim=1)
        per_point = self.bbox_embedder(pts)
        return per_point.sum(dim=1)

    def _compute_per_token_corner_pe(self, voxel_idx: torch.Tensor, real_counts: torch.Tensor, bbox_minmax: torch.Tensor) -> torch.Tensor:
        R = int(self.voxel_query_res)
        T = voxel_idx.shape[0]
        device = voxel_idx.device
        v = voxel_idx.float()
        c_offsets = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], device=device, dtype=v.dtype)
        corners_unit = (v.unsqueeze(1) + c_offsets.unsqueeze(0)) / R * 2.0 - 1.0
        part_ids = torch.arange(real_counts.shape[0], device=device).repeat_interleave(real_counts)
        mn = bbox_minmax[part_ids, 0]
        mx = bbox_minmax[part_ids, 1]
        center = 0.5 * (mn + mx)
        scale = 0.5 * (mx - mn).amax(dim=-1, keepdim=True)
        corners_global = corners_unit * scale.unsqueeze(1) + center.unsqueeze(1)
        corners_q = (corners_global + 2.0) / 4.0
        corners_q = corners_q.clamp(0.0, 1.0) * 256.0
        if self.part_loc_pe_with_id:
            pid = part_ids.view(T, 1, 1).expand(T, 8, 1).float()
            pts = torch.cat([corners_q, pid], dim=-1).view(T, 32)
        else:
            pts = corners_q.view(T, 24)
        return self.abs_pos_embedder(pts)

def apply_rotary_emb(x, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return x * cos + rotate_half(x) * sin

def precompute_freqs_cis_3d(dim: int, grid_indices: torch.Tensor, theta: float=10000.0):
    dim_x = dim // 3
    dim_y = dim // 3
    dim_z = dim - dim_x - dim_y
    device = grid_indices.device
    freqs_x = 1.0 / theta ** (torch.arange(0, dim_x, 2, device=device).float() / dim_x)
    freqs_y = 1.0 / theta ** (torch.arange(0, dim_y, 2, device=device).float() / dim_y)
    freqs_z = 1.0 / theta ** (torch.arange(0, dim_z, 2, device=device).float() / dim_z)
    x_idx = grid_indices[..., 0].float()
    y_idx = grid_indices[..., 1].float()
    z_idx = grid_indices[..., 2].float()
    args_x = x_idx.unsqueeze(-1) * freqs_x.unsqueeze(0).unsqueeze(0)
    args_y = y_idx.unsqueeze(-1) * freqs_y.unsqueeze(0).unsqueeze(0)
    args_z = z_idx.unsqueeze(-1) * freqs_z.unsqueeze(0).unsqueeze(0)
    args = torch.cat([args_x, args_y, args_z], dim=-1)
    args = torch.cat([args, args], dim=-1)
    return (torch.cos(args), torch.sin(args))

def precompute_freqs_cis_3d_interpolated(dim: int, grid_indices: torch.Tensor, theta: float=10000.0, trained_res: float=128.0, current_res: float=256.0):
    scale_factor = current_res / trained_res
    dim_x = dim // 3
    dim_y = dim // 3
    dim_z = dim - dim_x - dim_y
    device = grid_indices.device
    freqs_x = 1.0 / theta ** (torch.arange(0, dim_x, 2, device=device).float() / dim_x)
    freqs_y = 1.0 / theta ** (torch.arange(0, dim_y, 2, device=device).float() / dim_y)
    freqs_z = 1.0 / theta ** (torch.arange(0, dim_z, 2, device=device).float() / dim_z)
    num_freqs_x = dim_x // 2 + dim_x % 2
    num_freqs_y = dim_y // 2 + dim_y % 2
    target_len = dim // 2
    freqs_x = freqs_x[:num_freqs_x]
    freqs_y = freqs_y[:num_freqs_y]
    freqs_z = freqs_z[:target_len - len(freqs_x) - len(freqs_y)]
    input_x = grid_indices[..., 0].float()
    input_y = grid_indices[..., 1].float()
    input_z = grid_indices[..., 2].float()
    pos_x = input_x / scale_factor
    pos_y = input_y / scale_factor
    pos_z = input_z / scale_factor
    args_x = pos_x.unsqueeze(-1) * freqs_x.unsqueeze(0).unsqueeze(0)
    args_y = pos_y.unsqueeze(-1) * freqs_y.unsqueeze(0).unsqueeze(0)
    args_z = pos_z.unsqueeze(-1) * freqs_z.unsqueeze(0).unsqueeze(0)
    args = torch.cat([args_x, args_y, args_z], dim=-1)
    args = torch.cat([args, args], dim=-1)
    return (torch.cos(args), torch.sin(args))
