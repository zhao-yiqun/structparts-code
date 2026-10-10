# Modified for the StructParts inference-only release: retained padded inference and packed attention.
import torch
import torch.nn as nn
from . import dit_part as _dit_part_module
from .dit_part import PartDIT, apply_rotary_emb, precompute_freqs_cis_3d, precompute_freqs_cis_3d_interpolated

def _pack_rotary_pair_padded(rot_cos, rot_sin, padding_mask, num_cond):
    B, N, D = rot_cos.shape
    real_counts = padding_mask.sum(dim=1)
    seq_lens = num_cond + real_counts
    T_total = seq_lens.sum().item()
    device = rot_cos.device
    packed_cos = torch.empty(T_total, D, device=device, dtype=rot_cos.dtype)
    packed_sin = torch.empty(T_total, D, device=device, dtype=rot_sin.dtype)
    cond_cos = torch.ones(num_cond, D, device=device, dtype=rot_cos.dtype)
    cond_sin = torch.zeros(num_cond, D, device=device, dtype=rot_sin.dtype)
    offset = 0
    for i in range(B):
        n_real = real_counts[i].item()
        mask_i = padding_mask[i]
        packed_cos[offset:offset + num_cond] = cond_cos
        packed_sin[offset:offset + num_cond] = cond_sin
        packed_cos[offset + num_cond:offset + num_cond + n_real] = rot_cos[i, mask_i]
        packed_sin[offset + num_cond:offset + num_cond + n_real] = rot_sin[i, mask_i]
        offset += num_cond + n_real
    return (packed_cos, packed_sin)

def pack_sequences(x, padding_mask, cond_tokens, rotary_cos_vox, rotary_sin_vox):
    B, N, C = x.shape
    num_cond = cond_tokens.shape[1]
    head_dim = rotary_cos_vox.shape[-1]
    device = x.device
    real_counts = padding_mask.sum(dim=1)
    seq_lens = num_cond + real_counts
    T_total = seq_lens.sum().item()
    packed_x = torch.empty(T_total, C, device=device, dtype=x.dtype)
    packed_cos = torch.empty(T_total, head_dim, device=device, dtype=rotary_cos_vox.dtype)
    packed_sin = torch.empty(T_total, head_dim, device=device, dtype=rotary_sin_vox.dtype)
    cond_cos = torch.ones(num_cond, head_dim, device=device, dtype=rotary_cos_vox.dtype)
    cond_sin = torch.zeros(num_cond, head_dim, device=device, dtype=rotary_sin_vox.dtype)
    offset = 0
    for i in range(B):
        n_real = real_counts[i].item()
        seg_len = num_cond + n_real
        packed_x[offset:offset + num_cond] = cond_tokens[i]
        packed_cos[offset:offset + num_cond] = cond_cos
        packed_sin[offset:offset + num_cond] = cond_sin
        mask_i = padding_mask[i]
        packed_x[offset + num_cond:offset + seg_len] = x[i, mask_i]
        packed_cos[offset + num_cond:offset + seg_len] = rotary_cos_vox[i, mask_i]
        packed_sin[offset + num_cond:offset + seg_len] = rotary_sin_vox[i, mask_i]
        offset += seg_len
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    torch.cumsum(seq_lens.int(), dim=0, out=cu_seqlens[1:])
    max_seqlen = seq_lens.max().item()
    return (packed_x, packed_cos, packed_sin, cu_seqlens, max_seqlen, real_counts)

def compute_global_cu_seqlens(cu_seqlens_local, num_parts, device):
    cu_local = cu_seqlens_local.tolist()
    cu_global = [0]
    idx = 0
    global_lens = []
    for n_p in num_parts:
        obj_len = cu_local[idx + n_p] - cu_local[idx]
        global_lens.append(obj_len)
        cu_global.append(cu_global[-1] + obj_len)
        idx += n_p
    cu_seqlens_global = torch.tensor(cu_global, dtype=torch.int32, device=device)
    max_seqlen_global = max(global_lens) if global_lens else 0
    return (cu_seqlens_global, max_seqlen_global)

class VarlenSelfAttentionProcessor:

    def __init__(self, is_global: bool=False, apply_rope: bool=True, force_global_rope: bool=False):
        self.is_global = is_global
        self.apply_rope = apply_rope
        self.force_global_rope = force_global_rope

    def __call__(self, attn, hidden_states, cu_seqlens=None, max_seqlen=None, rotary_cos=None, rotary_sin=None, rotary_cos_global=None, rotary_sin_global=None, **kwargs):
        T, C = hidden_states.shape
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // attn.num_heads // 3
        qkv = qkv.view(T, attn.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if (self.is_global or self.force_global_rope) and rotary_cos_global is not None:
            rope_cos, rope_sin = (rotary_cos_global, rotary_sin_global)
        else:
            rope_cos, rope_sin = (rotary_cos, rotary_sin)
        if rope_cos is not None and self.apply_rope:
            cos = rope_cos.unsqueeze(1)
            sin = rope_sin.unsqueeze(1)

            def rotate_half(x):
                x1, x2 = x.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        out = _dit_part_module.flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False)
        out = out.reshape(T, -1)
        out = attn.out_proj(out)
        return out

class VarlenCrossAttentionWrapper(nn.Module):

    def __init__(self, cross_attn):
        super().__init__()
        self.inner = cross_attn

    def forward(self, x, y, cu_seqlens_q=None, max_seqlen_q=None, **kwargs):
        if cu_seqlens_q is None:
            return self.inner(x, y)
        ca = self.inner
        T = x.shape[0]
        if y.dim() == 2:
            y = y.unsqueeze(1)
        B, s2 = (y.shape[0], y.shape[1])
        q = ca.to_q(x).view(T, ca.num_heads, ca.head_dim)
        q = ca.q_norm(q)
        k_full = ca.to_k(y)
        v_full = ca.to_v(y)
        kv = torch.cat((k_full, v_full), dim=-1)
        split_size = kv.shape[-1] // ca.num_heads // 2
        kv = kv.view(1, -1, ca.num_heads, split_size * 2)
        k_full, v_full = torch.split(kv, split_size, dim=-1)
        k_full = k_full.reshape(B, s2, ca.num_heads, ca.head_dim)
        v_full = v_full.reshape(B, s2, ca.num_heads, ca.head_dim)
        k_full = ca.k_norm(k_full)
        k_packed = k_full.reshape(-1, ca.num_heads, ca.head_dim)
        v_packed = v_full.reshape(-1, ca.num_heads, ca.head_dim)
        cu_seqlens_kv = torch.arange(0, (B + 1) * s2, s2, dtype=torch.int32, device=x.device)
        max_seqlen_kv = s2
        q = q.to(torch.bfloat16)
        k_packed = k_packed.to(torch.bfloat16)
        v_packed = v_packed.to(torch.bfloat16)
        out = _dit_part_module.flash_attn_varlen_func(q, k_packed, v_packed, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, dropout_p=0.0, softmax_scale=None, causal=False)
        out = out.reshape(T, -1)
        out = ca.out_proj(out)
        return out

class PackedPartDIT(PartDIT):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._global_layers = set()
        if self.use_local_global:
            for layer in range(len(self.blocks)):
                if (layer + 1) % 2 == 0:
                    self._global_layers.add(layer)
        self._varlen_local_proc = VarlenSelfAttentionProcessor(is_global=False, apply_rope=self.rope_in_local, force_global_rope=getattr(self, 'use_global_rope_shape_in_local', False))
        self._varlen_global_proc = VarlenSelfAttentionProcessor(is_global=True, apply_rope=self.rope_in_global)
        self._ca_wrappers = [VarlenCrossAttentionWrapper(b.attn2) for b in self.blocks]

    def forward(self, x, t, contexts, **kwargs):
        return self._forward_varlen_from_padded(x, t, contexts, **kwargs)

    def _forward_varlen_from_padded(self, x, t, contexts, **kwargs):
        cond = contexts['main']
        padding_mask = kwargs.get('padding_mask')
        voxel_cond = kwargs.get('voxel_cond')
        num_parts = kwargs.get('num_parts')
        B, N, _ = x.shape
        device = x.device
        x_dtype = x.dtype
        t_emb = self.t_embedder(t, condition=kwargs.get('guidance_cond'))
        x = self.x_embedder(x)
        if self.use_part_embed and num_parts is not None:
            if isinstance(num_parts, torch.Tensor):
                num_parts_list = num_parts.tolist()
            else:
                num_parts_list = list(num_parts)
            part_embed_idx = torch.cat([torch.randperm(self.part_embed_valid_num, device=device, dtype=torch.long)[:n] if n <= self.part_embed_valid_num else torch.cat([torch.randperm(self.part_embed_valid_num, device=device, dtype=torch.long), torch.randint(self.part_embed_valid_num, (n - self.part_embed_valid_num,), device=device)]) for n in num_parts_list])
            x = x + self.part_embed[part_embed_idx].unsqueeze(1)
        else:
            num_parts_list = num_parts if num_parts is not None else None
        if getattr(self, 'use_part_loc_pe', False):
            bbox_mm = kwargs.get('part_global_bbox_minmax')
            if bbox_mm is not None:
                if bbox_mm.dim() == 4:
                    bbox_mm = bbox_mm.reshape(-1, 2, 3)
                bbox_mm = bbox_mm.to(device=device, dtype=x.dtype)
                if self.part_loc_pe_mode == 'per_part':
                    loc_pe = self._compute_part_loc_pe(bbox_mm)
                    x = x + loc_pe.unsqueeze(1)
                else:
                    vc = kwargs.get('voxel_cond')
                    if vc is not None:
                        B, N = vc.shape[:2]
                        fake_rc = torch.full((B,), N, device=vc.device, dtype=torch.long)
                        loc_pe = self._compute_per_token_corner_pe(vc.reshape(-1, 3), fake_rc, bbox_mm)
                        x = x + loc_pe.reshape(B, N, -1).to(dtype=x.dtype)
        c = t_emb
        num_cond = c.shape[1]
        head_dim = self.blocks[0].attn1.head_dim
        rotary_cos_vox, rotary_sin_vox = precompute_freqs_cis_3d_interpolated(head_dim, voxel_cond, current_res=self.voxel_query_res)
        rotary_cos_g_padded = rotary_sin_g_padded = None
        if getattr(self, 'use_global_rope_shape', False):
            part_bbox = kwargs.get('part_bbox')
            if part_bbox is None:
                raise ValueError("use_global_rope_shape=True requires 'part_bbox' in kwargs.")
            center_b = part_bbox[:, :3].unsqueeze(1)
            iso_b = part_bbox[:, 3].unsqueeze(1).unsqueeze(-1)
            part_unit = voxel_cond.float().add(0.5).mul(2.0 / self.voxel_query_res).sub(1.0)
            global_cont = part_unit * iso_b + center_b
            vc_global_padded = ((global_cont + 1.0) * (self.global_rope_shape_res * 0.5)).clamp(0.0, float(self.global_rope_shape_res - 1))
            rotary_cos_g_padded, rotary_sin_g_padded = precompute_freqs_cis_3d(head_dim, vc_global_padded, theta=self.global_rope_shape_theta)
        if padding_mask is None:
            padding_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        packed_x, packed_cos, packed_sin, cu_seqlens_local, max_seqlen_local, real_counts = pack_sequences(x, padding_mask, c, rotary_cos_vox, rotary_sin_vox)
        if rotary_cos_g_padded is not None:
            packed_cos_g, packed_sin_g = _pack_rotary_pair_padded(rotary_cos_g_padded, rotary_sin_g_padded, padding_mask, num_cond)
        else:
            packed_cos_g = packed_sin_g = None
        if num_parts_list is not None:
            cu_seqlens_global, max_seqlen_global = compute_global_cu_seqlens(cu_seqlens_local, num_parts_list, device)
        else:
            cu_seqlens_global = cu_seqlens_local
            max_seqlen_global = max_seqlen_local
        cu_seqlens_cross = cu_seqlens_local
        max_seqlen_cross = max_seqlen_local
        skip_value_list = []
        half = self.depth // 2

        def _run_block(packed_x, skip_value, layer_idx):
            blk = self.blocks[layer_idx]
            ca_wrapper = self._ca_wrappers[layer_idx]
            if layer_idx in self._global_layers:
                cu_sl, max_sl = (cu_seqlens_global, max_seqlen_global)
            else:
                cu_sl, max_sl = (cu_seqlens_local, max_seqlen_local)
            if blk.skip_linear is not None and skip_value is not None:
                cat = torch.cat([skip_value, packed_x], dim=-1)
                packed_x = blk.skip_linear(cat)
                packed_x = blk.skip_norm(packed_x)
            proc = self._varlen_global_proc if layer_idx in self._global_layers else self._varlen_local_proc
            attn_out = proc(blk.attn1, blk.norm1(packed_x), cu_seqlens=cu_sl, max_seqlen=max_sl, rotary_cos=packed_cos, rotary_sin=packed_sin, rotary_cos_global=packed_cos_g, rotary_sin_global=packed_sin_g)
            packed_x = packed_x + attn_out
            packed_x = packed_x + ca_wrapper(blk.norm2(packed_x), cond, cu_seqlens_q=cu_seqlens_cross, max_seqlen_q=max_seqlen_cross)
            mlp_input = blk.norm3(packed_x)
            if blk.use_moe:
                packed_x = packed_x + blk.moe(mlp_input.unsqueeze(0)).squeeze(0)
            else:
                packed_x = packed_x + blk.mlp(mlp_input)
            return packed_x
        for layer in range(self.depth):
            skip_value = skip_value_list.pop() if layer > half else None
            packed_x = _run_block(packed_x, skip_value, layer)
            if layer < half:
                skip_value_list.append(packed_x)
        packed_x = self.final_layer.norm_final(packed_x)
        out_parts = []
        for i in range(B):
            start = cu_seqlens_local[i].item() + num_cond
            end = cu_seqlens_local[i + 1].item()
            out_parts.append(self.final_layer.linear(packed_x[start:end]))
        out_channels = self.final_layer.linear.out_features
        output = torch.zeros(B, N, out_channels, device=device, dtype=x_dtype)
        for i in range(B):
            output[i, padding_mask[i]] = out_parts[i].to(x_dtype)
        return output
