"""Packed attention with an optional FlashAttention backend."""
import os
import torch
from torch.nn.functional import scaled_dot_product_attention

try:
    from flash_attn_interface import flash_attn_varlen_func as fa3
except ImportError:
    fa3 = None
try:
    from flash_attn import flash_attn_varlen_func as fa2
except ImportError:
    fa2 = None


def flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k,
                           dropout_p=0.0, softmax_scale=None, causal=False):
    backend = os.environ.get('STRUCTPARTS_ATTENTION', 'auto')
    if backend not in ('auto', 'sdpa', 'fa2', 'fa3'):
        raise ValueError('STRUCTPARTS_ATTENTION must be auto, sdpa, fa2 or fa3')
    if backend in ('fa2', 'fa3'):
        fn = fa2 if backend == 'fa2' else fa3
        if fn is None or not q.is_cuda:
            raise RuntimeError(f'{backend} requires CUDA and its installed extension')
    elif backend == 'auto' and q.is_cuda:
        fn = fa3 if fa3 is not None and torch.cuda.get_device_capability(q.device)[0] >= 9 else fa2
    else:
        fn = None
    if fn is fa3 and fn is not None:
        return fn(q, k, v, cu_q, cu_k, max_q, max_k,
                  softmax_scale=softmax_scale, causal=causal)
    if fn is not None:
        return fn(q, k, v, cu_q, cu_k, max_q, max_k,
                  dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    q_bounds, k_bounds = cu_q.tolist(), cu_k.tolist()
    outputs = []
    for qs, qe, ks, ke in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]):
        outputs.append(scaled_dot_product_attention(
            q[qs:qe].transpose(0, 1).unsqueeze(0),
            k[ks:ke].transpose(0, 1).unsqueeze(0),
            v[ks:ke].transpose(0, 1).unsqueeze(0),
            dropout_p=dropout_p, is_causal=causal, scale=softmax_scale,
        ).squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)
