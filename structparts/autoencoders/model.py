"""Decoder-only VAE. Parameter names match the original ShapeVAE decoder."""
import torch
from torch import nn
from .attention_blocks import FourierEmbedder, Transformer, CrossAttentionDecoder
from .volume_decoder import CoarseToFineVolumeDecoding
from .surface_extractor import extract_mesh


class VAEDecoder(nn.Module):
    def __init__(self, num_latents, embed_dim, num_freqs, include_pi, heads,
                 width, num_decoder_layers, qkv_bias, qk_norm, scale_factor,
                 geo_decoder_mlp_expand_ratio, geo_decoder_downsample_ratio,
                 geo_decoder_ln_post):
        super().__init__()
        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.post_kl = nn.Linear(embed_dim, width)
        self.transformer = Transformer(n_ctx=num_latents, width=width,
            layers=num_decoder_layers, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder, out_channels=1, num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=geo_decoder_ln_post, width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.volume_decoder = CoarseToFineVolumeDecoding()

    def forward(self, latents, padding_mask):
        x = self.transformer(self.post_kl(latents / self.scale_factor))
        return x * padding_mask.unsqueeze(-1).to(x.dtype)

    def mesh(self, decoded_latents, resolution, query_chunk_size, bounds, mc_level):
        grid = self.volume_decoder(decoded_latents, self.geo_decoder,
            bounds=float(bounds), num_chunks=query_chunk_size, mc_level=mc_level,
            octree_resolution=resolution, enable_pbar=True)
        # A decoder may stop refinement early; use the grid's actual resolution.
        return extract_mesh(grid[0], float(bounds), mc_level)
