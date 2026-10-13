"""Strict weight loading, Euler flow sampling and per-part mesh decoding."""
from pathlib import Path
import numpy as np
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from tqdm import tqdm
from .denoisers.dit_part_varlen import PackedPartDIT
from .autoencoders.model import VAEDecoder
from .conditioner import ImageConditioner


def load_models(config, root, device):
    factories = {
        'dit': lambda: PackedPartDIT(**config['dit']),
        'conditioner': lambda: ImageConditioner(config['dino'], config['preprocess']['image_size']),
        'vae': lambda: VAEDecoder(**config['vae']),
    }
    models = {}
    for name, factory in factories.items():
        path = Path(root) / config['weights'][name]
        if not path.is_file():
            if name == 'conditioner':
                raise FileNotFoundError(
                    f'Missing user-provided DINOv2 weights: {path}. '
                    'Run python scripts/download_dinov2.py from the release directory; '
                    'see README.md for the official download source. '
                    'DINOv2 weights are not bundled with StructParts.'
                )
            raise FileNotFoundError(f'Missing {name} weights: {path}')
        print(f'Loading {name}: {path.name}', flush=True)
        with init_empty_weights(include_buffers=False):
            model = factory()
        target = model.model if name == 'conditioner' else model
        state = load_file(str(path), device='cpu')
        if name == 'conditioner':
            # The original frozen encoder was loaded as FP16 before the final
            # checkpoint became BF16. Preserve that rounding with upstream FP32
            # weights rather than redistributing the embedded encoder copy.
            state = {key: value.to(torch.float16) for key, value in state.items()}
        target.load_state_dict(state, strict=True, assign=True)
        model = model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
        models[name] = model
    return models


@torch.inference_mode()
def sample(models, geometry, image, settings, device):
    dit, vae, dino = models['dit'], models['vae'], models['conditioner']
    dtype = torch.bfloat16
    torch.manual_seed(settings['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(settings['seed'])
    parts, tokens = geometry['padding_mask'].shape
    cond = dino(image.to(device=device, dtype=dtype)).expand(parts, -1, -1)
    contexts = {'main':torch.cat([cond, torch.zeros_like(cond)], dim=0)}
    kwargs = {'num_parts':[parts, parts]}
    for name in ('voxel_cond', 'padding_mask', 'part_bbox', 'part_global_bbox_minmax'):
        value = geometry[name].to(device)
        if name != 'padding_mask':
            value = value.to(dtype)
        kwargs[name] = torch.cat([value, value], dim=0)
    generator = torch.Generator(device=device).manual_seed(settings['seed'])
    latents = torch.randn((parts, tokens, vae.latent_shape[-1]), device=device, dtype=dtype, generator=generator)
    # Preserve the original pipeline's 0 -> 1 schedule, including its last zero-length step.
    sigmas = torch.from_numpy(np.linspace(0, 1, settings['steps'])).float().to(device)
    sigmas = torch.cat([sigmas, torch.ones(1, device=device)])
    for i in tqdm(range(settings['steps']), desc='Sampling'):
        timestep = (sigmas[i] * 1000).expand(parts * 2).to(dtype) / 1000
        pred = dit(torch.cat([latents, latents]), timestep, contexts, **kwargs)
        conditional, unconditional = pred.chunk(2)
        velocity = unconditional + settings['guidance_scale'] * (conditional - unconditional)
        latents = (latents.float() + (sigmas[i + 1] - sigmas[i]) * velocity).to(dtype)
    del contexts, cond, kwargs
    mask = geometry['padding_mask'].to(device)
    # Decode one part at a time to reduce peak memory; the VAE has no cross-part attention.
    for i in range(parts):
        decoded = vae(latents[i:i+1], mask[i:i+1])
        mesh = vae.mesh(decoded, settings['resolution'], settings['query_chunk_size'],
                        settings['bounds'], settings['mc_level'])
        bbox = geometry['part_bbox'][i].numpy()
        mesh.vertices = (mesh.vertices * bbox[3] + bbox[:3]) / geometry['shape_scale'] + geometry['shape_center']
        yield geometry['part_ids'][i], mesh
