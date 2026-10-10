#!/usr/bin/env python3
"""Generate part meshes from an image and a segmented scaffold."""
import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/inference.yaml')
    parser.add_argument('--input-dir', type=Path, help='Folder containing 0.png, mask_id/0.png, p3sam_mesh.glb and p3sam_labels.npy')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--mask', type=Path, help='Foreground or mask-ID image used for foreground cropping')
    parser.add_argument('--mesh', type=Path, help='Segmented scaffold mesh')
    parser.add_argument('--labels', type=Path, help='Integer face labels (.npy), one label per mesh face')
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--attention', choices=['auto','sdpa','fa2','fa3'], default='auto')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--guidance-scale', type=float)
    parser.add_argument('--check-weights', action='store_true', help='Strictly load all weights and exit; supports --device cpu')
    parser.add_argument('--prepare-only', action='store_true', help='Validate/preprocess inputs without loading weights')
    args = parser.parse_args()
    os.environ['STRUCTPARTS_ATTENTION'] = args.attention
    import torch
    import yaml
    from structparts.preprocess import prepare_geometry, prepare_image
    config = yaml.safe_load(args.config.read_text())
    settings = config['sampling'].copy()
    for key in ('steps','resolution','seed','guidance_scale'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if settings['steps'] < 2:
        parser.error('--steps must be >= 2')
    if settings['seed'] < 0:
        parser.error('--seed must be nonnegative')
    if not math.isfinite(settings['guidance_scale']) or settings['guidance_scale'] < 0:
        parser.error('--guidance-scale must be finite and nonnegative')
    resolution = settings['resolution']
    if resolution < 2 or resolution & (resolution - 1):
        parser.error('--resolution must be a power of two >= 2')
    if args.check_weights:
        from structparts.pipeline import load_models
        models = load_models(config, ROOT, torch.device(args.device))
        print('All three components loaded strictly:', ', '.join(models))
        return
    if args.input_dir:
        defaults = {'image':'0.png','mask':'mask_id/0.png','mesh':'p3sam_mesh.glb','labels':'p3sam_labels.npy'}
        for key, filename in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, args.input_dir / filename)
    for name in ('image','mask','mesh','labels'):
        path = getattr(args, name)
        if path is None or not path.is_file():
            parser.error(f'Provide an existing --{name} file, or --input-dir')
    geometry = prepare_geometry(args.mesh, args.labels, config['preprocess'], settings['seed'])
    if len(geometry['part_ids']) > config['dit']['part_embed_valid_num']:
        parser.error('Input exceeds the supported part count (50)')
    image = prepare_image(args.image, args.mask, config['preprocess'])
    print(f"Prepared {len(geometry['part_ids'])} parts; voxel counts: {geometry['padding_mask'].sum(1).tolist()}")
    if args.prepare_only:
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use a CUDA GPU for full inference or --device cpu --check-weights for validation')
    from structparts.pipeline import load_models, sample
    args.output.mkdir(parents=True, exist_ok=True)
    filenames = [args.output / f'part_{part_id}.glb' for part_id in geometry['part_ids']]
    if any(path.exists() for path in filenames) or (args.output / 'scene.glb').exists() or (args.output / 'result.json').exists():
        parser.error('Output already contains inference results; choose a new --output folder')
    models = load_models(config, ROOT, device)
    import trimesh
    scene = trimesh.Scene()
    parts = []
    for part_id, mesh in sample(models, geometry, image, settings, device):
        filename = f'part_{part_id}.glb'
        mesh.export(args.output / filename)
        scene.add_geometry(mesh, node_name=f'part_{part_id}', geom_name=f'part_{part_id}')
        parts.append({'part_id':part_id,'mesh':filename})
        print(f'Saved {filename}', flush=True)
    scene.export(args.output / 'scene.glb')
    (args.output / 'result.json').write_text(json.dumps({'parts':parts,'sampling':settings,
        'coordinates':'input mesh coordinates'}, indent=2) + '\n')
    print(f'Saved {len(parts)} parts and scene.glb to {args.output}')


if __name__ == '__main__':
    main()
