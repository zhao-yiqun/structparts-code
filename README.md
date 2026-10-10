# StructParts inference

Minimal image-conditioned part generation from a segmented 3D scaffold. The input segmentation and 2D foreground mask are supplied by the user. This repository contains the inference model, preprocessing, mesh decoder and one configuration.

## Setup

Use Python 3.10 and a CUDA GPU supporting BF16. Install the CUDA build of PyTorch appropriate for your system, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The three weight files belong in `checkpoints/`:

| File | Component | Size |
| --- | --- | --- |
| `dit.safetensors` | Part diffusion transformer | 6.10 GB |
| `dino.safetensors` | DINOv2-large image encoder | 0.61 GB |
| `vae_decoder.safetensors` | VAE mesh decoder | 0.43 GB |

All weights are BF16. They load locally without a Hugging Face download. `checkpoints/manifest.json` records their SHA-256 hashes. Keep weights as separate model artifacts when uploading the code repository; the `.gitignore` excludes large weight files from Git.

PyTorch SDPA works without FlashAttention. If installed, FlashAttention 3 is selected on Hopper GPUs, otherwise FlashAttention 2 is used when available. Set `--attention sdpa` to force PyTorch attention. For the original GPU surface extractor, install `cubvh` with `sparse_marching_cubes` support. Without it, the decoder uses scikit-image marching cubes, whose triangulation can differ.

## Inference

A prepared object directory has four files:

```text
object/
  0.png                 # input RGB or RGBA image
  mask_id/0.png         # image mask for foreground cropping
  p3sam_mesh.glb        # segmented scaffold mesh
  p3sam_labels.npy      # one integer part ID per triangle
```

Use the same mesh loading order as `trimesh.load(..., force="mesh", process=False)` when creating face labels. A binary foreground mask with values 0/255 is supported. Existing mask-ID images are also accepted; preprocessing uses the original foreground threshold of 76.5. An empty mask is rejected. The image and scaffold must represent the same object and view.

```bash
python infer.py --input-dir /path/to/object --output results/object
```

Or pass the four paths explicitly:

```bash
python infer.py --image image.png --mask foreground.png \
  --mesh scaffold.glb --labels face_labels.npy --output results/object
```

The default configuration uses 1,024 voxel tokens per part, a 1,022-pixel image, 50 Euler sampling iterations, CFG scale 5 and a 512-resolution mesh decoder. Up to 50 parts are supported. These values are in `configs/inference.yaml`; `--steps`, `--resolution`, `--seed` and `--guidance-scale` override sampling settings. Decoder resolution must be a power of two. For a lighter run, use `--resolution 256`.

Outputs are `part_<id>.glb`, `scene.glb` (separate named part geometries) and `result.json`. Part IDs come from the input face labels. Meshes are returned in the input scaffold's coordinate frame. Choose a new output folder for each run.

Validate inputs or strictly check all model weights:

```bash
python infer.py --input-dir /path/to/object --prepare-only
python infer.py --check-weights --device cpu
```

## Layout

```text
infer.py                   # command-line entry
configs/inference.yaml     # inference parameters
structparts/               # preprocessing, sampling and required model code
checkpoints/               # three standalone weight files and hash manifest
requirements.txt
LICENSE
Notice.txt
```

The model code derives from the existing StructParts / UltraShape / Hunyuan3D implementation. Original copyright and license notices are retained in `LICENSE`, `Notice.txt` and source headers. DINOv2 is used through Transformers; its model license is [Apache 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE), included in `licenses/DINOv2-Apache-2.0.txt`.
