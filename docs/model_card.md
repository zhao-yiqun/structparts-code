---
language:
- en
tags:
- image-to-3d
- part-decomposed-generation
- structparts
---
# StructParts

**Seeing Both the Forest and the Trees: Reusing Holistic 3D Priors for Part-Decomposed Generation**

Yiqun Zhao, Binbin Huang, Haobin Duan, Zibo Zhao, Shenghua Gao (corresponding author).

[Project page](https://zhao-yiqun.github.io/structparts/) · [Inference code and setup](https://github.com/zhao-yiqun/structparts-code)

## Weights

This repository distributes only our BF16 part diffusion transformer (`dit.safetensors`, 6.10 GB) and VAE mesh decoder (`vae_decoder.safetensors`, 0.43 GB). Download both into the inference repository's `checkpoints/` directory. `manifest.json` contains their SHA-256 hashes.

```bash
hf download Goatzyq/structparts dit.safetensors vae_decoder.safetensors manifest.json --local-dir checkpoints
```

The input is an image, a foreground mask, and a segmented 3D scaffold. See the inference README for dependencies, input preparation, usage, and limitations. Scaffold inaccuracies can persist in generated parts.

## DINOv2

DINOv2 weights are not included here. Obtain the compatible encoder directly from [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) using `python scripts/download_dinov2.py` in the inference repository. The helper pins revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`.

## Release status

Private preparation pending the arXiv release. The project page and code repositories may remain private until publication.

## Acknowledgments and licensing

We thank [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) and [UltraShape-1.0](https://github.com/PKU-YuanGroup/UltraShape-1.0) for their open-source models and implementations. Consult `LICENSE`, `Notice.txt`, and retained notices in the inference repository for applicable upstream terms. DINOv2 is downloaded separately under its upstream license.
