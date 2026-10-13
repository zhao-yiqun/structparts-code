#!/usr/bin/env python3
"""Download user-owned DINOv2 files directly from the facebook upstream repository."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'facebook/dinov2-large'
REVISION = '47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints/dinov2-large')
    args = parser.parse_args()
    snapshot_download(
        repo_id=REPOSITORY, revision=REVISION,
        allow_patterns=['config.json', 'model.safetensors'],
        local_dir=str(args.output.resolve()),
    )
    print(f'Downloaded upstream DINOv2-large to {args.output.resolve()}')


if __name__ == '__main__':
    main()
