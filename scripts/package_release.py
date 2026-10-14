#!/usr/bin/env python3
"""Package inference code and our two model artifacts, excluding all DINOv2 weights."""
import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-weights', action='store_true')
    args = parser.parse_args()
    # Explicit allowlist: never archive the checkpoints directory wholesale.
    files = [ROOT / name for name in ['README.md', 'requirements.txt', 'LICENSE', 'Notice.txt', 'infer.py']]
    for directory in ['configs', 'structparts', 'scripts', 'licenses', 'docs', 'assets']:
        files.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc', '.pth', '.bin', '.safetensors'])
    files.append(ROOT / 'checkpoints/manifest.json')
    if args.include_weights:
        files.extend(ROOT / 'checkpoints' / name for name in ['dit.safetensors', 'vae_decoder.safetensors'])
    output = args.output.resolve()
    if output in [p.resolve() for p in files]:
        parser.error('Output must not overwrite a release source file')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=str(Path('structparts') / path.relative_to(ROOT)), recursive=False)
    print(output)


if __name__ == '__main__':
    main()
