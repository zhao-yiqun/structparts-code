"""Upload only StructParts weights and metadata to the private model repository."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Goatzyq/structparts'
WEIGHTS = ('dit.safetensors', 'vae_decoder.safetensors')


def main():
    if not get_token():
        raise SystemExit('Run hf auth login on this server first; a write token is required.')
    api = HfApi()
    info = api.model_info(REPO)
    if not info.private:
        raise SystemExit('This release is staged privately until arXiv. Set the model repository to private in its Settings, then retry.')
    manifest = json.loads((ROOT / 'checkpoints/manifest.json').read_text())
    if set(manifest) != set(WEIGHTS):
        raise SystemExit('Unexpected manifest entries; refusing to upload.')
    operations = []
    for name in WEIGHTS:
        path = ROOT / 'checkpoints' / name
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != manifest[name]['bytes'] or digest.hexdigest() != manifest[name]['sha256']:
            raise SystemExit(f'Checksum or size mismatch: {name}')
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))
    operations.extend([
        CommitOperationAdd(path_in_repo='manifest.json', path_or_fileobj=str(ROOT / 'checkpoints/manifest.json')),
        CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(ROOT / 'docs/model_card.md')),
    ])
    result = api.create_commit(repo_id=REPO, repo_type='model', operations=operations,
                               commit_message='Upload StructParts DiT and VAE decoder weights')
    uploaded = api.model_info(REPO, files_metadata=True)
    files = {file.rfilename: file for file in uploaded.siblings}
    for name in WEIGHTS:
        if files[name].size != manifest[name]['bytes'] or files[name].lfs.sha256 != manifest[name]['sha256']:
            raise SystemExit(f'Remote verification failed: {name}')
    print(f'Uploaded and verified both weights: {result.commit_url}')


if __name__ == '__main__':
    main()
