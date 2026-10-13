# Uploading StructParts weights

Run these commands on the server containing the checkpoints. The existing CLI is installed at `/public/home/yiqun/anaconda3/envs/eval_baselines/bin/hf`; Homebrew is not needed.

1. Log in interactively with a token that can write to `Goatzyq/structparts`:

   ```bash
   /public/home/yiqun/anaconda3/envs/eval_baselines/bin/hf auth login
   ```

2. Keep the model repository private until arXiv publication, then upload the explicit allowlist:

   ```bash
   cd /public/home/yiqun/project/flexipart/release
   /public/home/yiqun/anaconda3/envs/eval_baselines/bin/python scripts/upload_weights.py
   ```

The uploader verifies local SHA-256 hashes, uploads only `dit.safetensors`, `vae_decoder.safetensors`, `manifest.json`, and the model card as `README.md`, then verifies the remote weight hashes and sizes. It never scans or uploads the checkpoint directory recursively. DINOv2 is excluded.

After successful upload, add the Hugging Face model URL and the download command from the model card to the inference README. Private model downloads require an authorized Hugging Face login.

## Server network setup

On this server, the inherited proxy failed during TLS negotiation with Hugging Face. The existing local proxy at `127.0.0.1:17890` was verified against the official HTTPS API (an unauthenticated request returned HTTP 401).

Before login and upload, run in the same terminal:

```bash
export http_proxy=http://127.0.0.1:17890
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export HF_ENDPOINT=https://huggingface.co
unset ALL_PROXY all_proxy NO_PROXY no_proxy
```

Then run the login and upload commands above. Keep TLS certificate verification enabled. These settings apply to the current shell only; the local proxy must remain running during upload.
