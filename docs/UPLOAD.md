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

The inference README contains the Hugging Face model URL and download command. Private model downloads require an authorized Hugging Face login. Only describe an upload as complete after the uploader verifies the remote hashes and sizes.

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

### Alternate storage route on this server

If Xet transfers stall, keep the API proxy settings above and use the HTTP storage path:

```bash
export HF_HUB_DISABLE_XET=1
export HF_UPLOAD_STORAGE_PROXY=http://10.10.7.2:38893
/public/home/yiqun/anaconda3/envs/eval_baselines/bin/python scripts/upload_weights.py
```

The optional storage proxy applies only to the official Hugging Face S3 upload host; other Hub requests keep using the API proxy. This route uploaded and remotely verified both artifacts on 2026-10-13. Omit these server-specific proxy settings on networks that connect normally.
