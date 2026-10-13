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
