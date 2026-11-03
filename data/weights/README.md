# Pretrained Weights

Run these commands from `data/` with its Python environment active. Weights
are stored under `assets/models/`, or under `HAND_PROCESS_ASSET_ROOT` when set.
They are not included in Git.

## Required models

| Component | Public source | Destination below `assets/models/` |
| --- | --- | --- |
| CaRe-Ego | [Official project](https://github.com/yuggiehk/CaRe-Ego), [checkpoint](https://drive.google.com/file/d/1F8QyhSeHaJfS7QLRfMKdgTVKa69JRp-i/view) | `care_ego/best_mIoU_ckpt.pth` |
| SAM 2.1 Large | [Official checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) | `sam2/sam2.1_hiera_large.pt` |
| ProPainter | [Official release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0) | `propainter/ProPainter.pth`, `raft-things.pth`, `recurrent_flow_completion.pth` |
| DiffuEraser | [Model repository](https://huggingface.co/lixiaowen/diffuEraser) | `diffueraser/diffuEraser/{brushnet,unet_main}/` |
| Stable Diffusion 1.5 | [Model repository](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `diffueraser/stable-diffusion-v1-5/` |
| SD VAE FT MSE | [Model repository](https://huggingface.co/stabilityai/sd-vae-ft-mse) | `diffueraser/sd-vae-ft-mse/` |
| PCM SD1.5 two-step adapter | [Model repository](https://huggingface.co/wangfuyun/PCM_Weights) | `diffueraser/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors` |

The low-memory profile needs only CaRe-Ego, SAM 2.1, and the three ProPainter
files. The default server profile requires every row. DiffuEraser uses
ProPainter as its prior generator.

The [asset manifest](../model_assets.json) is the authoritative list of
filenames, public URLs, sizes, and SHA-256 hashes. Diffusion repositories are
pinned to these revisions:

| Repository | Revision |
| --- | --- |
| `lixiaowen/diffuEraser` | `ad510dca07fa8e155d4bd8d002085bb8ec8f60e5` |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` |
| `stabilityai/sd-vae-ft-mse` | `31f26fdeee1355a5c34592e401dd41e45d25a493` |
| `wangfuyun/PCM_Weights` | `39560fead4ce00f94db3cb8e93dd8fba90ec0be6` |

## Download and verify

```bash
# Complete default pipeline.
./download.sh configs/server.yaml

# Smaller download for the ProPainter-only profile.
./download.sh configs/low_memory.yaml

# Check the default profile without starting models.
python -m hand_process.cli assets-manifest \
    --config configs/server.yaml --validate
```

The downloader uses public HTTPS sources and `gdown` for CaRe-Ego's Google
Drive checkpoint. Each download is written to a temporary file and published
only after size and SHA-256 verification. Existing valid files are skipped;
a failed download does not replace an existing file. Interrupted downloads
restart at the file level on the next invocation.

The downloader also creates the ProPainter weight links used by its standalone
runner. Profile validation recreates these links for manually installed weights.

## Manual and offline installation

If an automatic download fails, open the corresponding public source and place
the file at the manifest's exact relative path. For CaRe-Ego, download the
**best segmentation checkpoint**, not the separate Swin backbone initialization.
Do not bypass a hash mismatch; check that the selected file and revision match.

For offline use, download and verify all selected weights on a connected
machine, copy the complete `assets/models` directory, and run the validation
command on the target machine. Keep the subdirectory names unchanged.

The default directory layout is:

```text
assets/models/
├── care_ego/best_mIoU_ckpt.pth
├── sam2/sam2.1_hiera_large.pt
├── propainter/
│   ├── ProPainter.pth
│   ├── raft-things.pth
│   └── recurrent_flow_completion.pth
└── diffueraser/
    ├── diffuEraser/
    │   ├── brushnet/{config.json,diffusion_pytorch_model.safetensors}
    │   └── unet_main/{config.json,diffusion_pytorch_model.safetensors}
    ├── stable-diffusion-v1-5/
    │   ├── model_index.json
    │   ├── feature_extractor/
    │   ├── safety_checker/
    │   ├── scheduler/
    │   ├── text_encoder/
    │   └── tokenizer/
    ├── sd-vae-ft-mse/{config.json,diffusion_pytorch_model.safetensors}
    └── PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors
```

The default path uses the DiffuEraser UNet and separate VAE. It does not need
the Stable Diffusion repository's original `unet/`, `vae/`, or monolithic
checkpoint files. The manifest avoids those redundant downloads. Changing
`checkpoint_variant` requires obtaining the matching PCM adapter separately.

## Optional Grounding DINO

The optional detector uses [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny).
Download the Transformers-format model, tokenizer, and processor files:

```bash
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ.get('HAND_PROCESS_ASSET_ROOT', 'assets/models'))
snapshot_download(
    repo_id='IDEA-Research/grounding-dino-tiny',
    local_dir=root / 'grounding-dino-tiny',
    allow_patterns=['*.json', '*.txt', '*.safetensors'],
)
PY
./download.sh configs/grounding_dino.yaml
python -m hand_process.cli assets-manifest \
    --config configs/grounding_dino.yaml --validate
```

This optional snapshot is not covered by the pinned asset manifest. Retain its
resolved Hugging Face revision for reproducibility. The Grounding DINO profile
replaces CaRe-Ego but still needs SAM 2.1 and the complete inpainting stack.
It may remove all matching hands, including hands belonging to other people.

## Licenses

Use each weight under the terms of its source repository. ProPainter and its
weights are restricted to non-commercial use. The repository's main license
does not override model or third-party terms. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
