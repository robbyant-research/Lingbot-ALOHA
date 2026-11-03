# Video Hand Removal

Prepare egocentric videos for Mobile ALOHA policy training. The pipeline detects
hands, tracks their masks, and fills the masked regions while preserving the
input directory structure. Source videos are read-only.

The default pipeline uses **CaRe-Ego → SAM 2.1 → ProPainter prior → DiffuEraser**.
The low-memory profile uses **CaRe-Ego → SAM 2.1 → ProPainter** with staged model
residency. Model weights and input data are supplied separately.

## Installation

Use Linux, Python 3.12, and an NVIDIA GPU with a compatible CUDA driver.
Run the commands below from this `data/` directory. Keep this environment
separate from the policy-training environment at the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

The inference sources are included under `third_party/` and `vendor/`; no
submodule initialization is required. SAM 2 is imported directly from its
source directory. Its optional CUDA extension is not required by this setup.
CaRe-Ego uses pure-PyTorch MMSeg components through `mmcv-lite`.

The dependency versions are pinned for this integration. A clean Linux/CUDA
installation and end-to-end model inference still require validation on the
target machine. The CPU checks below do not establish GPU compatibility,
throughput, or output quality.

## Pretrained weights

Follow [weights/README.md](weights/README.md) for public sources, exact files,
download commands, and offline installation. To download the default profile:

```bash
./download.sh
python -m hand_process.cli assets-manifest \
    --config configs/server.yaml --validate
```

Downloads are checked against the sizes and SHA-256 hashes in
[model_assets.json](model_assets.json). Valid existing files are reused.
The default profile needs approximately 13.6 GB of model files, plus space for
temporary downloads, decoded frames, and outputs. No model files are tracked.

## Process videos

The launcher takes one or more **input/output directory pairs**. Output
directories must be absent or empty and must lie outside every input tree.
MP4 discovery is recursive and case-insensitive. Relative paths are retained.

```bash
# Select physical GPU 0. Without HAND_PROCESS_GPUS, the server uses all GPUs.
HAND_PROCESS_GPUS=0 scripts/run_server.sh /path/to/videos /path/to/processed

# Process datasets in the given order; videos within a dataset may run concurrently.
HAND_PROCESS_GPUS=0,1 scripts/run_server.sh \
    /path/to/dataset_a/videos /path/to/dataset_a/processed \
    /path/to/dataset_b/videos /path/to/dataset_b/processed
```

For example, `videos/chunk-000/camera/episode_000001.mp4` becomes
`processed/chunk-000/camera/episode_000001.mp4`.

The default profile processes at 480 pixels high and encodes H.264 video with
every frame as a keyframe. It preserves the video frame rate; output audio is
not retained. The low-memory profile processes at 360 pixels high and uses
GPU 0:

```bash
./download.sh configs/low_memory.yaml
scripts/run_edge.sh /path/to/videos /path/to/processed
```

The pipeline applies a camera convention based on the immediate parent
folder: names containing `left` rotate frames clockwise for inference; names
containing `right` rotate them counterclockwise. The output is rotated back.
Use neutral camera folder names when your videos do not use this convention.
A parent folder containing both words is rejected.

Models remain loaded after the launcher finishes. Stop this project's model
services when they are no longer needed:

```bash
./kill.sh --dry-run
./kill.sh
```

The stop command verifies each process's project directory and parent service
before sending a signal.

## Configuration and storage

| Profile | Detector | Inpainter | Residency | Processing height |
| --- | --- | --- | --- | --- |
| `configs/server.yaml` | CaRe-Ego | DiffuEraser with ProPainter prior | Resident | 480 |
| `configs/low_memory.yaml` | CaRe-Ego | ProPainter | Staged | 360 |
| `configs/grounding_dino.yaml` | Grounding DINO + SAM 2.1 | DiffuEraser with ProPainter prior | Resident | 480 |

The default is one service process per GPU and one 22-frame temporal window
per DiffuEraser forward pass. Larger `runtime.model_service.lanes_per_gpu` or
`models.inpainter.window_batch_size` values increase memory use. Validate them
on your hardware before processing a full dataset. A low-memory profile does
not guarantee that every video fits a particular GPU or host-memory size.

All bundled paths resolve relative to this `data/` directory, independently of
the current working directory. The configuration loader expands these tokens:

- `${project_root}`: this `data/` directory.
- `${asset_root}`: `data/assets/models`, overridden by `HAND_PROCESS_ASSET_ROOT`.
- `${work_root}`: a per-user, per-checkout directory under `/dev/shm`, overridden
  by `HAND_PROCESS_WORK_ROOT`.
- `${socket_root}`: a short per-user, per-checkout directory in the system
  temporary directory for local Unix sockets.

Model scratch must resolve beneath `/dev/shm` when `runtime.require_tmpfs` is
`true`, as it is in every bundled profile. Ensure `/dev/shm` is mounted as tmpfs
and check its capacity before a long run. Set `HAND_PROCESS_WORK_ROOT` to choose
a different subdirectory beneath `/dev/shm`.
To explicitly allow disk-backed scratch, set `runtime.require_tmpfs: false`
in a custom profile and choose a writable work directory. Runtime frames can
require substantially more space than compressed input videos.

The launcher writes generated configuration and run metadata under `runtime/`,
and logs under `logs/`. Full runs retain final videos and compact quality
reports. Temporary per-frame files are removed after successful processing.
Inspect failed runs before removing their remaining scratch files.

To customize a profile, create an ignored `configs/local.yaml`:

```yaml
extends: server.yaml
runtime:
  model_service:
    lanes_per_gpu: 1
models:
  inpainter:
    window_batch_size: 1
processing:
  target_height: 480
  paste_back_to_source: false
```

Select it with `HAND_PROCESS_SERVER_CONFIG=configs/local.yaml`. Shell scripts
use the active `python`; set `HAND_PROCESS_PYTHON` to choose an interpreter.
Set `HAND_PROCESS_CPU_THREADS` before launch to limit NumPy/BLAS threads.

With `paste_back_to_source: true`, inference still uses the processing
resolution. The pipeline upsamples the correction and composites it onto the
source-resolution frames, which increases host-memory use.

## Resume and quality reports

Each launcher run prints its run ID and writes its generated configuration to
`runtime/generated-configs/<run-id>.yaml`. A launcher rejects a nonempty output
directory, including on a repeated invocation. Resume an interrupted run with
the saved configuration and the same run ID instead:

```bash
python -m hand_process.cli run \
    --config runtime/generated-configs/<run-id>.yaml \
    --stage full --gpus 0 --run-id <run-id> --resume
```

If services have stopped, start them first with `python -m hand_process.cli serve
--config runtime/generated-configs/<run-id>.yaml --gpus 0` in another terminal.
Use the same GPU selection and model settings as the interrupted run.

Quality reports check mask area, temporal consistency, and changes outside
the masked regions. Inspect flagged videos before using them for training;
these metrics do not establish semantic correctness of generated content.

For a small first run, use a separate input directory with a few representative
videos and an empty output directory. Review the resulting videos before
processing the full dataset.

## Optional Grounding DINO detector

Download the additional weights described in [weights/README.md](weights/README.md),
then select `HAND_PROCESS_SERVER_CONFIG=configs/grounding_dino.yaml`.
Grounding DINO identifies hands by class and may remove another person's hands.
Use this profile only when every matching hand may be removed. The profile's
`acknowledge_non_ego_hand_removal` setting makes that behavior explicit.

## CPU checks

In a separate environment, install `requirements-test.txt` and run:

```bash
python -m pytest -q
python -m compileall -q hand_process third_party vendor
bash -n download.sh kill.sh scripts/run_server.sh scripts/run_edge.sh
```

These tests cover configuration, asset integrity, path handling, masks, video
I/O, temporary frame storage, and service orchestration. Tensor tests skip
when PyTorch is unavailable; the attention integration test also needs Diffusers. No pretrained model or GPU is needed for the
remaining checks.

## Attribution

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream projects,
licenses, and included modifications. ProPainter's license restricts use to
non-commercial purposes; that component is used by both default profiles.
Each pretrained model retains its own license and conditions.
