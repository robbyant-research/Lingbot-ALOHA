# LingBot ALOHA Policy

A Mobile ALOHA adaptation of [OpenPI](https://github.com/Physical-Intelligence/openpi), with RTC prefix conditioning for the JAX π₀.₅ model and training on multiple LeRobot datasets.

This release contains source code only. Datasets, pretrained weights, and trained checkpoints must be supplied separately. It registers one example, `pi05_mobile_aloha_rtc`, in [config.py](src/openpi/training/config.py). Both the standard and RTC configuration factories remain available for customization.

## Video data preparation

The [data pipeline](data/README.md) removes hands from egocentric videos using
CaRe-Ego, SAM 2.1, and DiffuEraser with a ProPainter prior. It includes recursive
video processing, a lower-memory ProPainter profile, and quality reports.
Follow the [pretrained weight guide](data/weights/README.md) before processing.
Use a separate environment for data preparation and policy training.
The pipeline produces processed videos; it does not convert them into a
LeRobot dataset or modify action and state metadata.

## Setup

Follow the [upstream OpenPI installation workflow](https://github.com/Physical-Intelligence/openpi#installation). OpenPI documents Ubuntu 22.04 with an NVIDIA GPU as its supported environment. Its approximate single-GPU memory requirements are more than 8 GB for inference, 22.5 GB for LoRA fine-tuning, and 70 GB for full fine-tuning. These are upstream estimates; this release's RTC example uses full fine-tuning, and its memory use depends on the configuration.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the root of this release:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

`GIT_LFS_SKIP_SMUDGE=1` is used when installing the LeRobot dependency. The final command activates the environment for the `python` commands below; alternatively, run them with `uv run python`.

The dependency definitions are in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). The optional [environment helpers](env_setup/README.md) use the same installation commands and the repository-local `.venv`.

The configuration loads initial π₀.₅ parameters from `gs://openpi-assets/checkpoints/pi05_base/params`. Make those parameters and tokenizer assets available before training, or change the factory's weight loader to your local parameters directory.

## Configure a run

Edit the single `_CONFIGS` entry in [config.py](src/openpi/training/config.py). Replace the placeholder dataset path. To train on two datasets, use the following entry in place of the existing one:

```python
make_pi05_mobile_aloha_rtc_config(
    # Registered name used by normalization, training, and serving commands.
    name="pi05_mobile_aloha_rtc",
    # LeRobot datasets to concatenate. Each must use the same feature schema.
    repo_id=["/path/to/dataset_a", "/path/to/dataset_b"],
    # Global batch size across all training devices.
    batch_size=128,
    # Total optimizer steps, not epochs or inference denoising steps.
    num_steps=6_000,
    # First N episodes from EACH dataset; -1 selects all episodes.
    use_episodes=-1,
    # Shared statistics directory: ./assets/mobile_aloha_example/.
    norm_name="mobile_aloha_example",
    # Accepted for compatibility but currently unused by this factory.
    use_delta_joint_actions=True,
    # Number of future action frames in a predicted chunk.
    action_horizon=25,
    # Past action frames loaded before the future chunk.
    history_length=1,
    # Training prefix length is sampled from 0 through this value minus one.
    rtc_max_delay=10,
    # Cached action frames used as a prefix during inference.
    rtc_prefix_len=6,
    # Action frames executed before requesting the next chunk.
    rtc_exec_steps=6,
    # Supply a separate diffusion-time condition for each action frame.
    per_step_time_rtc=True,
)
```

`norm_name` must be set when `repo_id` is a list. The fallback asset name assumes that `repo_id` is a string. The values above describe the example, not every factory default. In particular, the RTC factory defaults `per_step_time_rtc` to `False`; the registered example explicitly enables it.

Changing `use_delta_joint_actions` currently has no effect. To change action conversion, inspect and modify the data transforms rather than relying on this argument.

## Training with multiple datasets

### Dataset contract

The Mobile ALOHA factory uses the same repacking and model transforms for every dataset:

| Dataset feature | Use |
| --- | --- |
| `observation.image.head` | Head camera, mapped to `cam_high` |
| `observation.image.left` | Left wrist camera, mapped to `cam_left_wrist` |
| `observation.image.right` | Right wrist camera, mapped to `cam_right_wrist` |
| `observation.states.left` | Left arm state input |
| `observation.states.right` | Right arm state input |
| `action.left` | Left arm action sequence |
| `action.right` | Right arm action sequence |
| `action.base` | Mobile base action sequence |
| LeRobot task metadata | Task text used as `prompt` |

Datasets must agree on feature dimensions, camera conventions, action ordering, units, and task meanings. Concatenation does not reconcile different robot schemas. See [MobileAlohaInputs](src/openpi/policies/aloha_policy.py) for the exact state selection and action construction. It combines left arm actions, right arm actions, and the first and last base action channels before model padding.

### Sampling and episode selection

`MultiVLADataset` in [data_loader.py](src/openpi/training/data_loader.py) exposes the datasets as one concatenated frame index. Each dataset retains its own metadata and task-to-prompt mapping.

- Training samples come from shuffled frames in the combined dataset. There is no separate dataset-weight argument or automatic balancing by dataset.
- With 10,000 frames in dataset A and 30,000 in dataset B, the expected mixture is approximately 25% A and 75% B. These counts refer to the selected episodes.
- `use_episodes=N` selects the first N episodes of **each** dataset, capped by that dataset's episode count. It is neither a total limit across datasets nor a random subset. Use `-1` for all episodes.
- Repeated paths are not deduplicated. Listing a dataset twice repeats its contribution to the mixture.

For each action key, the loader requests `history_length` past frames followed by `action_horizon` future frames. Offsets are converted to seconds using each dataset's own FPS. This does not resample datasets to a common control rate: the same horizon covers different physical durations at different FPS. Use consistent control rates when the tasks require a common action timing convention.

The Mobile ALOHA transform removes the history portion from the action target and uses selected base history when constructing the state. `history_length` does not add that many frames to the predicted action chunk.

### Shared normalization

Compute statistics after finalizing the dataset list, episode selection, history length, and transforms:

```bash
python scripts/compute_norm_stats.py --config-name pi05_mobile_aloha_rtc
```

This command uses the configured combined dataset and input transforms to accumulate state and action statistics. The example writes a shared file at `assets/mobile_aloha_example/norm_stats.json`. The mixture uses this one set of statistics; it does not normalize each dataset independently.

For a shorter statistics pass, add `--max-frames 100000`. This caps the statistics pass at whole batches; it does not cap the training dataset. Recompute statistics when the data mixture or representation changes.

Start training with:

```bash
python scripts/train.py pi05_mobile_aloha_rtc --exp-name mobile_aloha_rtc_run
```

The factory sets AdamW, a peak learning rate of `5e-5`, 100 warmup steps, gradient clipping at `1.0`, EMA decay of `0.999`, and a checkpoint interval of 5,000 steps. Adjust these in the factory for your run. The current checkpoint writer omits the training state required by the restore path, so exact optimizer-state resume must not be assumed.

## RTC implementation

RTC conditions a new action chunk on a prefix taken from the previous prediction. The implementation has three parts: training with clean prefixes, retaining a prefix in policy inference, and scheduling requests in the client broker. The description below applies to the JAX π₀.₅ path; it does not establish equivalent RTC support in the PyTorch path.

### Training: clean prefix, noisy suffix

Let `H` be the action horizon and `D = rtc_max_delay`. When RTC is enabled and `D > 0`, each training example samples an integer prefix length `d` uniformly from `[0, D)`.

The first `d` action frames remain clean, with diffusion time zero. The remaining frames use the sampled diffusion time and the usual flow-matching interpolation between the target action and noise. Prefix frames condition the prediction but contribute no flow-matching loss. The remaining loss is scaled by `H / max(H - d, 1)` so that longer prefixes do not simply reduce its average magnitude.

With `per_step_time_rtc=True`, π₀.₅ receives a separate time condition for each action frame through the adaptive normalization path. With it disabled, π₀.₅ uses the final frame's time condition for the chunk, although the prefix inputs can still be clean. `rtc_max_delay=0` disables the training prefix masking.

Implementation: [pi0.py](src/openpi/models/pi0.py), [pi0_config.py](src/openpi/models/pi0_config.py), and [gemma.py](src/openpi/models/gemma.py).

### Inference: cached actions in model space

The policy stores the previous sampled chunk before output transforms, in normalized model action space. On the first request, there is no cached prefix. On later requests, with `s = rtc_exec_steps` and `d = rtc_prefix_len`, it supplies `previous_actions[s:s+d]` as the first `d` frames of the next prediction.

At the beginning of each denoising iteration, the sampler reapplies this prefix and sets its diffusion time to zero. The current sampler then updates all frames and does not apply a final prefix clamp. Consequently, returned prefix values are not guaranteed to be numerically identical to the cached actions.

Calling `reset()` clears the policy cache. Reset between independent episodes. When using the asynchronous broker, first let any outstanding inference finish; cancellation alone does not stop an already running request.

Implementation: [policy.py](src/openpi/policies/policy.py) and [pi0.py](src/openpi/models/pi0.py).

### Client scheduling and chunk handoff

[`RTCActionChunkBroker`](packages/openpi-client/src/openpi_client/rtc_action_chunk_broker.py) returns one action per `infer()` call:

1. Fetch the first chunk synchronously.
2. After serving `s` actions, start an asynchronous request using the observation supplied on that call.
3. Continue serving the next `d` actions from the old chunk while the request runs.
4. At `s + d`, wait for the result if needed and switch to the new chunk.

For the example (`H=25`, `s=6`, `d=6`), the request starts after old-chunk indices `0–5`; indices `6–11` are served during the overlap window. The current broker starts the new chunk at index `0`. **It does not skip the conditioned prefix**, so overlap actions can be replayed. Account for this behavior in the robot integration; this implementation does not guarantee seamless chunk transitions. With `d=0`, refreshing is synchronous after `s` actions.

All three timing parameters count action frames, not milliseconds. Keep server and client values consistent, with `s >= 1`, `d >= 0`, and `s + d <= H`. For training, use `0 < D <= H`; choosing `d < D` keeps the inference prefix length within the training range. These are configuration constraints to check, not guarantees enforced by the broker.

At control period `Δt`, the overlap window is `d × Δt`. Set it from measured inference and communication latency. If inference exceeds that window, the broker blocks at the handoff; it does not automatically adjust `d`.

## Serve and connect

Load a trained checkpoint with the same model configuration used for training:

```bash
python scripts/serve_policy.py \
    --port 8000 \
    --rtc-prefix-len 6 \
    --rtc-exec-steps 6 \
    --num-steps 10 \
    policy:checkpoint \
    --policy.config pi05_mobile_aloha_rtc \
    --policy.dir /path/to/trained/checkpoint
```

Here `--num-steps` controls inference denoising iterations. It is separate from the configuration factory's training-step argument. The RTC server flags override the policy's prefix and execution lengths, not the model's action horizon or training configuration.

In a client environment with `openpi-client` installed, construct the broker with matching values:

```python
from openpi_client.rtc_action_chunk_broker import RTCActionChunkBroker
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="127.0.0.1", port=8000)
broker = RTCActionChunkBroker(
    policy=client,       # Remote policy connection.
    action_horizon=25,  # Match the model's predicted chunk length.
    exec_steps=6,       # Match the server's RTC execution length.
    prefix_len=6,       # Match the server's RTC prefix length.
)

# In your control loop, supply a current observation in the policy input format:
# action = broker.infer(observation)["actions"]
# The result is one action frame, not a full chunk.
```

Adapt robot observations to the policy's serving input format in [aloha_policy.py](src/openpi/policies/aloha_policy.py). The training dataset feature names above describe the dataset repacking stage and are not a complete wire-format specification. Connect one independent control stream to a policy cache; do not interleave unrelated episodes through the same cached policy instance.

## Attribution and license

This repository is derived from [OpenPI](https://github.com/Physical-Intelligence/openpi), developed by [Physical Intelligence](https://www.physicalintelligence.company/) and the OpenPI contributors. The base model is [π₀.₅](https://www.physicalintelligence.company/blog/pi05). This is a modified downstream release and is not an official Physical Intelligence release.

The upstream OpenPI code is licensed under the **Apache License, Version 2.0**. The complete license is retained in [LICENSE](LICENSE). Copyright and other applicable notices in the upstream source remain with their respective owners. This README replaces the upstream overview to document the Mobile ALOHA, RTC, and multi-dataset changes in this release.

When redistributing this code, include the license, retain applicable upstream copyright and attribution notices, and identify modified files. Preserve any applicable upstream `NOTICE` notices if present in the material you redistribute. The license includes warranty and liability limitations; see [its full terms](https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE).

Model weights and third-party dependencies can have separate terms. No weights are bundled here; review the applicable model and dependency licenses when obtaining or redistributing them.
