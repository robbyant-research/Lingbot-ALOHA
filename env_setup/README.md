<!-- Modified 2026-10-22: replace the internal environment instructions with the upstream uv workflow. -->
# Python environment

Use the [upstream OpenPI installation workflow](https://github.com/Physical-Intelligence/openpi#installation). Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from the repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

The dependency definitions are maintained in `pyproject.toml` and `uv.lock`. See the [root README](../README.md#setup) for platform and GPU requirements.

The helper scripts use this same workflow:

```bash
bash env_setup/setup_env.sh
source env_setup/env.sh
python scripts/train.py --help
```

After replacing the example dataset path and computing its normalization statistics, start training with:

```bash
bash env_setup/run_train_jax.sh pi05_mobile_aloha_rtc --exp-name my-experiment
```

`setup_env.sh` installs into this repository's `.venv`. `env.sh` activates it, and `run_train_jax.sh` uses its Python interpreter. Model weights and datasets are supplied separately.
