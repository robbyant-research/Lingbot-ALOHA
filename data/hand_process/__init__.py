"""Offline egocentric hand-removal pipeline."""

import os

# Spawned GPU workers otherwise inherit the host's very large BLAS/OpenMP
# thread pools. Set these before NumPy, OpenCV, or PyTorch is imported.
_cpu_threads = os.environ.get("HAND_PROCESS_CPU_THREADS", "4")
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = _cpu_threads

__version__ = "0.3.0"
