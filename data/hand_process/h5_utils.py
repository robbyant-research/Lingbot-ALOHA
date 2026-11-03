from __future__ import annotations


def require_h5py():
    """Load HDF5 support for lossless temporary frame storage."""
    try:
        import h5py
    except ImportError as error:
        raise ValueError("HDF5 frame storage requires the 'h5py' package") from error
    return h5py
