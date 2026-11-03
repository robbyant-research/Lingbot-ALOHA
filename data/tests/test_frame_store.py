from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from hand_process.frame_store import (
    Hdf5VideoWriter,
    OpenCVHdf5Proxy,
    frame_store_count,
    load_frame_store,
)


h5py = pytest.importorskip("h5py")


def test_hdf5_video_writer_round_trips_bgr_as_lossless_rgb(tmp_path):
    destination = tmp_path / "frames.h5"
    writer = Hdf5VideoWriter(
        destination,
        expected_num_frames=2,
        fps=Fraction(30000, 1001),
        frame_size=(3, 2),
    )
    first_bgr = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    second_bgr = np.full((2, 3, 3), [2, 5, 9], dtype=np.uint8)
    writer.write(first_bgr)
    writer.write(second_bgr)
    writer.release()

    assert frame_store_count(destination) == 2
    actual = load_frame_store(destination)
    np.testing.assert_array_equal(actual[0], first_bgr[..., ::-1])
    np.testing.assert_array_equal(actual[1], second_bgr[..., ::-1])
    with h5py.File(destination, "r") as handle:
        assert handle["frames"].chunks is None
        assert handle["frames"].compression is None
        assert handle.attrs["fps_numerator"] == 30000
        assert handle.attrs["fps_denominator"] == 1001


def test_hdf5_video_writer_removes_partial_file_on_count_mismatch(tmp_path):
    destination = tmp_path / "frames.h5"
    writer = Hdf5VideoWriter(
        destination,
        expected_num_frames=2,
        fps=Fraction(24, 1),
        frame_size=(2, 2),
    )
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(RuntimeError, match="frame count mismatch"):
        writer.release()

    assert not destination.exists()
    assert not destination.with_suffix(".h5.tmp").exists()


def test_hdf5_video_writer_restores_requested_output_geometry(tmp_path):
    destination = tmp_path / "frames.h5"
    writer = Hdf5VideoWriter(
        destination,
        expected_num_frames=1,
        fps=Fraction(24, 1),
        frame_size=(6, 4),
    )
    writer.write(np.zeros((4, 4, 3), dtype=np.uint8))
    writer.release()

    assert load_frame_store(destination)[0].shape == (4, 6, 3)


def test_opencv_proxy_validates_single_completed_writer(tmp_path):
    proxy = OpenCVHdf5Proxy(
        SimpleNamespace(),
        tmp_path / "frames.h5",
        expected_num_frames=1,
        fps=Fraction(24, 1),
        frame_size=(2, 2),
    )
    writer = proxy.VideoWriter("ignored.mp4", 0, 24.0, (2, 2))
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    writer.release()
    proxy.require_completed()

    with pytest.raises(RuntimeError, match="more than one"):
        proxy.VideoWriter("ignored-again.mp4", 0, 24.0, (2, 2))
