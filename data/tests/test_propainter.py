import pytest

from hand_process.propainter import _mask_dilation


def test_mask_dilation_defaults_to_legacy_zero():
    assert _mask_dilation({}) == 0
    assert _mask_dilation({"mask_dilation": 3}) == 3


def test_mask_dilation_rejects_negative_radius():
    with pytest.raises(ValueError, match="must be >= 0"):
        _mask_dilation({"mask_dilation": -1})
