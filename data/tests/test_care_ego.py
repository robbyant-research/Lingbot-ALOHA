import numpy as np
import pytest

from hand_process.care_ego import build_hand_prompts


def _probabilities() -> np.ndarray:
    return np.array(
        [
            [[0.80, 0.10, 0.10], [0.34, 0.20, 0.70]],
            [[0.15, 0.45, 0.20], [0.33, 0.60, 0.20]],
            [[0.05, 0.45, 0.70], [0.33, 0.20, 0.10]],
        ],
        dtype=np.float32,
    )


def test_per_class_argmax_matches_legacy_prompt_formula_bitwise():
    probability = _probabilities()
    labels = probability.argmax(axis=0)
    expected_left = np.where(
        (labels == 1) & (probability[1] >= 0.4),
        probability[1],
        0.0,
    )
    expected_right = np.where(
        (labels == 2) & (probability[2] >= 0.4),
        probability[2],
        0.0,
    )

    left, right = build_hand_prompts(
        probability, 0.4, "per_class_argmax"
    )

    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)
    assert left.dtype == expected_left.dtype
    assert right.dtype == expected_right.dtype


def test_union_hand_probability_preserves_left_right_uncertainty():
    probability = _probabilities()

    union, unused = build_hand_prompts(
        probability, 0.45, "union_hand_probability"
    )

    expected = probability[1] + probability[2]
    expected = np.where(expected >= 0.45, expected, 0.0)
    np.testing.assert_array_equal(union, expected)
    np.testing.assert_array_equal(unused, np.zeros_like(expected))
    assert union[0, 1] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("probability", "threshold", "mode", "error"),
    [
        (
            np.zeros((2, 4, 4), dtype=np.float32),
            0.5,
            "per_class_argmax",
            "shape",
        ),
        (
            np.zeros((3, 4, 4), dtype=np.int32),
            0.5,
            "per_class_argmax",
            "floating-point",
        ),
        (
            np.zeros((3, 4, 4), dtype=np.float32),
            -0.1,
            "per_class_argmax",
            r"\[0, 1\]",
        ),
        (
            np.zeros((3, 4, 4), dtype=np.float32),
            0.5,
            "unknown",
            "prompt_mode",
        ),
    ],
)
def test_build_hand_prompts_rejects_invalid_inputs(
    probability, threshold, mode, error
):
    with pytest.raises((TypeError, ValueError), match=error):
        build_hand_prompts(probability, threshold, mode)
