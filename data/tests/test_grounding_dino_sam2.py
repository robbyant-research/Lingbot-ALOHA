import numpy as np
import pytest

from hand_process.grounding_dino_sam2 import (
    GroundingDinoSam2Segmenter,
    class_agnostic_nms,
    expand_and_clip_boxes,
    normalize_grounding_prompt,
    union_sam_masks,
)


def test_prompt_normalization_uses_grounding_dino_phrase_format():
    assert normalize_grounding_prompt(
        [" human hand ", "gloved hand.", "forearm"]
    ) == "human hand. gloved hand. forearm."
    assert normalize_grounding_prompt("hand") == "hand."
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_grounding_prompt([])


def test_class_agnostic_nms_suppresses_overlap_and_limits_results():
    boxes = np.asarray(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [30, 30, 40, 40],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    assert class_agnostic_nms(boxes, scores, 0.5).tolist() == [0, 2]
    assert class_agnostic_nms(
        boxes, scores, 0.5, max_detections=1
    ).tolist() == [0]


def test_box_expansion_clips_to_image_and_preserves_empty_shape():
    boxes = np.asarray(
        [[2, 3, 12, 13], [95, 75, 99, 79]], dtype=np.float32
    )
    expanded = expand_and_clip_boxes(
        boxes,
        width=100,
        height=80,
        expansion_ratio=0.2,
        min_expansion_px=3,
    )

    np.testing.assert_allclose(expanded[0], [0, 0, 15, 16])
    np.testing.assert_allclose(expanded[1], [92, 72, 99, 79])
    assert expand_and_clip_boxes(
        np.empty((0, 4), dtype=np.float32),
        width=100,
        height=80,
        expansion_ratio=0.2,
    ).shape == (0, 4)


def test_sam_masks_are_returned_as_union_and_zero_compatible_pair():
    first = np.zeros((8, 10), dtype=bool)
    second = np.zeros((1, 8, 10), dtype=bool)
    first[1:3, 2:4] = True
    second[:, 5:7, 6:9] = True

    union = union_sam_masks([first, second], height=8, width=10)
    zero = np.zeros_like(union)

    assert union.dtype == np.float32
    assert int(union.sum()) == 10
    assert not zero.any()


def test_sam_masks_reject_implausibly_large_regions():
    hand = np.zeros((10, 10), dtype=bool)
    background = np.zeros((10, 10), dtype=bool)
    hand[6:10, 2:5] = True
    background[4:10, :] = True

    union = union_sam_masks(
        [hand, background],
        height=10,
        width=10,
        max_area_ratio=0.25,
    )

    np.testing.assert_array_equal(union > 0, hand)
    with pytest.raises(ValueError, match="max_area_ratio"):
        union_sam_masks(
            [hand],
            height=10,
            width=10,
            max_area_ratio=1.1,
        )


def test_grounding_dino_requires_explicit_non_ego_acknowledgement():
    with pytest.raises(ValueError, match="every matching hand"):
        GroundingDinoSam2Segmenter({})
