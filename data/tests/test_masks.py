import numpy as np

from hand_process.masks import postprocess_probability, temporal_iou


CONFIG = {
    "min_component_area_ratio": 0.002,
    "dilation_ratio": 0.05,
    "dilation_min_px": 1,
    "dilation_max_px": 3,
    "feather_ratio": 0.02,
    "feather_min_px": 1,
    "feather_max_px": 2,
    "hard_alpha_threshold": 0.02,
}


def test_postprocess_removes_tiny_component_and_feathers():
    probability = np.zeros((32, 32), dtype=np.float32)
    probability[8:20, 8:20] = 0.9
    probability[1, 1] = 0.9
    hard, alpha, stats = postprocess_probability(probability, CONFIG)
    assert hard.dtype == np.uint8
    assert alpha.dtype == np.float32
    assert hard[1, 1] == 0
    assert hard[10, 10] == 255
    assert stats.num_components == 1
    assert 0 < stats.final_area_ratio < 1


def test_postprocess_preserves_component_entering_frame_edge():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[28:36, :44] = 0.9
    probability[22:42, 38:58] = 0.9
    hard, _alpha, _stats = postprocess_probability(probability, CONFIG)
    assert hard[32, 2] == 255
    assert hard[32, 50] == 255


def test_postprocess_preserves_internal_component():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[20:44, 20:44] = 0.9
    hard, _alpha, _stats = postprocess_probability(probability, CONFIG)
    assert hard[32, 32] == 255


def test_temporal_iou_handles_empty_and_overlap():
    empty = np.zeros((4, 4), dtype=np.uint8)
    assert temporal_iou(empty, empty) is None
    first = empty.copy()
    second = empty.copy()
    first[:2, :2] = 255
    second[1:3, 1:3] = 255
    assert temporal_iou(first, second) == 1 / 7
