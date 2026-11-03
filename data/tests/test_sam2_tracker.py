import numpy as np
import pytest

from hand_process.sam2_tracker import (
    Sam2Tracker,
    _validate_output_component_filter,
    filter_mask_components,
)


def test_conditioning_bounds_use_latest_prompt_for_reverse_tracking():
    nonempty = {1: [40, 100, 150], 2: [80, 140]}
    assert Sam2Tracker._conditioning_bounds(nonempty, [1, 2]) == (40, 150)


def test_valid_prompt_rejects_tiny_conditioning_noise():
    tracker = object.__new__(Sam2Tracker)
    tracker.min_prompt_area_ratio = 0.002
    tracker.max_prompt_area_ratio = 1.0
    tracker.max_prompt_components = 0
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[:2, :5] = 1.0
    assert not tracker._valid_prompt(probability)
    probability[:5, :5] = 1.0
    assert tracker._valid_prompt(probability)


def test_valid_prompt_rejects_huge_or_fragmented_conditioning():
    tracker = object.__new__(Sam2Tracker)
    tracker.min_prompt_area_ratio = 0.001
    tracker.max_prompt_area_ratio = 0.2
    tracker.max_prompt_components = 2

    huge = np.ones((100, 100), dtype=np.float32)
    assert not tracker._valid_prompt(huge)

    fragmented = np.zeros((100, 100), dtype=np.float32)
    fragmented[10:20, 10:20] = 1.0
    fragmented[40:50, 40:50] = 1.0
    fragmented[70:80, 70:80] = 1.0
    assert not tracker._valid_prompt(fragmented)


def test_merge_hands_uses_one_clipped_prompt_object():
    tracker = object.__new__(Sam2Tracker)
    tracker.merge_hands_for_tracking = True
    left = np.full((4, 4), 0.7, dtype=np.float32)
    right = np.full((4, 4), 0.6, dtype=np.float32)

    objects = tracker._prompt_objects(left, right)

    assert list(objects) == [1]
    assert np.all(objects[1] == 1.0)


def test_legacy_prompt_objects_keep_left_and_right_separate():
    tracker = object.__new__(Sam2Tracker)
    tracker.merge_hands_for_tracking = False
    left = np.ones((2, 2), dtype=np.float32)
    right = np.zeros((2, 2), dtype=np.float32)

    objects = tracker._prompt_objects(left, right)

    assert objects[1] is left
    assert objects[2] is right


def test_component_filter_defaults_preserve_baseline_object_bitwise():
    probability = np.random.default_rng(7).random((32, 48), dtype=np.float32)

    result = filter_mask_components(probability)

    assert result is probability
    np.testing.assert_array_equal(result, probability)


def test_component_filter_removes_only_small_foreground_islands():
    probability = np.full((100, 100), 0.1, dtype=np.float32)
    probability[20:40, 30:50] = 0.9
    probability[80:82, 90:92] = 0.95

    result = filter_mask_components(
        probability,
        min_area_ratio=0.001,
    )

    np.testing.assert_array_equal(result[20:40, 30:50], 0.9)
    np.testing.assert_array_equal(result[80:82, 90:92], 0.0)
    # Sub-threshold probability is untouched, preserving the soft mask outside
    # the removed foreground component.
    assert result[0, 0] == pytest.approx(0.1)


def test_component_ranking_treats_top_and_bottom_edges_symmetrically():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[0:5, 10:20] = 0.9
    probability[95:100, 80:90] = 0.9
    probability[45:55, 45:55] = 0.9

    result = filter_mask_components(
        probability,
        max_components=2,
        edge_margin_ratio=0.1,
        edge_score_bonus=1.0,
    )

    assert np.all(result[0:5, 10:20] == 0.9)
    assert np.all(result[95:100, 80:90] == 0.9)
    assert np.all(result[45:55, 45:55] == 0.0)


def test_ego_filter_keeps_any_boundary_and_removes_interior_hand():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[0:8, 10:20] = 0.9
    probability[40:50, 0:8] = 0.9
    probability[90:100, 30:40] = 0.9
    probability[60:70, 92:100] = 0.9
    probability[45:55, 45:55] = 0.95

    result = filter_mask_components(
        probability,
        prefer_edge_connected=True,
    )

    assert np.all(result[0:8, 10:20] == 0.9)
    assert np.all(result[40:50, 0:8] == 0.9)
    assert np.all(result[90:100, 30:40] == 0.9)
    assert np.all(result[60:70, 92:100] == 0.9)
    assert np.all(result[45:55, 45:55] == 0.0)


def test_ego_filter_falls_back_without_edge_connected_component():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[20:35, 20:35] = 0.9
    probability[60:80, 60:80] = 0.95

    result = filter_mask_components(
        probability,
        prefer_edge_connected=True,
        edge_margin_ratio=0.05,
    )

    np.testing.assert_array_equal(result, probability)


def test_tiny_edge_noise_does_not_override_safe_ego_fallback():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[0:2, 0:2] = 0.99
    probability[40:60, 40:60] = 0.9

    result = filter_mask_components(
        probability,
        min_area_ratio=0.001,
        prefer_edge_connected=True,
        edge_margin_ratio=0.05,
    )

    assert np.all(result[0:2, 0:2] == 0.0)
    assert np.all(result[40:60, 40:60] == 0.9)


def test_center_distance_is_only_a_soft_component_ranking_term():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[45:55, 45:55] = 0.9
    probability[10:20, 10:20] = 0.9

    result = filter_mask_components(
        probability,
        max_components=1,
        center_distance_score_weight=0.2,
    )

    assert np.all(result[10:20, 10:20] == 0.9)
    assert np.all(result[45:55, 45:55] == 0.0)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"enabled": "yes"}, "enabled"),
        ({"prefer_edge_connected": "yes"}, "prefer_edge_connected"),
        ({"min_area_ratio": 1.0}, "min_area_ratio"),
        ({"edge_margin_ratio": 0.6}, "edge_margin_ratio"),
        ({"max_components": -1}, "max_components"),
        ({"bottom_edge_only": True}, "unknown"),
    ],
)
def test_component_filter_rejects_invalid_settings(config, error):
    with pytest.raises(ValueError, match=error):
        _validate_output_component_filter(config)
