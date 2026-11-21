import numpy as np
import pytest

from graaal.Animation.SkeletalAnimation import Clip, Skeleton, blend_space_1d, blend_space_2d, model_matrices, sample_clip_pairs, skin_matrices
from graaal.Animation.PoseTasks import PosePlan


def test_blend_space_selects_neighbors_clamps_and_preserves_order():
	ids, weights = blend_space_1d([0, 2, 6], [4, 7, 9], [8, 1, -2, 2, 4, 6])
	np.testing.assert_array_equal(ids, [[7, 9], [4, 7], [4, 7], [7, 9], [7, 9], [7, 9]])
	np.testing.assert_allclose(weights, [[0, 1], [0.5, 0.5], [1, 0], [1, 0], [0.5, 0.5], [0, 1]])
	assert weights.dtype == np.float32


def test_blend_space_single_point_empty_batch_and_extreme_positions():
	ids, weights = blend_space_1d([5], [3], [-1, 5, 8])
	np.testing.assert_array_equal(ids, [[3, 3]] * 3)
	np.testing.assert_array_equal(weights, [[1, 0]] * 3)
	assert blend_space_1d([1, 2], [0, 1], [])[0].shape == (0, 2)
	_, weights = blend_space_1d([-1e308, 1e308], [0, 1], [0])
	np.testing.assert_allclose(weights, [[0.5, 0.5]])


@pytest.mark.parametrize("positions, ids, values", [
	([], [], [0]), ([1, 1], [0, 1], [0]), ([2, 1], [0, 1], [0]),
	([np.nan], [0], [0]), ([0], [-1], [0]), ([0], [0.0], [0]),
	([0, 1], [0], [0]), ([0], [0], [np.inf]), ([0], [0], 0),
])
def test_blend_space_rejects_invalid_inputs(positions, ids, values):
	with pytest.raises(ValueError): blend_space_1d(positions, ids, values)


def test_blend_space_samples_feed_existing_skin_matrices():
	skeleton = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clips = {i: Clip([[[x, 0, 0]]], [[[0, 0, 0, 1]]], [[[1, 1, 1]]]) for i, x in enumerate((0, 2, 6))}
	ids, weights = blend_space_1d([0, 2, 6], [0, 1, 2], [1, 4, 10])
	pose = sample_clip_pairs(skeleton, clips, ids, np.zeros((3, 2)), weights)
	matrices = skin_matrices(skeleton, model_matrices(skeleton, *pose))
	np.testing.assert_allclose(matrices[:, 0, 0, 3], [1, 4, 6])


def test_2d_blend_space_feeds_pose_plan_and_skin_matrices():
	skeleton = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clips = {i: Clip([[[x, 0, 0]]], [[[0, 0, 0, 1]]], [[[1, 1, 1]]]) for i, x in enumerate((0, 2, 10, 12))}
	ids, horizontal, vertical = blend_space_2d([0, 2], [0, 10], [[0, 1], [2, 3]], [[1, 5], [-1, 20], [2, 0]])
	plan = PosePlan(skeleton, [("sample", 0), ("sample", 1), ("blend", 0, 1, 0, None)])
	samples = [(ids[:, row], np.zeros((3, 2)), horizontal) for row in range(2)]
	pose = plan.evaluate(clips, samples, vertical[:, None])
	matrices = skin_matrices(skeleton, model_matrices(skeleton, *pose))
	np.testing.assert_allclose(matrices[:, 0, 0, 3], [6, 10, 2])


def test_2d_blend_space_degenerate_axes_and_empty_batch():
	ids, horizontal, vertical = blend_space_2d([3], [1], [[7]], [[-1, 5]])
	np.testing.assert_array_equal(ids, [[[7, 7], [7, 7]]])
	np.testing.assert_array_equal(horizontal, [[1, 0]])
	np.testing.assert_array_equal(vertical, [0])
	assert blend_space_2d([0], [0], [[1]], np.empty((0, 2)))[0].shape == (0, 2, 2)


@pytest.mark.parametrize("x, y, grid, values", [
	([], [0], [[]], [[0, 0]]), ([0, 0], [0], [[0, 1]], [[0, 0]]),
	([0], [0], [[-1]], [[0, 0]]), ([0], [0], [[1.0]], [[0, 0]]),
	([0], [0], [1], [[0, 0]]), ([0], [0], [[1]], [0, 0]),
	([0], [0], [[1]], [[0, np.nan]]),
])
def test_2d_blend_space_rejects_invalid_inputs(x, y, grid, values):
	with pytest.raises(ValueError): blend_space_2d(x, y, grid, values)
