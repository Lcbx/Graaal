import numpy as np
import pytest

from graaal.Animation.AnimationGraphPlayback import AnimationGraphPlayback
from graaal.Animation.AnimationLod import DistancePoseLod
from tests.test_animation_graph_playback import blend_graph, grid_graph


def test_camera_sweep_holds_preference_between_thresholds_and_during_blends():
	lod = DistancePoseLod(8, 12)

	def sweep(distance, hold=False):
		return lod.update([7], [[0, 0, 0]], [0, 0, distance], hold=[hold]).tolist()

	assert sweep(10) == [False]
	assert sweep(12, hold=True) == [False]
	assert sweep(12) == [True]
	assert sweep(10) == [True]
	assert sweep(8, hold=True) == [True]
	assert sweep(8) == [False]
	assert sweep(10) == [False]


def test_lod_keeps_full_handle_history_through_reordering_and_forgets_removed_owners():
	lod = DistancePoseLod(8, 12)
	camera = [0, 0, 0]
	owners = np.array([3, 4, 5], dtype=np.uint64)
	first = lod.update(owners, [[12, 0, 0], [12, 0, 0], [0, 0, 0]], camera)
	np.testing.assert_array_equal(first, [True, True, False])
	first[:] = False
	replacement = (1 << 63) + 4
	result = lod.update([5, replacement, 3], [[10, 0, 0]] * 3, camera)
	np.testing.assert_array_equal(result, [False, False, True])
	lod.update([replacement, replacement + 1], [[12, 0, 0], [0, 0, 0]], camera)
	np.testing.assert_array_equal(lod.update([replacement + 1, replacement], [[10, 0, 0]] * 2, camera), [False, True])
	assert lod.update([], np.empty((0, 3)), camera).tolist() == []
	np.testing.assert_array_equal(lod.update([replacement], [[10, 0, 0]], camera), [False])
	np.testing.assert_array_equal(lod.update(owners, [[10, 0, 0]] * 3, camera), [False, False, False])


def test_lod_rejects_invalid_input_without_changing_history():
	with pytest.raises(ValueError, match="distances"): DistancePoseLod(8, 8)
	with pytest.raises(ValueError, match="distances"): DistancePoseLod(0, np.inf)
	lod = DistancePoseLod(8, 12)
	lod.update([7], [[12, 0, 0]], [0, 0, 0])
	for invalid in ([7.0], [True], [-1], [1 << 64], [[7]]):
		with pytest.raises(ValueError, match="integer IDs"): lod.update(invalid, [[0, 0, 0]], [0, 0, 0])
	with pytest.raises(ValueError, match="distinct"): lod.update([7, 7], [[0, 0, 0]] * 2, [0, 0, 0])
	with pytest.raises(ValueError, match="finite"): lod.update([7], [[0, np.nan, 0]], [0, 0, 0])
	with pytest.raises(ValueError, match="boolean"): lod.update([7], [[0, 0, 0]], [0, 0, 0], hold=[1])
	assert lod.update([7], [[10, 0, 0]], [0, 0, 0]).tolist() == [True]


def test_graph_blending_flags_distinguish_grid_edges_and_samples():
	playback = AnimationGraphPlayback(grid_graph(), 4)
	parameters = [[0, 0, -1], [0, 0.5, -1], [0, 1, 0], [0, 1, 1]]
	assert playback.blending(parameters).tolist() == [False, True, True, False]


def test_graph_blending_flags_hold_during_fades_and_interior_1d_weights():
	playback = AnimationGraphPlayback(blend_graph(), 3)
	parameters = [[1, 0], [1, 0.5], [0, 0.5]]
	playback.advance(0, parameters)
	assert playback.blending(parameters).tolist() == [True, True, False]
	playback.advance(1, parameters)
	assert playback.blending(parameters).tolist() == [False, True, False]
