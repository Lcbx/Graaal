import numpy as np
import pytest

from graaal.AnimationStateMachine import AnimationStateMachine
from graaal.SkeletalAnimation import Clip, Skeleton, sample_clip_pairs


def test_selection_priority_active_fades_and_parameter_changes():
	machine = AnimationStateMachine([(0, 2, 0, ">=", 3, 0.2), (0, 1, 0, ">", 0, 0.4), (1, 0, 0, "<=", 0, 0.3)])
	states = np.array([0, 0, 0, 1, 1], dtype=np.uint32)
	parameters = np.array([[3], [1], [4], [0], [2]])
	selected, targets, durations = machine.select(states, parameters, [False, False, True, False, False])
	np.testing.assert_array_equal(selected, [True, True, False, True, False])
	np.testing.assert_array_equal(targets, [2, 1, 0, 0, 1])
	np.testing.assert_allclose(durations, [0.2, 0.4, 0, 0.3, 0])
	np.testing.assert_array_equal(states, [0, 0, 0, 1, 1])
	selected, targets, _ = machine.select(np.array([1]), [[0]], [False])
	assert selected[0] and targets[0] == 0


def test_selected_state_can_drive_existing_clip_pair_sampler():
	machine = AnimationStateMachine([(0, 1, 0, ">", 0, 0.5)])
	selected, targets, durations = machine.select(np.array([0, 0]), [[1], [0]], [False, False])
	ids = np.column_stack((np.zeros(2, dtype=np.uint64), targets))
	weights = np.column_stack((1 - selected * 0.5, selected * 0.5))
	skeleton = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clips = {i: Clip([[[2 * i, 0, 0]]], [[[0, 0, 0, 1]]], [[[1, 1, 1]]]) for i in range(2)}
	pose = sample_clip_pairs(skeleton, clips, ids, np.zeros((2, 2)), weights)
	np.testing.assert_allclose(pose[0][:, 0, 0], [1, 0])
	np.testing.assert_allclose(durations, [0.5, 0])


def test_all_conditions_must_match_and_rules_keep_priority():
	machine = AnimationStateMachine([(0, 2, [(0, ">", 0.5), (1, "==", 1)], 0.2), (0, 1, 0, ">", 0.5, 0.4)])
	selected, targets, durations = machine.select(np.zeros(4, dtype=np.uint32), [[1, 1], [1, 0], [0, 1], [1, 1]], [False, False, False, True])
	np.testing.assert_array_equal(selected, [True, True, False, False])
	np.testing.assert_array_equal(targets, [2, 1, 0, 0])
	np.testing.assert_allclose(durations, [0.2, 0.4, 0, 0])
	with pytest.raises(ValueError, match="parameter column"): machine.select(np.array([0]), [[1]], [False])


@pytest.mark.parametrize("edge", [
	(0, 0, 0, ">", 0, 1), (-1, 1, 0, ">", 0, 1), (True, 1, 0, ">", 0, 1),
	(0, 1, 0, "?", 0, 1), (0, 1, 0, ">", np.nan, 1), (0, 1, 0, ">", 0, 0),
	(0, 1, 0, ">", 0, np.inf), (0, 2**64, 0, ">", 0, 1), (0,),
	(0, 1, [], 1), (0, 1, [(0, ">", 0), (1,)], 1), (0, 1, [(0, ">", 0), (1, "?", 0)], 1),
])
def test_rejects_invalid_edges(edge):
	with pytest.raises(ValueError): AnimationStateMachine([edge])


def test_validation_and_empty_instances():
	machine = AnimationStateMachine([(0, 1, 0, "==", 1, 1)])
	assert machine.select(np.array([], dtype=np.uint32), np.empty((0, 1)), np.array([], dtype=bool))[0].shape == (0,)
	for states, params, active in (([-1], [[1]], [False]), ([0], [[np.nan]], [False]), ([0], np.empty((1, 0)), [False]), ([0], [[1]], [0]), ([0], [[1], [2]], [False])):
		with pytest.raises(ValueError): machine.select(states, params, active)
