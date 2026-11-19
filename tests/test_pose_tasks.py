import numpy as np
import pytest

from graaal.PoseTasks import PoseOp, PosePlan
from graaal.RootMotion import RootMotion
from graaal.SkeletalAnimation import Clip, Skeleton, additive_pose, blend_poses, model_matrices, sample_clip_pairs, skin_matrices


def resources():
	skeleton = Skeleton([-1, 0], np.zeros((2, 3)), np.tile([0, 0, 0, 1], (2, 1)), np.ones((2, 3)), np.tile(np.eye(4), (2, 1, 1)))
	clips = {i: Clip(np.full((1, 2, 3), i), np.tile([0, 0, 0, 1], (1, 2, 1)), np.ones((1, 2, 3))) for i in range(3)}
	return skeleton, clips


def sample(first, second, count=2):
	return (np.tile([first, second], (count, 1)), np.zeros((count, 2)), np.tile([0.25, 0.75], (count, 1)))


def test_compiled_plan_matches_existing_layer_operations_and_skinning():
	skeleton, clips = resources()
	mask = np.array([0, 1], dtype=np.float32)
	plan = PosePlan(skeleton, [("sample", 0), ("sample", 1), ("blend", 0, 1, 0, None), ("additive", 2, 1, 1, mask)])
	mask[:] = 0
	samples = [sample(0, 1), sample(1, 2)]
	weights = np.array([[0.5, 1], [1, 0.25]])
	a, b = (sample_clip_pairs(skeleton, clips, *item) for item in samples)
	expected = additive_pose(skeleton, blend_poses(skeleton, a, b, weights[:, 0]), b, weights[:, 1], mask=[0, 1])
	actual = plan.evaluate(clips, samples, weights)
	for result, reference in zip(actual, expected): np.testing.assert_allclose(result, reference)
	np.testing.assert_allclose(skin_matrices(skeleton, model_matrices(skeleton, *actual)), skin_matrices(skeleton, model_matrices(skeleton, *expected)))


def test_enum_and_string_operations_produce_identical_poses():
	skeleton, clips = resources()
	string_plan = PosePlan(skeleton, [("sample", 0), ("sample", 1), ("blend", 0, 1, 0, None)])
	enum_plan = PosePlan(skeleton, [(PoseOp.SAMPLE, 0), (PoseOp.SAMPLE, 1), (PoseOp.BLEND, 0, 1, 0, None)])
	samples, weights = [sample(0, 1), sample(1, 2)], np.array([[0.25], [0.75]])
	for actual, expected in zip(enum_plan.evaluate(clips, samples, weights), string_plan.evaluate(clips, samples, weights)):
		np.testing.assert_allclose(actual, expected)


def test_plan_reuses_buffers_and_preserves_shared_dependencies():
	skeleton, clips = resources()
	plan = PosePlan(skeleton, [("sample", 0), ("sample", 1), ("blend", 0, 1, 0, None), ("blend", 0, 2, 0, None)])
	samples = [sample(0, 0), sample(2, 2)]
	first = plan.evaluate(clips, samples, np.full((2, 1), 0.5))
	np.testing.assert_allclose(first[0], 0.5)
	second = plan.evaluate(clips, samples, np.ones((2, 1)))
	assert np.shares_memory(first[0], second[0])
	np.testing.assert_allclose(second[0], 2)
	assert plan.evaluate(clips, [sample(0, 0, 0), sample(2, 2, 0)], np.empty((0, 1)))[0].shape == (0, 2, 3)
	assert plan.evaluate(clips, [sample(0, 0, 4), sample(2, 2, 4)], np.ones((4, 1)))[0].shape == (4, 2, 3)


@pytest.mark.parametrize("tasks", [[], [("sample", -1)], [("sample", True)], [("blend", 0, 0, 0, None)], [("sample", 0), ("blend", 0, 2, 0, None)], [("sample", 0), ("additive", 0, 0, 0, [2, 0])], [("other", 0)]])
def test_plan_rejects_invalid_tasks(tasks):
	with pytest.raises(ValueError): PosePlan(resources()[0], tasks)


def test_plan_rejects_invalid_runtime_inputs():
	skeleton, clips = resources()
	plan = PosePlan(skeleton, [("sample", 0), ("blend", 0, 0, 0, None)])
	for samples, weights in (([], [[1]]), ([sample(0, 1)], [[1]]), ([sample(0, 1)], np.empty((2, 0))), ([sample(0, 1)], [[2], [0]])):
		with pytest.raises(ValueError): plan.evaluate(clips, samples, weights)


def test_in_place_task_masks_instances_and_preserves_shared_source():
	skeleton, clips = resources()
	plan = PosePlan(skeleton, [("sample", 0), ("in_place", 0, 0), ("blend", 0, 1, 0, None)])
	motion = RootMotion(skeleton, clips[2])
	args = [sample(2, 2)]
	pose = plan.evaluate(clips, args, np.ones((2, 1)), root_motions=[(motion, np.array([True, False]))])
	np.testing.assert_allclose(pose[0][:, 0, 0], [0, 2])
	np.testing.assert_allclose(pose[0][:, 1, 0], [2, 2])
	pose = plan.evaluate(clips, args, np.zeros((2, 1)), root_motions=[(motion, np.array([True, False]))])
	np.testing.assert_allclose(pose[0][:, 0, 0], [2, 2])
	pose = plan.evaluate(clips, args, np.ones((2, 1)), root_motions=[(None, np.array([False, False]))])
	np.testing.assert_allclose(pose[0][:, 0, 0], [2, 2])


def test_in_place_task_validates_dependencies_and_runtime_slots():
	skeleton, clips = resources()
	with pytest.raises(ValueError): PosePlan(skeleton, [("in_place", 0, 0)])
	plan = PosePlan(skeleton, [("sample", 0), ("in_place", 0, 0)])
	args = [sample(0, 0)]
	for slots in ([], [(None, [True, False])], [(None, [0, 0])], [(RootMotion(resources()[0], clips[0]), [True, True])]):
		with pytest.raises(ValueError): plan.evaluate(clips, args, np.empty((2, 0)), root_motions=slots)


def test_post_physics_pose_plan_applies_ik_and_preserves_input_pose():
	skeleton = Skeleton([-1, 0, 1], [[0, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 0, 0, 1]] * 3, np.ones((3, 3)), np.tile(np.eye(4), (3, 1, 1)))
	clip = Clip(skeleton.translation[None], skeleton.rotation[None], skeleton.scale[None])
	pre = PosePlan(skeleton, [("sample", 0)])
	post = PosePlan(skeleton, [("input", 0), ("two_bone_ik", 0, (0, 1, 2), 0, 0)])
	base = pre.evaluate({0: clip}, [sample(0, 0)], np.empty((2, 0)))
	before = tuple(value.copy() for value in base)
	goals = [(np.array([[1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 0], [0, 0, 1]]))]
	result = post.evaluate({}, [], np.array([[1], [0]]), poses=[base], goals=goals)
	matrices = skin_matrices(skeleton, model_matrices(skeleton, *result))
	np.testing.assert_allclose(matrices[:, 2, :, 3], [[1, 0, 0], [2, 0, 0]], atol=1e-6)
	for value, expected in zip(base, before): np.testing.assert_array_equal(value, expected)
	# A prior result may be supplied back as input without workspace corruption.
	result = post.evaluate({}, [], np.ones((2, 1)), poses=[result], goals=goals)
	np.testing.assert_allclose(model_matrices(skeleton, *result)[:, 2, :3, 3], goals[0][0], atol=1e-6)


def test_ik_plan_validates_inputs_and_chain():
	skeleton = Skeleton([-1, 0, 1], [[0, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 0, 0, 1]] * 3, np.ones((3, 3)), np.tile(np.eye(4), (3, 1, 1)))
	with pytest.raises(ValueError): PosePlan(skeleton, [("input", 0), ("two_bone_ik", 0, (0, 2, 1), 0, 0)])
	with pytest.raises(ValueError): PosePlan(skeleton, [("two_bone_ik", 0, (0, 1, 2), 0, 0)])
	plan = PosePlan(skeleton, [("input", 0), ("two_bone_ik", 0, (0, 1, 2), 0, 0)])
	pose = tuple(value[None] for value in (skeleton.translation, skeleton.rotation, skeleton.scale))
	for poses, goals, weights in (([], [], [[1]]), ([pose], [], [[1]]), ([pose], [(np.zeros((1, 3)), np.zeros((2, 3)))], [[1]]), ([pose], [(np.zeros((1, 3)), np.zeros((1, 3)))], np.empty((1, 0)))):
		with pytest.raises(ValueError): plan.evaluate({}, [], weights, poses=poses, goals=goals)


def test_align_axis_task_copies_axis_and_validates_goal_slot():
	skeleton = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	axis = np.array([0, 1, 0])
	plan = PosePlan(skeleton, [(PoseOp.INPUT, 0), (PoseOp.ALIGN_AXIS, 0, 0, 0, 0, axis)])
	axis[:] = 0
	pose = tuple(value[None] for value in (skeleton.translation, skeleton.rotation, skeleton.scale))
	result = plan.evaluate({}, [], [[1]], poses=[pose], goals=[(np.array([[0, 0, 1]]),)])
	np.testing.assert_allclose(model_matrices(skeleton, *result)[0, 0, :3, 1], [0, 0, 1], atol=1e-6)
	with pytest.raises(ValueError): plan.evaluate({}, [], [[1]], poses=[pose], goals=[(np.zeros((1, 3)), np.zeros((1, 3)))])
	with pytest.raises(ValueError): PosePlan(skeleton, [(PoseOp.INPUT, 0), (PoseOp.ALIGN_AXIS, 0, 0, 0, 0, [0, 0, 0])])
