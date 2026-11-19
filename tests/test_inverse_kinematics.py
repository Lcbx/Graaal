import numpy as np
import pytest

from graaal.InverseKinematics import align_bone_axis, solve_two_bone, two_bone_pose
from graaal.Math import Quaternion
from graaal.SkeletalAnimation import Skeleton, model_matrices, skin_matrices


def test_batch_reaches_targets_and_pole_selects_bend_side():
	root = np.zeros((3, 3))
	joint = np.tile([1, 0, 0], (3, 1))
	end = np.tile([2, 0, 0], (3, 1))
	target = np.tile([1, 0, 0], (3, 1))
	pole = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 1]])
	solved_joint, solved_end = solve_two_bone(root, joint, end, target, pole)
	np.testing.assert_allclose(solved_end, target)
	np.testing.assert_allclose(np.linalg.norm(solved_joint - root, axis=1), 1)
	np.testing.assert_allclose(np.linalg.norm(solved_end - solved_joint, axis=1), 1)
	np.testing.assert_allclose(solved_joint, [[0.5, np.sqrt(0.75), 0], [0.5, -np.sqrt(0.75), 0], [0.5, 0, np.sqrt(0.75)]])
	np.testing.assert_array_equal(joint, np.tile([1, 0, 0], (3, 1)))


def test_unreachable_targets_and_collapsed_goal_preserve_lengths():
	root = np.zeros((3, 3))
	joint = np.tile([2, 0, 0], (3, 1))
	end = np.tile([3, 0, 0], (3, 1))
	target = np.array([[5, 0, 0], [0.1, 0, 0], [0, 0, 0]])
	solved_joint, solved_end = solve_two_bone(root, joint, end, target, np.tile([0, 1, 0], (3, 1)))
	np.testing.assert_allclose(solved_end, [[3, 0, 0], [1, 0, 0], [1, 0, 0]])
	np.testing.assert_allclose(np.linalg.norm(solved_joint, axis=1), 2)
	np.testing.assert_allclose(np.linalg.norm(solved_end - solved_joint, axis=1), 1)
	joint, end = solve_two_bone([[0, 0, 0]], [[1, 0, 0]], [[2, 0, 0]], [[0, 0, 0]], [[0, 1, 0]])
	np.testing.assert_allclose(end, [[0, 0, 0]])
	np.testing.assert_allclose(np.linalg.norm(joint, axis=1), [1])


def test_collinear_pole_uses_existing_bend_or_deterministic_fallback():
	root = np.zeros((2, 3))
	joint = np.array([[0.5, -np.sqrt(0.75), 0], [1, 0, 0]])
	end = np.array([[1, 0, 0], [2, 0, 0]])
	solved_joint, solved_end = solve_two_bone(root, joint, end, np.tile([1, 0, 0], (2, 1)), np.tile([2, 0, 0], (2, 1)))
	assert solved_joint[0, 1] < 0
	assert solved_joint[1, 1] > 0
	np.testing.assert_allclose(np.linalg.norm(solved_end - solved_joint, axis=1), 1)


def test_random_chains_preserve_lengths_and_reach_clamped_distance():
	rng = np.random.default_rng(147)
	root = rng.normal(size=(500, 3))
	joint = root + rng.normal(size=(500, 3))
	end = joint + rng.normal(size=(500, 3))
	target = root + rng.normal(size=(500, 3)) * 4
	pole = root + rng.normal(size=(500, 3))
	solved_joint, solved_end = solve_two_bone(root, joint, end, target, pole)
	first_length = np.linalg.norm(joint - root, axis=1)
	second_length = np.linalg.norm(end - joint, axis=1)
	clamped = np.clip(np.linalg.norm(target - root, axis=1), np.abs(first_length - second_length), first_length + second_length)
	np.testing.assert_allclose(np.linalg.norm(solved_joint - root, axis=1), first_length, atol=1e-12)
	np.testing.assert_allclose(np.linalg.norm(solved_end - solved_joint, axis=1), second_length, atol=1e-12)
	np.testing.assert_allclose(np.linalg.norm(solved_end - root, axis=1), clamped, atol=1e-12)


def test_invalid_inputs_and_empty_batch():
	root = np.zeros((1, 3))
	joint = np.array([[1, 0, 0]])
	end = np.array([[2, 0, 0]])
	for args in ((root, root, end, end, joint), (root, joint, joint, end, joint), (root, joint, end, [[np.nan, 0, 0]], joint), (root[0], joint, end, end, joint), (root, joint, end, end, [[1e308, 0, 0]])):
		with pytest.raises(ValueError): solve_two_bone(*args)
	assert solve_two_bone(*(np.empty((0, 3)) for _ in range(5)))[0].shape == (0, 3)


def test_pose_ik_reaches_goals_with_rotated_parent_and_preserves_other_bones():
	parents = [-1, 0, 1, 2, 0]
	translation = np.array([[2, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, -1, 0]], dtype=np.float32)
	rotations = np.tile([0, 0, 0, 1], (5, 1)).astype(np.float32)
	rotations[0] = Quaternion.from_z_rotation(np.pi / 2)
	scales = np.ones((5, 3), dtype=np.float32)
	scales[0] = 2
	skeleton = Skeleton(parents, translation, rotations, scales, np.tile(np.eye(4), (5, 1, 1)))
	pose = (np.tile(translation, (2, 1, 1)), np.tile(rotations, (2, 1, 1)), np.tile(scales, (2, 1, 1)))
	before = tuple(value.copy() for value in pose)
	model = model_matrices(skeleton, *pose)
	root = model[:, 1, :3, 3]
	targets = root + np.array([[1, 1, 0], [1, -1, 0]])
	poles = root + np.array([[0, 0, 2], [0, 0, -2]])
	result = two_bone_pose(skeleton, pose, (1, 2, 3), targets, poles)
	actual = model_matrices(skeleton, *result)
	np.testing.assert_allclose(actual[:, 3, :3, 3], targets, atol=1e-5)
	np.testing.assert_allclose(actual[:, 4], model[:, 4], atol=1e-6)
	assert actual[0, 2, 2, 3] > 0 and actual[1, 2, 2, 3] < 0
	np.testing.assert_allclose(skin_matrices(skeleton, actual)[:, 3, :3, 3], targets, atol=1e-5)
	for value, original in zip(pose, before): np.testing.assert_array_equal(value, original)


def test_pose_ik_weights_unreachable_goals_and_invalid_chains():
	skeleton = Skeleton([-1, 0, 1], [[0, 0, 0], [1, 0, 0], [1, 0, 0]], [[0, 0, 0, 1]] * 3, np.ones((3, 3)), np.tile(np.eye(4), (3, 1, 1)))
	pose = (np.tile(skeleton.translation, (2, 1, 1)), np.tile(skeleton.rotation, (2, 1, 1)), np.tile(skeleton.scale, (2, 1, 1)))
	result = two_bone_pose(skeleton, pose, (0, 1, 2), [[0, 5, 0], [0, 5, 0]], [[0, 0, 1], [0, 0, 1]], weight=[0, 1])
	model = model_matrices(skeleton, *result)
	np.testing.assert_allclose(model[:, 2, :3, 3], [[2, 0, 0], [0, 2, 0]], atol=1e-6)
	for chain in ((0, 2, 1), (0, 1, 5), (True, 1, 2)):
		with pytest.raises(ValueError): two_bone_pose(skeleton, pose, chain, np.zeros((2, 3)), np.ones((2, 3)))
	bad_scale = pose[2].copy()
	bad_scale[0, 0] = [1, 2, 1]
	with pytest.raises(ValueError, match="uniform"): two_bone_pose(skeleton, (pose[0], pose[1], bad_scale), (0, 1, 2), np.zeros((2, 3)), np.ones((2, 3)))


def test_pose_ik_random_rotations_match_geometric_solution():
	rng = np.random.default_rng(1234)
	skeleton = Skeleton([-1, 0, 1], [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0, 1]] * 3, np.ones((3, 3)), np.tile(np.eye(4), (3, 1, 1)))
	count = 100
	translations = np.tile(skeleton.translation, (count, 1, 1))
	rotations = rng.normal(size=(count, 3, 4)).astype(np.float32)
	rotations /= np.linalg.norm(rotations, axis=2, keepdims=True)
	scales = np.tile(skeleton.scale, (count, 1, 1))
	pose = (translations, rotations, scales)
	before = model_matrices(skeleton, *pose)
	targets, poles = rng.normal(size=(2, count, 3))
	expected_joint, expected_end = solve_two_bone(before[:, 0, :3, 3], before[:, 1, :3, 3], before[:, 2, :3, 3], targets, poles)
	after = model_matrices(skeleton, *two_bone_pose(skeleton, pose, (0, 1, 2), targets, poles))
	np.testing.assert_allclose(after[:, 1, :3, 3], expected_joint, atol=2e-6)
	np.testing.assert_allclose(after[:, 2, :3, 3], expected_end, atol=2e-6)
	empty = (translations[:0], rotations[:0], scales[:0])
	assert two_bone_pose(skeleton, empty, (0, 1, 2), targets[:0], poles[:0])[0].shape == (0, 3, 3)


def test_align_bone_axis_handles_opposite_direction_and_weight():
	rotation = np.tile([0, 0, 0, 1], (3, 1)).astype(np.float32)
	rotation[0] = Quaternion.from_z_rotation(0.4)
	skeleton = Skeleton([-1, 0, 0], [[0, 0, 0], [1, 0, 0], [-1, 0, 0]], rotation, np.ones((3, 3)), np.tile(np.eye(4), (3, 1, 1)))
	pose = tuple(np.tile(value, (3, 1, 1)) for value in (skeleton.translation, skeleton.rotation, skeleton.scale))
	before = model_matrices(skeleton, *pose)
	directions = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]])
	result = align_bone_axis(skeleton, pose, 1, [0, 1, 0], directions, weight=[1, 1, 0])
	after = model_matrices(skeleton, *result)
	np.testing.assert_allclose(after[:2, 1, :3, 1], directions[:2], atol=1e-6)
	np.testing.assert_allclose(after[2, 1], before[2, 1], atol=1e-6)
	np.testing.assert_allclose(after[:, 1, :3, 3], before[:, 1, :3, 3], atol=1e-6)
	np.testing.assert_allclose(after[:, 2], before[:, 2], atol=1e-6)
	for axis, normal in (([0, 0, 0], directions), ([0, 1, 0], np.zeros((3, 3)))):
		with pytest.raises(ValueError): align_bone_axis(skeleton, pose, 1, axis, normal)
	bad_scale = pose[2].copy()
	bad_scale[0, 1] = [1, 2, 1]
	with pytest.raises(ValueError, match="uniform"): align_bone_axis(skeleton, (pose[0], pose[1], bad_scale), 1, [0, 1, 0], directions)
