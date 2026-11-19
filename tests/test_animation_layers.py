import numpy as np
import pytest

from graaal.SkeletalAnimation import Skeleton, additive_pose, blend_poses, model_matrices, subtree_mask


def rig():
	parents = [-1, 0, 1, 0, -1, 4]
	return Skeleton(parents, np.zeros((6, 3)), np.tile([0, 0, 0, 1], (6, 1)), np.ones((6, 3)), np.tile(np.eye(4), (6, 1, 1)))


def test_subtree_mask_respects_siblings_and_other_roots():
	skeleton = rig()
	np.testing.assert_array_equal(subtree_mask(skeleton, 1), [0, 1, 1, 0, 0, 0])
	np.testing.assert_array_equal(subtree_mask(skeleton, 0), [1, 1, 1, 1, 0, 0])
	np.testing.assert_array_equal(subtree_mask(skeleton, 4), [0, 0, 0, 0, 1, 1])
	for root in (-1, 6, True, 1.5):
		with pytest.raises(ValueError, match="root"): subtree_mask(skeleton, root)


def test_masked_blend_preserves_other_bones_and_uses_short_rotation_path():
	skeleton = rig()
	base = (np.zeros((2, 6, 3), np.float32), np.tile([0, 0, 0, 1], (2, 6, 1)), np.ones((2, 6, 3)))
	target = (np.full((2, 6, 3), 2), np.tile([0, 0, 0, -1], (2, 6, 1)), np.full((2, 6, 3), 3))
	result = blend_poses(skeleton, base, target, [1, 0.5], mask=subtree_mask(skeleton, 1))
	np.testing.assert_allclose(result[0][:, :, 0], [[0, 2, 2, 0, 0, 0], [0, 1, 1, 0, 0, 0]])
	np.testing.assert_allclose(result[2][:, :, 0], [[1, 3, 3, 1, 1, 1], [1, 2, 2, 1, 1, 1]])
	np.testing.assert_allclose(np.abs(result[1][..., 3]), 1)
	np.testing.assert_allclose(model_matrices(skeleton, *result)[0, 2, :3, 3], [8, 8, 8])
	np.testing.assert_array_equal(base[0], 0)


def test_masked_blend_soft_mask_and_invalid_inputs():
	skeleton = rig()
	pose = (np.zeros((1, 6, 3)), np.tile([0, 0, 0, 1], (1, 6, 1)), np.ones((1, 6, 3)))
	target = (np.ones((1, 6, 3)), pose[1], pose[2])
	result = blend_poses(skeleton, pose, target, 0.5, mask=[0, 0.5, 1, 0, 0, 0])
	np.testing.assert_allclose(result[0][0, :, 0], [0, 0.25, 0.5, 0, 0, 0])
	for weight, mask in ((-1, None), (np.nan, None), ([0, 1], None), (1, [1]), (1, [0, 0, 2, 0, 0, 0])):
		with pytest.raises(ValueError): blend_poses(skeleton, pose, target, weight, mask=mask)
	with pytest.raises(ValueError, match="zero quaternion"):
		blend_poses(skeleton, pose, (target[0], np.zeros((1, 6, 4)), target[2]), 0.5)


def test_additive_pose_composes_rotation_relative_to_nonidentity_reference():
	q = np.sqrt(0.5)
	skeleton = Skeleton([-1, 0], [[1, 0, 0], [0, 0, 0]], [[0, 0, q, q], [0, 0, 0, 1]], [[2, 1, 1], [1, 1, 1]], np.tile(np.eye(4), (2, 1, 1)))
	base = (np.tile([[3, 0, 0], [1, 0, 0]], (2, 1, 1)), np.tile([[0, q, 0, q], [0, 0, 0, 1]], (2, 1, 1)), np.tile([[3, 1, 1], [1, 1, 1]], (2, 1, 1)))
	overlay = (np.tile([[2, 0, 0], [5, 0, 0]], (2, 1, 1)), np.tile([[0.5, 0.5, 0.5, 0.5], [0, 0, 0, -1]], (2, 1, 1)), np.tile([[4, 1, 1], [4, 1, 1]], (2, 1, 1)))
	result = additive_pose(skeleton, base, overlay, [1, 0.5], mask=[1, 0])
	np.testing.assert_allclose(result[0][:, 0, 0], [4, 3.5])
	np.testing.assert_allclose(result[2][:, 0, 0], [5, 4])
	np.testing.assert_allclose(result[1][0, 0], [0.5, 0.5, -0.5, 0.5], atol=1e-6)
	a, b = np.sin(np.pi / 8) * q, np.cos(np.pi / 8) * q
	np.testing.assert_allclose(result[1][1, 0], [a, b, -a, b], atol=1e-6)
	np.testing.assert_allclose(result[0][:, 1], base[0][:, 1])
	np.testing.assert_allclose(result[1][:, 1], base[1][:, 1])
	np.testing.assert_allclose(base[0][:, 0, 0], 3)


def test_additive_reference_and_quaternion_sign_are_noops():
	skeleton = rig()
	base = (np.full((1, 6, 3), 2), np.tile([0, 0, 0, 1], (1, 6, 1)), np.full((1, 6, 3), 3))
	reference = (skeleton.translation[None], -skeleton.rotation[None], skeleton.scale[None])
	result = additive_pose(skeleton, base, reference, 1)
	for actual, expected in zip(result, base): np.testing.assert_allclose(actual, expected)
	for weight, mask in ((-1, None), (1, [0]), (np.nan, None)):
		with pytest.raises(ValueError): additive_pose(skeleton, base, reference, weight, mask=mask)


def test_batched_quaternion_operations_match_matrix_composition():
	from graaal.Math import Quaternion
	rng = np.random.default_rng(42)
	a = rng.normal(size=(3, 4, 4)).astype(np.float32)
	b = rng.normal(size=(3, 4, 4)).astype(np.float32)
	a /= np.linalg.norm(a, axis=-1, keepdims=True)
	b /= np.linalg.norm(b, axis=-1, keepdims=True)
	b[0] = -a[0]
	weight = rng.uniform(0, 1, size=(3, 4, 1)).astype(np.float32)
	product = Quaternion.multiply_batch(a, b)
	interpolated = Quaternion.nlerp_batch(a, b, weight)
	np.testing.assert_allclose(Quaternion.matrix_batch(product), Quaternion.matrix_batch(a) @ Quaternion.matrix_batch(b), atol=1e-6)
	np.testing.assert_allclose(np.linalg.norm(interpolated, axis=-1), 1, atol=1e-6)
	np.testing.assert_allclose(interpolated[0], a[0], atol=1e-6)


def test_crossfade_and_unmasked_pose_blend_agree():
	from graaal.SkeletalAnimation import Clip, sample_clip, sample_crossfade
	skeleton = rig()
	translation = np.zeros((2, 6, 3), dtype=np.float32)
	translation[1, :, 0] = 2
	rotation = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (2, 6, 1))
	rotation[1] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
	clip = Clip(translation, rotation, np.ones((2, 6, 3)), fps=1)
	base = sample_clip(clip, [0.1, 0.2], loop=False)
	overlay = sample_clip(clip, [0.7, 0.9], loop=False)
	actual = blend_poses(skeleton, base, overlay, [0.3, 0.8])
	expected = sample_crossfade(clip, clip, [0.1, 0.2], [0.7, 0.9], [0.3, 0.8], source_loop=False, target_loop=False)
	for a, b in zip(actual, expected): np.testing.assert_allclose(a, b, atol=1e-6)
