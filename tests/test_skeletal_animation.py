import numpy as np
import pytest

from graaal.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip, sample_crossfade, skin_matrices, skin_vertices


def skeleton(parents=(-1, 0, 1, 0)):
	bones = len(parents)
	return Skeleton(parents, np.zeros((bones, 3)), np.tile([0, 0, 0, 1], (bones, 1)), np.ones((bones, 3)), np.tile(np.eye(4), (bones, 1, 1)))


def test_skeleton_requires_depth_first_and_owns_reference_arrays():
	with pytest.raises(ValueError, match="depth-first"):
		skeleton((-1, 0, 0, 1))
	rig = skeleton()
	assert rig.depth.tolist() == [0, 1, 2, 1]
	assert not rig.translation.flags.writeable


def test_clip_sampling_loop_clamp_and_shortest_quaternion_path():
	translation = np.array([[[0, 0, 0]], [[2, 0, 0]], [[0, 0, 0]]], dtype=np.float32)
	rotation = np.array([[[0, 0, 0, 1]], [[0, 0, -1, 0]], [[0, 0, 0, -1]]], dtype=np.float32)
	clip = Clip(translation, rotation, np.ones_like(translation), fps=1)
	assert np.all(np.sum(clip.rotation[:-1] * clip.rotation[1:], axis=-1) >= 0)
	positions, rotations, scales = sample_clip(clip, [0.5, 2.5, -1.5])
	np.testing.assert_allclose(positions[:, 0, 0], [1, 1, 1])
	np.testing.assert_allclose(np.abs(rotations[:, 0, 2:]), np.sqrt(0.5), atol=1e-6)
	np.testing.assert_allclose(scales, 1)
	np.testing.assert_allclose(sample_clip(clip, [-1, 1, 3], loop=False)[0][:, 0, 0], [0, 2, 0])
	translation[:] = 9
	assert clip.translation[0, 0, 0] == 0


def test_crossfade_samples_independent_times_and_blends_local_poses():
	source = Clip([[[0, 0, 0]], [[2, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	target = Clip([[[4, 0, 0]], [[8, 0, 0]]], [[[0, 0, 0, -1]], [[0, 0, 0, -1]]], np.full((2, 1, 3), 3), fps=1)
	translation, rotation, scale = sample_crossfade(source, target, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0, 0.5, 1])
	np.testing.assert_allclose(translation[:, 0, 0], [1, 3.5, 6])
	np.testing.assert_allclose(np.abs(rotation[:, 0, 3]), 1)
	np.testing.assert_allclose(scale[:, 0, 0], [1, 2, 3])
	np.testing.assert_allclose(sample_crossfade(source, target, [2], [2], 0.5, source_loop=False, target_loop=False)[0][0, 0, 0], 5)
	assert sample_crossfade(source, target, [], [], 0.5)[1].shape == (0, 1, 4)


def test_crossfade_rejects_invalid_inputs():
	clip = Clip(np.zeros((1, 1, 3)), [[[0, 0, 0, 1]]], np.ones((1, 1, 3)))
	other = Clip(np.zeros((1, 2, 3)), [[[0, 0, 0, 1], [0, 0, 0, 1]]], np.ones((1, 2, 3)))
	with pytest.raises(ValueError, match="bone count"):
		sample_crossfade(clip, other, [0], [0], 0.5)
	with pytest.raises(ValueError, match="same length"):
		sample_crossfade(clip, clip, [0], [0, 0], 0.5)
	for weights in (-0.1, 1.1, np.nan, [0, 1]):
		with pytest.raises(ValueError, match="weights"):
			sample_crossfade(clip, clip, [0], [0], weights)
	with pytest.raises(ValueError, match="weights"):
		sample_crossfade(clip, clip, [], [], np.nan)


def test_hierarchy_and_skin_matrices_with_rotation_and_nonuniform_scale():
	rig = skeleton((-1, 0, 1, 0))
	positions = np.array([[[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
	rotations = np.tile([0, 0, 0, 1], (1, 4, 1)).astype(np.float32)
	rotations[0, 0] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
	scales = np.ones((1, 4, 3), dtype=np.float32)
	scales[0, 0] = [2, 1, 1]
	model = model_matrices(rig, positions, rotations, scales)
	np.testing.assert_allclose(model[0, :, :3, 3], [[1, 0, 0], [1, 2, 0], [1, 4, 0], [0, 0, 0]], atol=1e-6)
	skin = skin_matrices(rig, model)
	assert skin.shape == (1, 4, 3, 4) and skin.flags.c_contiguous
	np.testing.assert_allclose(skin, model[..., :3, :])


def test_single_frame_empty_batch_and_invalid_inputs():
	clip = Clip(np.zeros((1, 1, 3)), [[[0, 0, 0, 1]]], np.ones((1, 1, 3)))
	assert sample_clip(clip, [100, -100])[0].shape == (2, 1, 3)
	assert sample_clip(clip, [])[1].shape == (0, 1, 4)
	with pytest.raises(ValueError, match="zero quaternion"):
		Clip(np.zeros((1, 1, 3)), np.zeros((1, 1, 4)), np.ones((1, 1, 3)))


def test_inverse_bind_cancels_reference_pose():
	inverse_bind = np.eye(4, dtype=np.float32)[None, ...]
	inverse_bind[0, 0, 3] = -2
	rig = Skeleton([-1], [[2, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], inverse_bind)
	model = model_matrices(rig, rig.translation[None], rig.rotation[None], rig.scale[None])
	np.testing.assert_allclose(skin_matrices(rig, model)[0, 0], np.eye(4)[:3])


def test_skin_vertices_blends_positions_and_transforms_normals():
	matrices = np.tile(np.eye(4, dtype=np.float32)[:3], (2, 1, 1))
	matrices[1, 0, 3] = 2
	matrices[:, 0, 0] = 2
	positions, normals = skin_vertices([[1, 0, 0]], [[1, 1, 0]], [[0, 1]], [[0.25, 0.75]], matrices)
	np.testing.assert_allclose(positions, [[3.5, 0, 0]])
	np.testing.assert_allclose(normals, [[1/np.sqrt(5), 2/np.sqrt(5), 0]], atol=1e-6)
	with pytest.raises(ValueError, match="joint indices or skin weights"):
		skin_vertices([[0, 0, 0]], [[0, 1, 0]], [[2]], [[1]], matrices)
	with pytest.raises(ValueError, match="joint indices or skin weights"):
		skin_vertices([[0, 0, 0]], [[0, 1, 0]], [[0]], [[0.5]], matrices)


def test_skin_vertices_positions_only_supports_singular_transforms():
	matrices = np.tile(np.eye(4, dtype=np.float32)[:3], (2, 1, 1))
	matrices[:, 0, 0] = 0
	matrices[1, 0, 3] = 2
	positions, normals = skin_vertices([[1, 2, 3]], None, [[0, 1]], [[0.25, 0.75]], matrices)
	np.testing.assert_allclose(positions, [[1.5, 2, 3]])
	assert normals is None
	with pytest.raises(ValueError, match="singular"):
		skin_vertices([[1, 2, 3]], [[0, 1, 0]], [[0, 1]], [[0.25, 0.75]], matrices)
	with pytest.raises(ValueError, match="joint indices or skin weights"):
		skin_vertices([[1, 2, 3]], None, [[0]], [[0.5]], matrices)
	with pytest.raises(ValueError, match="finite"):
		skin_vertices([[np.nan, 2, 3]], None, [[0]], [[1]], matrices)
