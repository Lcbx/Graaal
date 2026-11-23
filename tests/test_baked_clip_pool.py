import numpy as np
import pytest

from graaal.Animation.BakedClipPool import BakedClipPool
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, skin_matrices
from graaal.Math import Quaternion


def test_pool_packs_frames_and_keeps_non_integral_duration_endpoint():
	rotation = [(0, 0, 0, 1)]
	rig = Skeleton([-1], [[0, 0, 0]], rotation, [[1, 1, 1]], [np.eye(4)])
	move = Clip([[[0, 0, 0]], [[2, 0, 0]]], [rotation, rotation], np.ones((2, 1, 3)), fps=3)
	static = Clip([[[0, 1, 0]]], [rotation], np.ones((1, 1, 3)))
	pool = BakedClipPool(rig, [move, static], fps=4)
	assert pool.packed.shape == (4, 4)
	assert pool.packed.nbytes == 4 * 16 and not pool.packed.flags.writeable
	assert pool.table["base"].tolist() == [0, 3]
	assert pool.table["frames"].tolist() == [3, 1]
	pose = pool.model([[0], [0], [0], [1]], [[0], [1 / 6], [1 / 3], [2]], [[1]] * 4)
	np.testing.assert_allclose(pose[:, 0, :3, 3], [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], atol=1e-6)
	assert pool.model(np.empty((0, 2), dtype=int), np.empty((0, 2)), np.empty((0, 2))).shape == (0, 1, 4, 4)
	with pytest.raises(ValueError, match="weights"): pool.model([[0, 1]], [[0, 1]], [[0.2, 0.3]])


def test_source_rate_bake_preserves_authored_samples():
	translation = np.zeros((32, 1, 3))
	translation[15, 0, 0] = 1
	rotation = np.tile([0, 0, 0, 1], (32, 1, 1))
	scale = np.ones((32, 1, 3))
	rig = Skeleton([-1], translation[0], rotation[0], scale[0], [np.eye(4)])
	clip = Clip(translation, rotation, scale, fps=30)
	pool = BakedClipPool(rig, [clip], fps=clip.fps)
	assert pool.table["frames"].tolist() == [32]
	times = np.arange(32)[:, None] / clip.fps
	pose = pool.model([[0]] * 32, times, [[1]] * 32)
	np.testing.assert_allclose(pose[:, :, :3, 3], translation, atol=1e-6)


def test_crossfade_validator_reports_model_space_shrinkage():
	translation = np.array([[[0, 0, 0], [1, 0, 0]]])
	rotation = np.tile([0, 0, 0, 1], (1, 2, 1))
	scale = np.ones((1, 2, 3))
	rig = Skeleton([-1, 0], translation[0], rotation[0], scale[0], np.tile(np.eye(4), (2, 1, 1)))
	idle = Clip(translation, rotation, scale)
	rotation = rotation.astype(float)
	rotation[0, 0] = Quaternion.from_z_rotation(np.pi / 2)
	turn = Clip(translation, rotation, scale)
	pool = BakedClipPool(rig, [idle, turn])
	ids, times, weights = [[0, 1]] * 3, np.zeros((3, 2)), [[1, 0], [0.5, 0.5], [0, 1]]
	approx = pool.model(ids, times, weights)
	np.testing.assert_allclose(approx[1, 1, :3, 3], [0.5, 0.5, 0], atol=1e-6)
	np.testing.assert_allclose(pool.crossfade_position_error(ids, times, weights), [0, 1 - np.sqrt(0.5), 0], atol=1e-6)
	bad = Clip(translation, rotation, [[[1, 1, 2], [1, 1, 1]]])
	with pytest.raises(ValueError, match="uniform local scale"): BakedClipPool(rig, [bad])


def test_gpu_crowd_matches_reference_and_preserves_cpu_palette_ranges():
	pytest.importorskip("wgpu")
	from graaal.RenderContext import RenderContext
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
	from graaal.RenderPasses.CrowdPose import CrowdPosePass

	translation = [[[0, 0, 0], [1, 0, 0]], [[3, 1, 0], [1, 0, 0]]]
	rotation = np.array([[[0, 0, 0, 1]] * 2, [Quaternion.from_z_rotation(0.7), Quaternion.from_x_rotation(0.2)]])
	scale = np.tile(np.array([[[1, 1, 1]], [[0.9, 0.9, 0.9]]]), (1, 2, 1))
	bind = np.tile(np.eye(4), (2, 1, 1))
	bind[1, 0, 3] = -1
	rig = Skeleton([-1, 0], translation[0], rotation[0], scale[0], bind)
	pool = BakedClipPool(rig, [Clip(translation, rotation, scale, fps=2), Clip(translation[:1], rotation[:1], scale[:1])])
	ids = [[0, 1, 0, 1], [1, 0, 0, 1]]
	times = [[-0.1, 0.7, 0.17, 0], [0, 1, 0.43, 0]]
	weights = [[0.1, 0.3, 0.4, 0.2], [0, 0.8, 0.2, 0]]
	expected = skin_matrices(rig, pool.model(ids, times, weights))
	RenderContext.setup_graphics_backend(True)
	palette = SkinMatrixBuffer(6)
	for owner in (9, 77, 17): palette.acquire(owner, 2)
	cpu = np.tile(np.eye(4, dtype=np.float32)[:3], (1, 2, 1, 1))
	palette.write([77], cpu)
	crowd = CrowdPosePass(pool, palette, 2)
	try:
		commands = RenderContext.commands("crowd reference")
		crowd.record(commands, [17, 9], ids, times, weights)
		RenderContext.submit(commands.finish())
		actual = np.frombuffer(RenderContext.device.queue.read_buffer(palette.buffer.handle), dtype=np.float32).reshape(3, 2, 3, 4)
		np.testing.assert_allclose(actual[[2, 0]], expected, atol=3e-6)
		np.testing.assert_array_equal(actual[1:2], cpu)
	finally:
		crowd.close()
		palette.close()
