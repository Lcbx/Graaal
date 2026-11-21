import unittest

import numpy as np

from graaal.ECS import ECS
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip, skin_matrices


def resources():
	rig = Skeleton([-1, 0], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0, 1]] * 2, np.ones((2, 3)), np.tile(np.eye(4), (2, 1, 1)))
	clip = Clip([[[0, 0, 0], [0, 1, 0]], [[2, 0, 0], [0, 2, 0]]], [[[0, 0, 0, 1]] * 2] * 2, np.ones((2, 2, 3)), fps=1)
	return rig, clip


class AnimationBufferTests(unittest.TestCase):
	def test_output_arrays_match_allocated_results(self):
		rig, clip = resources()
		times = [0.25, 0.75]
		expected = sample_clip(clip, times)
		out = (np.empty((2, 2, 3), np.float32), np.empty((2, 2, 4), np.float32), np.empty((2, 2, 3), np.float32))
		actual = sample_clip(clip, times, out=out)
		for value, buffer, expected_value in zip(actual, out, expected):
			self.assertIs(value, buffer)
			np.testing.assert_allclose(value, expected_value)
		np.testing.assert_allclose(actual[0][:, 0, 0], [0.5, 1.5])
		np.testing.assert_allclose(actual[0][:, 1, 1], [1.25, 1.75])
		model = np.empty((2, 2, 4, 4), np.float32)
		skin = np.empty((2, 2, 3, 4), np.float32)
		self.assertIs(model_matrices(rig, *actual, out=model), model)
		self.assertIs(skin_matrices(rig, model, out=skin), skin)
		np.testing.assert_allclose(model, model_matrices(rig, *expected))
		np.testing.assert_allclose(skin, skin_matrices(rig, model))
		np.testing.assert_allclose(skin[:, 1, :3, 3], [[0.5, 1.25, 0], [1.5, 1.75, 0]])

	def test_invalid_or_overlapping_output_is_rejected(self):
		rig, clip = resources()
		pose = sample_clip(clip, [0])
		with self.assertRaisesRegex(ValueError, "out"): sample_clip(clip, [0], out=(pose[0], pose[1]))
		with self.assertRaisesRegex(ValueError, "overlap"): sample_clip(clip, [0], out=(pose[0], pose[1], pose[0]))
		with self.assertRaisesRegex(ValueError, "model out"): model_matrices(rig, *pose, out=np.empty((1, 2, 4, 4), np.float64))
		model = model_matrices(rig, *pose)
		with self.assertRaisesRegex(ValueError, "skin out"): skin_matrices(rig, model, out=np.empty((1, 2, 4, 4), np.float32))

	def test_adapter_reuses_grows_and_clears_buffers(self):
		rig, clip = resources()
		world = ECS()
		world.register(SkeletonRef, ClipPlayer)
		first = world.create()[0]
		world.add(first, SkeletonRef(0), ClipPlayer(0, loop=False))
		adapter = EcsAnimationAdapter(world, {0: rig}, {0: clip})
		initial, = adapter.update(0.25)
		copy = initial.matrices.copy()
		next_batch, = adapter.update(0.25)
		self.assertTrue(np.shares_memory(initial.matrices, next_batch.matrices))
		self.assertNotEqual(copy[0, 0, 0, 3], next_batch.matrices[0, 0, 0, 3])
		second = world.create()[0]
		world.add(second, SkeletonRef(0), ClipPlayer(0, loop=False))
		grown, = adapter.update(0.25)
		self.assertFalse(np.shares_memory(next_batch.matrices, grown.matrices))
		np.testing.assert_allclose(grown.matrices[:, 0, 0, 3], [1.5, 0.5])
		world.delete(second)
		shrunk, = adapter.update(0)
		self.assertTrue(np.shares_memory(grown.matrices, shrunk.matrices))
		adapter.clear_pose_buffers()
		reallocated, = adapter.update(0)
		self.assertFalse(np.shares_memory(shrunk.matrices, reallocated.matrices))
		np.testing.assert_allclose(reallocated.matrices, shrunk.matrices)


if __name__ == "__main__":
	unittest.main()
