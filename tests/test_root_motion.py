import unittest

import numpy as np

from graaal.Animation.RootMotion import RootMotion, _blend_deltas
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip


def resources(turn=False):
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	q = [0, 0, np.sqrt(0.5), np.sqrt(0.5)] if turn else [0, 0, 0, 1]
	clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [q]], np.ones((2, 1, 3)), fps=1)
	return rig, clip


class RootMotionTests(unittest.TestCase):
	def test_translation_loops_reverse_and_clamping(self):
		rig, clip = resources()
		motion = RootMotion(rig, clip)
		delta = motion.delta([0.75, 0.25, 0, 2], [1.25, -0.25, 3, -2])
		np.testing.assert_allclose(delta[:, 0, 3], [0.5, -0.5, 3, -4], atol=1e-6)
		np.testing.assert_allclose(motion.delta([-2, 0.75], [2, 2], loop=False)[:, 0, 3], [1, 0.25])
		self.assertEqual(motion.delta([], []).shape, (0, 4, 4))

	def test_rotation_composes_over_loops_and_reverses(self):
		rig, clip = resources(turn=True)
		motion = RootMotion(rig, clip)
		two_cycles = motion.delta([0], [2])[0]
		np.testing.assert_allclose(two_cycles[:3, 3], [1, 1, 0], atol=1e-6)
		np.testing.assert_allclose(two_cycles[:3, :3], np.diag([-1, -1, 1]), atol=1e-6)
		forward = motion.delta([0.5], [1.5])[0]
		reverse = motion.delta([1.5], [0.5])[0]
		np.testing.assert_allclose(forward @ reverse, np.eye(4), atol=1e-6)
		np.testing.assert_allclose(motion.delta([-1], [0])[0], motion.delta([0], [1])[0], atol=1e-6)

	def test_delta_uses_start_root_frame_and_single_frame_is_stationary(self):
		rig, _ = resources()
		q = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
		clip = Clip([[[3, 0, 0]], [[4, 0, 0]]], [[q], [q]], np.ones((2, 1, 3)), fps=1)
		motion = RootMotion(rig, clip)
		np.testing.assert_allclose(motion.delta([0], [1], loop=False)[0, :3, 3], [0, -1, 0], atol=1e-6)
		np.testing.assert_allclose(motion.delta([0.5], [2.5])[0, :3, 3], [0, -2, 0], atol=1e-6)
		stationary = Clip([[[3, 0, 0]]], [[q]], np.ones((1, 1, 3)))
		np.testing.assert_allclose(RootMotion(rig, stationary).delta([-100], [100])[0], np.eye(4), atol=1e-6)

	def test_invalid_root_scale_and_times(self):
		rig, clip = resources()
		with self.assertRaises(ValueError): RootMotion(rig, clip, True)
		with self.assertRaises(ValueError): RootMotion(rig, clip, 1)
		bad = Clip(clip.translation, clip.rotation, np.full((2, 1, 3), 2), fps=1)
		with self.assertRaisesRegex(ValueError, "unit root scale"): RootMotion(rig, bad)
		motion = RootMotion(rig, clip)
		for start, end in (([0], [0, 1]), ([np.nan], [0]), ([0], [np.inf])):
			with self.assertRaises(ValueError): motion.delta(start, end)

	def test_in_place_pose_and_character_delta_preserve_world_pose(self):
		q = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
		rig = Skeleton([-1, 0], [[2, 1, 0], [0, 1, 0]], [q, [0, 0, 0, 1]], np.ones((2, 3)), np.tile(np.eye(4), (2, 1, 1)))
		clip = Clip([[[3, 0, 0], [0, 1, 0]], [[4, 1, 0], [0, 2, 0]]], [[[0, 0, 0, 1], [0, 0, 0, 1]], [q, [0, 0, 0, 1]]], np.ones((2, 2, 3)), fps=1)
		motion = RootMotion(rig, clip)
		pose = sample_clip(clip, [1], loop=False)
		original = tuple(value.copy() for value in pose)
		in_place = motion.in_place(*pose)
		for value, before in zip(pose, original): np.testing.assert_array_equal(value, before)
		np.testing.assert_array_equal(in_place[0][0, 1], pose[0][0, 1])
		local_model = model_matrices(rig, *in_place)[0]
		original_model = model_matrices(rig, *pose)[0]
		root_start = model_matrices(rig, *sample_clip(clip, [0], loop=False))[0, 0]
		reference = model_matrices(rig, rig.translation[None], rig.rotation[None], rig.scale[None])[0, 0]
		expected = reference @ np.linalg.inv(root_start) @ original_model[1]
		actual = motion.character_delta([0], [1], loop=False)[0] @ local_model[1]
		np.testing.assert_allclose(actual, expected, atol=1e-6)

	def test_in_place_rejects_invalid_poses_and_nonunit_reference_scale(self):
		rig, clip = resources()
		motion = RootMotion(rig, clip)
		pose = sample_clip(clip, [0])
		with self.assertRaises(ValueError): motion.in_place(pose[0], pose[1][:, :0], pose[2])
		with self.assertRaisesRegex(ValueError, "unit pose root scale"): motion.in_place(pose[0], pose[1], pose[2] * 2)
		bad_rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[2, 1, 1]], [np.eye(4)])
		bad_motion = RootMotion(bad_rig, clip)
		with self.assertRaisesRegex(ValueError, "unit reference root scale"): bad_motion.in_place(*pose)
		with self.assertRaisesRegex(ValueError, "unit reference root scale"): bad_motion.character_delta([0], [1])

	def test_delta_blend_handles_half_turn_and_shortest_rotation(self):
		rig, _ = resources()
		q = [np.sqrt(0.5), -np.sqrt(0.5), 0, 0]
		target = model_matrices(rig, np.array([[[0, 0, 0]]]), np.array([[q]]), np.ones((1, 1, 3)))[:, 0]
		identity = np.eye(4)[None]
		half = _blend_deltas(identity, target, [0.5])
		np.testing.assert_allclose(half[0] @ half[0], target[0], atol=1e-6)
		np.testing.assert_allclose(_blend_deltas(identity, target, [0]), identity, atol=1e-6)
		np.testing.assert_allclose(_blend_deltas(identity, target, [1]), target, atol=1e-6)
		q = [0, 0, np.sin(3 * np.pi / 4), np.cos(3 * np.pi / 4)]
		target = model_matrices(rig, np.array([[[0, 0, 0]]]), np.array([[q]]), np.ones((1, 1, 3)))[:, 0]
		short = _blend_deltas(identity, target, [0.5])[0]
		np.testing.assert_allclose(short[:2, :2], [[np.sqrt(0.5), np.sqrt(0.5)], [-np.sqrt(0.5), np.sqrt(0.5)]], atol=1e-6)


if __name__ == "__main__":
	unittest.main()
