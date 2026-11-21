import unittest

import numpy as np

from graaal.ECS import ECS
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.Animation.RootMotion import RootMotion
from graaal.Animation.PoseTasks import PosePlan
from graaal.Animation.SkeletalAnimation import Clip, Skeleton


def setup():
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	return world, rig, clip


class EcsRootMotionTests(unittest.TestCase):
	def test_sample_batch_root_motion_slot_matches_existing_evaluator(self):
		world, rig, clip = setup()
		ids = world.create(2)
		world.add(ids, SkeletonRef(0), ClipPlayer(0, loop=False))
		world.get(ClipPlayer)[ids].assign(extract_root_motion=[True, False])
		adapter = EcsAnimationAdapter(world, {0: rig}, {0: clip}, root_motions={(0, 0): RootMotion(rig, clip)})
		adapter.update(0.5, evaluate_poses=False)
		batch, = adapter.sample_batches
		plan = PosePlan(rig, [("sample", 0), ("in_place", 0, 0)])
		pose = plan.evaluate({0: clip}, [(batch.clip_ids, batch.times, batch.weights)], np.empty((2, 0)), root_motions=[adapter.in_place_motion(batch)])
		np.testing.assert_allclose(pose[0][:, 0, 0], [0, 0.5])
		np.testing.assert_allclose(adapter.evaluate_samples()[0].matrices[:, 0, 0, 3], pose[0][:, 0, 0])
		self.assertAlmostEqual(adapter.root_motion_batches[0].matrices[0, 0, 3], 0.5)
		adapter.root_motions.clear()
		with self.assertRaisesRegex(ValueError, "root motion"): adapter.in_place_motion(batch)

	def test_requests_and_in_place_poses_mixed_with_regular_players(self):
		world, rig, clip = setup()
		ids = world.create(4)
		world.add(ids, SkeletonRef(0), ClipPlayer(0))
		world.get(ClipPlayer)[ids].assign(time=[0.75, 0.25, 0.25, 0.8], speed=[1, -1, 1, 1], loop=[True, False, True, False], playing=[True, True, True, False], extract_root_motion=[True, True, False, True])
		adapter = EcsAnimationAdapter(world, {0: rig}, {0: clip}, root_motions={(0, 0): RootMotion(rig, clip)})
		pose, = adapter.update(0.5)
		np.testing.assert_allclose(pose.matrices[:, 0, 0, 3], [0, 0, 0.75, 0], atol=1e-6)
		requests = {int(entity): matrix for batch in adapter.root_motion_batches for entity, matrix in zip(batch.entities, batch.matrices)}
		self.assertEqual(set(requests), {int(ids[0]), int(ids[1]), int(ids[3])})
		np.testing.assert_allclose([requests[int(entity)][0, 3] for entity in ids[[0, 1, 3]]], [0.5, -0.25, 0], atol=1e-6)
		world.delete(ids)
		self.assertEqual(adapter.update(0), [])
		self.assertEqual(adapter.root_motion_batches, [])

	def test_invalid_configuration_does_not_change_players_or_outputs(self):
		world, rig, clip = setup()
		ids = world.create(2)
		world.add(ids, SkeletonRef(0), ClipPlayer(0, extract_root_motion=True))
		adapter = EcsAnimationAdapter(world, {0: rig}, {0: clip}, root_motions={(0, 0): RootMotion(rig, clip)})
		adapter.update(0.25)
		old_batches = adapter.root_motion_batches
		players = world.get(ClipPlayer)[ids]
		players.assign(transitioning=[False, True], target_clip_id=99)
		with self.assertRaisesRegex(ValueError, "Unknown target clip"):
			adapter.update(0.5)
		np.testing.assert_allclose(players.time, [0.25, 0.25])
		self.assertIs(adapter.root_motion_batches, old_batches)
		players.assign(transitioning=False)
		adapter.root_motions.clear()
		with self.assertRaisesRegex(ValueError, "root motion"):
			adapter.update(0.5)
		np.testing.assert_allclose(players.time, [0.25, 0.25])
		self.assertIs(adapter.root_motion_batches, old_batches)

	def test_crossfade_blends_motion_and_applies_target_remainder(self):
		world, rig, source = setup()
		target = Clip([[[0, 0, 0]], [[2, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
		entity = world.create()[0]
		world.add(entity, SkeletonRef(0), ClipPlayer(0, transitioning=True, target_clip_id=1, transition_duration=1, extract_root_motion=True))
		adapter = EcsAnimationAdapter(world, {0: rig}, {0: source, 1: target}, root_motions={(0, 0): RootMotion(rig, source), (0, 1): RootMotion(rig, target)})
		pose, = adapter.update(0.5)
		self.assertAlmostEqual(adapter.root_motion_batches[0].matrices[0, 0, 3], 0.625)
		self.assertAlmostEqual(pose.matrices[0, 0, 0, 3], 0)
		pose, = adapter.update(0.75)
		self.assertAlmostEqual(adapter.root_motion_batches[0].matrices[0, 0, 3], 1.375)
		self.assertAlmostEqual(pose.matrices[0, 0, 0, 3], 0)
		players = world.get(ClipPlayer)[[entity]]
		self.assertEqual(players.clip_id[0], 1)
		self.assertFalse(players.transitioning[0])
		self.assertAlmostEqual(players.time[0], 0.25)

	def test_crossfade_pause_and_missing_target_motion(self):
		world, rig, clip = setup()
		entity = world.create()[0]
		world.add(entity, SkeletonRef(0), ClipPlayer(0, playing=False, transitioning=True, target_clip_id=1, extract_root_motion=True))
		adapter = EcsAnimationAdapter(world, {0: rig}, {0: clip, 1: clip}, root_motions={(0, 0): RootMotion(rig, clip)})
		with self.assertRaisesRegex(ValueError, "target clip"):
			adapter.update(0.5)
		self.assertEqual(world.get(ClipPlayer)[[entity]].time[0], 0)
		adapter.root_motions[(0, 1)] = RootMotion(rig, clip)
		adapter.update(0.5)
		np.testing.assert_allclose(adapter.root_motion_batches[0].matrices[0], np.eye(4), atol=1e-7)
		self.assertEqual(world.get(ClipPlayer)[[entity]].transition_time[0], 0)

	def test_crossfade_reverse_looping_and_rotation(self):
		world, rig, source = setup()
		q = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
		target = Clip([[[0, 0, 0]], [[0, 0, 0]]], [[[0, 0, 0, 1]], [q]], np.ones((2, 1, 3)), fps=1)
		entity = world.create()[0]
		world.add(entity, SkeletonRef(0), ClipPlayer(0, time=0.25, speed=-1, transitioning=True, target_clip_id=1, target_time=0.25, target_speed=-1, extract_root_motion=True))
		adapter = EcsAnimationAdapter(world, {0: rig}, {0: source, 1: target}, root_motions={(0, 0): RootMotion(rig, source), (0, 1): RootMotion(rig, target)})
		pose, = adapter.update(0.5)
		request = adapter.root_motion_batches[0].matrices[0]
		self.assertLess(request[0, 3], 0)
		self.assertLess(request[1, 0], 0)
		np.testing.assert_allclose(pose.matrices[0, 0], np.eye(4)[:3], atol=1e-6)


if __name__ == "__main__":
	unittest.main()
