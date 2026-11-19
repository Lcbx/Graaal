import unittest

import numpy as np

from graaal.AnimationEvents import EventTrack
from graaal.ECS import ECS
from graaal.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.SkeletalAnimation import Clip, Skeleton


class AnimationEventTests(unittest.TestCase):
	def test_interval_boundaries_reverse_and_multiple_loops(self):
		track = EventTrack(1, [0.75, 0, 0.25, 0.25], ["late", "start", "left", "right"])
		self.assertEqual([event.name for event in track.between(0.25, 1.25)], ["late", "start", "left", "right"])
		self.assertEqual([event.name for event in track.between(1.25, 0.25)], ["start", "late", "left", "right"])
		self.assertEqual([event.time for event in track.between(0.5, 2.5) if event.name == "start"], [1, 2])
		self.assertEqual([event.name for event in track.between(0.75, -1, loop=False)], ["left", "right", "start"])
		self.assertEqual(track.between(0.25, 0.25), [])

	def test_track_validation_and_ownership(self):
		times = np.array([0.5])
		track = EventTrack(1, times, ["step"])
		times[0] = 0.2
		self.assertEqual(track.times[0], 0.5)
		self.assertFalse(track.times.flags.writeable)
		for duration, values, names in ((0, [], []), (1, [1], ["end"]), (1, [-0.1], ["bad"]), (1, [0.5], []), (1, [0.5], [""])):
			with self.assertRaises(ValueError): EventTrack(duration, values, names)

	def test_ecs_transition_events_stop_source_and_keep_target(self):
		rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
		clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
		world = ECS()
		world.register(SkeletonRef, ClipPlayer)
		entity = world.create()[0]
		world.add(entity, SkeletonRef(0), ClipPlayer(0, transitioning=True, target_clip_id=1, transition_duration=0.5))
		tracks = {0: EventTrack(1, [0.25, 0.75], ["source early", "source late"]), 1: EventTrack(1, [0.25, 0.75], ["target early", "target late"])}
		animation = EcsAnimationAdapter(world, {0: rig}, {0: clip, 1: clip}, event_tracks=tracks)
		animation.update(0.8)
		self.assertEqual([(event.entity, event.role, event.name) for event in animation.events], [(int(entity), "source", "source early"), (int(entity), "target", "target early"), (int(entity), "target", "target late")])
		animation.update(0)
		self.assertEqual(animation.events, [])
		world.get(ClipPlayer)[[entity]].assign(time=0.2, speed=-1, playing=True)
		animation.update(0.5)
		self.assertEqual([event.name for event in animation.events], ["target late"])
		world.get(ClipPlayer)[[entity]].assign(playing=False)
		animation.update(1)
		self.assertEqual(animation.events, [])
		world.delete(entity)
		animation.update(1)
		self.assertEqual(animation.events, [])

	def test_invalid_track_does_not_commit_state_or_events(self):
		rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
		clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
		world = ECS()
		world.register(SkeletonRef, ClipPlayer)
		entity = world.create()[0]
		world.add(entity, SkeletonRef(0), ClipPlayer(0))
		animation = EcsAnimationAdapter(world, {0: rig}, {0: clip}, event_tracks={0: EventTrack(2, [0.25], ["step"])})
		with self.assertRaisesRegex(ValueError, "duration"):
			animation.update(0.5)
		self.assertEqual(world.get(ClipPlayer)[[entity]].time[0], 0)
		self.assertEqual(animation.events, [])

	def test_nonlooping_seeks_outside_clip_do_not_emit_early_source_or_target_events(self):
		rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
		clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
		for target in (False, True):
			for time, speed, marker in ((2, -1, 0.8), (-1, 1, 0.2)):
				with self.subTest(target=target, time=time):
					world = ECS()
					world.register(SkeletonRef, ClipPlayer)
					entity = world.create()[0]
					player = ClipPlayer(0, time=time, speed=speed, loop=False)
					if target:
						player = ClipPlayer(0, speed=0, transitioning=True, target_clip_id=1, target_time=time, target_speed=speed, target_loop=False, transition_duration=2)
					world.add(entity, SkeletonRef(0), player)
					tracks = {int(target): EventTrack(1, [marker], ["step"])}
					adapter = EcsAnimationAdapter(world, {0: rig}, {0: clip, 1: clip}, event_tracks=tracks)
					adapter.update(0.25)
					self.assertEqual(adapter.events, [])
					# A later seek crosses into the authored interval during this update.
					seek = 1.1 if speed < 0 else -0.1
					selection = world.get(ClipPlayer)[[entity]]
					if target: selection.assign(target_time=seek)
					else: selection.assign(time=seek)
					adapter.update(0.4)
					self.assertEqual([event.name for event in adapter.events], ["step"])


if __name__ == "__main__":
	unittest.main()
