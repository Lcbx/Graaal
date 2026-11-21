"""Headless ECS animation events: python -m scenes.animation_events_example."""

import numpy as np

from graaal.Animation.AnimationEvents import EventTrack
from graaal.ECS import ECS, Scheduler, component
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.Animation.SkeletalAnimation import Clip, Skeleton


@component
class Footsteps:
	count: np.uint32 = 0


def run_scene():
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, Footsteps)
	entities = world.create(2)
	world.add(entities[0], SkeletonRef(0), ClipPlayer(0), Footsteps())
	world.add(entities[1], SkeletonRef(0), ClipPlayer(0, speed=-1.0), Footsteps())

	# A two-second clip and example markers; real games author markers for each clip.
	skeleton = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[0, 1, 0]], [[0, 0, 0]]], [[[0, 0, 0, 1]]] * 3, np.ones((3, 1, 3)), fps=1)
	track = EventTrack(clip.duration, [0.5, 1.5], ["left_step", "right_step"])
	animation = EcsAnimationAdapter(world, {0: skeleton}, {0: clip}, event_tracks={0: track})

	def handle_events(world, commands, dt):
		# Consume events after every evaluation: the next update replaces the list.
		for event in animation.events:
			steps = world.get(Footsteps)[[event.entity]]
			steps.assign(count=steps.count + 1)
			print(f"entity={event.entity} event={event.name} clip={event.clip_id} role={event.role} steps={steps.count[0]}")

	schedule = Scheduler(world)
	schedule.add("animation", animation.evaluate, phase="fixed_update")
	schedule.add("footsteps", handle_events, phase="fixed_update", after="animation")
	for _ in range(8):
		schedule.run(0.5, fixed_steps=1, fixed_dt=0.5)


if __name__ == "__main__":
	run_scene()
