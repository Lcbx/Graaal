"""Fixed-step ECS locomotion with requested and accepted root movement.

Run without graphics: python -m scenes.root_motion_example
"""

import numpy as np

from graaal.ECS import ECS, Scheduler, component
from graaal.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.RootMotion import RootMotion
from graaal.SkeletalAnimation import Clip, Skeleton


@component
class Movement:
	x: np.float32 = 0.0
	requested: np.float32 = 0.0
	accepted: np.float32 = 0.0


def make_clip(speed):
	"""Create a two-second looping pose with a root moving along X."""
	translation = np.zeros((3, 2, 3), dtype=np.float32)
	translation[:, 0, 0] = np.array([0, 1, 2]) * speed
	translation[:, 1, 1] = [1, 1.2 if speed else 1, 1]
	rotation = np.broadcast_to([0, 0, 0, 1], (3, 2, 4))
	return Clip(translation, rotation, np.ones((3, 2, 3)), fps=1)


def create_demo(limit=2.0):
	"""Create an animation controller and a bounded movement system."""
	rig = Skeleton([-1, 0], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0, 1]] * 2, np.ones((2, 3)), np.tile(np.eye(4), (2, 1, 1)))
	clips = {0: make_clip(0), 1: make_clip(1), 2: make_clip(2)}
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, Movement)
	entity = world.create()[0]
	world.add(entity, SkeletonRef(0), ClipPlayer(0, extract_root_motion=True), Movement())
	root_motions = {(0, clip_id): RootMotion(rig, clip) for clip_id, clip in clips.items()}
	animation = EcsAnimationAdapter(world, {0: rig}, clips, root_motions=root_motions)

	def accept_movement(world, commands, dt):
		for batch in animation.root_motion_batches:
			movement = world.get(Movement)[batch.entities]
			requested = batch.matrices[:, 0, 3].astype(np.float32)
			position = np.clip(movement.x + requested, np.float32(0), np.float32(limit))
			movement.assign(requested=requested, accepted=position - movement.x, x=position)

	schedule = Scheduler(world)
	schedule.add("animation_control", animation.advance, phase="fixed_update")
	schedule.add("movement", accept_movement, phase="fixed_update", after="animation_control")
	schedule.add("animation_pose", animation.finalize, phase="fixed_update", after="movement")
	return world, entity, animation, schedule


def crossfade(world, entity, target, duration=0.5):
	world.get(ClipPlayer)[[entity]].assign(transitioning=True, target_clip_id=target, target_time=0, target_speed=1, target_loop=True, transition_time=0, transition_duration=duration)


def run_scene():
	world, entity, animation, schedule = create_demo()
	print("Synthetic idle/walk/run root motion; movement is limited to x = 2.0")
	for label, target, ticks in (("idle", None, 2), ("walk", 1, 8), ("run", 2, 8), ("idle", 0, 4)):
		if target is not None: crossfade(world, entity, target)
		for _ in range(ticks):
			schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
			movement = world.get(Movement)[[entity]]
			print(f"{label:4} x={movement.x[0]:.2f} requested={movement.requested[0]:.2f} accepted={movement.accepted[0]:.2f}")


if __name__ == "__main__":
	run_scene()
