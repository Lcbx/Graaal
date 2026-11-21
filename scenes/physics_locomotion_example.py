"""Headless root-motion locomotion against native collision geometry.

Run: python -m scenes.physics_locomotion_example
"""
import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.Animation.RootMotion import RootMotion
from graaal.Animation.SkeletalAnimation import Skeleton
from scenes.root_motion_example import Movement, make_clip


def create_demo():
	import pyjoltphysics as pj

	rig = Skeleton([-1, 0], [[0, 0, 0], [0, 1, 0]], [[0, 0, 0, 1]] * 2, np.ones((2, 3)), np.tile(np.eye(4), (2, 1, 1)))
	clip = make_clip(1)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, Movement, MaterialTransform, Collider, RigidBody)
	character, obstacle = world.create(2)
	world.add(character, SkeletonRef(0), ClipPlayer(0, extract_root_motion=True), Movement(), MaterialTransform(position=(0, 1, 0)), Collider(0), RigidBody(motion=1, gravity_factor=0))
	world.add(obstacle, MaterialTransform(position=(2, 1, 0)), Collider(1))
	physics = EcsPhysicsAdapter(world, {0: pj.Shape.sphere(0.25), 1: pj.Shape.box((0.5, 0.5, 0.5))}, settings=pj.WorldSettings(tick_hz=4, threads=1))
	animation = EcsAnimationAdapter(world, {0: rig}, {0: clip}, root_motions={(0, 0): RootMotion(rig, clip)})

	def accept_movement(world, commands, dt):
		for batch in animation.root_motion_batches:
			movement = world.get(Movement)[batch.entities]
			transforms = world.get(MaterialTransform)[batch.entities]
			requested = batch.matrices[:, 0, 3].astype(np.float32)
			position = transforms.position.copy()
			accepted = np.zeros(len(requested), dtype=np.float32)
			for i, entity in enumerate(batch.entities):
				if abs(requested[i]) < 1e-8: continue
				start = position[i]
				end = start + [requested[i], 0, 0]
				hit = physics.sweep(0, start, end, exclude=[entity])
				fraction = 1 if hit is None else max(0, min(1, hit.fraction - 0.005 / abs(requested[i])))
				accepted[i] = requested[i] * fraction
			position[:, 0] += accepted
			movement.assign(requested=requested, accepted=accepted, x=position[:, 0])
			transforms.assign(position=position)

	schedule = Scheduler(world)
	schedule.add("animation_control", animation.advance, phase="fixed_update")
	schedule.add("physics_sync", physics.reconcile, phase="fixed_update", after="animation_control")
	schedule.add("movement", accept_movement, phase="fixed_update", after="physics_sync")
	schedule.add("physics_input", physics.prepare_inputs, phase="fixed_update", after="movement")
	schedule.add("physics_step", physics.step, phase="fixed_update", after="physics_input")
	schedule.add("physics_output", physics.pull, phase="fixed_update", after="physics_step")
	schedule.add("animation_pose", animation.finalize, phase="fixed_update", after="physics_output")
	return world, character, obstacle, animation, physics, schedule


def run_scene():
	world, character, obstacle, animation, physics, schedule = create_demo()
	with physics:
		for tick in range(12):
			schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
			movement = world.get(Movement)[[character]]
			print(f"tick={tick} x={movement.x[0]:.3f} requested={movement.requested[0]:.3f} accepted={movement.accepted[0]:.3f}")


if __name__ == "__main__":
	run_scene()
