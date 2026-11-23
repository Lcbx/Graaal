"""Hinge and slider motors with limits, plus a breakable fixed joint.

Run: python -m scenes.physics_joints_example
"""

import argparse

import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, PhysicsJoint, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform


TICK_HZ = 120
BOX_SIZE = (0.4, 0.4, 0.4)


def run_scene(*, on_pose=None):
	import pyjoltphysics as pj

	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsJoint)
	hinge, slider, breakable = ecs.create(3)
	entities = [hinge, slider, breakable]
	ecs.add(entities, Collider(0), RigidBody(gravity_factor=0), MaterialTransform())
	transforms = ecs.get(MaterialTransform)
	transforms[entities].position = [[1, 3, 0], [4, 3, 0], [7, 3, 0]]
	ecs.add(hinge, PhysicsJoint(kind=2, anchor=(0, 3, 0), axis=(0, 0, 1), lower=(-0.5,) * 6, upper=(0.5,) * 6, motor_mode=(1, 0, 0, 0, 0, 0), motor_target=(2, 0, 0, 0, 0, 0), motor_max=(50,) * 6))
	ecs.add(slider, PhysicsJoint(kind=3, anchor=(4, 3, 0), lower=(-1,) * 6, upper=(1,) * 6, motor_mode=(1, 0, 0, 0, 0, 0), motor_target=(1, 0, 0, 0, 0, 0), motor_max=(50,) * 6))
	ecs.add(breakable, PhysicsJoint(key=7, break_force=0.01))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box(BOX_SIZE)}, settings=pj.WorldSettings(tick_hz=TICK_HZ, threads=1), joint_type=PhysicsJoint) as physics:
		schedule = Scheduler(ecs)
		for name, callback in (("sync", physics.reconcile), ("input", physics.prepare_inputs), ("step", physics.step), ("pull", physics.pull), ("events", physics.dispatch_events)):
			schedule.add(name, callback, phase="fixed_update")
		def report(world, commands, events):
			broken = events[events["phase"] == pj.EventPhase.BREAK]
			for event in broken:
				print(f"broken joint: entity={event['entity_a']} key={event['joint_key']}")
		physics.subscribe(report, kind=pj.EventKind.JOINT)
		physics.add_force([breakable], (0, 100, 0))
		if on_pose is not None:
			pose = transforms[entities]
			on_pose(pose.position, pose.rotation)
		fixed_dt = 1 / TICK_HZ
		for tick in range(2 * TICK_HZ):
			if tick == TICK_HZ:
				physics.set_joint_motor([hinge, slider], 0, mode=pj.MotorMode.VELOCITY, target=-1, max_force=50)
			schedule.run(fixed_dt, fixed_steps=1, fixed_dt=fixed_dt)
			if on_pose is not None:
				pose = transforms[entities]
				on_pose(pose.position, pose.rotation)
			if tick % 60 == 59:
				joints = physics.world.latest().joints
				print(f"tick={tick + 1} joint_values={np.round(joints.value, 3)} states={joints.state}")


def run_viewer(*, offscreen=False):
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh
	from graaal.Math import Camera
	from scenes.physics_geometry import cube_vertices
	from scenes.physics_viewing_utils import has_blue_pixels, view_poses

	samples = []
	run_scene(on_pose=lambda position, rotation: samples.append((position, rotation)))
	mesh = MaterialMesh(cube_vertices(), DefaultMaterialShader())
	colors = ((0.12, 0.55, 0.95, 1), (0.95, 0.50, 0.13, 1), (0.30, 0.75, 0.23, 1))
	boxes = [DrawInstance(mesh, material={"base_color": color}, scale=BOX_SIZE) for color in colors]
	camera = Camera((4.5, 6, 11), (3.5, 3.3, 0), (0, 1, 0), 43, near=0.05, far=60)
	view_poses("Physics joints | hinge, slider, breakable", camera, samples, boxes, sample_hz=TICK_HZ, offscreen=offscreen, check_frame=has_blue_pixels)


if __name__ == "__main__":
	from scenes.physics_viewing_utils import add_viewing_options

	parser = argparse.ArgumentParser(description=__doc__)
	add_viewing_options(parser)
	args = parser.parse_args()
	if args.headless:
		run_scene()
	else:
		run_viewer(offscreen=args.offscreen)
