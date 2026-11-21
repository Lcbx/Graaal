"""Headless hinge and slider motors with limits, plus a breakable fixed joint.

Run: python -m scenes.physics_joints_example
"""

import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, PhysicsJoint, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform


def run_scene():
	import pyjoltphysics as pj

	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsJoint)
	hinge, slider, breakable = ecs.create(3)
	ecs.add([hinge, slider, breakable], Collider(0), RigidBody(gravity_factor=0), MaterialTransform())
	ecs.get(MaterialTransform)[[hinge, slider, breakable]].position = [[1, 3, 0], [4, 3, 0], [7, 3, 0]]
	ecs.add(hinge, PhysicsJoint(kind=2, anchor=(0, 3, 0), axis=(0, 0, 1), lower=(-0.5,) * 6, upper=(0.5,) * 6, motor_mode=(1, 0, 0, 0, 0, 0), motor_target=(2, 0, 0, 0, 0, 0), motor_max=(50,) * 6))
	ecs.add(slider, PhysicsJoint(kind=3, anchor=(4, 3, 0), lower=(-1,) * 6, upper=(1,) * 6, motor_mode=(1, 0, 0, 0, 0, 0), motor_target=(1, 0, 0, 0, 0, 0), motor_max=(50,) * 6))
	ecs.add(breakable, PhysicsJoint(key=7, break_force=0.01))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box((0.4, 0.4, 0.4))}, settings=pj.WorldSettings(threads=1), joint_type=PhysicsJoint) as physics:
		schedule = Scheduler(ecs)
		for name, callback in (("sync", physics.reconcile), ("input", physics.prepare_inputs), ("step", physics.step), ("pull", physics.pull), ("events", physics.dispatch_events)):
			schedule.add(name, callback, phase="fixed_update")
		def report(world, commands, events):
			broken = events[events["phase"] == pj.EventPhase.BREAK]
			for event in broken:
				print(f"broken joint: entity={event['entity_a']} key={event['joint_key']}")
		physics.subscribe(report, kind=pj.EventKind.JOINT)
		physics.add_force([breakable], (0, 100, 0))
		for tick in range(240):
			if tick == 120:
				physics.set_joint_motor([hinge, slider], 0, mode=pj.MotorMode.VELOCITY, target=-1, max_force=50)
			schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
			if tick % 60 == 59:
				joints = physics.world.latest().joints
				print(f"tick={tick + 1} joint_values={np.round(joints.value, 3)} states={joints.state}")


if __name__ == "__main__":
	run_scene()
