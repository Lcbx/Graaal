"""Native ray batches drive the existing foot-IK demo over collision geometry.

Run: python -m scenes.physics_queries_example
"""

import numpy as np

from graaal.ECS import ECS
from graaal.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
from graaal.MaterialTransform import MaterialTransform
from scenes.foot_ik_example import create_demo, terrain_height


def terrain_mesh(pj):
	x = np.linspace(-4, 8, 97)
	vertices = np.column_stack((np.repeat(x, 2), np.repeat(terrain_height(x), 2), np.tile([-2, 2], len(x))))
	i = np.arange(len(x) - 1) * 2
	triangles = np.concatenate((np.column_stack((i, i + 1, i + 2)), np.column_stack((i + 1, i + 3, i + 2))))
	return pj.Shape.mesh(vertices, triangles)


def query_ground(physics, points):
	hits = physics.raycast_batch(points + [0, 3, 0], np.tile((0, -1, 0), (len(points), 1)), 6)
	return hits.point[:, 1], hits.normal


def run_scene():
	import pyjoltphysics as pj

	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform)
	ground = ecs.create(1)[0]
	ecs.add(ground, Collider(0), MaterialTransform())
	with EcsPhysicsAdapter(ecs, {0: terrain_mesh(pj)}, settings=pj.WorldSettings(threads=1)) as physics:
		physics.reconcile()
		physics.step()
		physics.pull()
		physics.dispatch_events()
		hit = physics.raycast((1, 3, 0), (0, -1, 0), 6)
		print(f"ray entity={physics.entities_of(hit.uid)} point={np.round(hit.point, 3)}")
		swept = physics.sweep(pj.Shape.sphere(0.1), (1, 3, 0), (1, -1, 0))
		print(f"sweep entity={physics.entities_of(swept.uid)} distance={swept.distance:.3f}")
		world, entities, animation, schedule, frames = create_demo(lambda points: query_ground(physics, points))
		for tick in range(8):
			schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
			for owners, matrices in frames["batches"]:
				print(f"tick={tick} foot_heights={matrices[:, 2, 1, 3].round(3)}")


if __name__ == "__main__":
	run_scene()
