"""Native ray batches drive the existing foot-IK demo over collision geometry.

Run: python -m scenes.physics_queries_example
"""

import argparse

import numpy as np

from graaal.ECS import ECS
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from scenes.foot_ik_example import Character, create_demo, terrain_height


TICK_HZ = 4
QUERY_START = (1, 3, 0)
SWEEP_RADIUS = 0.1


def terrain_geometry():
	x = np.linspace(-4, 8, 97)
	vertices = np.column_stack((np.repeat(x, 2), np.repeat(terrain_height(x), 2), np.tile([-2, 2], len(x))))
	i = np.arange(len(x) - 1) * 2
	triangles = np.concatenate((np.column_stack((i, i + 1, i + 2)), np.column_stack((i + 1, i + 3, i + 2))))
	return vertices, triangles


def terrain_mesh(pj):
	return pj.Shape.mesh(*terrain_geometry())


def query_ground(physics, points):
	hits = physics.raycast_batch(points + [0, 3, 0], np.tile((0, -1, 0), (len(points), 1)), 6)
	return hits.point[:, 1], hits.normal


def run_scene(*, on_sample=None):
	"""Optionally emit world-space leg joints and the ray and sweep hit locations."""
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
		hit = physics.raycast(QUERY_START, (0, -1, 0), 6)
		print(f"ray entity={physics.entities_of(hit.uid)} point={np.round(hit.point, 3)}")
		swept = physics.sweep(pj.Shape.sphere(SWEEP_RADIUS), QUERY_START, (1, -1, 0))
		print(f"sweep entity={physics.entities_of(swept.uid)} distance={swept.distance:.3f}")
		world, entities, animation, schedule, frames = create_demo(lambda points: query_ground(physics, points))
		if on_sample is not None:
			ray_hit = np.array(hit.point)
			sweep_center = np.asarray(QUERY_START) - [0, swept.distance, 0]
		fixed_dt = 1 / TICK_HZ
		for tick in range(8):
			schedule.run(fixed_dt, fixed_steps=1, fixed_dt=fixed_dt)
			legs = []
			for owners, matrices in frames["batches"]:
				print(f"tick={tick} foot_heights={matrices[:, 2, 1, 3].round(3)}")
				if on_sample is not None:
					joints = matrices[..., :3, 3].copy()  # This demo uses identity inverse-bind matrices.
					joints[..., 0] += world.get(Character)[owners].x[:, None]
					legs.append(joints)
			if on_sample is not None:
				on_sample(np.concatenate(legs), ray_hit, sweep_center)


def run_viewer(*, offscreen=False):
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
	from graaal.Math import Camera
	from scenes.physics_geometry import cube_vertices, rounded_vertices, segment_transform, triangle_vertices
	from scenes.physics_viewing_utils import has_blue_pixels, play_frames, viewing_target

	samples = []
	run_scene(on_sample=lambda joints, ray_hit, sweep_center: samples.append((joints, ray_hit, sweep_center)))
	shader = DefaultMaterialShader()
	cube = cube_vertices()
	box = MaterialMesh(cube, shader)
	terrain = MaterialMesh(triangle_vertices(*terrain_geometry()), shader)
	ground = DrawInstance(terrain, material={"base_color": (0.30, 0.34, 0.30, 1)})
	colors = ((0.12, 0.55, 0.95, 1), (0.95, 0.45, 0.14, 1))
	legs = [[DrawInstance(box, material={"base_color": color}) for _ in range(2)] for color in colors]
	ray = DrawInstance(box, material={"base_color": (0.95, 0.85, 0.15, 1)})
	sphere = MaterialMesh(rounded_vertices(cube, SWEEP_RADIUS, 0), shader)
	sweep = DrawInstance(sphere, material={"base_color": (0.35, 0.90, 0.25, 1)})

	def update(index):
		joints, ray_hit, sweep_center = samples[index]
		for parts, points in zip(legs, joints, strict=True):
			for part, start, end in zip(parts, points[:-1], points[1:], strict=True):
				part.position, part.rotation, part.scale = segment_transform(start, end, 0.09)
		ray.position, ray.rotation, ray.scale = segment_transform(np.asarray(QUERY_START), ray_hit, 0.025)
		sweep.position = sweep_center

	camera = Camera((3.2, 2.5, 5.4), (0.55, 1.4, 0), (0, 1, 0), 40, near=0.05, far=60)
	title = "Physics queries | blue: foot IK, yellow: ray, green: sphere sweep"
	with viewing_target(f"{title} | Space: pause, R: restart", offscreen=offscreen) as output:
		renderer = MaterialRenderer([ground, ray, sweep, *legs[0], *legs[1]])
		play_frames(renderer, camera, output, update, label="Physics queries", sample_count=len(samples), sample_hz=TICK_HZ, check_frame=has_blue_pixels)


if __name__ == "__main__":
	from scenes.physics_viewing_utils import add_viewing_options

	parser = argparse.ArgumentParser(description=__doc__)
	add_viewing_options(parser)
	args = parser.parse_args()
	if args.headless:
		run_scene()
	else:
		run_viewer(offscreen=args.offscreen)
