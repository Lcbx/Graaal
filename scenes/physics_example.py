"""Falling ECS cubes, static obstacles, a moving platform and an occupancy-tinted sensor.

python main.py scenes/physics_example.py
python -m scenes.physics_example --headless --frames 240
python -m scenes.physics_example --offscreen --frames 2
"""

import argparse

import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
from scenes.physics_geometry import cube_vertices
from scenes.physics_utils import advance_fixed_steps
from scenes.physics_viewing_utils import draw_frame, read_pixels, viewing_target


def make_world():
	"""Build physics declarations without loading any rendering modules."""
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform)
	boxes = ecs.create(12)
	ecs.add(boxes, Collider(0), RigidBody(), MaterialTransform())
	i = np.arange(len(boxes))
	ecs.get(MaterialTransform)[boxes].position = np.column_stack(((i % 4 - 1.5) * 1.6, 4 + i // 4 * 1.5, np.zeros(len(i))))
	floor, obstacle, platform, sensor = ecs.create(4)
	ecs.add(floor, Collider(0), MaterialTransform(position=(0, -0.5, 0), scale=(18, 1, 12)))
	ecs.add(obstacle, Collider(0), MaterialTransform(position=(4, 0.75, 0), scale=(1, 1.5, 4)))
	ecs.add(platform, Collider(0), RigidBody(motion=1), MaterialTransform(position=(-2, 2, 0), scale=(4, 0.3, 3)))
	ecs.add(sensor, Collider(0, sensor=True), MaterialTransform(position=(0, 1, 0), scale=(6, 1.5, 4)))
	return ecs, boxes, floor, obstacle, platform, sensor


def make_schedule(ecs, physics, platform):
	schedule = Scheduler(ecs)
	def move_platform(world, commands, dt):
		time = (physics.world.tick + 1) / physics.world.settings.tick_hz
		world.get(MaterialTransform)[[platform]].position = (-2 + np.sin(time) * 2, 2, 0)
	schedule.add("platform", move_platform, phase="fixed_update")
	for name, callback, after in (("physics_sync", physics.reconcile, "platform"), ("physics_input", physics.prepare_inputs, "physics_sync"), ("physics_step", physics.step, "physics_input"), ("physics_output", physics.pull, "physics_step")):
		schedule.add(name, callback, phase="fixed_update", after=after)
	schedule.add("physics_events", physics.dispatch_events, phase="fixed_update", after="physics_output")
	return schedule


def update_sensor_occupants(physics, sensor, boxes, inside, events):
	"""Track current box UIDs from trigger transitions, excluding other bodies."""
	sensor_uid = physics.body_uids.get(int(sensor))
	box_uids = np.array([physics.body_uids.get(int(box), 0) for box in boxes], dtype=np.uint32)
	if sensor_uid is None: return np.empty(0, dtype=np.uint32)
	triggers = events[(events["kind"] == physics.pj.EventKind.TRIGGER) & (events["a"] == sensor_uid)]
	if len(triggers):
		# The last transition wins if a pair begins and ends within this batch.
		_, last = np.unique(triggers["b"][::-1], return_index=True)
		latest = triggers[::-1][last]
		active = (latest["phase"] == physics.pj.EventPhase.BEGIN) | (latest["phase"] == physics.pj.EventPhase.PERSIST)
		inside = np.union1d(np.setdiff1d(inside, latest["b"]), latest["b"][active])
	return inside[np.isin(inside, box_uids)]


def run_headless(ticks=240):
	import pyjoltphysics as pj

	ecs, boxes, floor, obstacle, platform, sensor = make_world()
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, settings=pj.WorldSettings(threads=1)) as physics:
		schedule = make_schedule(ecs, physics, platform)
		fixed_dt = 1 / physics.world.settings.tick_hz
		contacts = triggers = 0
		inside = np.empty(0, dtype=np.uint32)
		for _ in range(ticks):
			schedule.run(fixed_dt, fixed_steps=1, fixed_dt=fixed_dt)
			events = physics.events
			inside = update_sensor_occupants(physics, sensor, boxes, inside, events)
			contacts += np.count_nonzero(events["kind"] == pj.EventKind.CONTACT)
			triggers += np.count_nonzero(events["kind"] == pj.EventKind.TRIGGER)
		positions = ecs.get(MaterialTransform)[boxes].position
		print(f"ticks={physics.world.tick} contacts={contacts} triggers={triggers} inside={len(inside)} heights={positions[:, 1].round(2)}")
		return positions


def run_scene(*, offscreen=False, frames=None):
	import pyjoltphysics as pj
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.Materials.MaterialRenderer import MaterialMesh, MaterialRenderer
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext

	ecs, boxes, floor, obstacle, platform, sensor = make_world()
	physics = EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, interpolate=True)
	with physics, viewing_target("Physics | sensor: green when empty, redder with more boxes", offscreen=offscreen) as output:
		cube = cube_vertices()
		mesh = MaterialMesh(cube, DefaultMaterialShader())
		sensor_mesh = MaterialMesh(cube, DefaultMaterialShader(transparent=True))
		ecs.register(MaterialInstance)
		ecs.add(boxes, MaterialInstance(mesh, {"base_color": (0.9, 0.35, 0.1, 1)}))
		ecs.add(floor, MaterialInstance(mesh, {"base_color": (0.3, 0.3, 0.35, 1)}))
		ecs.add(obstacle, MaterialInstance(mesh, {"base_color": (0.5, 0.5, 0.6, 1)}))
		ecs.add(platform, MaterialInstance(mesh, {"base_color": (0.1, 0.4, 0.9, 1)}))
		sensor_material = {"base_color": (0.1, 0.9, 0.3, 0.15), "alpha_mode": 2}
		ecs.add(sensor, MaterialInstance(sensor_mesh, sensor_material))
		renderer = MaterialRenderer(meshes=[mesh, sensor_mesh])
		tick_hz = physics.world.settings.tick_hz
		fixed_dt = 1 / tick_hz
		accumulator, displayed = 0.0, 0
		drawing = EcsMaterialAdapter(renderer, ecs, transform_source=lambda owners: physics.render_transforms(owners, alpha=accumulator * tick_hz))
		schedule = make_schedule(ecs, physics, platform)
		inside = np.empty(0, dtype=np.uint32)
		def update_sensor(world, commands, dt):
			nonlocal inside
			previous = len(inside)
			inside = update_sensor_occupants(physics, sensor, boxes, inside, physics.events)
			if len(inside) == previous: return
			weight = 1 - 0.65 ** len(inside)
			sensor_material["base_color"] = (0.1 + 0.9 * weight, 0.9 - 0.75 * weight, 0.3 - 0.25 * weight, 0.15)
			drawing.touch_materials(sensor_material)
		schedule.add("sensor_tint", update_sensor, phase="fixed_update", after="physics_events")
		camera = Camera((12, 10, 16), (0, 2, 0), (0, 1, 0), 50, near=0.1, far=100)
		if offscreen and frames is None: frames = 1
		while frames is None or displayed < frames:
			if not offscreen and not RenderContext.window_loop(): break
			elapsed = 1 / 60 if offscreen else RenderContext.frame_time
			fixed_steps, accumulator = advance_fixed_steps(accumulator, min(elapsed, 0.1), tick_hz)
			schedule.run(elapsed, fixed_steps=fixed_steps, fixed_dt=fixed_dt)
			drawing.sync()
			draw_frame(renderer, camera, output, "physics")
			displayed += 1
		if offscreen:
			if not np.any(read_pixels(output)[..., :3]): raise RuntimeError("physics scene rendered no colored pixels")
			print(f"Physics scene rendered {displayed} offscreen frames")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("--headless", action="store_true")
	mode.add_argument("--offscreen", action="store_true")
	parser.add_argument("--frames", type=int)
	args = parser.parse_args()
	if args.frames is not None and args.frames <= 0: parser.error("--frames must be positive")
	if args.headless: run_headless(240 if args.frames is None else args.frames)
	else: run_scene(offscreen=args.offscreen, frames=args.frames)
