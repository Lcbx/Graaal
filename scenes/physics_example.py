"""Falling ECS cubes, static obstacles, a moving platform and an occupancy-tinted sensor.

python main.py scenes/physics_example.py
python -m scenes.physics_example --headless --frames 240
python -m scenes.physics_example --offscreen --frames 2
"""

import argparse

import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.MaterialTransform import MaterialTransform
from graaal.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody


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
		contacts = triggers = 0
		inside = np.empty(0, dtype=np.uint32)
		for _ in range(ticks):
			schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
			events = physics.events
			inside = update_sensor_occupants(physics, sensor, boxes, inside, events)
			contacts += np.count_nonzero(events["kind"] == pj.EventKind.CONTACT)
			triggers += np.count_nonzero(events["kind"] == pj.EventKind.TRIGGER)
		positions = ecs.get(MaterialTransform)[boxes].position
		print(f"ticks={physics.world.tick} contacts={contacts} triggers={triggers} inside={len(inside)} heights={positions[:, 1].round(2)}")
		return positions


def cube_vertices():
	from graaal.TransparencyData import VERTEX_DTYPE

	# Each face has its own normals and UVs; collision geometry is a separate asset.
	normals = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.float32)
	up = np.array([(0, 1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (0, 1, 0)], dtype=np.float32)
	right = np.cross(up, normals)
	uv = np.array([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)], dtype=np.float32)
	vertices = np.zeros((6, 6), dtype=VERTEX_DTYPE)
	vertices["position"] = normals[:, None] * 0.5 + right[:, None] * (uv[None, :, :1] - 0.5) + up[:, None] * (uv[None, :, 1:] - 0.5)
	vertices["normal"] = normals[:, None]
	vertices["uv"] = uv
	return vertices.reshape(-1)


def run_scene(*, offscreen=False, frames=None):
	import pyjoltphysics as pj
	from graaal.DefaultMaterial import DefaultMaterialShader
	from graaal.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.MaterialRenderer import MaterialMesh, MaterialRenderer
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext, Texture, TextureUsage

	ecs, boxes, floor, obstacle, platform, sensor = make_world()
	physics = EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, interpolate=True)
	try:
		if offscreen:
			RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
		else:
			RenderContext.init_window(1000, 760, "Physics | sensor: green when empty, redder with more boxes", required_gpu_features=["indirect-first-instance"])
		mesh = MaterialMesh(cube_vertices(), DefaultMaterialShader())
		sensor_mesh = MaterialMesh(cube_vertices(), DefaultMaterialShader(transparent=True))
		ecs.register(MaterialInstance)
		ecs.add(boxes, MaterialInstance(mesh, {"base_color": (0.9, 0.35, 0.1, 1)}))
		ecs.add(floor, MaterialInstance(mesh, {"base_color": (0.3, 0.3, 0.35, 1)}))
		ecs.add(obstacle, MaterialInstance(mesh, {"base_color": (0.5, 0.5, 0.6, 1)}))
		ecs.add(platform, MaterialInstance(mesh, {"base_color": (0.1, 0.4, 0.9, 1)}))
		sensor_material = {"base_color": (0.1, 0.9, 0.3, 0.15), "alpha_mode": 2}
		ecs.add(sensor, MaterialInstance(sensor_mesh, sensor_material))
		renderer = MaterialRenderer(meshes=[mesh, sensor_mesh])
		accumulator, displayed = 0.0, 0
		drawing = EcsMaterialAdapter(renderer, ecs, transform_source=lambda owners: physics.render_transforms(owners, alpha=accumulator * 120))
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
		size = (640, 480)
		output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC) if offscreen else None
		if offscreen and frames is None: frames = 1
		while frames is None or displayed < frames:
			if not offscreen and not RenderContext.window_loop(): break
			accumulator += 1 / 60 if offscreen else min(RenderContext.frame_time, 0.1)
			fixed_steps = int(accumulator * 120)
			accumulator = max(0.0, accumulator - fixed_steps / 120)
			schedule.run(1 / 60 if offscreen else RenderContext.frame_time, fixed_steps=fixed_steps, fixed_dt=1 / 120)
			drawing.sync()
			commands = RenderContext.commands("physics")
			size = size if offscreen else RenderContext.window_dimensions
			renderer.record(commands, size, camera, output.color_attachment() if offscreen else RenderContext.screen())
			RenderContext.submit(commands.finish())
			displayed += 1
		if offscreen:
			pixels = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": size[0] * 4}, (*size, 1))
			if not np.any(np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 4)[:, :3]): raise RuntimeError("physics scene rendered no colored pixels")
			print(f"Physics scene rendered {displayed} offscreen frames")
	finally:
		physics.close()
		RenderContext.cleanup()


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
