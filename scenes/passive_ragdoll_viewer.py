"""Playback of the passive ragdoll example using the engine's direct draw API."""

from pathlib import Path

import numpy as np

from graaal.Math import Quaternion
from scenes.passive_ragdoll_example import make_humanoid, run_scene, terrain_boxes
from scenes.physics_geometry import cube_vertices


DEFAULT_DUMMY_MODEL = Path(__file__).with_name("resources") / "dummy_rigged.glb"


def rounded_vertices(template, radius, cylinder_length):
	u = np.linspace(0, 2 * np.pi, 25)
	v = np.concatenate((np.linspace(0, np.pi / 2, 9), np.linspace(np.pi / 2, np.pi, 9)))
	u, v = np.meshgrid(u, v)
	normals = np.stack((np.sin(v) * np.cos(u), np.cos(v), np.sin(v) * np.sin(u)), axis=-1)
	grid = np.zeros(u.shape, dtype=template.dtype)
	grid["position"], grid["normal"] = normals * radius, normals
	# Duplicate the equator to connect the two hemispheres with a cylinder.
	grid["position"][:9, :, 1] += cylinder_length / 2
	grid["position"][9:, :, 1] -= cylinder_length / 2
	grid["uv"] = np.stack((u / (2 * np.pi), v / np.pi), axis=-1)
	return np.stack((grid[:-1, :-1], grid[1:, 1:], grid[1:, :-1],
		grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:]), axis=2).reshape(-1)


def dummy_skin(settings, height, path):
	"""Fit the supplied T-pose rig to the playback's joint rotations."""
	from graaal.Animation.SkeletalAnimation import model_matrices, skin_matrices
	from graaal.Animation.SkeletalGltf import load_skinned_gltf

	asset = load_skinned_gltf(path)
	rig = asset.skeleton
	# These are glTF node IDs for the supplied dummy, in the simulation's bone order.
	nodes = (51, 40, 0, 19, 18, 38, 37, 46, 45, 44, 50, 49, 48)
	if not set((*nodes, 17, 36)).issubset(asset.joint_nodes) or asset.joint_nodes[0] != nodes[0]:
		raise ValueError(f"Model '{path}' must use the dummy_rigged.glb skeleton")
	mapped = np.array([asset.joint_nodes.index(node) for node in nodes])
	rest = model_matrices(rig, rig.translation[None], rig.rotation[None], rig.scale[None])[0]
	aligned = rest[mapped, :3, :3].copy()
	for upper, lower, hand in (("upper_arm_l", "lower_arm_l", 17), ("upper_arm_r", "lower_arm_r", 36)):
		upper, lower = settings.bone_index(upper), settings.bone_index(lower)
		hand = asset.joint_nodes.index(hand)
		for part, tip in ((upper, mapped[lower]), (lower, hand)):
			direction = rest[tip, :3, 3] - rest[mapped[part], :3, 3]
			direction /= np.linalg.norm(direction)
			down = np.array([0, -1, 0])
			axis = np.cross(direction, down)
			rotation = Quaternion.from_axis_rotation(axis / np.linalg.norm(axis), np.arccos(np.clip(direction @ down, -1, 1)))
			aligned[part] = rotation.matrix33 @ aligned[part]
	local = Quaternion.matrix_batch(rig.rotation)
	levels = [np.flatnonzero(rig.depth == depth) for depth in range(1, int(rig.depth.max()) + 1)]

	def pose(turns):
		world = np.broadcast_to(np.eye(4, dtype=np.float32), rest.shape).copy()
		targets = Quaternion.matrix_batch(turns) @ aligned
		world[0, :3, :3] = targets[0]
		for joints in levels:
			parent = world[rig.parents[joints]]
			world[joints, :3, 3] = (parent[:, :3, :3] @ rig.translation[joints, :, None])[..., 0] + parent[:, :3, 3]
			world[joints, :3, :3] = parent[:, :3, :3] @ local[joints]
			selected = np.isin(mapped, joints)
			world[mapped[selected], :3, :3] = targets[selected]
		return skin_matrices(rig, world[None])

	vertices = np.concatenate([part.vertices["position"] for part in asset.primitives])
	return asset, pose, height / float(np.ptp(vertices[:, 1]))


def run_viewer(scenario, *, offscreen=False, frames=960, height=1.8, mass=70.0, model=False, **options):
	"""Replay a scenario with colliders, the default dummy rig, or a supplied rig path."""
	if model:
		path = DEFAULT_DUMMY_MODEL if model is True else Path(model)
		if not path.is_file():
			raise FileNotFoundError(f"Dummy model was not found at '{path}'. Supply your copy of dummy_rigged.glb with --model PATH or place it at '{DEFAULT_DUMMY_MODEL}'.")

	import pyjoltphysics as pj
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext, Texture, TextureUsage

	settings, *_ = make_humanoid(height=height, mass=mass)
	if model:
		from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer

		asset, skin_pose, mesh_scale = dummy_skin(settings, height, path)
	else:
		offsets = np.array([bone.shape_offset for bone in settings.bones])
		shape_rotations = np.array([bone.shape_rotation for bone in settings.bones])

	positions, orientations = [], []

	def capture(position, orientation):
		positions.append(position.copy())
		orientations.append(orientation.copy())

	result = run_scene(scenario, frames=frames, height=height, mass=mass, on_pose=capture, **options)
	ground_positions, ground_sizes = terrain_boxes(scenario, height / 1.8)
	positions, orientations = np.array(positions), np.array(orientations)
	try:
		features = ["indirect-first-instance"]
		if offscreen:
			RenderContext.setup_graphics_backend(True, required_features=features)
		else:
			RenderContext.init_window(1000, 760, f"Passive ragdoll | {scenario.replace('_', ' ')} | Space: pause, R: restart", required_gpu_features=features)

		shader = DefaultMaterialShader()
		cube = cube_vertices()
		box_mesh = MaterialMesh(cube, shader)
		terrain = [DrawInstance(box_mesh, material={"base_color": (0.27, 0.30, 0.32, 1)}, position=position, scale=size) for position, size in zip(ground_positions, ground_sizes)]
		if model:
			bones = len(asset.skeleton.parents)
			palette = SkinMatrixBuffer(bones * len(asset.primitives))
			body_parts = [DrawInstance(MaterialMesh(part.vertices, shader, indices=part.indices, bounds=((0, 0, 0), (36, 36, 36))), material={"base_color": (0.15, 0.55, 0.93, 1)}, scale=(mesh_scale,) * 3) for part in asset.primitives]
			skin_ids = [part.surface_id for part in body_parts]
			for owner in skin_ids:
				palette.acquire(owner, bones)
		else:
			palette = None
			body_parts = []
			for bone in settings.bones:
				color = (0.12, 0.56, 0.92, 1) if bone.name.endswith("_l") else (0.96, 0.40, 0.13, 1) if bone.name.endswith("_r") else (0.78, 0.83, 0.91, 1)
				if bone.shape.kind == pj.ShapeKind.BOX:
					mesh, scale = box_mesh, bone.shape.params[:3]
				else:
					radius = bone.shape.params[0]
					cylinder_length = bone.shape.params[1] if bone.shape.kind == pj.ShapeKind.CAPSULE else 0
					mesh = MaterialMesh(rounded_vertices(cube, radius, cylinder_length), shader)
					scale = (1, 1, 1)
				body_parts.append(DrawInstance(mesh, material={"base_color": color}, scale=scale))
		renderer = MaterialRenderer(body_parts + terrain, skin_palette=palette)
		center = (positions.min(axis=(0, 1)) + positions.max(axis=(0, 1))) / 2
		distance = max(height, float(np.ptp(positions, axis=0).max()))
		camera_offset = (-0.35, 0.45, 1.68) if scenario == "stairs" else (1.1, 0.7, 2.0)
		camera = Camera(center + distance * np.array(camera_offset), center, (0, 1, 0), 43, near=0.02 * height, far=100 * height)
		size = (640, 480)
		output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC) if offscreen else None
		cursor, paused = 0.0, False
		if not offscreen:
			import glfw

			def key_pressed(key, scancode, action, mods):
				nonlocal cursor, paused
				if action == glfw.PRESS:
					if key == glfw.KEY_SPACE:
						paused = not paused
					elif key == glfw.KEY_R:
						cursor = 0.0

			RenderContext.subscribe_event("key", key_pressed)

		displayed = 0
		while displayed < 2 or not offscreen:
			if not offscreen and not RenderContext.window_loop():
				break
			tick = min(int(cursor), frames)
			turns = orientations[tick]
			if model:
				for part in body_parts:
					part.position = positions[tick, 0]
				palette.write(skin_ids, np.repeat(skin_pose(turns), len(skin_ids), axis=0))
			else:
				centers = positions[tick] + np.einsum("bij,bj->bi", Quaternion.matrix_batch(turns), offsets)
				rotations = Quaternion.multiply_batch(turns, shape_rotations)
				for part, position, rotation in zip(body_parts, centers, rotations):
					part.position, part.rotation = position, rotation
			renderer.sync()
			commands = RenderContext.commands("Passive ragdoll")
			renderer.record(commands, size if offscreen else RenderContext.window_dimensions, camera, output.color_attachment() if offscreen else RenderContext.screen())
			RenderContext.submit(commands.finish())
			displayed += 1
			if offscreen:
				pixels = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": size[0] * 4}, (*size, 1))
				rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 4).astype(np.int16)
				if not np.any((rgb[:, 2] > rgb[:, 0] + 20) & (rgb[:, 2] > 80)):
					raise RuntimeError("passive ragdoll rendered no blue body pixels")
				cursor = frames if scenario == "zero_gravity" else min(frames, 120)
			elif not paused:
				# Hold the last pose for one second before replaying the fall.
				cursor = (cursor + min(RenderContext.frame_time, 0.1) * 120) % (frames + 120)
		if offscreen: print(f"Passive ragdoll rendered {displayed} offscreen frames")
		return result
	finally:
		RenderContext.cleanup()
