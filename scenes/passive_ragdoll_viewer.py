"""Playback of the passive ragdoll example using the engine's direct draw API."""

from contextlib import nullcontext
from functools import cache
from pathlib import Path

import numpy as np

from graaal.Math import Quaternion
from scenes.passive_ragdoll_example import SCENARIOS, make_humanoid, run_scene, terrain_boxes
from scenes.physics_geometry import cube_vertices, rounded_vertices
from scenes.physics_viewing_utils import has_blue_pixels, play_frames, viewing_target


DEFAULT_DUMMY_MODEL = Path(__file__).with_name("resources") / "dummy_rigged.glb"
SCENARIO_LABELS = tuple(name.replace("_", " ").capitalize() for name in SCENARIOS)


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
	"""Replay with colliders or a dummy rig and return the last viewed scenario's report."""
	if model:
		path = DEFAULT_DUMMY_MODEL if model is True else Path(model)
		if not path.is_file():
			raise FileNotFoundError(f"Dummy model was not found at '{path}'. Supply your copy of dummy_rigged.glb with --model PATH or place it at '{DEFAULT_DUMMY_MODEL}'.")

	import pyjoltphysics as pj
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
	from graaal.Math import Camera

	settings, *_ = make_humanoid(height=height, mass=mass)
	if model:
		from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer

		asset, skin_pose, mesh_scale = dummy_skin(settings, height, path)
	else:
		offsets = np.array([bone.shape_offset for bone in settings.bones])
		shape_rotations = np.array([bone.shape_rotation for bone in settings.bones])

	@cache
	def recording(name):
		positions, orientations = [], []

		def capture(position, orientation):
			positions.append(position.copy())
			orientations.append(orientation.copy())

		result = run_scene(name, frames=frames, height=height, mass=mass, on_pose=capture, **options)
		return result, np.array(positions), np.array(orientations)

	def camera_points(name, poses):
		center = (poses.min(axis=(0, 1)) + poses.max(axis=(0, 1))) / 2
		distance = max(height, float(np.ptp(poses, axis=0).max()))
		offset = (-0.35, 0.45, 1.68) if name == "stairs" else (1.1, 0.7, 2.0)
		return center + distance * np.array(offset), center

	def window_title(index):
		keys = ", ".join(f"{i} {label}" for i, label in enumerate(SCENARIO_LABELS, 1))
		return f"Passive ragdoll | {SCENARIO_LABELS[index]} | {keys} | Space: pause, R: restart"

	result, positions, orientations = recording(scenario)
	selected = SCENARIOS.index(scenario)
	with viewing_target(window_title(selected), offscreen=offscreen) as output:
		shader = DefaultMaterialShader()
		cube = cube_vertices()
		box_mesh = MaterialMesh(cube, shader)

		@cache
		def terrain(name):
			ground_positions, ground_sizes = terrain_boxes(name, height / 1.8)
			return [DrawInstance(box_mesh, material={"base_color": (0.27, 0.30, 0.32, 1)}, position=position, scale=size) for position, size in zip(ground_positions, ground_sizes)]

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
		meshes = [box_mesh, *(part.mesh for part in body_parts)]
		renderer = MaterialRenderer(body_parts + terrain(scenario), meshes=meshes, skin_palette=palette)
		camera = Camera(*camera_points(scenario, positions), (0, 1, 0), 43, near=0.02 * height, far=100 * height)

		def update(tick):
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

		selector = nullcontext()
		if not offscreen:
			import glfw
			from graaal.RenderContext import RenderContext
			from graaal.UI.ImGui import ImGui

			def select_key(key, scancode, action, mods):
				nonlocal selected
				if action == glfw.PRESS and glfw.KEY_1 <= key <= glfw.KEY_4:
					selected = key - glfw.KEY_1
					return True
				return False

			RenderContext.subscribe_event("key", select_key)
			try:
				selector = ImGui()
			except ModuleNotFoundError as error:
				if error.name != "imgui_bundle":
					raise

		def switch_scenario():
			nonlocal scenario, result, positions, orientations
			name = SCENARIOS[selected]
			if name == scenario:
				return False
			result, positions, orientations = recording(name)
			scenario = name
			renderer.instances[len(body_parts):] = terrain(name)
			camera.position[:], camera.target[:] = camera_points(name, positions)
			glfw.set_window_title(RenderContext.window, window_title(selected))
			return True

		sample_hz = result["tick_hz"]
		offscreen_samples = (0, frames if scenario == "zero_gravity" else min(frames, sample_hz))
		with selector as gui:
			def draw_selector():
				nonlocal selected
				imgui = gui.imgui
				imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
				flags = imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_collapse
				expanded, _ = imgui.begin("Playback", flags=flags)
				try:
					if expanded:
						imgui.set_next_item_width(144)
						_, selected = imgui.combo("Scenario", selected, SCENARIO_LABELS)
				finally:
					imgui.end()

			play_frames(renderer, camera, output, update, label="Passive ragdoll", sample_count=len(positions), sample_hz=sample_hz,
				offscreen_samples=offscreen_samples, check_frame=has_blue_pixels, before_frame=switch_scenario,
				draw_overlay=(lambda: gui.render(draw_selector)) if gui is not None else None)
	return result
