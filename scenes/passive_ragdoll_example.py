"""Python-authored passive humanoid: zero gravity, floor drop, stairs and impact.

python -m scenes.passive_ragdoll_example --headless --scenario all --frames 960
python -m scenes.passive_ragdoll_example --scenario stairs
"""

import argparse
import json
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import numpy as np

from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import joint_limit_torques, joint_spring_torques, ragdoll_limit_parameters
from scenes.physics_geometry import collider_bounds, collider_poses, cube_vertices, cylinder_vertices, rounded_vertices
from scenes.physics_utils import joint_error
from scenes.physics_viewing_utils import add_viewing_options, has_blue_pixels, play_frames, viewing_target


SCENARIOS = ("zero_gravity", "drop", "stairs", "impact")
HUMANOID_NAMES = ("pelvis", "chest", "head", "upper_arm_l", "lower_arm_l", "upper_arm_r", "lower_arm_r", "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r")
HUMANOID_PARENTS = (-1, 0, 1, 1, 3, 1, 5, 0, 7, 8, 0, 10, 11)
DEFAULT_DUMMY_MODEL = Path(__file__).with_name("resources") / "dummy_rigged.glb"
SCENARIO_LABELS = tuple(name.replace("_", " ").capitalize() for name in SCENARIOS)


def make_humanoid(*, height=1.8, mass=70.0):
	"""Return an example 13-body asset, relaxed local rotations and passive gains.

	Translations/shapes bake height; masses and gains scale with mass. Gains
	scale with length squared to retain their reference angular frequencies.
	These are illustrative authored settings, evaluated at 120 Hz.
	"""
	import pyjoltphysics as pj

	if not np.isfinite(height) or not np.isfinite(mass) or height <= 0 or mass <= 0: raise ValueError("height and mass must be finite and positive")
	scale = height / 1.8
	translation = np.array([(0, 0.99, 0), (0, 0.30, 0), (0, 0.32, 0), (0.19, 0.24, 0), (0, -0.30, 0), (-0.19, 0.24, 0), (0, -0.30, 0), (0.10, -0.06, 0), (0, -0.44, 0), (0, -0.39, 0), (-0.10, -0.06, 0), (0, -0.44, 0), (0, -0.39, 0)]) * scale
	radius = np.array([0.12, 0.13, 0.10, 0.055, 0.045, 0.055, 0.045, 0.075, 0.055, 0.06, 0.075, 0.055, 0.06]) * scale
	settings = pj.RagdollSettings.from_skeleton(HUMANOID_NAMES, HUMANOID_PARENTS, translation, radius=radius, mass=mass, self_collision=True, friction=0.8, restitution=0, linear_damping=0.05, angular_damping=0.3, root_force=0, root_torque=0, flags=pj.BodyFlag.REPORT | pj.BodyFlag.NO_SLEEP)
	masses = np.array([14, 22, 5, 2, 1.5, 2, 1.5, 7, 3, 1, 7, 3, 1]) * (mass / 70)
	for bone, weight in zip(settings.bones, masses):
		bone.mass = float(weight)
		bone.strength = 0
		bone.friction = 0
	limits = dict(chest=((-0.3, 0.3), (0.4, 0.3), 0), head=((-0.7, 0.7), (0.5, 0.4), 0), upper_arm=((-0.6, 0.6), (1.4, 1.2), 0), lower_arm=((-0.6, 0.6), (0.7, 0.05), -0.7), thigh=((-0.4, 0.4), (1.0, 0.5), -0.5), shin=((-0.1, 0.1), (0.65, 0.05), 0.65), foot=((-0.15, 0.15), (0.55, 0.25), 0.10))
	for bone in settings.bones[1:]:
		name = bone.name[:-2] if bone.name.endswith(("_l", "_r")) else bone.name
		bone.twist, bone.swing, bone.bend = limits[name]
		if name in ("lower_arm", "thigh", "shin", "foot"): bone.plane_axis = (1, 0, 0)
		if name == "thigh":
			# Extend toward the pelvis while leaving the knee end in place.
			radius, cylinder_length = bone.shape.params[:2]
			extension = 0.085 * scale
			bone.shape = pj.Shape.capsule(radius, cylinder_length + extension)
			bone.shape_offset = tuple(np.asarray(bone.shape_offset) - np.asarray(bone.twist_axis) * (extension / 2))
		if name == "foot":
			bone.twist_axis = (0, 1, 0)
			bone.shape = pj.Shape.box(tuple(np.array([0.14, 0.10, 0.26]) * scale))
			# The shin ends at the ankle, where the top of the foot box meets it.
			bone.shape_offset = (0, -0.05 * scale, 0.055 * scale)
			bone.shape_rotation = (0, 0, 0, 1)
	head = settings.bones[settings.bone_index("head")]
	head.shape = pj.Shape.sphere(0.11 * scale)
	head.shape_offset = (0, 0.08 * scale, 0)
	head.shape_rotation = (0, 0, 0, 1)
	neutral = np.array([Quaternion.from_x_rotation(angle) for angle in (0, -0.06, 0.06, -0.28, -0.35, -0.28, -0.35, -0.48, 0.76, -0.28, -0.48, 0.76, -0.28)])
	for name, angle in (("upper_arm_l", 0.25), ("upper_arm_r", -0.25), ("thigh_l", 0.22), ("thigh_r", -0.22)):
		i = settings.bone_index(name)
		neutral[i] = Quaternion.multiply_batch(Quaternion.from_z_rotation(angle), neutral[i])
	inertia_scale = (mass / 70) * scale**2
	spring = dict(stiffness=np.array([0, 1, 0.15, 0.3, 0.10, 0.3, 0.10, 1, 0.6, 0.12, 1, 0.6, 0.12]) * inertia_scale,
		damping=np.array([0, 0.05, 0.008, 0.008, 0.003, 0.008, 0.003, 0.03, 0.012, 0.003, 0.03, 0.012, 0.003]) * inertia_scale,
		max_torque=np.array([0, 1, 0.15, 0.25, 0.10, 0.25, 0.10, 1, 0.5, 0.10, 1, 0.5, 0.10]) * inertia_scale)
	limit_scale = 1.25 * inertia_scale
	limit = dict(ragdoll_limit_parameters(settings), margin=0.08,
		stiffness=np.array([0, 3, 1, 0.5, 0.25, 0.5, 0.25, 3, 2, 0.35, 3, 2, 0.35]) * limit_scale,
		damping=np.array([0, 0.05, 0.01, 0.02, 0.01, 0.02, 0.01, 0.08, 0.05, 0.006, 0.08, 0.05, 0.006]) * limit_scale,
		max_torque=np.array([0, 2, 0.3, 0.4, 0.2, 0.4, 0.2, 3, 2, 0.25, 3, 2, 0.25]) * limit_scale)
	return settings, neutral, spring, limit


def terrain_boxes(scenario, scale):
	positions = np.empty((0, 3))
	sizes = np.empty((0, 3))
	if scenario != "zero_gravity":
		positions = np.array([[0, -0.5, 0]])
		sizes = np.array([[20, 1, 20]])
	if scenario == "stairs":
		i = np.arange(6)
		top = (i + 1) * 0.22
		positions = np.concatenate((positions, np.column_stack(((i + 0.5) * 0.4, top / 2, np.zeros(6)))))
		sizes = np.concatenate((sizes, np.column_stack((np.full(6, 0.4), top, np.full(6, 3)))))
	return positions * scale, sizes * scale


def _rest_collider_bounds(settings, rotation=(0, 0, 0, 1)):
	from pyjoltphysics.ragdolls import forward_kinematics

	positions, orientations = forward_kinematics(settings, None, (0, 0, 0), rotation)
	return collider_bounds(settings.bones, positions, orientations)


def _body_peak_speeds(state):
	linear = float(np.max(np.linalg.norm(state.linear_velocity, axis=-1)))
	angular = float(np.max(np.linalg.norm(state.angular_velocity, axis=-1)))
	return linear, angular


def run_scene(scenario="drop", *, frames=960, threads=1, height=None, mass=None, tuning=None, passive=True, on_pose=None):
	"""Run a bounded passive scene and return detached results.

	tuning can be a loaded RagdollProject with its own dimensions and masses.
	on_pose receives joint positions and orientations at tick zero and after each step.
	"""
	import pyjoltphysics as pj

	if scenario not in SCENARIOS: raise ValueError("unknown passive scenario")
	if frames < 1 or threads < 1: raise ValueError("frames and threads must be positive")
	if tuning is None:
		height = 1.8 if height is None else height
		mass = 70.0 if mass is None else mass
		settings, neutral, spring, limit = make_humanoid(height=height, mass=mass)
	else:
		if height is not None or mass is not None:
			raise ValueError("tuning supplies its own height and mass")
		settings, neutral, spring, limit = tuning.settings, tuning.neutral, tuning.spring, tuning.limit_parameters()
		lower, upper = _rest_collider_bounds(settings)
		height = float(upper[:, 1].max() - lower[:, 1].min())
		mass = sum(bone.mass for bone in settings.bones)
	parents, translation = settings.parents, settings.rest_translations
	masses = np.array([bone.mass for bone in settings.bones])
	bone_indices = {bone.name: i for i, bone in enumerate(settings.bones)}
	foot_indices = [bone_indices[name] for name in ("foot_l", "foot_r") if name in bone_indices]
	chest = bone_indices.get("chest", 0)
	scale = height / 1.8
	gravity = (0, 0, 0) if scenario == "zero_gravity" else (0, -9.81, 0)
	with pj.World(pj.WorldSettings(tick_hz=120, threads=threads, gravity=gravity, allow_sleeping=False)) as world:
		positions, sizes = terrain_boxes(scenario, scale)
		static = np.empty(0, dtype=np.uint32)
		if len(positions):
			static, _ = world.add_bodies(pj.BodyBatch.boxes(positions, size=sizes, motion=pj.MotionType.STATIC, friction=0.8))
		floor, steps = static[:1], static[1:]
		position, rotation = (0, 0, 0), Quaternion()
		if scenario == "drop": position, rotation = (0, 1.6 * scale, 0), Quaternion.from_z_rotation(0.35)
		elif scenario == "stairs":
			rotation = Quaternion.from_z_rotation(0.7)
			step_top = positions[-1, 1] + sizes[-1, 1] / 2
			position = (2.2 * scale, step_top + 0.01 * scale, 0)
		elif scenario == "impact": position = (0, 0.05 * scale, 0)
		if scenario == "stairs" or (tuning is not None and scenario != "zero_gravity"):
			lower, upper = _rest_collider_bounds(settings, rotation)
			lowest = np.argmin(lower[:, 1])
			position = np.asarray(position, dtype=float)
			position[1] -= lower[lowest, 1]
			if tuning is not None and scenario == "stairs":
				position[[0, 2]] = positions[-1, [0, 2]] - (lower[lowest, [0, 2]] + upper[lowest, [0, 2]]) / 2
		handle = world.add_ragdoll(settings, position, rotation, mode=pj.RagdollMode.LIMP)
		handle.set_strength(np.zeros(len(settings)))
		if scenario == "stairs": world.set_velocities(handle.bones, linear=np.tile([-1.2 * scale, 0, 0], (len(settings), 1)))
		world.flush()
		state = world.bodies_state(handle.bones)
		if on_pose is not None: on_pose(state.position, state.orientation)
		initial_error = float(np.mean(joint_error(parents, state.orientation, neutral))) if len(parents) > 1 else 0.0
		feet = handle.bones[foot_indices]
		counts = dict(contact_records=0, floor_contact_records=0, stair_contact_records=0, foot_contact_records=0, self_contact_records=0)
		peak_anchor = peak_depth = peak_impulse = 0.0
		peak_linear, peak_angular = _body_peak_speeds(state)
		late_linear = late_angular = 0.0
		impact = None
		# Strike after foot contact, while the body is still upright.
		impact_tick = min(24, frames // 3)
		for tick in range(frames):
			if scenario == "impact" and tick == impact_tick:
				offset = Quaternion.matrix_batch(state.orientation[chest]) @ (np.array([0, 0.08, 0.08]) * scale)
				impulse = np.array([20, 3, 0]) * (mass / 70)
				handle.hit(chest, impulse, point=state.position[chest] + offset)
				world.flush()
				after = world.bodies_state(handle.bones)
				impact = dict(tick=world.tick, impulse_ns=impulse.copy(), offset_m=offset.copy(), delta_linear_velocity=after.linear_velocity[chest] - state.linear_velocity[chest], delta_angular_velocity=after.angular_velocity[chest] - state.angular_velocity[chest])
				state = after
				linear_speed, angular_speed = _body_peak_speeds(state)
				peak_linear = max(peak_linear, linear_speed)
				peak_angular = max(peak_angular, angular_speed)
			if passive:
				torque = joint_spring_torques(parents, state.orientation[None], state.angular_velocity[None], neutral, **spring)
				torque += joint_limit_torques(orientations=state.orientation[None], angular_velocity=state.angular_velocity[None], **limit)
				world.add_torques(handle.bones, torque[0])
			world.run_ticks(1)
			state = world.bodies_state(handle.bones)
			if not state.found.all() or not all(np.isfinite(value).all() for value in (state.position, state.orientation, state.linear_velocity, state.angular_velocity)): raise RuntimeError("missing or nonfinite passive ragdoll state")
			if on_pose is not None: on_pose(state.position, state.orientation)
			anchors = state.position[1:] - state.position[parents[1:]] - np.einsum("bij,bj->bi", Quaternion.matrix_batch(state.orientation[parents[1:]]), translation[1:])
			peak_anchor = max(peak_anchor, float(np.max(np.linalg.norm(anchors, axis=-1), initial=0)))
			linear_speed, angular_speed = _body_peak_speeds(state)
			peak_linear = max(peak_linear, linear_speed)
			peak_angular = max(peak_angular, angular_speed)
			if tick >= max(frames - 120, 0):
				late_linear = max(late_linear, linear_speed)
				late_angular = max(late_angular, angular_speed)
			events = world.drain_events()
			contacts = events[events["kind"] == pj.EventKind.CONTACT]
			if len(contacts):
				a, b = contacts["a"], contacts["b"]
				counts["contact_records"] += len(contacts)
				counts["floor_contact_records"] += int(np.count_nonzero(np.isin(a, floor) | np.isin(b, floor)))
				counts["stair_contact_records"] += int(np.count_nonzero(np.isin(a, steps) | np.isin(b, steps)))
				counts["foot_contact_records"] += int(np.count_nonzero((np.isin(a, feet) | np.isin(b, feet)) & (np.isin(a, static) | np.isin(b, static))))
				counts["self_contact_records"] += int(np.count_nonzero(np.isin(a, handle.bones) & np.isin(b, handle.bones)))
				peak_depth = max(peak_depth, float(np.max(contacts["depth"])))
				peak_impulse = max(peak_impulse, float(np.max(contacts["impulse"])))
		final_error = float(np.mean(joint_error(parents, state.orientation, neutral))) if len(parents) > 1 else 0.0
		return dict(scenario=scenario, passive=passive, height_m=height, mass_kg=mass, threads=threads, tick_hz=120, ticks=world.tick, bodies=len(settings),
			mode=int(world.latest().ragdolls.mode[0]), active_strength=float(world.latest().ragdolls.strength[0]),
			initial_joint_error_rad=initial_error, final_joint_error_rad=final_error, max_anchor_error_m=peak_anchor,
			max_linear_speed_m_s=peak_linear, max_angular_speed_rad_s=peak_angular, final_linear_kinetic_energy_j=float(0.5 * np.sum(masses * np.sum(state.linear_velocity**2, axis=-1))),
			late_max_linear_speed_m_s=late_linear, late_max_angular_speed_rad_s=late_angular,
			peak_contact_depth_m=peak_depth, peak_reported_contact_impulse_ns=peak_impulse, impact=impact, position=state.position.copy(), orientation=state.orientation.copy(),
			stats=world.stats(), **counts)


def dummy_skin(settings, height, path):
	"""Fit the supplied T-pose rig to the playback's joint rotations."""
	from graaal.Animation.SkeletalAnimation import model_matrices, skin_matrices
	from graaal.Animation.SkeletalGltf import load_skinned_gltf

	if tuple(bone.name for bone in settings.bones) != HUMANOID_NAMES or tuple(settings.parents) != HUMANOID_PARENTS:
		raise ValueError("--model requires the passive humanoid's bone names, parents and order")
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


def run_viewer(scenario, *, offscreen=False, frames=960, height=None, mass=None, tuning=None, model=False, **options):
	"""Replay with colliders or a dummy rig and return the last viewed scenario's report."""
	if model:
		path = DEFAULT_DUMMY_MODEL if model is True else Path(model)
		if not path.is_file():
			raise FileNotFoundError(f"Dummy model was not found at '{path}'. Supply your copy of dummy_rigged.glb with --model PATH or place it at '{DEFAULT_DUMMY_MODEL}'.")

	import pyjoltphysics as pj
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
	from graaal.Math import Camera

	@cache
	def recording(name):
		positions, orientations = [], []

		def capture(position, orientation):
			positions.append(position.copy())
			orientations.append(orientation.copy())

		result = run_scene(name, frames=frames, height=height, mass=mass, tuning=tuning, on_pose=capture, **options)
		return result, np.array(positions), np.array(orientations)

	def camera_points(name):
		if tuning is None:
			center = (positions.min(axis=(0, 1)) + positions.max(axis=(0, 1))) / 2
			distance = max(scene_height, float(np.ptp(positions, axis=0).max()))
		else:
			lower, upper = collider_bounds(settings.bones, positions, orientations)
			lower, upper = lower.min(axis=(0, 1)), upper.max(axis=(0, 1))
			center = (lower + upper) / 2
			distance = max(scene_height, float((upper - lower).max()))
		offset = (-0.35, 0.45, 1.68) if name == "stairs" else (1.1, 0.7, 2.0)
		far = 100 * scene_height if tuning is None else max(100 * scene_height, 10 * distance)
		return center + distance * np.array(offset), center, far

	def window_title(index):
		keys = ", ".join(f"{i} {label}" for i, label in enumerate(SCENARIO_LABELS, 1))
		return f"Passive ragdoll | {SCENARIO_LABELS[index]} | {keys} | Space: pause, R: restart"

	result, positions, orientations = recording(scenario)
	scene_height = result["height_m"]
	settings = tuning.settings if tuning is not None else make_humanoid(height=scene_height, mass=result["mass_kg"])[0]
	if model:
		from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer

		asset, skin_pose, mesh_scale = dummy_skin(settings, scene_height, path)
	else:
		offsets = np.array([bone.shape_offset for bone in settings.bones])
		shape_rotations = np.array([bone.shape_rotation for bone in settings.bones])
	selected = SCENARIOS.index(scenario)
	with viewing_target(window_title(selected), offscreen=offscreen) as output:
		shader = DefaultMaterialShader()
		cube = cube_vertices()
		box_mesh = MaterialMesh(cube, shader)

		@cache
		def terrain(name):
			ground_positions, ground_sizes = terrain_boxes(name, scene_height / 1.8)
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
			blue_bone = next((i for i, bone in enumerate(settings.bones) if bone.name.endswith("_l")), 0)
			for i, bone in enumerate(settings.bones):
				color = (0.12, 0.56, 0.92, 1) if i == blue_bone or bone.name.endswith("_l") else (0.96, 0.40, 0.13, 1) if bone.name.endswith("_r") else (0.78, 0.83, 0.91, 1)
				if bone.shape.kind == pj.ShapeKind.BOX:
					mesh, scale = box_mesh, bone.shape.params[:3]
				elif bone.shape.kind == pj.ShapeKind.CYLINDER:
					mesh, scale = MaterialMesh(cylinder_vertices(*bone.shape.params[:2]), shader), (1, 1, 1)
				else:
					radius = bone.shape.params[0]
					cylinder_length = bone.shape.params[1] if bone.shape.kind == pj.ShapeKind.CAPSULE else 0
					mesh = MaterialMesh(rounded_vertices(cube, radius, cylinder_length), shader)
					scale = (1, 1, 1)
				body_parts.append(DrawInstance(mesh, material={"base_color": color}, scale=scale))
		meshes = [box_mesh, *(part.mesh for part in body_parts)]
		renderer = MaterialRenderer(body_parts + terrain(scenario), meshes=meshes, skin_palette=palette)
		eye, target, far = camera_points(scenario)
		camera = Camera(eye, target, (0, 1, 0), 43, near=0.02 * scene_height, far=far)

		def update(tick):
			turns = orientations[tick]
			if model:
				for part in body_parts:
					part.position = positions[tick, 0]
				palette.write(skin_ids, np.repeat(skin_pose(turns), len(skin_ids), axis=0))
			else:
				centers, rotations = collider_poses(positions[tick], turns, offsets, shape_rotations)
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
			camera.position[:], camera.target[:], camera.far = camera_points(name)
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


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--scenario", choices=(*SCENARIOS, "all"))
	parser.add_argument("--frames", type=int, default=960)
	parser.add_argument("--threads", type=int, default=1)
	parser.add_argument("--height", type=float, help="height of the built-in humanoid in metres (default: 1.8)")
	parser.add_argument("--mass", type=float, help="mass of the built-in humanoid in kilograms (default: 70)")
	parser.add_argument("--tuning", type=Path, metavar="PATH", help="load a ragdoll tuner project or native ragdoll settings JSON")
	parser.add_argument("--no-passive", action="store_true", help="keep native hard constraints but disable Python passive forces")
	parser.add_argument("--output", type=Path)
	add_viewing_options(parser)
	parser.add_argument("--model", nargs="?", const=True, default=False, type=Path, metavar="PATH", help="draw the supplied dummy rig in place of ragdoll colliders (default path: scenes/resources/dummy_rigged.glb)")
	args = parser.parse_args()
	if args.frames < 1 or args.threads < 1: parser.error("frames and threads must be positive")
	if any(value is not None and (not np.isfinite(value) or value <= 0) for value in (args.height, args.mass)):
		parser.error("height and mass must be finite and positive")
	if args.model and args.headless: parser.error("--model requires playback")
	scenario = args.scenario or ("all" if args.headless else "drop")
	if scenario == "all" and not args.headless: parser.error("choose one initial scenario for playback, or use --headless --scenario all")
	tuning = None
	if args.tuning is not None:
		if args.height is not None or args.mass is not None:
			parser.error("--tuning cannot be combined with --height or --mass")
		from tools.ragdoll_project import read_project

		try:
			tuning = read_project(args.tuning)
		except (OSError, ValueError) as error:
			parser.error(f"could not load tuning file '{args.tuning}': {error}")
	options = dict(frames=args.frames, threads=args.threads, height=args.height, mass=args.mass, tuning=tuning, passive=not args.no_passive)
	if args.headless:
		scenarios = SCENARIOS if scenario == "all" else (scenario,)
		results = [run_scene(name, **options) for name in scenarios]
	else:
		try:
			results = [run_viewer(scenario, offscreen=args.offscreen, model=args.model, **options)]
		except (FileNotFoundError, ValueError) as error:
			parser.error(str(error))
	if args.output is not None: args.output.write_text(json.dumps(results, indent=2, default=lambda value: value.tolist(), allow_nan=False) + "\n", encoding="utf-8")
	for result in results:
		print(f"{result['scenario']}: ticks={result['ticks']} error={result['initial_joint_error_rad']:.4f}->{result['final_joint_error_rad']:.4f} rad contacts={result['contact_records']} anchor_error={result['max_anchor_error_m']:.4f} m")


if __name__ == "__main__":
	main()
