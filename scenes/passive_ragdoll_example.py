"""Python-authored passive humanoid: zero gravity, floor drop, stairs and impact.

python -m scenes.passive_ragdoll_example --scenario all --frames 960
python -m scenes.passive_ragdoll_example --view --scenario stairs
"""

import argparse
import json
from pathlib import Path

import numpy as np

from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import joint_limit_torques, joint_spring_torques, ragdoll_limit_parameters
from scenes.physics_utils import joint_error


def make_humanoid(*, height=1.8, mass=70.0):
	"""Return an example 13-body asset, relaxed local rotations and passive gains.

	Translations/shapes bake height; masses and gains scale with mass. Gains
	scale with length squared to retain their reference angular frequencies.
	These are illustrative authored settings, evaluated at 120 Hz.
	"""
	import pyjoltphysics as pj

	if not np.isfinite(height) or not np.isfinite(mass) or height <= 0 or mass <= 0: raise ValueError("height and mass must be finite and positive")
	scale = height / 1.8
	names = ("pelvis", "chest", "head", "upper_arm_l", "lower_arm_l", "upper_arm_r", "lower_arm_r", "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r")
	parents = [-1, 0, 1, 1, 3, 1, 5, 0, 7, 8, 0, 10, 11]
	translation = np.array([(0, 0.99, 0), (0, 0.30, 0), (0, 0.32, 0), (0.19, 0.24, 0), (0, -0.30, 0), (-0.19, 0.24, 0), (0, -0.30, 0), (0.10, -0.06, 0), (0, -0.44, 0), (0, -0.39, 0), (-0.10, -0.06, 0), (0, -0.44, 0), (0, -0.39, 0)]) * scale
	radius = np.array([0.12, 0.13, 0.10, 0.055, 0.045, 0.055, 0.045, 0.075, 0.055, 0.06, 0.075, 0.055, 0.06]) * scale
	settings = pj.RagdollSettings.from_skeleton(names, parents, translation, radius=radius, mass=mass, self_collision=True, friction=0.8, restitution=0, linear_damping=0.05, angular_damping=0.3, root_force=0, root_torque=0, flags=pj.BodyFlag.REPORT | pj.BodyFlag.NO_SLEEP)
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


def _stair_spawn_height(settings, rotation, step_top, clearance):
	import pyjoltphysics as pj

	translations = settings.rest_translations
	local_rotations = Quaternion.matrix_batch(settings.rest_rotations)
	positions = np.empty_like(translations)
	rotations = np.empty_like(local_rotations)
	root_rotation, origin = rotation.matrix33, np.zeros(3)
	lowest = np.inf
	for i, bone in enumerate(settings.bones):
		parent = bone.parent
		parent_rotation = root_rotation if parent == -1 else rotations[parent]
		parent_position = origin if parent == -1 else positions[parent]
		positions[i] = parent_position + parent_rotation @ translations[i]
		rotations[i] = parent_rotation @ local_rotations[i]
		center_y = positions[i, 1] + rotations[i, 1] @ bone.shape_offset
		shape_y = (rotations[i] @ Quaternion(bone.shape_rotation).matrix33)[1]
		shape = bone.shape
		if shape.kind == pj.ShapeKind.BOX:
			extent = np.abs(shape_y) @ np.asarray(shape.params[:3]) / 2
		elif shape.kind == pj.ShapeKind.CAPSULE:
			extent = shape.params[0] + abs(shape_y[1]) * shape.params[1] / 2
		else:  # Sphere
			extent = shape.params[0]
		lowest = min(lowest, center_y - extent)
	return float(step_top - lowest + clearance)


def _body_peak_speeds(state):
	linear = float(np.max(np.linalg.norm(state.linear_velocity, axis=-1)))
	angular = float(np.max(np.linalg.norm(state.angular_velocity, axis=-1)))
	return linear, angular


def run_scene(scenario="drop", *, frames=960, threads=1, height=1.8, mass=70.0, passive=True, on_pose=None):
	"""Run a bounded passive scene and return detached results.

	on_pose receives joint positions and orientations at tick zero and after each step.
	"""
	import pyjoltphysics as pj

	if scenario not in ("zero_gravity", "drop", "stairs", "impact"): raise ValueError("unknown passive scenario")
	if frames < 1 or threads < 1: raise ValueError("frames and threads must be positive")
	settings, neutral, spring, limit = make_humanoid(height=height, mass=mass)
	parents, translation = settings.parents, settings.rest_translations
	masses = np.array([bone.mass for bone in settings.bones])
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
			step_top = np.max(positions[1:, 1] + sizes[1:, 1] / 2)
			position = (2.2 * scale, _stair_spawn_height(settings, rotation, step_top, 0.01 * scale), 0)
		elif scenario == "impact": position = (0, 0.05 * scale, 0)
		handle = world.add_ragdoll(settings, position, rotation, mode=pj.RagdollMode.LIMP)
		handle.set_strength(np.zeros(len(settings)))
		if scenario == "stairs": world.set_velocities(handle.bones, linear=np.tile([-1.2 * scale, 0, 0], (len(settings), 1)))
		world.flush()
		state = world.bodies_state(handle.bones)
		if on_pose is not None: on_pose(state.position, state.orientation)
		initial_error = float(np.mean(joint_error(parents, state.orientation, neutral)))
		feet = handle.bones[[settings.bone_index("foot_l"), settings.bone_index("foot_r")]]
		counts = dict(contact_records=0, floor_contact_records=0, stair_contact_records=0, foot_contact_records=0, self_contact_records=0)
		peak_anchor = peak_depth = peak_impulse = 0.0
		peak_linear, peak_angular = _body_peak_speeds(state)
		late_linear = late_angular = 0.0
		impact = None
		impact_tick = min(120, frames // 3)
		for tick in range(frames):
			if scenario == "impact" and tick == impact_tick:
				chest = settings.bone_index("chest")
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
			peak_anchor = max(peak_anchor, float(np.max(np.linalg.norm(anchors, axis=-1))))
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
		return dict(scenario=scenario, passive=passive, height_m=height, mass_kg=mass, threads=threads, tick_hz=120, ticks=world.tick, bodies=len(settings),
			mode=int(world.latest().ragdolls.mode[0]), active_strength=float(world.latest().ragdolls.strength[0]),
			initial_joint_error_rad=initial_error, final_joint_error_rad=float(np.mean(joint_error(parents, state.orientation, neutral))), max_anchor_error_m=peak_anchor,
			max_linear_speed_m_s=peak_linear, max_angular_speed_rad_s=peak_angular, final_linear_kinetic_energy_j=float(0.5 * np.sum(masses * np.sum(state.linear_velocity**2, axis=-1))),
			late_max_linear_speed_m_s=late_linear, late_max_angular_speed_rad_s=late_angular,
			peak_contact_depth_m=peak_depth, peak_reported_contact_impulse_ns=peak_impulse, impact=impact, position=state.position.copy(), orientation=state.orientation.copy(),
			stats=world.stats(), **counts)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--scenario", choices=("zero_gravity", "drop", "stairs", "impact", "all"))
	parser.add_argument("--frames", type=int, default=960)
	parser.add_argument("--threads", type=int, default=1)
	parser.add_argument("--height", type=float, default=1.8)
	parser.add_argument("--mass", type=float, default=70)
	parser.add_argument("--no-passive", action="store_true", help="keep native hard constraints but disable Python passive forces")
	parser.add_argument("--output", type=Path)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("--view", action="store_true", help="open a window with looping playback")
	mode.add_argument("--offscreen", action="store_true", help="check rendered frames without opening a window")
	parser.add_argument("--model", nargs="?", const=True, default=False, type=Path, metavar="PATH", help="draw the supplied dummy rig in place of ragdoll colliders (default path: scenes/resources/dummy_rigged.glb)")
	args = parser.parse_args()
	if args.frames < 1 or args.threads < 1 or not np.isfinite((args.height, args.mass)).all() or min(args.height, args.mass) <= 0: parser.error("frames/threads must be positive; height/mass must be finite and positive")
	visual = args.view or args.offscreen
	if args.model and not visual: parser.error("--model requires --view or --offscreen")
	scenario = args.scenario or ("drop" if visual else "all")
	options = dict(frames=args.frames, threads=args.threads, height=args.height, mass=args.mass, passive=not args.no_passive)
	if visual:
		if scenario == "all": parser.error("choose one scenario for playback")
		from scenes.passive_ragdoll_viewer import run_viewer
		try:
			results = [run_viewer(scenario, offscreen=args.offscreen, model=args.model, **options)]
		except FileNotFoundError as error:
			parser.error(str(error))
	else:
		scenarios = ("zero_gravity", "drop", "stairs", "impact") if scenario == "all" else (scenario,)
		results = [run_scene(name, **options) for name in scenarios]
	if args.output is not None: args.output.write_text(json.dumps(results, indent=2, default=lambda value: value.tolist(), allow_nan=False) + "\n", encoding="utf-8")
	for result in results:
		print(f"{result['scenario']}: ticks={result['ticks']} error={result['initial_joint_error_rad']:.4f}->{result['final_joint_error_rad']:.4f} rad contacts={result['contact_records']} anchor_error={result['max_anchor_error_m']:.4f} m")


if __name__ == "__main__":
	main()
