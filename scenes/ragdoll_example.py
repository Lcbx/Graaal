"""Authored humanoid: animation, powered balance, shove, limp and recovery.

python -m scenes.ragdoll_example --headless --frames 960
python -m scenes.ragdoll_example --standing
python main.py scenes/ragdoll_example.py
"""

import argparse

import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, PoseBatch, SkeletonRef
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, PhysicsRagdoll, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import RagdollAsset, joint_limit_torques, joint_spring_torques
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip_pairs
from scenes.passive_ragdoll_example import make_humanoid
from scenes.physics_geometry import collider_bounds, collider_poses, cube_vertices
from scenes.physics_viewing_utils import add_viewing_options, has_blue_pixels, ragdoll_collider_instances, view_poses
from scenes.ragdoll_control import STANDING_MOTOR_CAPS, make_ankle_balance, observe_ragdoll, standing_pose


TICK_HZ = 120
FLOOR_POSITION = (0, -0.5, 0)
FLOOR_SIZE = (8, 1, 8)


def create_demo(*, standing=False):
	import pyjoltphysics as pj
	from pyjoltphysics.ragdolls import forward_kinematics

	settings, neutral, spring, limits = make_humanoid()
	reference = standing_pose(settings)
	feet = np.array([settings.bone_index(name) for name in ("foot_l", "foot_r")])
	caps = np.array(STANDING_MOTOR_CAPS)
	caps[feet] = 0
	for bone, cap in zip(settings.bones, caps, strict=True):
		bone.strength = float(cap)
	positions, orientations = forward_kinematics(settings, reference, (0, 0, 0), (0, 0, 0, 1))
	lower, _ = collider_bounds(settings.bones, positions, orientations)
	spawn = (0, 0.001 - lower[:, 1].min(), 0)
	balance_torques = make_ankle_balance(settings, positions, orientations, feet)
	count = len(settings)
	rig = Skeleton(settings.parents, settings.rest_translations, settings.rest_rotations, np.ones((count, 3)), np.tile(np.eye(4), (count, 1, 1)))
	rest = (rig.translation[None], rig.rotation[None], rig.scale[None])
	rig = Skeleton(rig.parents, rig.translation, rig.rotation, rig.scale, np.linalg.inv(model_matrices(rig, *rest)[0]))
	asset = RagdollAsset(rig, settings)
	translation, rotation, scale = (np.tile(value, (5, 1, 1)) for value in (rig.translation, reference, rig.scale))
	arms = [settings.bone_index(name) for name in ("upper_arm_l", "upper_arm_r")]
	wave = np.array([Quaternion.from_z_rotation(angle) for angle in (0, 0.3, 0, 0.15, 0)])
	rotation[:, arms] = Quaternion.multiply_batch(rotation[:, arms], np.stack((wave, wave * [-1, -1, -1, 1]), axis=1))
	clip = Clip(translation, rotation, scale, fps=2)
	ecs = ECS()
	ecs.register(Collider, RigidBody, PhysicsRagdoll, MaterialTransform, SkeletonRef, ClipPlayer)
	owner, floor = ecs.create(2)
	ecs.add(owner, PhysicsRagdoll(0, mode=int(standing)), MaterialTransform(position=spawn), SkeletonRef(0), ClipPlayer(0))
	ecs.add(floor, Collider(0), RigidBody(motion=0, friction=0.8), MaterialTransform(position=FLOOR_POSITION))
	physics = EcsPhysicsAdapter(ecs, {0: pj.Shape.box(FLOOR_SIZE)}, settings=pj.WorldSettings(tick_hz=TICK_HZ, threads=1, allow_sleeping=False), ragdolls={0: asset})
	animation = EcsAnimationAdapter(ecs, {0: rig}, {0: clip})
	frames = {"batches": [], "stage": "powered" if standing else "animated", "observation": None}
	motor_action = np.zeros((1, count, 4))
	motor_action[0, :, 3] = caps > 0
	standing_target = (rig.translation[None], reference[None], rig.scale[None])

	def control(world, commands, dt):
		if standing:
			return
		tick = physics.world.tick % (8 * TICK_HZ)
		if tick == 0:
			world.get(PhysicsRagdoll)[[owner]].assign(mode=0, blend=0)
			frames["stage"] = "animated"
		elif tick == 120:
			world.get(PhysicsRagdoll)[[owner]].assign(mode=1, blend=0.2)
			frames["stage"] = "powered"
		elif tick == 180:
			physics.ragdolls[int(owner)].hit(settings.bone_index("chest"), (45, 5, 0))
		elif tick == 360:
			world.get(PhysicsRagdoll)[[owner]].assign(mode=2, blend=0.2)
			frames["stage"] = "limp"
		elif tick == 600:
			world.get(PhysicsRagdoll)[[owner]].assign(mode=0, blend=1)
			frames["stage"] = "recovery"

	def targets(world, commands, dt):
		if frames["stage"] == "powered":
			physics.set_ragdoll_pose([owner], standing_target)
			return
		for batch in animation.sample_batches:
			pose = sample_clip_pairs(rig, {0: clip}, batch.clip_ids, batch.times, batch.weights)
			physics.set_ragdoll_pose(batch.entities, pose)

	def joint_control(world, commands, dt):
		state = frames["observation"]
		if state is None or frames["stage"] not in ("powered", "limp"):
			return
		torques = joint_spring_torques(settings.parents, state.orientation, state.angular_velocity, neutral, **spring)
		torques += joint_limit_torques(orientations=state.orientation, angular_velocity=state.angular_velocity, **limits)
		if frames["stage"] == "powered":
			torques[0] += balance_torques(state)
		physics.world.add_torques(physics.ragdolls[int(owner)].bones, torques[0])

	def finalize(world, commands, dt):
		bones = physics.ragdolls[int(owner)].bones[None]
		action = motor_action if frames["stage"] == "powered" else np.zeros_like(motor_action)
		frames["observation"] = observe_ragdoll(physics.world, bones, settings, physics.events, reference, action, floor=[physics.body_uids[int(floor)]])
		frames["batches"] = [PoseBatch(0, 0, np.array([owner]), physics.ragdoll_skin([owner]))]

	schedule = Scheduler(ecs)
	schedule.add("animation", animation.advance, phase="fixed_update")
	schedule.add("control", control, phase="fixed_update", after="animation")
	schedule.add("physics_sync", physics.reconcile, phase="fixed_update", after="control")
	schedule.add("pose_targets", targets, phase="fixed_update", after="physics_sync")
	schedule.add("physics_input", physics.prepare_inputs, phase="fixed_update", after="pose_targets")
	schedule.add("joint_control", joint_control, phase="fixed_update", after="physics_input")
	schedule.add("physics_step", physics.step, phase="fixed_update", after="joint_control")
	schedule.add("physics_output", physics.pull, phase="fixed_update", after="physics_step")
	schedule.add("physics_events", physics.dispatch_events, phase="fixed_update", after="physics_output")
	schedule.add("final_pose", finalize, phase="fixed_update", after="physics_events")
	return ecs, owner, floor, physics, schedule, frames, asset


def simulate(frames, *, standing=False, on_pose=None):
	if frames < 1:
		raise ValueError("frames must be positive")
	_, owner, _, physics, schedule, state, asset = create_demo(standing=standing)
	with physics:
		fixed_dt = 1 / TICK_HZ
		for _ in range(frames):
			schedule.run(fixed_dt, fixed_steps=1, fixed_dt=fixed_dt)
			if not np.isfinite(state["batches"][0].matrices).all(): raise RuntimeError("nonfinite ragdoll palette")
			if on_pose is not None:
				on_pose(*physics.ragdolls[int(owner)].pose())
			if physics.world.tick % TICK_HZ == 0:
				positions, _ = physics.ragdolls[int(owner)].pose()
				print(f"tick={physics.world.tick} stage={state['stage']} pelvis_y={positions[0, 1]:.3f}")
	return asset.settings


def run_headless(frames=960, *, standing=False):
	simulate(frames, standing=standing)


def run_scene(*, offscreen=False, frames=None, standing=False):
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh
	from graaal.Math import Camera

	poses = []
	frames = (10 if standing else 8) * TICK_HZ if frames is None else frames
	settings = simulate(frames, standing=standing, on_pose=lambda position, turn: poses.append((position.copy(), turn.copy())))
	positions, orientations = map(np.stack, zip(*poses, strict=True))
	bones = settings.bones
	offsets = [bone.shape_offset for bone in bones]
	shape_rotations = [bone.shape_rotation for bone in bones]
	centers, turns = collider_poses(positions, orientations, offsets, shape_rotations)
	cube, shader = cube_vertices(), DefaultMaterialShader()
	box_mesh = MaterialMesh(cube, shader)
	body_parts = ragdoll_collider_instances(bones, cube, box_mesh, shader)
	ground = DrawInstance(box_mesh, material={"base_color": (0.27, 0.30, 0.32, 1)}, position=FLOOR_POSITION, scale=FLOOR_SIZE)
	lower, upper = collider_bounds(bones, positions, orientations)
	target = (lower.min(axis=(0, 1)) + upper.max(axis=(0, 1))) / 2
	distance = max(1.8, float(np.ptp(centers, axis=0).max()))
	camera = Camera(target + distance * np.array((1.1, 0.7, 2.0)), target, (0, 1, 0), 43, near=0.02, far=100)
	title = "Ragdoll | Ankle balance" if standing else "Ragdoll | Animation, ankle balance, shove, fall and recovery"
	offscreen_samples = np.linspace(0, frames - 1, min(frames, 6), dtype=int)
	view_poses(title, camera, list(zip(centers, turns)), body_parts, sample_hz=TICK_HZ, static=[ground], offscreen=offscreen, offscreen_samples=offscreen_samples, check_frame=has_blue_pixels)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	add_viewing_options(parser)
	parser.add_argument("--frames", type=int, help="simulation ticks to record at 120 Hz")
	parser.add_argument("--standing", action="store_true", help="show continuous joint-only standing without the shove and recovery sequence")
	args = parser.parse_args()
	if args.frames is not None and args.frames <= 0: parser.error("--frames must be positive")
	if args.headless:
		run_headless(args.frames or (10 if args.standing else 8) * TICK_HZ, standing=args.standing)
	else:
		run_scene(offscreen=args.offscreen, frames=args.frames, standing=args.standing)


if __name__ == "__main__":
	main()
