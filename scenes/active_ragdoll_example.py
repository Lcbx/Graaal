"""Headless reference-pose tracking and root-assisted standing balance.

Run with python -m scenes.active_ragdoll_example.
"""

import argparse

import numpy as np

from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import joint_limit_torques, joint_spring_torques
from graaal.PhysicsRagdollControl import apply_ragdoll_motors, ragdoll_observations
from scenes.passive_ragdoll_example import make_humanoid
from scenes.physics_utils import joint_error


def standing_action(observation, feet, chest, *, balance=True):
	"""Use PD feedback relative to the contacting feet."""
	action = np.zeros_like(observation.previous_action)
	action[..., 3] = 1
	force = np.zeros_like(observation.center_of_mass)
	if balance:
		contacts = observation.contacts[:, feet]
		count = contacts.sum(axis=1)
		support = (observation.position[:, feet] * contacts[..., None]).sum(axis=1) / np.maximum(count, 1)[:, None]
		error = support - observation.center_of_mass
		force = error * 650 - observation.center_of_mass_velocity * 130
		force[:, 1] = 0
		force *= np.minimum(1, 200 / np.maximum(np.linalg.norm(force, axis=1), 1))[:, None] * (count > 0)[:, None]
		error[:, 1] = 0
		world_lean = np.cross([0, 1, 0], error * 0.6)
		frame = Quaternion.matrix_batch(observation.orientation[:, chest])
		action[:, chest, :3] = (world_lean[:, None, :] @ frame)[:, 0] * (count > 0)[:, None]
	return action, force


def run_scene(*, frames=480, impulse=3.0, balance=True, motors=True):
	"""Build a fresh isolated world and run a repeatable support/impact probe."""
	import pyjoltphysics as pj

	if frames < 1 or not np.isfinite(impulse) or impulse < 0: raise ValueError("frames must be positive and impulse finite and nonnegative")
	settings, neutral, spring, limits = make_humanoid()
	if motors and balance: settings.root_torque = 180
	reference = settings.rest_rotations
	feet = [settings.bone_index(name) for name in ("foot_l", "foot_r")]
	chest = settings.bone_index("chest")
	strength = np.array([0, 160, 35, 30, 18, 30, 18, 180, 140, 100, 180, 140, 100]) if motors else 0
	with pj.World(pj.WorldSettings(tick_hz=120, threads=1, allow_sleeping=False)) as world:
		floor, _ = world.add_bodies(pj.BodyBatch.boxes([(0, -0.5, 0)], size=(8, 1, 8), motion=pj.MotionType.STATIC, friction=0.8))
		handle = world.add_ragdoll(settings, (0, 0, 0), mode=pj.RagdollMode.POWERED)
		uids = handle.bones[None]
		previous = np.zeros(uids.shape + (4,))
		world.flush()

		def observe(action):
			events = world.drain_events()
			active = events[(events["kind"] == pj.EventKind.CONTACT) & np.isin(events["phase"], (pj.EventPhase.BEGIN, pj.EventPhase.PERSIST))]
			pairs = np.column_stack((active["a"], active["b"]))
			return ragdoll_observations(world, uids, settings, pairs, reference, action, support_bodies=floor)

		observation = observe(previous)
		min_height, support_ticks, peak_force = float(observation.position[0, 0, 1]), 0, 0.0
		for tick in range(frames):
			if tick == frames // 3 and impulse: handle.hit(chest, (0, 0, impulse))
			action, force = standing_action(observation, feet, chest, balance=balance and motors)
			if not motors: action[..., 3] = 0
			apply_ragdoll_motors([handle], reference, action, max_offset=0.22, max_strength=strength)
			if balance and motors: world.add_forces(uids[:, 0], force)
			torque = joint_spring_torques(settings.parents, observation.orientation, observation.angular_velocity, neutral, **spring)
			torque += joint_limit_torques(orientations=observation.orientation, angular_velocity=observation.angular_velocity, **limits)
			world.add_torques(uids.ravel(), torque[0])
			world.run_ticks(1)
			observation = observe(action)
			min_height = min(min_height, float(observation.position[0, 0, 1]))
			support_ticks += int(observation.contacts[:, feet].any())
			peak_force = max(peak_force, float(np.linalg.norm(force[0])))
		return dict(ticks=world.tick, min_pelvis_height_m=min_height, support_ticks=support_ticks,
			peak_balance_force_n=peak_force, final_tracking_error_rad=float(np.mean(joint_error(settings.parents, observation.orientation, reference))),
			position=observation.position[0], joint_rotation=observation.joint_rotation[0])


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--frames", type=int, default=480)
	parser.add_argument("--impulse", type=float, default=3, help="chest impulse in N s")
	parser.add_argument("--no-balance", action="store_true")
	parser.add_argument("--no-motors", action="store_true")
	args = parser.parse_args()
	if args.frames < 1 or not np.isfinite(args.impulse) or args.impulse < 0: parser.error("frames must be positive and impulse finite and nonnegative")
	result = run_scene(frames=args.frames, impulse=args.impulse, balance=not args.no_balance, motors=not args.no_motors)
	print(f"ticks={result['ticks']} support_ticks={result['support_ticks']} min_pelvis={result['min_pelvis_height_m']:.3f} m tracking_error={result['final_tracking_error_rad']:.3f} rad")


if __name__ == "__main__":
	main()
