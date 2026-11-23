"""Standing pose and joint feedback shared by the humanoid examples."""

import numpy as np

from graaal.Math import Quaternion
from graaal.PhysicsRagdollControl import ragdoll_observations
from scenes.physics_geometry import collider_bounds


STANDING_MOTOR_CAPS = (0, 160, 35, 30, 18, 30, 18, 180, 140, 100, 180, 140, 100)
STANDING_ANGLES = (0, 0, 0, 0, 0, 0, 0, -0.025, 0.05, -0.025, -0.025, 0.05, -0.025)


def standing_pose(settings):
	turns = np.array([Quaternion.from_x_rotation(angle) for angle in STANDING_ANGLES])
	return Quaternion.multiply_batch(settings.rest_rotations, turns)


def observe_ragdoll(world, uids, settings, events, reference, action, *, floor):
	import pyjoltphysics as pj

	active = events[(events["kind"] == pj.EventKind.CONTACT) & np.isin(events["phase"], (pj.EventPhase.BEGIN, pj.EventPhase.PERSIST))]
	pairs = np.column_stack((active["a"], active["b"]))
	return ragdoll_observations(world, uids, settings, pairs, reference, action, support_bodies=floor)


def make_ankle_balance(settings, positions, orientations, feet):
	"""Prepare the paired ankle torque controller from the reference foot bounds."""
	if any(settings.bones[i].shape.kind.name != "BOX" for i in feet):
		raise ValueError("ankle balance requires box foot colliders")
	lower, upper = collider_bounds(settings.bones, positions, orientations)
	ankles = positions[feet][:, [0, 2]]
	lower, upper = lower[feet][:, [0, 2]] - ankles, upper[feet][:, [0, 2]] - ankles
	shins = np.asarray(settings.parents)[feet]
	caps = np.asarray(STANDING_MOTOR_CAPS)[feet]
	weight = 9.81 * sum(bone.mass for bone in settings.bones)

	def torques(state, residual=None):
		torque = np.zeros_like(state.position[0])
		planted = state.contacts[0, feet]
		if not planted.any():
			return torque
		ankles = state.position[0, feet][:, [0, 2]]
		center = state.center_of_mass[0, [0, 2]]
		speed = state.center_of_mass_velocity[0, [0, 2]]
		# Put pressure beyond the COM in the direction of drift. Approximate
		# vertical load as shared evenly by the contacting feet.
		pressure = center + 0.35 * (center - ankles.mean(axis=0)) + 0.45 * speed
		load = weight / planted.sum()
		offset = pressure - ankles[planted].mean(axis=0)
		if residual is not None:
			# Convert world X/Z torque into equivalent sole X/Z offsets.
			offset = offset + residual[planted][:, [1, 0]] * [-1, 1] / load
		arms = np.clip(offset, lower[planted], upper[planted])
		turns = np.cross([0, 1, 0], np.column_stack((arms[:, 0], np.zeros(len(arms)), arms[:, 1])))
		turns *= load
		turns *= np.minimum(1, caps[planted] / np.maximum(np.linalg.norm(turns, axis=1), 1))[:, None]
		torque[feet[planted]] = turns
		torque[shins[planted]] = -turns
		return torque

	return torques
