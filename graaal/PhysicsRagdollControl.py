"""Batched observations and bounded motor targets for powered ragdolls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .Animation.SkeletalAnimation import _rotations
from .Math import Quaternion
from .PhysicsRagdoll import _joint_state


if TYPE_CHECKING:
	import pyjoltphysics as pj


@dataclass(frozen=True, slots=True)
class RagdollObservation:
	position: np.ndarray
	orientation: np.ndarray
	linear_velocity: np.ndarray
	angular_velocity: np.ndarray
	joint_rotation: np.ndarray
	joint_angular_velocity: np.ndarray
	contacts: np.ndarray
	center_of_mass: np.ndarray
	center_of_mass_velocity: np.ndarray
	reference: np.ndarray
	previous_action: np.ndarray


def _finite(value: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
	array = np.asarray(value, dtype=np.float64)
	if array.shape != shape or not np.isfinite(array).all(): raise ValueError(f"{name} must be finite with shape {shape}")
	return array


def _bone_controls(value: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
	array = np.broadcast_to(np.asarray(value, dtype=np.float64), shape)
	if not np.isfinite(array).all() or np.any(array < 0): raise ValueError(f"{name} must be finite and nonnegative")
	return array


def ragdoll_observations(world: pj.World, uids: ArrayLike, settings: pj.RagdollSettings, contact_pairs: ArrayLike, reference: ArrayLike, previous_action: ArrayLike, *, support_bodies: ArrayLike) -> RagdollObservation:
	"""Collect one post-step observation in (instances, bones, ...) UID order.

	The caller passes active body UID pairs from this tick and the UIDs that count as
	support. Body fields and mass centers are in world space, while joints and
	references are parent-local. Shape offsets locate the centers of the authored
	symmetric bone shapes. Previous actions store three offsets and one strength
	fraction per bone. All returned arrays own their data.
	"""
	ids = np.asarray(uids)
	if ids.ndim != 2 or ids.shape[1] != len(settings) or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids > np.iinfo(np.uint32).max):
		raise ValueError("uids must contain native body IDs in (instances, bones) order")
	count, bones = ids.shape
	state = world.bodies_state(ids.ravel())
	if not state.found.all(): raise ValueError("ragdoll observation contains missing body UIDs")
	position, orientation, linear, angular = (np.asarray(value).reshape(count, bones, width) for value, width in (
		(state.position, 3), (state.orientation, 4), (state.linear_velocity, 3), (state.angular_velocity, 3),
	))
	position = _finite(position, (count, bones, 3), "position")
	linear = _finite(linear, position.shape, "linear velocity")
	parents, turns, angular = _joint_state(settings.parents, orientation, angular)
	world_angular = np.asarray(angular, dtype=np.float64)
	parent_turns = turns[:, parents[1:]]
	inverse = parent_turns * [-1, -1, -1, 1]
	joint = turns.copy()
	joint[:, 1:] = Quaternion.multiply_batch(inverse, turns[:, 1:])
	joint_velocity = world_angular.copy()
	joint_velocity[:, 1:] = (Quaternion.matrix_batch(inverse) @ (world_angular[:, 1:] - world_angular[:, parents[1:]])[..., None])[..., 0]

	support = np.asarray(support_bodies)
	pairs = np.asarray(contact_pairs)
	if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu": raise ValueError("contact_pairs must contain (contact, 2) body UID rows")
	a, b = pairs.T
	contacts = np.isin(ids, np.concatenate((a[np.isin(b, support)], b[np.isin(a, support)])))
	masses = np.array([bone.mass for bone in settings.bones], dtype=np.float64)
	if np.any(masses <= 0) or not np.isfinite(masses.sum()): raise ValueError("ragdoll observations require explicit positive bone masses")
	weights = masses[None, :, None] / masses.sum()
	offsets = np.array([bone.shape_offset for bone in settings.bones])
	centers = position + (Quaternion.matrix_batch(turns) @ offsets[None, :, :, None])[..., 0]
	ref = _rotations(np.broadcast_to(reference, (count, bones, 4)), (count, bones, 4), "reference")
	previous = _finite(previous_action, (count, bones, 4), "previous_action")
	return RagdollObservation(position.copy(), turns.copy(), linear.copy(), world_angular.copy(), joint, joint_velocity, contacts,
		np.sum(weights * np.asarray(centers, dtype=np.float64), axis=1), np.sum(weights * np.asarray(linear, dtype=np.float64), axis=1), ref.copy(), previous.copy())


def ragdoll_motor_targets(reference: ArrayLike, action: ArrayLike, *, max_offset: ArrayLike, max_strength: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
	"""Convert (instances, bones, 4) policy actions to local poses and N m caps.

	The first three channels are a reference-local rotation vector in radians.
	Clip its length to the per-bone cap. The fourth channel is a strength
	fraction clipped to [0, 1]. The root is left at its reference orientation
	with zero motor strength. Roots may be controlled separately by the caller.
	"""
	action = np.asarray(action, dtype=np.float64)
	if action.ndim != 3 or action.shape[-1] != 4 or not action.shape[1] or not np.isfinite(action).all():
		raise ValueError("action must be finite with shape (instances, bones, 4)")
	shape = action.shape[:-1]
	ref = _rotations(np.broadcast_to(reference, action.shape), action.shape, "reference")
	offset_cap = _bone_controls(max_offset, shape, "max_offset")
	if np.any(offset_cap > np.pi): raise ValueError("max_offset must be at most pi radians")
	strength = _bone_controls(max_strength, shape, "max_strength") * np.clip(action[..., 3], 0, 1)
	vectors = action[..., :3]
	# Normalize without overflowing when a policy returns a large finite action.
	magnitude = np.max(np.abs(vectors), axis=-1)
	axis = np.divide(vectors, magnitude[..., None], out=np.zeros_like(vectors), where=magnitude[..., None] > 0)
	length = np.linalg.norm(axis, axis=-1)
	axis /= np.where(length > 0, length, 1)[..., None]
	half = np.minimum(magnitude, offset_cap / np.maximum(length, 1)) * length * 0.5
	delta = np.concatenate((axis * np.sin(half)[..., None], np.cos(half)[..., None]), axis=-1)
	target = Quaternion.multiply_batch(ref, delta)
	target[:, 0], strength[:, 0] = ref[:, 0], 0
	return target, strength


def apply_ragdoll_motors(handles: Sequence[pj.Ragdoll], reference: ArrayLike, action: ArrayLike, *, max_offset: ArrayLike, max_strength: ArrayLike) -> np.ndarray:
	"""Queue validated pose targets and per-bone caps for one control tick."""
	target, strength = ragdoll_motor_targets(reference, action, max_offset=max_offset, max_strength=max_strength)
	if len(handles) != len(target) or any(len(handle.bones) != target.shape[1] for handle in handles):
		raise ValueError("actions must match the motor handles and bone counts")
	for handle, rotations, caps in zip(handles, target, strength, strict=True):
		handle.set_pose(rotations)
		handle.set_strength(caps)
	return strength
