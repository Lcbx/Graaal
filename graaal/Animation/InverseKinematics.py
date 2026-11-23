"""Standalone batched inverse-kinematics geometry."""
from __future__ import annotations

from operator import index
from typing import Sequence, SupportsIndex, TYPE_CHECKING

from numpy.typing import ArrayLike
import numpy as np

from ..Math import Quaternion
from .SkeletalAnimation import _rotations, model_matrices


if TYPE_CHECKING:
	from .SkeletalAnimation import Skeleton


def solve_two_bone(root: ArrayLike, joint: ArrayLike, end: ArrayLike, target: ArrayLike, pole: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
	"""Solve joint and end positions for two-bone chains in a common space.

	Inputs are finite (instances, 3) arrays. The pole is a point indicating the
	preferred bend direction. Bone lengths are preserved and unreachable targets
	are clamped to the nearest reachable distance. Returns float64 joint and end
	positions; neither inputs nor pose rotations are modified.
	"""
	root, joint, end, target, pole = (np.asarray(value, dtype=np.float64) for value in (root, joint, end, target, pole))
	if root.ndim != 2 or root.shape[1:] != (3,) or any(value.shape != root.shape for value in (joint, end, target, pole)) or not all(np.isfinite(value).all() for value in (root, joint, end, target, pole)):
		raise ValueError("positions must be finite arrays of shape (instances, 3)")
	with np.errstate(over="ignore", invalid="ignore"):
		first, second, goal, pole_vector = joint - root, end - joint, target - root, pole - root
		length1, length2, distance, pole_length = (np.linalg.norm(value, axis=1) for value in (first, second, goal, pole_vector))
		maximum = length1 + length2
	if not all(np.isfinite(value).all() for value in (first, second, goal, pole_vector, length1, length2, distance, pole_length, maximum)) or np.any(length1 <= 0) or np.any(length2 <= 0):
		raise ValueError("bone lengths must be finite and positive, and positions must have finite differences")
	axis = np.empty_like(goal)
	moving = distance > 0
	axis[moving] = goal[moving] / distance[moving, None]
	axis[~moving] = first[~moving] / length1[~moving, None]
	clamped = np.clip(distance, np.abs(length1 - length2), maximum)
	# Scaling the triangle avoids squaring the original bone lengths.
	scale = np.maximum(length1, length2)
	l1, l2, d = length1 / scale, length2 / scale, clamped / scale
	along = np.zeros_like(d)
	nonzero = d > 0
	along[nonzero] = ((l1[nonzero] ** 2 - l2[nonzero] ** 2 + d[nonzero] ** 2) / (2 * d[nonzero])) * scale[nonzero]
	along = np.clip(along, -length1, length1)
	height = length1 * np.sqrt(np.maximum(0, 1 - (along / length1) ** 2))
	bend = pole_vector - np.sum(pole_vector * axis, axis=1)[:, None] * axis
	bend_length = np.linalg.norm(bend, axis=1)
	# A collinear pole uses the animated bend, then a deterministic axis.
	degenerate = bend_length <= pole_length * 1e-12
	if np.any(degenerate):
		bend[degenerate] = first[degenerate] - np.sum(first[degenerate] * axis[degenerate], axis=1)[:, None] * axis[degenerate]
		bend_length = np.linalg.norm(bend, axis=1)
		degenerate = bend_length <= length1 * 1e-12
		if np.any(degenerate):
			basis = np.eye(3)[np.argmin(np.abs(axis[degenerate]), axis=1)]
			bend[degenerate] = basis - np.sum(basis * axis[degenerate], axis=1)[:, None] * axis[degenerate]
			bend_length = np.linalg.norm(bend, axis=1)
	bend /= bend_length[:, None]
	with np.errstate(over="ignore", invalid="ignore"):
		solved_joint = root + axis * along[:, None] + bend * height[:, None]
		solved_end = root + axis * clamped[:, None]
	if not np.isfinite(solved_joint).all() or not np.isfinite(solved_end).all(): raise ValueError("solved positions exceed finite range")
	return solved_joint, solved_end


def _between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
	"""Return shortest-arc rotations for nonzero vectors."""
	a = source / np.linalg.norm(source, axis=1, keepdims=True)
	b = target / np.linalg.norm(target, axis=1, keepdims=True)
	dot = np.clip(np.sum(a * b, axis=1), -1, 1)
	rotation = np.column_stack((np.cross(a, b), 1 + dot))
	opposite = dot < -1 + 1e-10
	if np.any(opposite):
		basis = np.eye(3)[np.argmin(np.abs(a[opposite]), axis=1)]
		rotation[opposite, :3] = np.cross(a[opposite], basis)
		rotation[opposite, 3] = 0
	return rotation / np.linalg.norm(rotation, axis=1, keepdims=True)


def _validate_chain(skeleton: Skeleton, chain: Sequence[int]) -> tuple[int, int, int]:
	if not isinstance(chain, (tuple, list)) or len(chain) != 3: raise ValueError("chain must contain root, joint and end bone indices")
	try:
		if any(isinstance(value, (bool, np.bool_)) for value in chain): raise ValueError
		root, joint, end = (index(value) for value in chain)
	except (TypeError, ValueError) as exc: raise ValueError("chain must contain valid bone indices") from exc
	if min(root, joint, end) < 0 or max(root, joint, end) >= len(skeleton.parents) or skeleton.parents[joint] != root or skeleton.parents[end] != joint:
		raise ValueError("chain must contain three directly connected bones")
	return root, joint, end


def two_bone_pose(skeleton: Skeleton, pose: Sequence[ArrayLike], chain: Sequence[int], targets: ArrayLike, poles: ArrayLike, *, weight: ArrayLike=1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return a local pose with a direct two-bone chain aimed at model-space goals.

	The three chain indices are root, joint and end. Ancestors and the root and
	joint bones must have positive uniform local scale. Weight blends rotations
	with the original pose; partial weights do not guarantee reaching the goal.
	"""
	root, joint, end = _validate_chain(skeleton, chain)
	if not isinstance(pose, (tuple, list)) or len(pose) != 3: raise ValueError("pose must contain translation, rotation and scale arrays")
	translation, rotation, scale = (np.asarray(value, dtype=np.float32) for value in pose)
	model = model_matrices(skeleton, translation, rotation, scale)
	count = len(model)
	weight = np.asarray(weight, dtype=np.float32)
	if weight.ndim == 0: weight = np.broadcast_to(weight, (count,))
	if weight.shape != (count,) or not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)): raise ValueError("weight must be finite and in [0, 1] for each instance")
	ancestors, bone = [joint], root
	while bone != -1:
		ancestors.append(bone)
		bone = int(skeleton.parents[bone])
	chain_scale = scale[:, ancestors]
	if np.any(chain_scale <= 0) or not np.allclose(chain_scale, chain_scale[..., :1], rtol=1e-5, atol=0):
		raise ValueError("chain and ancestor scales must be positive and uniform")
	positions = model[:, :, :3, 3][:, [root, joint, end]].astype(np.float64)
	solved_joint, solved_end = solve_two_bone(positions[:, 0], positions[:, 1], positions[:, 2], targets, poles)
	root_delta = _between_vectors(positions[:, 1] - positions[:, 0], solved_joint - positions[:, 0])
	second = positions[:, 2] - positions[:, 1]
	rotated_second = (Quaternion.matrix_batch(root_delta) @ second[..., None])[..., 0]
	joint_delta = _between_vectors(rotated_second, solved_end - solved_joint)
	parent = int(skeleton.parents[root])
	if parent == -1: parent_rotation = np.broadcast_to(np.array([0., 0., 0., 1.]), (count, 4))
	else:
		linear = model[:, parent, :3, :3].astype(np.float64)
		parent_rotation = Quaternion.from_matrices_batch(linear / np.linalg.norm(linear[:, :, 0], axis=1)[:, None, None])
	root_linear = model[:, root, :3, :3].astype(np.float64)
	root_rotation = Quaternion.from_matrices_batch(root_linear / np.linalg.norm(root_linear[:, :, 0], axis=1)[:, None, None])
	new_root_global = Quaternion.multiply_batch(root_delta, root_rotation)
	parent_inverse = parent_rotation * [-1, -1, -1, 1]
	new_root = Quaternion.multiply_batch(parent_inverse, new_root_global)
	joint_local = rotation[:, joint] / np.linalg.norm(rotation[:, joint], axis=1, keepdims=True)
	new_joint = Quaternion.multiply_batch(Quaternion.multiply_batch(new_root_global * [-1, -1, -1, 1], joint_delta), Quaternion.multiply_batch(new_root_global, joint_local))
	result_rotation = rotation.copy()
	root_local = rotation[:, root] / np.linalg.norm(rotation[:, root], axis=1, keepdims=True)
	result_rotation[:, root] = Quaternion.nlerp_batch(root_local, new_root, weight[:, None])
	result_rotation[:, joint] = Quaternion.nlerp_batch(joint_local, new_joint, weight[:, None])
	return translation.copy(), result_rotation, scale.copy()


def _validate_axis(axis: ArrayLike) -> np.ndarray:
	axis = np.asarray(axis, dtype=np.float64)
	if axis.shape != (3,) or not np.isfinite(axis).all(): raise ValueError("axis must be a finite nonzero 3D vector")
	length = np.linalg.norm(axis)
	if not np.isfinite(length) or length == 0: raise ValueError("axis must be a finite nonzero 3D vector")
	return axis / length


def limit_twist(rotations: ArrayLike, reference: ArrayLike, axis: ArrayLike, minimum: float, maximum: float) -> np.ndarray:
	"""Clamp twist in radians around a reference-local axis, preserving swing.

	Rotations are (instances, 4) xyzw rows. Reference may be a single quaternion
	or one per row. The signed range is [-pi, pi] with exact half turns mapped
	to +pi. At a 180-degree swing around a perpendicular axis the twist is
	undefined, so use zero as its input angle.
	"""
	turn = np.asarray(rotations)
	if turn.ndim != 2 or turn.shape[1] != 4: raise ValueError("rotations must have shape (instances, 4)")
	if not (-np.pi <= minimum <= maximum <= np.pi): raise ValueError("twist limits must satisfy -pi <= minimum <= maximum <= pi")
	axis = _validate_axis(axis)
	turn = _rotations(turn, turn.shape, "rotations")
	ref = _rotations(np.broadcast_to(reference, turn.shape), turn.shape, "reference")
	conjugate = np.array([-1, -1, -1, 1])
	relative = Quaternion.multiply_batch(ref * conjugate, turn)
	projected = np.sum(relative[:, :3] * axis, axis=1)
	scalar = relative[:, 3]
	flip = (scalar < 0) | ((scalar == 0) & (projected < 0))
	projected = np.where(flip, -projected, projected)
	undefined = np.hypot(projected, scalar) <= 1e-8
	half = np.where(undefined, 0, np.arctan2(projected, np.abs(scalar)))
	twist_inverse = np.column_stack((-np.sin(half)[:, None] * axis, np.cos(half)))
	clamped = np.clip(half, minimum * 0.5, maximum * 0.5)
	correction = Quaternion.multiply_batch(twist_inverse, np.column_stack((np.sin(clamped)[:, None] * axis, np.cos(clamped))))
	return Quaternion.multiply_batch(turn, correction)


def foot_plant(targets: ArrayLike, normals: ArrayLike, stance: ArrayLike, locked: ArrayLike, anchors: ArrayLike, anchor_normals: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
	"""Retain world-space foot goals for rows still in a supported stance.

	Positions and surface normals are (feet, 3) rows. `locked` identifies the
	previous stance. Pass this tick's stance and returned goals as next tick's
	lock state and anchors. Clear a row's lock on teleports or owner changes.
	"""
	targets, normals, anchors, anchor_normals = (np.asarray(value, dtype=np.float64) for value in (targets, normals, anchors, anchor_normals))
	stance, locked = np.asarray(stance), np.asarray(locked)
	if targets.ndim != 2 or targets.shape[1] != 3 or any(value.shape != targets.shape for value in (normals, anchors, anchor_normals)):
		raise ValueError("foot goals and anchors must have shape (feet, 3)")
	if any(value.shape != (len(targets),) or value.dtype.kind != "b" for value in (stance, locked)):
		raise ValueError("stance and locked must be boolean rows aligned with the feet")
	keep = (stance & locked)[:, None]
	position = np.where(keep, anchors, targets)
	normal = np.where(keep, anchor_normals, normals)
	if not np.isfinite(position).all() or not np.isfinite(normal).all(): raise ValueError("resolved foot goals must be finite")
	return position, normal


def align_bone_axis(skeleton: Skeleton, pose: Sequence[ArrayLike], bone: SupportsIndex, axis: ArrayLike, directions: ArrayLike, *, weight: ArrayLike=1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Align a bone's local axis to model-space directions using local rotation.

	The bone and its ancestors require positive uniform scale. Weights blend the
	original and aligned local rotations; the bone position is unchanged.
	"""
	if isinstance(bone, (bool, np.bool_)): raise ValueError("bone must be a valid index")
	try: bone = index(bone)
	except TypeError as exc: raise ValueError("bone must be a valid index") from exc
	if bone < 0 or bone >= len(skeleton.parents): raise ValueError("bone must be a valid index")
	axis = _validate_axis(axis)
	if not isinstance(pose, (tuple, list)) or len(pose) != 3: raise ValueError("pose must contain translation, rotation and scale arrays")
	translation, rotation, scale = (np.asarray(value, dtype=np.float32) for value in pose)
	model = model_matrices(skeleton, translation, rotation, scale)
	count = len(model)
	directions = np.asarray(directions, dtype=np.float64)
	if directions.shape != (count, 3) or not np.isfinite(directions).all(): raise ValueError("directions must be finite (instances, 3) vectors")
	length = np.linalg.norm(directions, axis=1)
	if not np.isfinite(length).all() or np.any(length == 0): raise ValueError("directions must be nonzero finite vectors")
	weight = np.asarray(weight, dtype=np.float32)
	if weight.ndim == 0: weight = np.broadcast_to(weight, (count,))
	if weight.shape != (count,) or not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)): raise ValueError("weight must be finite and in [0, 1] for each instance")
	ancestors, node = [], bone
	while node != -1:
		ancestors.append(node)
		node = int(skeleton.parents[node])
	bone_scale = scale[:, ancestors]
	if np.any(bone_scale <= 0) or not np.allclose(bone_scale, bone_scale[..., :1], rtol=1e-5, atol=0): raise ValueError("bone and ancestor scales must be positive and uniform")
	linear = model[:, bone, :3, :3].astype(np.float64)
	bone_rotation = Quaternion.from_matrices_batch(linear / np.linalg.norm(linear[:, :, 0], axis=1)[:, None, None])
	current = (Quaternion.matrix_batch(bone_rotation) @ axis[:, None])[..., 0]
	delta = _between_vectors(current, directions)
	parent = int(skeleton.parents[bone])
	if parent == -1: parent_rotation = np.broadcast_to(np.array([0., 0., 0., 1.]), (count, 4))
	else:
		linear = model[:, parent, :3, :3].astype(np.float64)
		parent_rotation = Quaternion.from_matrices_batch(linear / np.linalg.norm(linear[:, :, 0], axis=1)[:, None, None])
	aligned = Quaternion.multiply_batch(parent_rotation * [-1, -1, -1, 1], Quaternion.multiply_batch(delta, bone_rotation))
	local = rotation[:, bone] / np.linalg.norm(rotation[:, bone], axis=1, keepdims=True)
	result_rotation = rotation.copy()
	result_rotation[:, bone] = Quaternion.nlerp_batch(local, aligned, weight[:, None])
	return translation.copy(), result_rotation, scale.copy()
