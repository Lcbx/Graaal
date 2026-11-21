"""CPU ragdoll pose conversion and joint springs, independent of ECS and rendering."""

import numpy as np

from .Math import Quaternion
from .Animation.SkeletalAnimation import _rotations, blend_poses, model_matrices, skin_matrices


def _rigid_pose(skeleton, pose):
	model = model_matrices(skeleton, *pose)
	if not np.allclose(pose[2], 1, atol=1e-6):
		raise ValueError("ragdoll poses require unit local scale; bake scale into the skeleton and shapes")
	return model


def _frame(position, rotation, count):
	position = np.array(np.broadcast_to(position, (count, 3)), dtype=np.float64, copy=True)
	if not np.all(np.isfinite(position)): raise ValueError("frame position must be finite")
	rotation = _rotations(np.broadcast_to(rotation, (count, 4)), (count, 4), "frame rotation")
	result = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
	result[:, :3, :3], result[:, :3, 3] = Quaternion.matrix_batch(rotation), position
	return result


def _joint_state(parents, orientations, angular_velocity):
	parents = np.asarray(parents)
	if parents.ndim != 1 or not len(parents) or parents.dtype.kind not in "iu" or parents[0] != -1 or np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, len(parents))):
		raise ValueError("parents must describe one parent-before-child tree")
	parents = parents.astype(np.intp, copy=False)
	orientations = np.asarray(orientations)
	if orientations.ndim != 3 or orientations.shape[1:] != (len(parents), 4):
		raise ValueError("orientations must have shape (instances, bones, 4)")
	count, bones = orientations.shape[:2]
	turns = _rotations(orientations, (count, bones, 4), "orientations")
	angular = np.asarray(angular_velocity, dtype=np.float64)
	if angular.shape != (count, bones, 3) or not np.isfinite(angular).all():
		raise ValueError("angular velocity must be finite (instances, bones, 3)")
	return parents, turns, angular


def _rotation_vector(error):
	error = np.where(error[..., 3:] < 0, -error, error)
	sine = np.linalg.norm(error[..., :3], axis=-1, keepdims=True)
	factor = np.divide(2 * np.arctan2(sine, error[..., 3:]), sine, out=np.full_like(sine, 2), where=sine > 1e-8)
	return error[..., :3] * factor


def _joint_reactions(parents, torque, limit):
	length = np.linalg.norm(torque, axis=-1, keepdims=True)
	torque = torque * np.minimum(1, np.divide(limit, length, out=np.ones_like(length), where=length > 0))
	result = np.zeros((len(torque), len(parents), 3), dtype=np.float64)
	result[:, 1:] = torque
	# Several joints can react on the same parent; accumulate those contributions.
	np.add.at(result, (np.arange(len(torque))[:, None], parents[None, 1:]), -torque)
	return result


def joint_spring_torques(parents, orientations, angular_velocity, neutral_rotations, *, stiffness, damping, max_torque):
	"""Return bounded world-space joint spring reactions as (instances, bones, 3).

	Orientations/velocities are world-space; neutral xyzw rotations are relative
	to each parent. Gains and each joint's torque cap broadcast to (instances,
	bones). Stiffness is N m/rad, damping N m s/rad, and the cap N m. The root
	has no world spring; every child torque has an equal opposite parent torque.
	"""
	parents, turns, angular = _joint_state(parents, orientations, angular_velocity)
	count, bones = turns.shape[:2]
	neutral = _rotations(np.broadcast_to(neutral_rotations, (count, bones, 4)), (count, bones, 4), "neutral rotations")
	controls = []
	for value, name in ((stiffness, "stiffness"), (damping, "damping"), (max_torque, "max torque")):
		values = np.broadcast_to(np.asarray(value, dtype=np.float64), (count, bones))
		if not np.isfinite(values).all() or np.any(values < 0): raise ValueError(f"{name} must be finite and nonnegative")
		controls.append(values[:, 1:, None])
	stiffness, damping, limit = controls
	desired = Quaternion.multiply_batch(turns[:, parents[1:]], neutral[:, 1:])
	error = Quaternion.multiply_batch(desired, turns[:, 1:] * [-1, -1, -1, 1])
	torque = stiffness * _rotation_vector(error) - damping * (angular[:, 1:] - angular[:, parents[1:]])
	return _joint_reactions(parents, torque, limit)


def ragdoll_limit_parameters(settings):
	"""Copy authored Bone axes, bend and limits into detached soft-limit parameters.

	Call once per immutable native RagdollSettings asset. No backend is imported;
	the returned arrays are read-only and can be passed to joint_limit_torques.
	"""
	parents = np.array(settings.parents, dtype=np.intp, copy=True)
	bones = len(parents)
	rotations = _rotations(settings.rest_rotations, (bones, 4), "rest rotations")
	translations = np.asarray(settings.rest_translations, dtype=np.float64)
	first_child = {}
	for i, parent in enumerate(parents):
		if parent >= 0: first_child.setdefault(int(parent), i)
	axes, planes = [], []
	for i, bone in enumerate(settings.bones):
		if bone.twist_axis is not None: axis = np.asarray(bone.twist_axis, dtype=np.float64)
		elif i in first_child: axis = translations[first_child[i]]
		else:
			axis = Quaternion.matrix_batch(rotations[i]).T @ translations[i]
		if bone.twist_axis is None and np.linalg.norm(axis) < 1e-9: axis = np.array([0, 1, 0.0])
		if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8: raise ValueError("twist axis must be a finite nonzero vector")
		axis = axis / np.linalg.norm(axis)
		plane = np.cross(axis, (0, 0, 1) if abs(axis[2]) < 0.9 else (1, 0, 0)) if bone.plane_axis is None else np.asarray(bone.plane_axis, dtype=np.float64)
		if plane.shape != (3,) or not np.isfinite(plane).all(): raise ValueError("plane axis must be a finite vector")
		plane = plane - axis * np.dot(axis, plane)
		if np.linalg.norm(plane) < 1e-8: raise ValueError("plane axis must not be parallel to twist axis")
		axes.append(axis)
		planes.append(plane / np.linalg.norm(plane))
	axes, planes = np.asarray(axes), np.asarray(planes)
	child_frames = Quaternion.from_matrices_batch(np.stack((axes, planes, np.cross(axes, planes)), axis=-1))
	bend = np.asarray([bone.bend for bone in settings.bones], dtype=np.float64)
	if not np.isfinite(bend).all(): raise ValueError("bend must be finite")
	bend_rotations = np.column_stack((np.zeros(bones), np.sin(bend / 2), np.zeros(bones), np.cos(bend / 2)))
	parent_frames = Quaternion.multiply_batch(Quaternion.multiply_batch(rotations, child_frames), bend_rotations)
	result = dict(parents=parents, parent_frames=parent_frames, child_frames=child_frames, twist=np.array([bone.twist for bone in settings.bones], dtype=np.float64), swing=np.array([bone.swing for bone in settings.bones], dtype=np.float64))
	for value in result.values(): value.flags.writeable = False
	return result


def joint_limit_torques(parents, orientations, angular_velocity, parent_frames, child_frames, *, twist, swing, margin, stiffness, damping, max_torque):
	"""Return progressive swing/twist soft-limit torques, with parent reactions.

	Frames are bone-local xyzw, limits are (min, max) twist and two swing half
	angles in radians, broadcasting to (instances, bones, 4/2). The comfortable
	range is inset by margin radians. Outside it, spring magnitude is
	stiffness * distance**2 / margin; damping opposes only outward motion and
	fades in across the margin. Gains/cap use the same units as joint springs.
	"""
	parents, turns, angular = _joint_state(parents, orientations, angular_velocity)
	count, bones = turns.shape[:2]
	frames = [_rotations(np.broadcast_to(value, (count, bones, 4)), (count, bones, 4), name)[:, 1:] for value, name in ((parent_frames, "parent frames"), (child_frames, "child frames"))]
	parent = Quaternion.multiply_batch(turns[:, parents[1:]], frames[0])
	child = Quaternion.multiply_batch(turns[:, 1:], frames[1])
	relative = Quaternion.multiply_batch(parent * [-1, -1, -1, 1], child)
	relative = np.where(relative[..., 3:] < 0, -relative, relative)
	limits = [np.broadcast_to(np.asarray(value, dtype=np.float64), (count, bones, 2)) for value in (twist, swing)]
	twist, swing = limits
	if not np.isfinite(twist).all() or np.any(twist[..., 0] > twist[..., 1]) or np.any(np.abs(twist) > np.pi + 1e-6): raise ValueError("twist limits must be ordered finite angles in [-pi, pi]")
	if not np.isfinite(swing).all() or np.any(swing < 0) or np.any(swing > np.pi + 1e-6): raise ValueError("swing half angles must be finite in [0, pi]")
	twist, swing = np.clip(twist[:, 1:], -np.pi, np.pi), np.clip(swing[:, 1:], 0, np.pi)
	controls = []
	for value, name in ((margin, "margin"), (stiffness, "stiffness"), (damping, "damping"), (max_torque, "max torque")):
		values = np.broadcast_to(np.asarray(value, dtype=np.float64), (count, bones))
		if not np.isfinite(values).all() or np.any(values < 0) or (name == "margin" and np.any(values == 0)): raise ValueError(f"{name} must be finite and {'positive' if name == 'margin' else 'nonnegative'}")
		controls.append(values[:, 1:, None])
	margin, stiffness, damping, limit = controls
	# q = swing * twist, with twist around constraint X and swing in Y/Z.
	twist_q = np.zeros_like(relative)
	length = np.hypot(relative[..., 0], relative[..., 3])
	twist_q[..., 0] = np.divide(relative[..., 0], length, out=np.zeros_like(length), where=length > 1e-8)
	twist_q[..., 3] = np.divide(relative[..., 3], length, out=np.ones_like(length), where=length > 1e-8)
	swing_q = Quaternion.multiply_batch(relative, twist_q * [-1, -1, -1, 1])
	angle = 2 * np.arctan2(twist_q[..., 0:1], twist_q[..., 3:])
	midpoint = np.mean(twist, axis=-1, keepdims=True)
	clamped = np.clip(angle, np.minimum(twist[..., :1] + margin, midpoint), np.maximum(twist[..., 1:] - margin, midpoint))
	free_twist = (twist[..., :1] <= -np.pi + 1e-6) & (twist[..., 1:] >= np.pi - 1e-6)
	clamped = np.where(free_twist, angle, clamped)
	twist_target = np.concatenate((np.sin(clamped / 2), np.zeros(clamped.shape[:-1] + (2,)), np.cos(clamped / 2)), axis=-1)
	radii = np.where(swing >= np.pi - 1e-6, 1, np.sin(np.maximum(swing - margin, 0) / 2))
	projected = np.where(radii > 1e-8, swing_q[..., 1:3], 0)
	radius = np.linalg.norm(np.divide(projected, radii, out=np.zeros_like(projected), where=radii > 1e-8), axis=-1, keepdims=True)
	projected = projected / np.maximum(radius, 1)
	swing_target = np.concatenate((np.zeros(projected.shape[:-1] + (1,)), projected, np.sqrt(np.maximum(1 - np.sum(projected**2, axis=-1, keepdims=True), 0))), axis=-1)
	clamp_swing = (radius > 1) | np.any((radii <= 1e-8) & (np.abs(swing_q[..., 1:3]) > 1e-8), axis=-1, keepdims=True)
	swing_target = np.where(clamp_swing, swing_target, swing_q)
	desired = Quaternion.multiply_batch(parent, Quaternion.multiply_batch(swing_target, twist_target))
	error = _rotation_vector(Quaternion.multiply_batch(desired, child * [-1, -1, -1, 1]))
	distance = np.linalg.norm(error, axis=-1, keepdims=True)
	direction = np.divide(error, distance, out=np.zeros_like(error), where=distance > 1e-7)
	outward_speed = np.maximum(-np.sum((angular[:, 1:] - angular[:, parents[1:]]) * direction, axis=-1, keepdims=True), 0)
	weight = distance / margin
	torque = direction * (stiffness * distance * weight + damping * outward_speed * np.minimum(weight, 1))
	return _joint_reactions(parents, torque, limit)


class RagdollAsset:
	"""Map native joint frames to selected render bones, preserving other local poses.

	``bones`` maps native bone order to render indices. Native rest joint frames
	must coincide with the selected render joints. Shapes may have arbitrary
	native offsets and rotations. Unit skeleton/entity scale is required.
	"""
	def __init__(self, skeleton, settings=None, *, bones=None, names=None, **native_settings):
		import pyjoltphysics as pj
		self.skeleton = skeleton
		if bones is None: bones = np.arange(len(skeleton.parents))
		bones = np.asarray(bones)
		if bones.ndim != 1 or not len(bones) or bones.dtype.kind not in "iu" or np.any(bones < 0) or np.any(bones >= len(skeleton.parents)) or len(np.unique(bones)) != len(bones):
			raise ValueError("bones must be distinct render bone indices in native order")
		self.bones = bones.astype(np.intp, copy=True)
		self.bones.flags.writeable = False
		self._native_by_render = {int(bone): i for i, bone in enumerate(bones)}
		parents = []
		for bone in bones:
			parent = int(skeleton.parents[bone])
			while parent >= 0 and parent not in self._native_by_render: parent = int(skeleton.parents[parent])
			parents.append(self._native_by_render.get(parent, -1))
		self.parents = np.array(parents, dtype=np.intp)
		self.parents.flags.writeable = False
		if self.parents[0] != -1 or np.any(self.parents[1:] < 0) or np.any(self.parents[1:] >= np.arange(1, len(bones))):
			raise ValueError("physical bones must form one parent-before-child hierarchy")
		rest = (skeleton.translation[None], skeleton.rotation[None], skeleton.scale[None])
		translations, rotations = self.target_pose(rest)
		if settings is None:
			names = [str(int(bone)) for bone in bones] if names is None else names
			if len(names) != len(bones): raise ValueError("names must match the physical bone count")
			settings = pj.RagdollSettings.from_skeleton(names, self.parents, translations[0], rotations[0], **native_settings)
		elif native_settings or names is not None:
			raise ValueError("native settings and names are only used when generating fallback settings")
		if not isinstance(settings, pj.RagdollSettings) or len(settings.bones) != len(bones):
			raise ValueError("settings must contain one native Bone per mapped render bone")
		if not np.array_equal(settings.parents, self.parents) or not np.allclose(settings.rest_translations, translations[0], atol=1e-5) or not np.allclose(np.abs(np.sum(settings.rest_rotations * rotations[0], axis=-1)), 1, atol=1e-5):
			raise ValueError("native rest joint frames must match the mapped skeleton rest pose")
		self.settings = settings

	def target_pose(self, pose):
		"""Convert batched render-local TRS into native-parent joint targets (T, R)."""
		model = _rigid_pose(self.skeleton, pose)[:, self.bones].astype(np.float64)
		local = model.copy()
		children = np.flatnonzero(self.parents >= 0)
		local[:, children] = np.linalg.inv(model[:, self.parents[children]]) @ model[:, children]
		rotation = Quaternion.from_matrices_batch(local[..., :3, :3].reshape(-1, 3, 3)).reshape(local.shape[:2] + (4,))
		return local[..., :3, 3].copy(), rotation

	def local_pose(self, positions, orientations, base, *, position=(0, 0, 0), rotation=(0, 0, 0, 1), weight=1):
		"""Overlay native world joint poses on animation, returning detached local TRS.

		Inputs are (instances, physical bones, 3/4) and batched render-local TRS.
		The entity frame is removed before skinning. Unmapped bones retain their
		animation-local pose and follow the final pose of their parent.
		"""
		model = _rigid_pose(self.skeleton, base).astype(np.float64)
		count = len(model)
		positions = np.asarray(positions, dtype=np.float64)
		if positions.shape != (count, len(self.bones), 3) or not np.all(np.isfinite(positions)):
			raise ValueError("joint positions must be finite (instances, physical bones, 3)")
		orientations = _rotations(orientations, (count, len(self.bones), 4), "joint orientations")
		physical = np.tile(np.eye(4), (count, len(self.bones), 1, 1))
		physical[..., :3, :3], physical[..., :3, 3] = Quaternion.matrix_batch(orientations), positions
		physical = np.linalg.inv(_frame(position, rotation, count))[:, None] @ physical
		translation, turns, scale = (np.array(value, dtype=np.float32, copy=True) for value in base)
		turns = _rotations(turns, (count, len(self.skeleton.parents), 4), "base rotation")
		# Skeleton order is topological; unmapped bones inherit the corrected parent.
		for bone, parent in enumerate(self.skeleton.parents):
			if bone in self._native_by_render:
				model[:, bone] = physical[:, self._native_by_render[bone]]
				local = model[:, bone] if parent < 0 else np.linalg.inv(model[:, parent]) @ model[:, bone]
				translation[:, bone] = local[:, :3, 3]
				turns[:, bone] = Quaternion.from_matrices_batch(local[:, :3, :3])
			else:
				local = np.tile(np.eye(4), (count, 1, 1))
				local[:, :3, :3], local[:, :3, 3] = Quaternion.matrix_batch(turns[:, bone]), translation[:, bone]
				model[:, bone] = local if parent < 0 else model[:, parent] @ local
		return blend_poses(self.skeleton, base, (translation, turns, scale), weight)

	def skin(self, positions, orientations, base, **kwargs):
		"""Produce the existing float32 (instances, render bones, 3, 4) palette."""
		return skin_matrices(self.skeleton, model_matrices(self.skeleton, *self.local_pose(positions, orientations, base, **kwargs)))
