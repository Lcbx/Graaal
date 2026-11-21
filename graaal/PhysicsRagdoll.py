"""CPU skeleton/native-ragdoll pose conversion, independent of ECS and rendering."""

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
