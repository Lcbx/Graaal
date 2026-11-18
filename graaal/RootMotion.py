"""Batched rigid root-motion extraction, independent of ECS and physics."""
from operator import index

import numpy as np

from .SkeletalAnimation import sample_clip


def _inverse(matrix):
	result = np.zeros_like(matrix)
	result[..., :3, :3] = matrix[..., :3, :3].swapaxes(-1, -2)
	result[..., :3, 3] = -(result[..., :3, :3] @ matrix[..., :3, 3, None])[..., 0]
	result[..., 3, 3] = 1
	return result


def _transform(translation, rotation):
	rotation = np.asarray(rotation, dtype=np.float64)
	rotation = rotation / np.linalg.norm(rotation, axis=-1, keepdims=True)
	x, y, z, w = np.moveaxis(rotation, -1, 0)
	result = np.zeros(rotation.shape[:-1] + (4, 4), dtype=np.float64)
	result[..., 0, 0] = 1 - 2 * (y*y + z*z)
	result[..., 0, 1] = 2 * (x*y - z*w)
	result[..., 0, 2] = 2 * (x*z + y*w)
	result[..., 1, 0] = 2 * (x*y + z*w)
	result[..., 1, 1] = 1 - 2 * (x*x + z*z)
	result[..., 1, 2] = 2 * (y*z - x*w)
	result[..., 2, 0] = 2 * (x*z - y*w)
	result[..., 2, 1] = 2 * (y*z + x*w)
	result[..., 2, 2] = 1 - 2 * (x*x + y*y)
	result[..., :3, 3] = translation
	result[..., 3, 3] = 1
	return result


def _powers(matrix, exponents):
	result = np.broadcast_to(np.eye(4), (len(exponents), 4, 4)).copy()
	base = np.broadcast_to(matrix, result.shape).copy()
	base[exponents < 0] = _inverse(matrix)
	remaining = np.abs(exponents)
	while np.any(remaining):
		selected = (remaining & 1) != 0
		result[selected] = result[selected] @ base[selected]
		base = base @ base
		remaining //= 2
	return result


def _quaternions(matrices):
	"""Convert batches of rigid rotation matrices to xyzw quaternions."""
	m = matrices[..., :3, :3]
	diagonal = np.stack((1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], 1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]), axis=1)
	choice = np.argmax(diagonal, axis=1)
	result = np.empty((len(m), 4), dtype=np.float64)
	for branch in range(4):
		selected = choice == branch
		part = m[selected]
		s = 2 * np.sqrt(np.maximum(diagonal[selected, branch], 0))
		if branch == 0:
			values = ((part[:, 2, 1] - part[:, 1, 2]) / s, (part[:, 0, 2] - part[:, 2, 0]) / s, (part[:, 1, 0] - part[:, 0, 1]) / s, s / 4)
		elif branch == 1:
			values = (s / 4, (part[:, 0, 1] + part[:, 1, 0]) / s, (part[:, 0, 2] + part[:, 2, 0]) / s, (part[:, 2, 1] - part[:, 1, 2]) / s)
		elif branch == 2:
			values = ((part[:, 0, 1] + part[:, 1, 0]) / s, s / 4, (part[:, 1, 2] + part[:, 2, 1]) / s, (part[:, 0, 2] - part[:, 2, 0]) / s)
		else:
			values = ((part[:, 0, 2] + part[:, 2, 0]) / s, (part[:, 1, 2] + part[:, 2, 1]) / s, s / 4, (part[:, 1, 0] - part[:, 0, 1]) / s)
		result[selected] = np.stack(values, axis=1)
	return result / np.linalg.norm(result, axis=1, keepdims=True)


def _blend_deltas(source, target, weight):
	"""Blend same-frame rigid requests using linear translation and quaternion nlerp."""
	weight = np.asarray(weight, dtype=np.float64)[:, None]
	q0, q1 = _quaternions(source), _quaternions(target)
	q1 = np.where(np.sum(q0 * q1, axis=1, keepdims=True) < 0, -q1, q1)
	translation = source[:, :3, 3] * (1 - weight) + target[:, :3, 3] * weight
	return _transform(translation, q0 * (1 - weight) + q1 * weight)


class RootMotion:
	"""Extract a root clip's rigid movement without changing its pose or ECS state."""
	def __init__(self, skeleton, clip, root=0):
		if isinstance(root, (bool, np.bool_)): raise ValueError("root must be a bone index")
		try: root = index(root)
		except TypeError as exc: raise ValueError("root must be a bone index") from exc
		if root < 0 or root >= len(skeleton.parents) or skeleton.parents[root] != -1:
			raise ValueError("root must identify a skeleton root bone")
		if clip.translation.shape[1] != len(skeleton.parents): raise ValueError("Clip and skeleton bone counts differ")
		if not np.allclose(clip.scale[:, root], 1, rtol=0, atol=1e-6):
			raise ValueError("root motion requires unit root scale")
		self.skeleton, self.clip, self.root = skeleton, clip, root
		endpoints = self._sample(np.array([0, clip.duration]))
		self._start = endpoints[0]
		self._cycle = _inverse(endpoints[0]) @ endpoints[1]

	def _sample(self, times):
		translation, rotation, _ = sample_clip(self.clip, times, loop=False)
		return _transform(translation[:, self.root], rotation[:, self.root])

	def _unwrapped(self, times, loop):
		if not loop or not self.clip.duration: return self._sample(times)
		cycles = np.floor(times / self.clip.duration)
		if not np.isfinite(cycles).all() or np.any(np.abs(cycles) >= 2**52):
			raise ValueError("root motion interval exceeds supported loop count")
		local = np.mod(times, self.clip.duration)
		return self._start @ _powers(self._cycle, cycles.astype(np.int64)) @ _inverse(self._start) @ self._sample(local)

	def delta(self, start, end, *, loop=True):
		"""Return (N, 4, 4) transforms from start root frames to end root frames."""
		start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
		if start.ndim != 1 or end.shape != start.shape or not np.isfinite(start).all() or not np.isfinite(end).all():
			raise ValueError("root motion times must be finite 1D arrays of equal length")
		return _inverse(self._unwrapped(start, loop)) @ self._unwrapped(end, loop)

	def _reference(self):
		if not np.allclose(self.skeleton.scale[self.root], 1, rtol=0, atol=1e-6):
			raise ValueError("in-place root motion requires unit reference root scale")
		return _transform(self.skeleton.translation[self.root], self.skeleton.rotation[self.root])

	def character_delta(self, start, end, *, loop=True):
		"""Return deltas in character coordinates for poses anchored to the reference root."""
		reference = self._reference()
		return reference @ self.delta(start, end, loop=loop) @ _inverse(reference)

	def in_place(self, translation, rotation, scale):
		"""Return copied local pose arrays with the root fixed to its reference pose."""
		self._reference()
		translation = np.array(translation, dtype=np.float32, copy=True)
		rotation = np.array(rotation, dtype=np.float32, copy=True)
		scale = np.array(scale, dtype=np.float32, copy=True)
		bones = len(self.skeleton.parents)
		if translation.ndim != 3 or translation.shape[1:] != (bones, 3):
			raise ValueError("translation must have shape (instances, bones, 3)")
		count = len(translation)
		if rotation.shape != (count, bones, 4) or scale.shape != (count, bones, 3):
			raise ValueError("rotation and scale must match pose instances and bones")
		if not all(np.isfinite(value).all() for value in (translation, rotation, scale)):
			raise ValueError("pose values must be finite")
		if not np.allclose(scale[:, self.root], 1, rtol=0, atol=1e-6):
			raise ValueError("in-place root motion requires unit pose root scale")
		translation[:, self.root] = self.skeleton.translation[self.root]
		rotation[:, self.root] = self.skeleton.rotation[self.root]
		scale[:, self.root] = self.skeleton.scale[self.root]
		return translation, rotation, scale
