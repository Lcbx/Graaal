"""CPU skeletal animation resources and batched pose evaluation.

Transforms use column vectors, xyzw quaternions and float32 arrays.
"""

import numpy as np


def _array(value, shape, name):
	result = np.array(value, dtype=np.float32, copy=True)
	if result.shape != shape or not np.isfinite(result).all():
		raise ValueError(f"{name} must be finite with shape {shape}")
	return result


def _rotations(value, shape, name):
	result = _array(value, shape, name)
	length = np.linalg.norm(result, axis=-1, keepdims=True)
	if np.any(length < 1e-8): raise ValueError(f"{name} contains a zero quaternion")
	return result / length


def _freeze(*arrays):
	for array in arrays: array.flags.writeable = False


class Skeleton:
	"""A depth-first, parent-before-child skeleton and its local reference pose."""
	def __init__(self, parents, translation, rotation, scale, inverse_bind):
		raw = np.asarray(parents)
		if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in "iu":
			raise ValueError("parents must be a nonempty integer array")
		parents = raw.astype(np.int64, copy=True)
		stack = []
		depth = np.empty(len(parents), dtype=np.intp)
		for i, parent in enumerate(parents):
			if parent == -1:
				stack.clear()
			elif parent < 0 or parent >= i or parent not in stack:
				raise ValueError("parents must describe a depth-first preorder forest")
			else:
				while stack[-1] != parent: stack.pop()
			depth[i] = len(stack)
			stack.append(i)
		self.parents = parents
		self.depth = depth
		self.translation = _array(translation, (len(parents), 3), "translation")
		self.rotation = _rotations(rotation, (len(parents), 4), "rotation")
		self.scale = _array(scale, (len(parents), 3), "scale")
		self.inverse_bind = _array(inverse_bind, (len(parents), 4, 4), "inverse_bind")
		if not np.allclose(self.inverse_bind[:, 3, :], [0, 0, 0, 1]):
			raise ValueError("inverse_bind matrices must be affine")
		_freeze(self.parents, self.depth, self.translation, self.rotation, self.scale, self.inverse_bind)


class Clip:
	"""Uniformly sampled local poses. The last frame is at (frame_count - 1) / fps."""
	def __init__(self, translation, rotation, scale, fps=30):
		values = np.asarray(translation)
		if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] != 3:
			raise ValueError("translation must have shape (frames, bones, 3)")
		frames, bones = values.shape[:2]
		if not np.isfinite(fps) or fps <= 0: raise ValueError("fps must be positive and finite")
		self.fps = float(fps)
		self.translation = _array(translation, (frames, bones, 3), "translation")
		self.rotation = _rotations(rotation, (frames, bones, 4), "rotation")
		self.scale = _array(scale, (frames, bones, 3), "scale")
		# Accumulated signs make each consecutive pair take the short arc.
		dots = np.sum(self.rotation[1:] * self.rotation[:-1], axis=-1)
		signs = np.concatenate((np.ones((1, bones), dtype=np.float32), np.cumprod(np.where(dots < 0, -1, 1), axis=0)), axis=0)
		self.rotation *= signs[..., None]
		_freeze(self.translation, self.rotation, self.scale)

	@property
	def duration(self): return (len(self.translation) - 1) / self.fps


def sample_clip(clip, times, *, loop=True):
	"""Return batched local translation, rotation and scale for a 1D time array.

	Looping treats the final frame as the endpoint at duration and wraps there;
	seamless loops should therefore have matching first and last poses.
	"""
	times = np.asarray(times, dtype=np.float64)
	if times.ndim != 1 or not np.isfinite(times).all(): raise ValueError("times must be a finite 1D array")
	if clip.duration == 0:
		frame = np.zeros_like(times)
	else:
		frame = (np.mod(times, clip.duration) if loop else np.clip(times, 0, clip.duration)) * clip.fps
	frame = np.minimum(frame, len(clip.translation) - 1)
	first = np.floor(frame).astype(np.intp)
	second = np.minimum(first + 1, len(clip.translation) - 1)
	weight = (frame - first).astype(np.float32)[:, None, None]
	translation = clip.translation[first] * (1 - weight) + clip.translation[second] * weight
	scale = clip.scale[first] * (1 - weight) + clip.scale[second] * weight
	q0, q1 = clip.rotation[first], clip.rotation[second]
	q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
	rotation = q0 * (1 - weight) + q1 * weight
	rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
	return translation, rotation, scale


def model_matrices(skeleton, translation, rotation, scale):
	"""Convert batched local TRS to model matrices, processing one depth at a time."""
	translation = np.asarray(translation, dtype=np.float32)
	if translation.ndim != 3 or translation.shape[1:] != (len(skeleton.parents), 3):
		raise ValueError("translation must have shape (instances, bones, 3)")
	n, bones = translation.shape[:2]
	translation = _array(translation, (n, bones, 3), "translation")
	rotation = _rotations(rotation, (n, bones, 4), "rotation")
	scale = _array(scale, (n, bones, 3), "scale")
	x, y, z, w = np.moveaxis(rotation, -1, 0)
	local = np.zeros((n, bones, 4, 4), dtype=np.float32)
	local[..., 0, 0] = 1 - 2 * (y*y + z*z)
	local[..., 0, 1] = 2 * (x*y - z*w)
	local[..., 0, 2] = 2 * (x*z + y*w)
	local[..., 1, 0] = 2 * (x*y + z*w)
	local[..., 1, 1] = 1 - 2 * (x*x + z*z)
	local[..., 1, 2] = 2 * (y*z - x*w)
	local[..., 2, 0] = 2 * (x*z - y*w)
	local[..., 2, 1] = 2 * (y*z + x*w)
	local[..., 2, 2] = 1 - 2 * (x*x + y*y)
	local[..., :3, :3] *= scale[..., None, :]
	local[..., :3, 3] = translation
	local[..., 3, 3] = 1
	result = np.empty_like(local)
	for depth in range(int(skeleton.depth.max()) + 1):
		bones_at_depth = np.flatnonzero(skeleton.depth == depth)
		if depth == 0: result[:, bones_at_depth] = local[:, bones_at_depth]
		else: result[:, bones_at_depth] = result[:, skeleton.parents[bones_at_depth]] @ local[:, bones_at_depth]
	return result


def skin_matrices(skeleton, model):
	"""Return contiguous row-major 3x4 affine skin matrices (model @ inverse bind)."""
	model = np.asarray(model, dtype=np.float32)
	if model.ndim != 4 or model.shape[1:] != (len(skeleton.parents), 4, 4) or not np.isfinite(model).all():
		raise ValueError("model must be finite with shape (instances, bones, 4, 4)")
	return np.ascontiguousarray((model @ skeleton.inverse_bind)[..., :3, :])


def skin_vertices(positions, normals, joints, weights, matrices):
	"""Skin one mesh on the CPU, returning positions and inverse-transpose normals.

	Joint indices and weights have shape (vertices, influences); matrices have
	shape (bones, 3, 4). Weights must be nonnegative and sum to one per vertex.
	"""
	positions = np.asarray(positions, dtype=np.float32)
	normals = np.asarray(normals, dtype=np.float32)
	joints = np.asarray(joints)
	weights = np.asarray(weights, dtype=np.float32)
	matrices = np.asarray(matrices, dtype=np.float32)
	if positions.ndim != 2 or positions.shape[1:] != (3,) or normals.shape != positions.shape:
		raise ValueError("positions and normals must have shape (vertices, 3)")
	if joints.ndim != 2 or joints.shape[0] != len(positions) or joints.shape[1] == 0 or joints.dtype.kind not in "iu" or weights.shape != joints.shape:
		raise ValueError("joints and weights must have shape (vertices, influences)")
	if matrices.ndim != 3 or matrices.shape[1:] != (3, 4) or not len(matrices):
		raise ValueError("matrices must have shape (bones, 3, 4)")
	if not all(np.isfinite(value).all() for value in (positions, normals, weights, matrices)):
		raise ValueError("skin inputs must be finite")
	if np.any(joints < 0) or np.any(joints >= len(matrices)) or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, atol=1e-5):
		raise ValueError("invalid joint indices or skin weights")
	blended = np.sum(matrices[joints] * weights[..., None, None], axis=1)
	points = np.einsum("nij,nj->ni", blended[..., :3], positions) + blended[..., 3]
	try:
		directions = np.linalg.solve(blended[..., :3].transpose(0, 2, 1), normals[..., None])[..., 0]
	except np.linalg.LinAlgError as exc:
		raise ValueError("skinning produced a singular normal transform") from exc
	lengths = np.linalg.norm(directions, axis=1, keepdims=True)
	return points, directions / np.maximum(lengths, 1e-8)
