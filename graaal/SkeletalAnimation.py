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


def _output(value, shape, name):
	if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape or not value.flags.writeable or not value.flags.c_contiguous:
		raise ValueError(f"{name} must be a writable C-contiguous float32 array with shape {shape}")
	return value


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


def sample_clip(clip, times, *, loop=True, out=None):
	"""Return batched local translation, rotation and scale for a 1D time array.

	Looping treats the final frame as the endpoint at duration and wraps there;
	seamless loops should therefore have matching first and last poses.
	"""
	times = np.asarray(times, dtype=np.float64)
	if times.ndim != 1 or not np.isfinite(times).all(): raise ValueError("times must be a finite 1D array")
	shape = (len(times), clip.translation.shape[1])
	if out is None:
		translation = np.empty(shape + (3,), dtype=np.float32)
		rotation = np.empty(shape + (4,), dtype=np.float32)
		scale = np.empty(shape + (3,), dtype=np.float32)
	else:
		if not isinstance(out, (tuple, list)) or len(out) != 3: raise ValueError("out must contain translation, rotation and scale arrays")
		translation = _output(out[0], shape + (3,), "translation out")
		rotation = _output(out[1], shape + (4,), "rotation out")
		scale = _output(out[2], shape + (3,), "scale out")
		if any(np.shares_memory(a, b) for a, b in ((translation, rotation), (translation, scale), (rotation, scale))):
			raise ValueError("pose output arrays must not overlap")
	if clip.duration == 0:
		frame = np.zeros_like(times)
	else:
		frame = (np.mod(times, clip.duration) if loop else np.clip(times, 0, clip.duration)) * clip.fps
	frame = np.minimum(frame, len(clip.translation) - 1)
	first = np.floor(frame).astype(np.intp)
	second = np.minimum(first + 1, len(clip.translation) - 1)
	weight = (frame - first).astype(np.float32)[:, None, None]
	np.multiply(clip.translation[first], 1 - weight, out=translation)
	translation += clip.translation[second] * weight
	np.multiply(clip.scale[first], 1 - weight, out=scale)
	scale += clip.scale[second] * weight
	q0, q1 = clip.rotation[first], clip.rotation[second]
	q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
	np.multiply(q0, 1 - weight, out=rotation)
	rotation += q1 * weight
	rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
	return translation, rotation, scale


def sample_crossfade(source, target, source_times, target_times, weights, *, source_loop=True, target_loop=True):
	"""Sample and blend two clips in local space, with one weight per instance.

	Weights range from zero (source) to one (target); a scalar applies to all
	instances. Both clips must use the same skeleton and their times must have
	the same length. Rotation uses shortest-path normalized linear interpolation.
	"""
	if source.translation.shape[1] != target.translation.shape[1]:
		raise ValueError("clips must have the same bone count and order")
	source_pose = sample_clip(source, source_times, loop=source_loop)
	target_pose = sample_clip(target, target_times, loop=target_loop)
	if source_pose[0].shape != target_pose[0].shape:
		raise ValueError("source and target times must have the same length")
	weights = np.asarray(weights, dtype=np.float32)
	if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
		raise ValueError("weights must be finite values in [0, 1], one per instance")
	if weights.ndim == 0: weights = np.broadcast_to(weights, (len(source_pose[0]),))
	if weights.shape != (len(source_pose[0]),):
		raise ValueError("weights must be finite values in [0, 1], one per instance")
	weight = weights[:, None, None]
	translation = source_pose[0] * (1 - weight) + target_pose[0] * weight
	q0, q1 = source_pose[1], target_pose[1]
	q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
	rotation = q0 * (1 - weight) + q1 * weight
	rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
	scale = source_pose[2] * (1 - weight) + target_pose[2] * weight
	return translation, rotation, scale


def sample_clip_pairs(skeleton, clips, clip_ids, times, weights, *, out=None):
	"""Evaluate two clamped clip samples per instance into batched local poses.

	`clip_ids`, `times` and `weights` have shape (instances, 2). Weights are
	nonnegative and sum to one per instance. The caller handles playback time
	wrapping and root-motion removal separately.
	"""
	clip_ids = np.asarray(clip_ids)
	times = np.asarray(times, dtype=np.float64)
	weights = np.asarray(weights, dtype=np.float32)
	if clip_ids.ndim != 2 or clip_ids.shape[1:] != (2,) or clip_ids.dtype.kind not in "iu" or np.any(clip_ids < 0):
		raise ValueError("clip_ids must be a nonnegative integer array of shape (instances, 2)")
	if times.shape != clip_ids.shape or not np.isfinite(times).all():
		raise ValueError("times must be finite with shape (instances, 2)")
	if weights.shape != clip_ids.shape or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)) or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-6):
		raise ValueError("weights must be nonnegative, finite, and sum to one per instance")
	for clip_id in np.unique(clip_ids):
		clip_id = int(clip_id)
		if clip_id not in clips: raise ValueError(f"Unknown clip {clip_id}")
		if clips[clip_id].translation.shape[1] != len(skeleton.parents): raise ValueError("Clip and skeleton bone counts differ")
	shape = (len(clip_ids), len(skeleton.parents))
	if out is None:
		pose = tuple(np.empty(shape + (size,), dtype=np.float32) for size in (3, 4, 3))
	else:
		if not isinstance(out, (tuple, list)) or len(out) != 3: raise ValueError("out must contain translation, rotation and scale arrays")
		pose = tuple(_output(value, shape + (size,), "pose out") for value, size in zip(out, (3, 4, 3)))
		if any(np.shares_memory(a, b) for a, b in ((pose[0], pose[1]), (pose[0], pose[2]), (pose[1], pose[2]))):
			raise ValueError("pose output arrays must not overlap")
	# Endpoint weights require only one sample; this includes ordinary clip players.
	for slot, selected in ((0, weights[:, 1] == 0), (1, weights[:, 1] == 1)):
		for clip_id in np.unique(clip_ids[selected, slot]):
			rows = np.flatnonzero(selected & (clip_ids[:, slot] == clip_id))
			local = sample_clip(clips[int(clip_id)], times[rows, slot], loop=False)
			for output, values in zip(pose, local): output[rows] = values
	intermediate = (weights[:, 1] > 0) & (weights[:, 1] < 1)
	for source, target in np.unique(clip_ids[intermediate], axis=0):
		rows = np.flatnonzero(intermediate & (clip_ids[:, 0] == source) & (clip_ids[:, 1] == target))
		mixed = sample_crossfade(clips[int(source)], clips[int(target)], times[rows, 0], times[rows, 1], weights[rows, 1], source_loop=False, target_loop=False)
		for output, values in zip(pose, mixed): output[rows] = values
	return pose


def model_matrices(skeleton, translation, rotation, scale, *, out=None):
	"""Convert batched local TRS to model matrices, processing one depth at a time."""
	translation = np.asarray(translation, dtype=np.float32)
	if translation.ndim != 3 or translation.shape[1:] != (len(skeleton.parents), 3):
		raise ValueError("translation must have shape (instances, bones, 3)")
	n, bones = translation.shape[:2]
	result = np.empty((n, bones, 4, 4), dtype=np.float32) if out is None else _output(out, (n, bones, 4, 4), "model out")
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
	for depth in range(int(skeleton.depth.max()) + 1):
		bones_at_depth = np.flatnonzero(skeleton.depth == depth)
		if depth == 0: result[:, bones_at_depth] = local[:, bones_at_depth]
		else: result[:, bones_at_depth] = result[:, skeleton.parents[bones_at_depth]] @ local[:, bones_at_depth]
	return result


def skin_matrices(skeleton, model, *, out=None):
	"""Return contiguous row-major 3x4 affine skin matrices (model @ inverse bind)."""
	model = np.asarray(model, dtype=np.float32)
	if model.ndim != 4 or model.shape[1:] != (len(skeleton.parents), 4, 4) or not np.isfinite(model).all():
		raise ValueError("model must be finite with shape (instances, bones, 4, 4)")
	shape = (len(model), len(skeleton.parents), 3, 4)
	result = np.empty(shape, dtype=np.float32) if out is None else _output(out, shape, "skin out")
	np.matmul(model[..., :3, :], skeleton.inverse_bind, out=result)
	return result


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
