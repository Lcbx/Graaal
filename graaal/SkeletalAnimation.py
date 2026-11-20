"""CPU skeletal animation resources and batched pose evaluation.

Transforms use column vectors, xyzw quaternions and float32 arrays.
"""

from operator import index

import numpy as np

from .Math import Quaternion


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


def subtree_mask(skeleton, root):
	"""Return a float32 mask selecting a bone and its depth-first subtree."""
	if isinstance(root, (bool, np.bool_)): raise ValueError("root must be a bone index")
	try: root = index(root)
	except TypeError as exc: raise ValueError("root must be a bone index") from exc
	if root < 0 or root >= len(skeleton.parents): raise ValueError("root must be a valid bone index")
	following = np.flatnonzero(skeleton.depth[root + 1:] <= skeleton.depth[root])
	end = root + 1 + int(following[0]) if len(following) else len(skeleton.parents)
	mask = np.zeros(len(skeleton.parents), dtype=np.float32)
	mask[root:end] = 1
	return mask


def _layer_inputs(skeleton, base, overlay, weight, mask):
	if not isinstance(base, (tuple, list)) or len(base) != 3 or not isinstance(overlay, (tuple, list)) or len(overlay) != 3:
		raise ValueError("poses must contain translation, rotation and scale arrays")
	translation = np.asarray(base[0], dtype=np.float32)
	bones = len(skeleton.parents)
	if translation.ndim != 3 or translation.shape[1:] != (bones, 3):
		raise ValueError("translation must have shape (instances, bones, 3)")
	n = len(translation)
	base_t = _array(translation, (n, bones, 3), "base translation")
	base_r = _rotations(base[1], (n, bones, 4), "base rotation")
	base_s = _array(base[2], (n, bones, 3), "base scale")
	overlay_t = _array(overlay[0], (n, bones, 3), "overlay translation")
	overlay_r = _rotations(overlay[1], (n, bones, 4), "overlay rotation")
	overlay_s = _array(overlay[2], (n, bones, 3), "overlay scale")
	weight = np.asarray(weight, dtype=np.float32)
	if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
		raise ValueError("weights must be finite values in [0, 1]")
	if weight.ndim == 0: weight = np.broadcast_to(weight, (n,))
	if weight.shape != (n,): raise ValueError("weight must be scalar or one value per instance")
	if mask is None: mask = np.ones(bones, dtype=np.float32)
	mask = np.asarray(mask, dtype=np.float32)
	if mask.shape != (bones,) or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
		raise ValueError("mask must contain one finite value in [0, 1] per bone")
	w = weight[:, None, None] * mask[None, :, None]
	return (base_t, base_r, base_s), (overlay_t, overlay_r, overlay_s), w


def _blend_local_poses(base, overlay, weight):
	return (base[0] * (1 - weight) + overlay[0] * weight,
		Quaternion.nlerp_batch(base[1], overlay[1], weight),
		base[2] * (1 - weight) + overlay[2] * weight)


def blend_poses(skeleton, base, overlay, weight, *, mask=None):
	"""Blend batched local poses using per-instance weights and a per-bone mask."""
	base, overlay, w = _layer_inputs(skeleton, base, overlay, weight, mask)
	return _blend_local_poses(base, overlay, w)


def additive_pose(skeleton, base, overlay, weight, *, mask=None):
	"""Apply an overlay's local-pose delta from the skeleton reference to a base pose."""
	(base_t, base_r, base_s), (overlay_t, overlay_r, overlay_s), w = _layer_inputs(skeleton, base, overlay, weight, mask)
	reference_inverse = skeleton.rotation.copy()
	reference_inverse[:, :3] *= -1
	delta = Quaternion.multiply_batch(reference_inverse[None], overlay_r)
	identity = np.zeros_like(delta)
	identity[..., 3] = 1
	delta = Quaternion.nlerp_batch(identity, delta, w)
	rotation = Quaternion.multiply_batch(base_r, delta)
	rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
	return base_t + (overlay_t - skeleton.translation) * w, rotation, base_s + (overlay_s - skeleton.scale) * w


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
	Quaternion.nlerp_batch(q0, q1, weight, out=rotation)
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
	return _blend_local_poses(source_pose, target_pose, weights[:, None, None])


def blend_space_1d(positions, clip_ids, values):
	"""Select two clips and their weights for each value on a 1D axis.

	Positions must be strictly increasing. Values outside the axis clamp to
	the nearest endpoint; the returned arrays have shape (instances, 2).
	Playback times are managed separately by the caller.
	"""
	positions = np.asarray(positions, dtype=np.float64)
	clip_ids = np.asarray(clip_ids)
	values = np.asarray(values, dtype=np.float64)
	if positions.ndim != 1 or not len(positions) or not np.isfinite(positions).all() or np.any(positions[1:] <= positions[:-1]):
		raise ValueError("positions must be a nonempty, finite, strictly increasing 1D array")
	if clip_ids.shape != positions.shape or clip_ids.dtype.kind not in "iu" or np.any(clip_ids < 0):
		raise ValueError("clip_ids must contain one nonnegative integer per position")
	if values.ndim != 1 or not np.isfinite(values).all():
		raise ValueError("values must be a finite 1D array")
	if len(positions) == 1:
		return np.broadcast_to(clip_ids, (len(values), 2)).copy(), np.tile(np.array([1, 0], dtype=np.float32), (len(values), 1))
	upper = np.clip(np.searchsorted(positions, values, side="right"), 1, len(positions) - 1)
	lower = upper - 1
	clamped = np.clip(values, positions[lower], positions[upper])
	# Halving before subtraction avoids overflow for widely separated endpoints.
	with np.errstate(over="ignore"): width = positions[upper] - positions[lower]
	wide = np.isinf(width)
	weight = np.empty(len(values), dtype=np.float64)
	weight[wide] = (clamped[wide] * 0.5 - positions[lower[wide]] * 0.5) / (positions[upper[wide]] * 0.5 - positions[lower[wide]] * 0.5)
	weight[~wide] = (clamped[~wide] - positions[lower[~wide]]) / width[~wide]
	weight = weight.astype(np.float32)
	return np.stack((clip_ids[lower], clip_ids[upper]), axis=1), np.stack((1 - weight, weight), axis=1)


def blend_space_2d(x_positions, y_positions, clip_ids, values):
	"""Select two horizontal clip pairs and a vertical blend for a grid.

	The clip grid has shape (y positions, x positions), and values have shape
	(instances, 2). Returns IDs (instances, 2, 2), horizontal pair weights
	(instances, 2), and vertical blend weights (instances,).
	"""
	grid = np.asarray(clip_ids)
	x_positions = np.asarray(x_positions)
	y_positions = np.asarray(y_positions)
	values = np.asarray(values, dtype=np.float64)
	if values.ndim != 2 or values.shape[1:] != (2,) or not np.isfinite(values).all():
		raise ValueError("values must be a finite array of shape (instances, 2)")
	if x_positions.ndim != 1 or y_positions.ndim != 1 or grid.shape != (len(y_positions), len(x_positions)) or grid.dtype.kind not in "iu" or np.any(grid < 0):
		raise ValueError("clip_ids must be a grid of nonnegative integers matching the axes")
	x_indices, x_weights = blend_space_1d(x_positions, np.arange(len(x_positions)), values[:, 0])
	y_indices, y_weights = blend_space_1d(y_positions, np.arange(len(y_positions)), values[:, 1])
	ids = grid[y_indices[:, :, None], x_indices[:, None, :]]
	return ids, x_weights, y_weights[:, 1]


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
	local = np.zeros((n, bones, 4, 4), dtype=np.float32)
	local[..., :3, :3] = Quaternion.matrix_batch(rotation)
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
	Pass normals=None for positions only; the returned normal value is then None.
	"""
	positions = np.asarray(positions, dtype=np.float32)
	normals = None if normals is None else np.asarray(normals, dtype=np.float32)
	joints = np.asarray(joints)
	weights = np.asarray(weights, dtype=np.float32)
	matrices = np.asarray(matrices, dtype=np.float32)
	if positions.ndim != 2 or positions.shape[1:] != (3,) or (normals is not None and normals.shape != positions.shape):
		raise ValueError("positions and normals must have shape (vertices, 3)")
	if joints.ndim != 2 or joints.shape[0] != len(positions) or joints.shape[1] == 0 or joints.dtype.kind not in "iu" or weights.shape != joints.shape:
		raise ValueError("joints and weights must have shape (vertices, influences)")
	if matrices.ndim != 3 or matrices.shape[1:] != (3, 4) or not len(matrices):
		raise ValueError("matrices must have shape (bones, 3, 4)")
	if not all(np.isfinite(value).all() for value in (positions, weights, matrices)) or (normals is not None and not np.isfinite(normals).all()):
		raise ValueError("skin inputs must be finite")
	if np.any(joints < 0) or np.any(joints >= len(matrices)) or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, atol=1e-5):
		raise ValueError("invalid joint indices or skin weights")
	blended = np.sum(matrices[joints] * weights[..., None, None], axis=1)
	points = np.einsum("nij,nj->ni", blended[..., :3], positions) + blended[..., 3]
	if normals is None: return points, None
	try:
		directions = np.linalg.solve(blended[..., :3].transpose(0, 2, 1), normals[..., None])[..., 0]
	except np.linalg.LinAlgError as exc:
		raise ValueError("skinning produced a singular normal transform") from exc
	lengths = np.linalg.norm(directions, axis=1, keepdims=True)
	return points, directions / np.maximum(lengths, 1e-8)
