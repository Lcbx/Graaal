"""Packed model-space clips and a CPU reference for crowd pose evaluation."""
from __future__ import annotations

from typing import Self, Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..Math import Quaternion
from .SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip, sample_clip_pairs


if TYPE_CHECKING:
	from .AnimationGraph import CompiledAnimationGraph


CLIP_TABLE_DTYPE = np.dtype([("base", "<u4"), ("frames", "<u4"), ("fps", "<f4"), ("duration", "<f4")])


class BakedClipPool:
	"""Frame-major packed rotations, translations and uniform scales for one rig.
	
	`packed` contains four uint32 pairs of half floats per bone/frame, ordered
	qx, qy, qz, qw, x, y, z, scale. `table` stores base offsets in bone/frames.
	Both arrays own immutable storage. The last frame is the exact clip endpoint
	before quantization. Times supplied to the reference evaluator are clamped.
	"""
	def __init__(self, skeleton: Skeleton, clips: Sequence[Clip], *, fps: float = 30) -> None:
		if not len(clips) or not np.isfinite(fps) or fps <= 0: raise ValueError("baking requires clips and a finite positive sample rate")
		fps = float(fps)
		self.skeleton, self.clips = skeleton, tuple(clips)
		self.bones = len(skeleton.parents)
		table = np.empty(len(clips), dtype=CLIP_TABLE_DTYPE)
		records = []
		base = 0
		for i, clip in enumerate(clips):
			if np.any(clip.scale <= 0) or not np.allclose(clip.scale, clip.scale[..., :1], rtol=1e-6, atol=0):
				raise ValueError(f"clip {i}: baking requires positive uniform local scales")
			duration = clip.duration
			source_intervals = len(clip.translation) - 1
			requested = source_intervals * (fps / clip.fps) if source_intervals else 0
			if not np.isfinite(requested) or requested > np.iinfo(np.uint32).max:
				raise ValueError("baked clip pool exceeds uint32 indexing")
			intervals = max(1, int(np.ceil(requested))) if source_intervals else 0
			count = (intervals + 1) * self.bones
			if base + count > np.iinfo(np.uint32).max: raise ValueError("baked clip pool exceeds uint32 indexing")
			same_grid = intervals == source_intervals
			rate = (clip.fps if same_grid else intervals / duration) if intervals else 0
			if max(duration, rate) > np.finfo(np.float32).max: raise ValueError("baked clip clock exceeds float32 range")
			table[i] = base, intervals + 1, rate, duration
			if same_grid:
				pose = clip.translation, clip.rotation, clip.scale
			else:
				pose = sample_clip(clip, np.linspace(0, duration, intervals + 1), loop=False)
			model = model_matrices(skeleton, *pose)
			linear = model[..., :3, :3].astype(np.float64)
			scale = np.linalg.norm(linear[..., :, 0], axis=-1, keepdims=True)
			if not np.isfinite(model).all() or np.any(scale < np.finfo(np.float16).smallest_subnormal):
				raise ValueError(f"clip {i}: baked model transform exceeds half-float range")
			rotation = Quaternion.from_matrices_batch((linear / scale[..., None]).reshape(-1, 3, 3)).reshape(model.shape[:2] + (4,))
			values = np.concatenate((rotation, model[..., :3, 3], scale), axis=-1)
			if np.any(np.abs(values) > np.finfo(np.float16).max):
				raise ValueError(f"clip {i}: baked model transform exceeds half-float range")
			records.append(values.reshape(count, 8).astype("<f2"))
			base += count
		storage = np.concatenate(records)
		storage.flags.writeable = False
		packed = storage.view("<u4")
		packed.flags.writeable = False
		table.flags.writeable = False
		self.packed, self.table = packed, table

	@classmethod
	def from_graph(cls, graph: CompiledAnimationGraph, *, fps: float = 30) -> Self:
		"""Bake the graph's clips in their compiled ID order."""
		return cls(graph.plan.skeleton, tuple(graph.clips[i] for i in range(len(graph.clips))), fps=fps)

	def model(self, clip_ids: ArrayLike, times: ArrayLike, weights: ArrayLike) -> np.ndarray:
		"""Evaluate 1–4 clamped clip samples per instance into approximate models."""
		ids, sample_times, sample_weights = np.asarray(clip_ids), np.asarray(times, dtype=np.float64), np.asarray(weights, dtype=np.float32)
		if ids.ndim != 2 or not 1 <= ids.shape[1] <= 4 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(self.table)):
			raise ValueError("clip IDs must be valid (instances, 1–4) sample rows")
		if sample_times.shape != ids.shape or not np.isfinite(sample_times).all(): raise ValueError("times must be finite and match clip IDs")
		if sample_weights.shape != ids.shape or not np.isfinite(sample_weights).all() or np.any(sample_weights < 0) or not np.allclose(sample_weights.sum(axis=1), 1, rtol=0, atol=1e-6):
			raise ValueError("weights must be nonnegative and sum to one per instance")
		meta = self.table[ids]
		last = meta["frames"] - 1
		frame = np.clip(np.clip(sample_times, 0, meta["duration"]) * meta["fps"], 0, last)
		frame = np.where(sample_times >= meta["duration"], last, frame)
		left = frame.astype(np.int64)
		right = np.minimum(left + 1, last)
		bones = np.arange(self.bones)
		data = self.packed.view("<f2")
		a, b = (data[(meta["base"] + index * self.bones)[..., None] + bones].astype(np.float32) for index in (left, right))
		alpha = (frame - left)[..., None, None].astype(np.float32)
		turns = Quaternion.nlerp_batch(a[..., :4], b[..., :4], alpha)
		normalized = (sample_weights / sample_weights.sum(axis=1, keepdims=True))[..., None, None]
		dominant = np.argmax(normalized[:, :, 0, 0], axis=1)
		anchor = turns[np.arange(len(dominant)), dominant][:, None]
		aligned = np.where(np.sum(turns * anchor, axis=-1, keepdims=True) < 0, -turns, turns)
		rotation = np.sum(normalized * aligned, axis=1)
		rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
		translation_scale = np.sum(normalized * (a[..., 4:] + (b[..., 4:] - a[..., 4:]) * alpha), axis=1)
		result = np.zeros((len(ids), self.bones, 4, 4), dtype=np.float32)
		result[..., :3, :3] = Quaternion.matrix_batch(rotation) * translation_scale[..., None, 3:]
		result[..., :3, 3] = translation_scale[..., :3]
		result[..., 3, 3] = 1
		return result

	def crossfade_position_error(self, clip_ids: ArrayLike, times: ArrayLike, weights: ArrayLike) -> np.ndarray:
		"""Per-row max bone error in rig units against an exact local clip pair."""
		local = sample_clip_pairs(self.skeleton, dict(enumerate(self.clips)), clip_ids, times, weights)
		exact = model_matrices(self.skeleton, *local)[..., :3, 3]
		approx = self.model(clip_ids, times, weights)[..., :3, 3]
		return np.max(np.linalg.norm(approx - exact, axis=-1), axis=1)
