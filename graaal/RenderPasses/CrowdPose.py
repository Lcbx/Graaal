"""Evaluate packed crowd clips directly into a GPU skin palette."""
from __future__ import annotations

from operator import index
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ..Animation.BakedClipPool import BakedClipPool
from ..RenderContext import BufferUsage, CommandContext, ComputePipeline, GpuBuffer, RenderContext, Shader
from ..RenderContext.SkinMatrixBuffer import SkinMatrixBuffer


SAMPLE_DTYPE = np.dtype([("clip", "<u4"), ("time", "<f4"), ("weight", "<f4")])
ROW_DTYPE = np.dtype([("base", "<u4"), ("count", "<u4"), ("padding", "<u4", 2), ("samples", SAMPLE_DTYPE, 4)])


class CrowdPosePass:
	"""Record one skeleton's model-space clip blend before its material draws.

	The caller resolves clip clocks and crowd eligibility and owns palette ranges.
	Record once per submitted command batch through this instance. The staging
	buffer is reused on subsequent calls, so submit before staging the next batch.
	"""
	def __init__(self, pool: BakedClipPool, palette: SkinMatrixBuffer, max_instances: int) -> None:
		self.capacity = index(max_instances)
		self.pool, self.palette = pool, palette
		self._bone_groups = (pool.bones + 63) // 64
		limits = RenderContext.device.limits
		workgroups = limits["max-compute-workgroups-per-dimension"]
		if not 1 <= self.capacity <= workgroups or self._bone_groups > workgroups:
			raise ValueError("crowd dispatch exceeds the device workgroup limit")
		sizes = (pool.packed.nbytes, pool.table.nbytes, pool.bones * 64, self.capacity * ROW_DTYPE.itemsize, palette.buffer.binding_size)
		if max(sizes) > limits["max-storage-buffer-binding-size"]:
			raise ValueError("crowd buffers exceed the device storage binding size")
		shader_path = Path(__file__).resolve().parents[1] / "shaders" / "crowd_pose.shader"
		self._shader = Shader(filepath=str(shader_path), label="crowd pose")
		self._pipeline = ComputePipeline(self._shader)
		self._rows = np.zeros(self.capacity, dtype=ROW_DTYPE)
		self._inputs = GpuBuffer(self._rows, BufferUsage.STORAGE | BufferUsage.COPY_DST)
		self._packed = GpuBuffer(pool.packed, BufferUsage.STORAGE)
		self._table = GpuBuffer(pool.table, BufferUsage.STORAGE)
		inverse = np.ascontiguousarray(pool.skeleton.inverse_bind.swapaxes(-1, -2))
		self._inverse = GpuBuffer(inverse, BufferUsage.STORAGE)
		self._bindings = self._shader.bind_group(0, packed=self._packed, clips=self._table, inverse_bind=self._inverse, rows=self._inputs, palette=palette.buffer)

	def record(self, commands: CommandContext, owners: ArrayLike, clip_ids: ArrayLike, times: ArrayLike, weights: ArrayLike) -> None:
		"""Write 1–4 clamped clip samples per owner and record a compute dispatch."""
		owner_ids, ids = np.asarray(owners), np.asarray(clip_ids)
		sample_times, sample_weights = np.asarray(times, dtype=np.float64), np.asarray(weights, dtype=np.float32)
		if owner_ids.ndim != 1 or owner_ids.dtype.kind not in "iu" or len(owner_ids) > self.capacity or len(np.unique(owner_ids)) != len(owner_ids):
			raise ValueError("owners must be unique integer IDs within the crowd capacity")
		if ids.ndim != 2 or not 1 <= ids.shape[1] <= 4 or len(ids) != len(owner_ids) or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(self.pool.table)):
			raise ValueError("clip IDs must be valid (owners, 1–4) sample rows")
		if sample_times.shape != ids.shape or not np.isfinite(sample_times).all(): raise ValueError("times must be finite and match clip IDs")
		if sample_weights.shape != ids.shape or not np.isfinite(sample_weights).all() or np.any(sample_weights < 0) or not np.allclose(sample_weights.sum(axis=1), 1, rtol=0, atol=1e-6):
			raise ValueError("weights must be nonnegative and sum to one per owner")
		if not len(owner_ids): return
		ranges = self.palette.ranges(owner_ids)
		if np.any(ranges[:, 1] != self.pool.bones): raise ValueError("owner palette ranges must match the crowd skeleton")
		staged = self._rows[:len(ids)]
		staged["base"], staged["count"] = ranges[:, 0], ids.shape[1]
		samples = staged["samples"][:, :ids.shape[1]]
		samples["clip"] = ids
		samples["time"] = np.clip(sample_times, 0, self.pool.table["duration"][ids])
		samples["weight"] = sample_weights / sample_weights.sum(axis=1, keepdims=True)
		self._inputs.upload_range(0, len(staged))
		with commands.compute_pass(label="crowd pose") as compute:
			compute.set_pipeline(self._pipeline)
			compute.set_bind_group(self._bindings)
			compute.dispatch(self._bone_groups, len(owner_ids))

	def close(self) -> None:
		"""Release owned resources once recorded work has been submitted."""
		for buffer in (self._inputs, self._packed, self._table, self._inverse): buffer.close()
