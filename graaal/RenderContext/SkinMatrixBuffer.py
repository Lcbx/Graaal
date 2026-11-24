"""Fixed-capacity GPU skin-matrix storage, independent of ECS and rendering."""
from __future__ import annotations

from operator import index
from typing import SupportsIndex

from numpy.typing import ArrayLike
import numpy as np

from .core import BufferUsage, CommandContext, GpuBuffer


MATRIX_BYTES = 3 * 4 * np.dtype(np.float32).itemsize
STAGING_USAGE = BufferUsage.COPY_SRC | BufferUsage.COPY_DST


def _uint(value: SupportsIndex, label: str, *, positive: bool=False) -> int:
	if isinstance(value, (bool, np.bool_)): raise ValueError(f"{label} must be an integer")
	try: value = index(value)
	except TypeError as exc: raise ValueError(f"{label} must be an integer") from exc
	if value < int(positive) or value > np.iinfo(np.uint64).max: raise ValueError(f"Invalid {label}")
	return value


class SkinMatrixBuffer:
	"""A stable GPU buffer of row-major 3x4 float32 matrices.

	Each owner receives a contiguous range. Owners can be full generational ECS
	handles or any other unique uint64 IDs. The caller manages their lifetimes and
	ensures recorded work no longer needs a range before releasing or rewriting it.
	"""
	def __init__(self, capacity: int) -> None:
		capacity = _uint(capacity, "capacity", positive=True)
		if capacity > np.iinfo(np.uint32).max: raise ValueError("capacity exceeds uint32 indexing")
		self.capacity = capacity
		self.buffer = GpuBuffer(np.zeros((capacity, 3, 4), dtype=np.float32), BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC)
		self._free = [(0, capacity)]
		self._ranges: dict[int, tuple[int, int]] = {}
		self._staging: GpuBuffer | None = None

	def acquire(self, owner: SupportsIndex, bones: int) -> int:
		"""Allocate a range and return its base matrix index."""
		owner, bones = _uint(owner, "owner"), _uint(bones, "bones", positive=True)
		if owner in self._ranges: raise ValueError("Owner already has a skin range")
		for i, (base, count) in enumerate(self._free):
			if bones > count: continue
			if bones == count: self._free.pop(i)
			else: self._free[i] = (base + bones, count - bones)
			self._ranges[owner] = (base, bones)
			return base
		raise ValueError("Skin matrix buffer has insufficient contiguous capacity")

	def range(self, owner: SupportsIndex) -> tuple[int, int]:
		"""Return (base, bone_count) for an allocated owner."""
		return self._ranges[_uint(owner, "owner")]

	def ranges(self, owners: ArrayLike) -> np.ndarray:
		"""Return (base, bone_count) rows in the requested owner order."""
		ids = np.asarray(owners)
		if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
			raise ValueError("owners must be a one-dimensional array of nonnegative integers")
		return np.asarray([self._ranges[owner] for owner in ids.tolist()], dtype=np.uint32).reshape(-1, 2)

	def release(self, owner: SupportsIndex) -> None:
		"""Release a range for reuse; previously uploaded bytes are not cleared."""
		allocation = self._ranges.pop(_uint(owner, "owner"))
		self._free.append(allocation)
		self._free.sort()
		merged: list[tuple[int, int]] = []
		for base, count in self._free:
			if merged and merged[-1][0] + merged[-1][1] == base:
				merged[-1] = (merged[-1][0], merged[-1][1] + count)
			else: merged.append((base, count))
		self._free = merged

	def _write_batch(self, owners: ArrayLike, matrices: ArrayLike) -> tuple[np.ndarray, np.ndarray, list[int]]:
		owners = np.asarray(owners)
		matrices = np.asarray(matrices, dtype=np.float32)
		if owners.ndim != 1 or owners.dtype.kind not in "iu" or np.any(owners < 0):
			raise ValueError("owners must be a one-dimensional array of nonnegative integers")
		if len(np.unique(owners)) != len(owners): raise ValueError("owner IDs must be distinct")
		if matrices.ndim != 4 or matrices.shape[0] != len(owners) or matrices.shape[2:] != (3, 4) or not np.isfinite(matrices).all():
			raise ValueError("matrices must be finite with shape (owners, bones, 3, 4)")
		try:
			ranges = self.ranges(owners)
		except KeyError as exc:
			raise ValueError(f"No skin range for owner {exc.args[0]}") from exc
		if np.any(ranges[:, 1] != matrices.shape[1]):
			raise ValueError("Pose bone count does not match allocated range")
		bases = ranges[:, 0].astype(np.intp)
		if np.any(bases[1:] < bases[:-1]):
			order = np.argsort(bases)
			bases, matrices = bases[order], matrices[order]
		boundaries = np.r_[0, np.flatnonzero(np.diff(bases) != matrices.shape[1]) + 1, len(bases)].tolist()
		return bases, matrices, boundaries

	def write(self, owners: ArrayLike, matrices: ArrayLike) -> None:
		"""Upload same-skeleton poses in owner order after validating the whole batch."""
		bases, poses, boundaries = self._write_batch(owners, matrices)
		if not len(bases): return
		for start, end in zip(boundaries, boundaries[1:]):
			self.buffer.write_range(poses[start:end], offset=int(bases[start]) * MATRIX_BYTES)

	def record_write(self, commands: CommandContext, owners: ArrayLike, matrices: ArrayLike, *, staging: GpuBuffer | None = None) -> None:
		"""Stage one pose batch and record copies to the owners' palette ranges.

		Use distinct COPY_SRC | COPY_DST staging buffers for each write recorded
		before one submission. By default the palette owns one reusable buffer.
		Submit before reusing a staging buffer or closing it or the palette.
		"""
		bases, poses, boundaries = self._write_batch(owners, matrices)
		if not len(bases): return
		if staging is None:
			if self._staging is None:
				self._staging = GpuBuffer(poses, STAGING_USAGE, upload=False, label="skin palette staging")
			staging = self._staging
		if staging is self.buffer or staging.usage & STAGING_USAGE != STAGING_USAGE:
			raise ValueError("staging must be a separate buffer with COPY_SRC and COPY_DST usage")
		mirror = self.buffer.content
		assert mirror is not None
		staging.write(poses)
		bones = poses.shape[1]
		for start, end in zip(boundaries, boundaries[1:]):
			base = int(bases[start])
			count = (end - start) * bones
			commands.copy_buffer(staging, self.buffer, source_offset=start * bones * MATRIX_BYTES,
				destination_offset=base * MATRIX_BYTES, size=count * MATRIX_BYTES)
			mirror[base:base + count] = poses[start:end].reshape(count, 3, 4)

	def close(self) -> None:
		if self._staging is not None:
			self._staging.close()
			self._staging = None
		self.buffer.close()
		self._ranges.clear()
		self._free.clear()
