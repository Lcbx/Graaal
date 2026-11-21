"""Fixed-capacity GPU skin-matrix storage, independent of ECS and rendering."""
from operator import index

import numpy as np

from .core import BufferUsage, GpuBuffer


def _uint(value, label, *, positive=False):
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
	def __init__(self, capacity):
		capacity = _uint(capacity, "capacity", positive=True)
		if capacity > np.iinfo(np.uint32).max: raise ValueError("capacity exceeds uint32 indexing")
		self.capacity = capacity
		self.buffer = GpuBuffer(np.zeros((capacity, 3, 4), dtype=np.float32), BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC)
		self._free = [(0, capacity)]
		self._ranges = {}

	def acquire(self, owner, bones):
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

	def range(self, owner):
		"""Return (base, bone_count) for an allocated owner."""
		return self._ranges[_uint(owner, "owner")]

	def release(self, owner):
		"""Release a range for reuse; previously uploaded bytes are not cleared."""
		allocation = self._ranges.pop(_uint(owner, "owner"))
		self._free.append(allocation)
		self._free.sort()
		merged = []
		for base, count in self._free:
			if merged and merged[-1][0] + merged[-1][1] == base:
				merged[-1] = (merged[-1][0], merged[-1][1] + count)
			else: merged.append((base, count))
		self._free = merged

	def write(self, owners, matrices):
		"""Upload same-skeleton poses in owner order after validating the whole batch."""
		owners = np.asarray(owners)
		matrices = np.asarray(matrices, dtype=np.float32)
		if owners.ndim != 1 or owners.dtype.kind not in "iu" or np.any(owners < 0):
			raise ValueError("owners must be a one-dimensional array of nonnegative integers")
		if len(np.unique(owners)) != len(owners): raise ValueError("owner IDs must be distinct")
		if matrices.ndim != 4 or matrices.shape[0] != len(owners) or matrices.shape[2:] != (3, 4) or not np.isfinite(matrices).all():
			raise ValueError("matrices must be finite with shape (owners, bones, 3, 4)")
		entries = []
		for owner, pose in zip(owners, matrices):
			if int(owner) not in self._ranges: raise ValueError(f"No skin range for owner {int(owner)}")
			base, count = self._ranges[int(owner)]
			if len(pose) != count: raise ValueError("Pose bone count does not match allocated range")
			entries.append((base, count, pose))
		entries.sort(key=lambda entry: entry[0])
		start = 0
		while start < len(entries):
			end = start + 1
			boundary = entries[start][0] + entries[start][1]
			while end < len(entries) and entries[end][0] == boundary:
				boundary += entries[end][1]
				end += 1
			poses = [entry[2] for entry in entries[start:end]]
			data = poses[0] if len(poses) == 1 else np.concatenate(poses)
			self.buffer.write_range(data, offset=entries[start][0] * 48)
			start = end

	def close(self):
		self.buffer.close()
		self._ranges.clear()
		self._free.clear()
