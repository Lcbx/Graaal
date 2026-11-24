"""Distance-based selection for exact and crowd pose evaluators."""
from __future__ import annotations

from operator import index

import numpy as np
from numpy.typing import ArrayLike


class DistancePoseLod:
	"""Remember the preferred evaluator by owner for one view and live batch.

	Enter the GPU tier at or beyond `far` and return to the CPU at or within
	`near`. New owners start with a CPU preference. Omitted owners are forgotten.
	"""
	def __init__(self, near: float, far: float) -> None:
		self.near, self.far = float(near), float(far)
		if not 0 <= self.near < self.far or not np.isfinite(self.far):
			raise ValueError("distances must satisfy 0 <= near < far < infinity")
		self._owners = np.empty(0, dtype=np.uint64)
		self._gpu = np.empty(0, dtype=bool)

	def update(self, owners: ArrayLike, positions: ArrayLike, camera_position: ArrayLike, *, hold: ArrayLike | None = None) -> np.ndarray:
		"""Return the GPU preference mask in owner order.

		Owners are uint64 IDs. An empty batch clears the preference history.
		Positions and the camera use world coordinates. True entries in `hold`
		keep their previous preference, for example during a pose blend.
		Combine the result with graph eligibility before evaluating poses.
		"""
		# Mixed-width Python integers can otherwise be promoted to float64.
		ids = np.asarray(owners) if isinstance(owners, np.ndarray) else np.asarray(owners, dtype=object)
		owner_error = "owners must be one-dimensional nonnegative integer IDs within uint64 range"
		if ids.ndim != 1:
			raise ValueError(owner_error)
		if ids.dtype.kind in "iu":
			if np.any(ids < 0): raise ValueError(owner_error)
			ids = ids.astype(np.uint64, copy=False)
		else:
			if ids.dtype.kind != "O": raise ValueError(owner_error)
			values = ids.tolist()
			if any(isinstance(owner, (bool, np.bool_)) for owner in values): raise ValueError(owner_error)
			try:
				ids = np.asarray([index(owner) for owner in values], dtype=np.uint64)
			except (TypeError, OverflowError) as exc:
				raise ValueError(owner_error) from exc
		points, camera = (np.asarray(value, dtype=np.float64) for value in (positions, camera_position))
		if points.shape != (len(ids), 3) or camera.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(camera).all():
			raise ValueError("positions must be finite (owners, 3) rows and camera_position a finite vec3")
		held = None if hold is None else np.asarray(hold)
		if held is not None and (held.shape != ids.shape or held.dtype.kind != "b"):
			raise ValueError("hold must contain one boolean per owner")
		previous = self._gpu
		changed = not np.array_equal(ids, self._owners)
		if changed:
			if len(np.unique(ids)) != len(ids): raise ValueError("owners must be distinct")
			_, current_rows, previous_rows = np.intersect1d(ids, self._owners, assume_unique=True, return_indices=True)
			previous = np.zeros(len(ids), dtype=bool)
			previous[current_rows] = self._gpu[previous_rows]
		with np.errstate(over="ignore"):
			dx, dy, dz = (points - camera).T
			distance = np.hypot(np.hypot(dx, dy), dz)
		selected = np.where(previous, distance > self.near, distance >= self.far)
		if held is not None: selected = np.where(held, previous, selected)
		if changed: self._owners = ids.copy()
		self._gpu = selected.copy()
		return selected
