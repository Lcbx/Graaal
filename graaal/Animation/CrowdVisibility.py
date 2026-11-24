"""CPU frustum selection for crowd pose batches."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..Math import extract_frustum_planes


def frustum_pose_mask(view_projection: ArrayLike, centers: ArrayLike, radii: ArrayLike) -> NDArray[np.bool_]:
	"""Select pose rows whose world-space spheres may be visible.

	Use the draw view's row-vector ``view @ projection`` and conservative
	animated bounds at the render positions. ``centers`` contains one vec3
	per pose row. ``radii`` is a shared scalar or one radius per row.
	Spheres touching a frustum plane remain selected.
	"""
	matrix = np.asarray(view_projection, dtype=np.float32)
	points = np.asarray(centers, dtype=np.float32)
	radius = np.asarray(radii, dtype=np.float32)
	if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
		raise ValueError("view_projection must be a finite (4, 4) matrix")
	if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
		raise ValueError("centers must contain finite (poses, 3) rows")
	if radius.shape not in ((), (len(points),)) or not np.isfinite(radius).all() or np.any(radius < 0):
		raise ValueError("radii must be a finite nonnegative scalar or one value per pose")
	planes = extract_frustum_planes(matrix)
	if not np.isfinite(planes).all(): raise ValueError("view_projection produced nonfinite frustum planes")
	distances = points @ planes[:, :3].T + planes[:, 3]
	return ~np.any(distances < -radius[..., None], axis=1)
