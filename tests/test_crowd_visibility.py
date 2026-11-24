import numpy as np
import pytest

from graaal.Animation.CrowdVisibility import frustum_pose_mask


def test_frustum_pose_mask_accepts_shared_and_per_pose_radii():
	view_projection = np.eye(4)
	centers = [[0, 0, 0.5], [1.25, 0, 0.5], [1.5, 0, 0.5], [0, 0, -0.25], [0, 0, 1.25]]
	np.testing.assert_array_equal(frustum_pose_mask(view_projection, centers, 0.25), [True, True, False, True, True])
	np.testing.assert_array_equal(frustum_pose_mask(view_projection, centers, [0, 0, 0.5, 0, 0]), [True, False, True, False, False])
	assert frustum_pose_mask(view_projection, np.empty((0, 3)), []).shape == (0,)


def test_frustum_pose_mask_checks_bounds_and_matrix():
	with pytest.raises(ValueError, match="view_projection"):
		frustum_pose_mask(np.eye(3), [[0, 0, 0.5]], 1)
	with pytest.raises(ValueError, match="centers"):
		frustum_pose_mask(np.eye(4), [[0, 0, np.nan]], 1)
	for invalid in (-1, np.inf, [[1]], [1, 2]):
		with pytest.raises(ValueError, match="radii"):
			frustum_pose_mask(np.eye(4), [[0, 0, 0.5]], invalid)
