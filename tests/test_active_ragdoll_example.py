import numpy as np
import pytest

from graaal.Math import Quaternion
from graaal.PhysicsRagdollControl import ragdoll_motor_targets


def test_motor_action_clips_rotation_magnitude_and_strength_in_reference_frame():
	reference = np.array([[0, 0, 0, 1], Quaternion.from_y_rotation(0.8), [0, 0, 0, 1]])
	actions = [[[0, 1, 0, 1], [1e308, 0, 0, 2], [0, 2, 0, -1]]]
	target, caps = ragdoll_motor_targets(reference, actions, max_offset=[0.1, 0.25, 0.5], max_strength=[10, 45, 30])
	expected = [reference[0], Quaternion.multiply_batch(reference[1], Quaternion.from_x_rotation(0.25)), Quaternion.from_y_rotation(0.5)]
	np.testing.assert_allclose(target[0], expected, atol=1e-7)
	np.testing.assert_array_equal(caps, [[0, 45, 0]])
	with pytest.raises(ValueError, match="max_offset"): ragdoll_motor_targets(reference, actions, max_offset=np.pi * 2, max_strength=1)


def test_assisted_reference_tracks_and_restarts_reproducibly():
	pytest.importorskip("pyjoltphysics")
	from scenes.active_ragdoll_example import run_scene
	assisted = run_scene(frames=360)
	tracking_only = run_scene(frames=360, balance=False)
	restarted = run_scene(frames=360)
	assert assisted["min_pelvis_height_m"] > 0.9
	assert assisted["support_ticks"] > 340
	assert 0 < assisted["peak_balance_force_n"] <= 200 + 1e-6
	assert assisted["final_tracking_error_rad"] < 0.05
	assert tracking_only["min_pelvis_height_m"] < assisted["min_pelvis_height_m"] - 0.3
	np.testing.assert_array_equal(restarted["position"], assisted["position"])
	np.testing.assert_array_equal(restarted["joint_rotation"], assisted["joint_rotation"])
