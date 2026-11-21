"""Progressive swing/twist limit forces use authored frames and internal reactions."""

from types import SimpleNamespace

import numpy as np
import pytest

from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import joint_limit_torques, ragdoll_limit_parameters


def inputs(count=1):
	turns = np.tile([0, 0, 0, 1.0], (count, 2, 1))
	angular = np.zeros((count, 2, 3))
	frames = np.tile([0, 0, 0, 1.0], (2, 1))
	parameters = dict(parents=[-1, 0], parent_frames=frames, child_frames=frames, twist=(-0.6, 0.6), swing=(np.pi, np.pi))
	controls = dict(margin=0.2, stiffness=4, damping=1, max_torque=10)
	return turns, angular, parameters, controls


def test_comfortable_range_has_no_spring_or_damping():
	turns, angular, parameters, controls = inputs()
	turns[:, 1] = Quaternion.from_x_rotation(0.2)
	angular[:, 1] = [20, -30, 40]
	np.testing.assert_allclose(joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls), 0, atol=1e-12)


def test_twist_resistance_grows_quadratically_through_soft_margin():
	turns, angular, parameters, controls = inputs(3)
	for row, angle in enumerate((0.45, 0.5, 0.55)): turns[row, 1] = Quaternion.from_x_rotation(angle)
	torque = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	np.testing.assert_allclose(torque[:, 1, 0], [-0.05, -0.2, -0.45], atol=1e-6)
	np.testing.assert_allclose(torque.sum(axis=1), 0, atol=1e-14)
	assert torque.dtype == np.float64
	assert not np.shares_memory(torque, angular)


def test_asymmetric_twist_bounds_restore_toward_the_interior():
	turns, angular, parameters, controls = inputs(2)
	parameters["twist"] = (-0.4, 0.8)
	turns[0, 1], turns[1, 1] = Quaternion.from_x_rotation(-0.3), Quaternion.from_x_rotation(0.7)
	torque = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	np.testing.assert_allclose(torque[:, 1, 0], [0.2, -0.2], atol=1e-6)


def test_damping_opposes_only_relative_outward_motion():
	turns, angular, parameters, controls = inputs(3)
	turns[:, 1] = Quaternion.from_x_rotation(0.5)
	angular[:, 0, 0] = 1
	angular[:, 1, 0] = [1, 3, -1]
	torque = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	np.testing.assert_allclose(torque[:, 1, 0], [-0.2, -1.2, -0.2], atol=1e-6)


def test_swing_uses_an_elliptic_quaternion_cone():
	turns, angular, parameters, controls = inputs()
	parameters.update(twist=(-np.pi, np.pi), swing=(0.8, 0.8))
	part = np.sin(0.5 / 2)
	turns[:, 1] = [0, part, part, np.sqrt(1 - 2 * part**2)]
	torque = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	assert torque[0, 1, 1] < -0.01 and torque[0, 1, 2] < -0.01
	np.testing.assert_allclose(torque[0, 1, 0], 0, atol=1e-7)
	np.testing.assert_allclose(torque.sum(axis=1), 0, atol=1e-14)


def test_anisotropic_swing_axes_use_the_authored_half_angles():
	turns, angular, parameters, controls = inputs(3)
	parameters.update(twist=(-np.pi, np.pi), swing=(0.8, 0.4))
	controls["margin"] = 0.1
	turns[0, 1] = Quaternion.from_y_rotation(0.6)
	turns[1, 1] = Quaternion.from_y_rotation(0.75)
	turns[2, 1] = Quaternion.from_z_rotation(0.35)
	torque = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	np.testing.assert_allclose(torque[0], 0, atol=1e-12)
	np.testing.assert_allclose(torque[1, 1], [0, -0.1, 0], atol=1e-6)
	np.testing.assert_allclose(torque[2, 1], [0, 0, -0.1], atol=1e-6)


def test_free_axes_have_no_artificial_limit_at_the_quaternion_wrap():
	turns, angular, parameters, controls = inputs(2)
	parameters.update(twist=(-np.pi, np.pi), swing=(np.pi, np.pi))
	turns[0, 1], turns[1, 1] = Quaternion.from_x_rotation(3.13), Quaternion.from_y_rotation(3.13)
	angular[:] = 10
	np.testing.assert_allclose(joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls), 0, atol=1e-12)


@pytest.mark.parametrize("swing", ((0, 0), (0, 0.6), (0.6, 0)))
def test_locked_swing_axes_and_antiparallel_pose_remain_finite_and_bounded(swing):
	turns, angular, parameters, controls = inputs()
	parameters["swing"] = swing
	controls["max_torque"] = 0.3
	turns[:, 1] = [0, 1, 0, 0]
	angular[:, 1] = [100, 100, 100]
	torque = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	assert np.isfinite(torque).all()
	np.testing.assert_allclose(np.linalg.norm(torque[:, 1], axis=-1), 0.3, atol=1e-12)
	np.testing.assert_allclose(torque.sum(axis=1), 0, atol=1e-14)


def test_limit_torques_preserve_frames_quaternion_signs_and_world_rotation():
	turns, angular, parameters, controls = inputs()
	parameters["parent_frames"] = np.tile(Quaternion.from_z_rotation(0.4), (2, 1))
	parameters["child_frames"] = np.tile(Quaternion.from_y_rotation(0.2), (2, 1))
	turns[:, 1] = Quaternion.from_x_rotation(0.9)
	angular[0, 1] = [0.3, -0.1, 0.4]
	expected = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	assert np.linalg.norm(expected) > 0.01
	world = Quaternion.from_y_rotation(0.8)
	actual = joint_limit_torques(orientations=Quaternion.multiply_batch(world, -turns * 3), angular_velocity=angular @ world.matrix33.T, **parameters, **controls)
	np.testing.assert_allclose(actual, expected @ world.matrix33.T, atol=2e-6)


@pytest.mark.parametrize("bad", ("twist_order", "twist_range", "swing_negative", "swing_nan", "margin_zero", "margin_negative", "stiffness_negative", "cap_infinite", "frames_zero", "velocity_shape"))
def test_invalid_soft_limit_parameters_rejected(bad):
	turns, angular, parameters, controls = inputs()
	if bad == "twist_order": parameters["twist"] = (0.5, -0.5)
	elif bad == "twist_range": parameters["twist"] = (-4, 4)
	elif bad == "swing_negative": parameters["swing"] = (-0.1, 0.5)
	elif bad == "swing_nan": parameters["swing"] = (np.nan, 0.5)
	elif bad == "margin_zero": controls["margin"] = 0
	elif bad == "margin_negative": controls["margin"] = -0.1
	elif bad == "stiffness_negative": controls["stiffness"] = -1
	elif bad == "cap_infinite": controls["max_torque"] = np.inf
	elif bad == "frames_zero": parameters["child_frames"] = np.zeros((2, 4))
	else: angular = angular[0]
	with pytest.raises(ValueError): joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)


def test_implicit_authored_axes_are_prepared_without_loading_a_backend():
	rotations = np.array([[0, 0, 0, 1], Quaternion.from_z_rotation(0.4), Quaternion.from_y_rotation(0.2)])
	settings = SimpleNamespace(parents=np.array([-1, 0, 1]), rest_rotations=rotations, rest_translations=np.array([[0, 0, 0], [0, 0, 0], [0.8, 0, 0]]),
		bones=[SimpleNamespace(twist_axis=None, plane_axis=None, bend=0, twist=(-0.3, 0.3), swing=(0.6, 0.6)) for _ in range(3)])
	parameters = ragdoll_limit_parameters(settings)
	axes = Quaternion.matrix_batch(parameters["child_frames"])[..., 0]
	np.testing.assert_allclose(axes, [[0, 1, 0], [1, 0, 0], Quaternion.from_y_rotation(0.2).matrix33.T @ [1, 0, 0]], atol=1e-6)
	settings.rest_rotations[:] = 0
	assert np.isfinite(parameters["parent_frames"]).all()


def test_per_instance_caps_and_empty_or_root_only_batches():
	turns, angular, parameters, controls = inputs(2)
	turns[:, 1] = Quaternion.from_x_rotation(1)
	controls["max_torque"] = [[0, 0.2], [0, 0.4]]
	torque = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	np.testing.assert_allclose(torque[:, 1, 0], [-0.2, -0.4], atol=1e-6)
	controls["max_torque"] = 1
	empty = joint_limit_torques(orientations=turns[:0], angular_velocity=angular[:0], **parameters, **controls)
	assert empty.shape == (0, 2, 3)
	parameters["parents"] = [-1]
	parameters["parent_frames"], parameters["child_frames"] = parameters["parent_frames"][:1], parameters["child_frames"][:1]
	np.testing.assert_array_equal(joint_limit_torques(orientations=turns[:, :1], angular_velocity=angular[:, :1], **parameters, **controls), 0)


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


def native_settings(pj, *, bend=0, reference=(0, 0, 0, 1)):
	bones = [pj.Bone("root", -1, (0, 0, 0), shape=pj.Shape.sphere(0.5), mass=100, friction=0),
		pj.Bone("child", 0, (0.8, 0, 0), rotation=reference, shape=pj.Shape.sphere(0.15), mass=1, friction=0, twist_axis=(0, 1, 0), plane_axis=(1, 0, 0), twist=(-0.6, 0.6), swing=(0.6, 0.4), bend=bend)]
	return pj.RagdollSettings(bones, self_collision=False, linear_damping=0, angular_damping=0, flags=pj.BodyFlag.NO_SLEEP)


def test_authored_limit_parameters_copy_native_frames_bend_and_axes(pj):
	settings = native_settings(pj, bend=0.3, reference=Quaternion.from_z_rotation(0.4))
	parameters = ragdoll_limit_parameters(settings)
	for value in parameters.values(): assert not value.flags.writeable
	settings.bones[1].twist = (-0.2, 0.2)
	np.testing.assert_array_equal(parameters["twist"][1], [-0.6, 0.6])
	turns, angular, _, controls = inputs()
	controls["margin"] = 0.1
	# The cone is centred on +0.3 around the authored plane axis in the rest frame.
	turns[:, 1] = Quaternion.multiply_batch(settings.rest_rotations[1], Quaternion.from_x_rotation(0.3))
	np.testing.assert_allclose(joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls), 0, atol=1e-12)
	turns[:, 1] = Quaternion.multiply_batch(settings.rest_rotations[1], Quaternion.from_x_rotation(0.85))
	torque = joint_limit_torques(orientations=turns, angular_velocity=angular, **parameters, **controls)
	np.testing.assert_allclose(torque[0, 1], Quaternion.from_z_rotation(0.4).matrix33 @ [-0.1, 0, 0], atol=1e-6)


@pytest.mark.parametrize("field,value", (("twist_axis", (0, 0, 0)), ("plane_axis", (0, 1, 0)), ("bend", np.nan)))
def test_bad_authored_limit_frames_are_rejected(pj, field, value):
	settings = native_settings(pj)
	setattr(settings.bones[1], field, value)
	with pytest.raises(ValueError): ragdoll_limit_parameters(settings)


@pytest.mark.parametrize("mode", ("LIMP", "POWERED"))
@pytest.mark.parametrize("threads", (1, 2))
@pytest.mark.parametrize("kind", ("twist", "swing", "bend"))
def test_python_soft_limits_resist_pressure_before_jolt_hard_limits(pj, mode, threads, kind):
	settings = native_settings(pj, bend=0.3 if kind == "bend" else 0, reference=Quaternion.from_z_rotation(0.3))
	parameters = ragdoll_limit_parameters(settings)
	axis_index = 1 if kind == "twist" else 0
	axis_local = np.eye(3)[axis_index]
	upper = 0.9 if kind == "bend" else 0.6
	start = upper - 0.15
	def angle_of(state):
		relative = Quaternion.multiply_batch(settings.rest_rotations[1] * [-1, -1, -1, 1], Quaternion.multiply_batch(state.orientation[0] * [-1, -1, -1, 1], state.orientation[1]))
		return 2 * np.arctan2(relative[axis_index], relative[3])
	with pj.World(pj.WorldSettings(threads=threads, gravity=(0, 0, 0))) as world:
		handle = world.add_ragdoll(settings, mode=pj.RagdollMode.KINEMATIC)
		pose = settings.rest_rotations
		pose[1] = Quaternion.multiply_batch(settings.rest_rotations[1], Quaternion.from_axis_rotation(axis_local, start))
		handle.set_pose(pose)
		world.run_ticks(2)
		handle.set_strength([0, 0])
		handle.set_mode(getattr(pj.RagdollMode, mode))
		world.set_velocities(handle.bones, linear=np.zeros((2, 3)), angular=np.zeros((2, 3)))
		world.flush()
		angles = []
		for _ in range(360):
			state = world.bodies_state(handle.bones)
			torque = joint_limit_torques(orientations=state.orientation[None], angular_velocity=state.angular_velocity[None], margin=0.2, stiffness=4, damping=0.1, max_torque=0.5, **parameters)
			np.testing.assert_allclose(torque.sum(axis=1), 0, atol=1e-14)
			axis = Quaternion.matrix_batch(Quaternion.multiply_batch(state.orientation[0], settings.rest_rotations[1])) @ axis_local
			world.add_torques(handle.bones, torque[0] + np.array([-0.1 * axis, 0.1 * axis]))
			world.run_ticks(1)
			state = world.bodies_state(handle.bones)
			angles.append(angle_of(state))
		assert np.isfinite(angles).all()
		assert start < np.mean(angles[-120:]) < upper - 0.1
		assert np.ptp(angles[-120:]) < 0.02
		assert np.max(angles) < upper - 0.05
		# Pressure beyond the soft torque cap still meets Jolt's hard backstop.
		for _ in range(120):
			state = world.bodies_state(handle.bones)
			torque = joint_limit_torques(orientations=state.orientation[None], angular_velocity=state.angular_velocity[None], margin=0.2, stiffness=4, damping=0.1, max_torque=0.5, **parameters)
			axis = Quaternion.matrix_batch(Quaternion.multiply_batch(state.orientation[0], settings.rest_rotations[1])) @ axis_local
			world.add_torques(handle.bones, torque[0] + np.array([-0.8 * axis, 0.8 * axis]))
			world.run_ticks(1)
		assert angle_of(world.bodies_state(handle.bones)) == pytest.approx(upper, abs=0.02)
		assert world.stats()["rejected"] == 0
