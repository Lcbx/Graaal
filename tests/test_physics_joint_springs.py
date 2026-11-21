"""Neutral joint springs stay internal, bounded and independent of native motors."""

import numpy as np
import pytest

from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import joint_spring_torques


def inputs(count=1, bones=2):
	return np.tile([0, 0, 0, 1.0], (count, bones, 1)), np.zeros((count, bones, 3)), np.tile([0, 0, 0, 1.0], (bones, 1))


def test_spring_direction_and_relative_angular_damping():
	turns, angular, neutral = inputs()
	turns[:, 1] = Quaternion.from_z_rotation(0.4)
	angular[:, 0, 1], angular[:, 1, 1] = 1, 2
	torque = joint_spring_torques([-1, 0], turns, angular, neutral, stiffness=10, damping=2, max_torque=100)
	np.testing.assert_allclose(torque, [[[0, 2, 4], [0, -2, -4]]], atol=1e-6)
	assert torque.dtype == np.float64
	assert not np.shares_memory(torque, angular)


def test_branch_reactions_caps_and_per_instance_joint_gains():
	turns, angular, neutral = inputs(2, 3)
	turns[:, 1:] = Quaternion.from_z_rotation(0.5)
	torque = joint_spring_torques([-1, 0, 0], turns, angular, neutral, stiffness=[[0, 10, 20], [0, 1, 2]], damping=0, max_torque=[0, 1, 2])
	np.testing.assert_allclose(torque[..., 2], [[3, -1, -2], [1.5, -0.5, -1]], atol=1e-6)
	np.testing.assert_allclose(torque.sum(axis=1), 0, atol=1e-14)


def test_large_rotation_uses_the_shortest_restoring_direction():
	turns, angular, neutral = inputs()
	turns[:, 1] = Quaternion.from_z_rotation(3.5)
	torque = joint_spring_torques([-1, 0], turns, angular, neutral, stiffness=1, damping=0, max_torque=10)
	assert torque[0, 1, 2] == pytest.approx(2 * np.pi - 3.5, abs=1e-6)


def test_springs_are_covariant_under_world_rotation_and_ignore_quaternion_sign():
	turns, angular, neutral = inputs()
	neutral[1] = Quaternion.from_y_rotation(0.2)
	turns[:, 1] = Quaternion.from_z_rotation(0.4)
	angular[0] = [[1, 0.2, 0], [0, -0.1, 1]]
	expected = joint_spring_torques([-1, 0], turns, angular, neutral, stiffness=3, damping=0.4, max_torque=2)
	world = Quaternion.from_x_rotation(0.7)
	rotated_turns = Quaternion.multiply_batch(world, turns)
	rotated_angular = angular @ world.matrix33.T
	actual = joint_spring_torques([-1, 0], rotated_turns, rotated_angular, neutral, stiffness=3, damping=0.4, max_torque=2)
	np.testing.assert_allclose(actual, expected @ world.matrix33.T, atol=1e-6)
	before = turns.copy()
	negated = joint_spring_torques([-1, 0], -turns * 3, angular, -neutral * 2, stiffness=3, damping=0.4, max_torque=2)
	np.testing.assert_allclose(negated, expected, atol=1e-6)
	np.testing.assert_array_equal(turns, before)


def test_no_root_world_spring_and_zero_torque_cap():
	turns, angular, neutral = inputs(2)
	turns[:] = Quaternion.from_y_rotation(0.8)
	angular[:] = [1, 2, 3]
	np.testing.assert_allclose(joint_spring_torques([-1, 0], turns, angular, neutral, stiffness=4, damping=3, max_torque=2), 0, atol=1e-6)
	turns[:, 1] = Quaternion.from_x_rotation(0.2)
	np.testing.assert_array_equal(joint_spring_torques([-1, 0], turns, angular, neutral, stiffness=4, damping=3, max_torque=0), 0)
	np.testing.assert_array_equal(joint_spring_torques([-1], turns[:, :1], angular[:, :1], neutral[:1], stiffness=4, damping=3, max_torque=2), 0)


@pytest.mark.parametrize("parents", ([], [-1, -1], [-1, 1], [-1, 2], [-1.0, 0.0], [0, 0]))
def test_invalid_spring_hierarchy_rejected(parents):
	turns, angular, neutral = inputs()
	with pytest.raises(ValueError): joint_spring_torques(parents, turns, angular, neutral, stiffness=1, damping=1, max_torque=1)


@pytest.mark.parametrize("bad", ("zero_rotation", "nan_rotation", "bad_shape", "nan_velocity", "bad_neutral", "negative_stiffness", "negative_damping", "infinite_cap", "bad_gain_shape"))
def test_invalid_spring_values_rejected(bad):
	turns, angular, neutral = inputs()
	controls = dict(stiffness=1, damping=1, max_torque=1)
	if bad == "zero_rotation": turns[:, 1] = 0
	elif bad == "nan_rotation": turns[:, 1, 0] = np.nan
	elif bad == "bad_shape": turns = turns[0]
	elif bad == "nan_velocity": angular[:, 1] = np.nan
	elif bad == "bad_neutral": neutral[1] = 0
	elif bad == "negative_stiffness": controls["stiffness"] = -1
	elif bad == "negative_damping": controls["damping"] = -1
	elif bad == "infinite_cap": controls["max_torque"] = np.inf
	else: controls["stiffness"] = np.zeros(3)
	with pytest.raises(ValueError): joint_spring_torques([-1, 0], turns, angular, neutral, **controls)


def native_settings(pj):
	bones = [pj.Bone("root", -1, (0, 0, 0), shape=pj.Shape.sphere(0.5), mass=100, friction=0),
		pj.Bone("left", 0, (-0.8, 0, 0), shape=pj.Shape.sphere(0.15), mass=1, friction=0, twist_axis=(0, 1, 0), swing=(1, 1)),
		pj.Bone("right", 0, (0.8, 0, 0), shape=pj.Shape.sphere(0.15), mass=1, friction=0, twist_axis=(0, 1, 0), swing=(1, 1))]
	return pj.RagdollSettings(bones, self_collision=False, linear_damping=0, angular_damping=0, flags=pj.BodyFlag.NO_SLEEP)


@pytest.mark.parametrize("mode", ("LIMP", "POWERED"))
@pytest.mark.parametrize("threads", (1, 2))
def test_native_neutral_spring_relaxes_with_zero_active_motor_strength(mode, threads):
	pj = pytest.importorskip("pyjoltphysics")
	settings = native_settings(pj)
	with pj.World(pj.WorldSettings(threads=threads, gravity=(0, 0, 0))) as world:
		handle = world.add_ragdoll(settings, mode=pj.RagdollMode.KINEMATIC)
		turns = settings.rest_rotations
		turns[1], turns[2] = Quaternion.from_z_rotation(0.5), Quaternion.from_z_rotation(-0.5)
		handle.set_pose(turns)
		world.run_ticks(2)
		handle.set_pose(settings.rest_rotations)
		handle.set_strength([0, 0, 0])
		handle.set_mode(getattr(pj.RagdollMode, mode))
		world.set_velocities(handle.bones, linear=np.zeros((3, 3)), angular=np.zeros((3, 3)))
		world.flush()
		initial = world.bodies_state(handle.bones)
		for _ in range(240):
			state = world.bodies_state(handle.bones)
			torque = joint_spring_torques(settings.parents, state.orientation[None], state.angular_velocity[None], settings.rest_rotations, stiffness=1, damping=0.1, max_torque=0.5)
			np.testing.assert_allclose(torque.sum(axis=1), 0, atol=1e-14)
			world.add_torques(handle.bones, torque[0])
			world.run_ticks(1)
		final = world.bodies_state(handle.bones)
		relative = Quaternion.multiply_batch(final.orientation[0] * [-1, -1, -1, 1], final.orientation[1:])
		np.testing.assert_allclose(relative, [[0, 0, 0, 1]] * 2, atol=0.003)
		assert np.max(np.abs(initial.orientation[1:, 2])) > 0.2
		assert world.stats()["rejected"] == 0


def test_native_per_bone_strength_weakens_only_the_selected_joint():
	pj = pytest.importorskip("pyjoltphysics")
	settings = native_settings(pj)
	with pj.World(pj.WorldSettings(threads=1, gravity=(0, 0, 0))) as world:
		handle = world.add_ragdoll(settings, mode=pj.RagdollMode.POWERED)
		pose = settings.rest_rotations
		pose[1:] = Quaternion.from_z_rotation(0.5)
		handle.set_pose(pose)
		handle.set_strength([0, 20, 0])
		world.run_ticks(120)
		state = world.bodies_state(handle.bones)
		relative = Quaternion.multiply_batch(state.orientation[0] * [-1, -1, -1, 1], state.orientation[1:])
		angles = 2 * np.arctan2(np.linalg.norm(relative[:, :3], axis=1), np.abs(relative[:, 3]))
		assert angles[0] == pytest.approx(0.5, abs=0.02)
		assert angles[1] < 0.1
		assert world.stats()["rejected"] == 0


@pytest.mark.parametrize("threads", (1, 2))
def test_independent_native_worlds_repeat_spring_trajectories_and_isolate_impacts(threads):
	pj = pytest.importorskip("pyjoltphysics")
	settings = native_settings(pj)
	with pj.World(pj.WorldSettings(threads=threads, gravity=(0, 0, 0))) as a, pj.World(pj.WorldSettings(threads=threads, gravity=(0, 0, 0))) as b:
		handles = [world.add_ragdoll(settings, mode=pj.RagdollMode.LIMP) for world in (a, b)]
		for world, handle in zip((a, b), handles):
			world.run_ticks(1)
			world.add_torques(handle.bones[1:2], [[0, 0, 0.5]])
			world.run_ticks(1)
		for _ in range(30):
			states = []
			for world, handle in zip((a, b), handles):
				state = world.bodies_state(handle.bones)
				torque = joint_spring_torques(settings.parents, state.orientation[None], state.angular_velocity[None], settings.rest_rotations, stiffness=1, damping=0.1, max_torque=0.5)
				world.add_torques(handle.bones, torque[0])
				world.run_ticks(1)
				states.append(world.bodies_state(handle.bones))
			for name in ("position", "orientation", "linear_velocity", "angular_velocity"):
				np.testing.assert_allclose(getattr(states[0], name), getattr(states[1], name), atol=1e-6, rtol=0)
		before = b.bodies_state(handles[1].bones).position.copy()
		handles[0].hit(1, (10, 0, 0))
		a.run_ticks(3)
		np.testing.assert_array_equal(b.bodies_state(handles[1].bones).position, before)
		assert a.tick == b.tick + 3
