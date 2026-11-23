from dataclasses import replace

import numpy as np
import pytest

from scenes.benchmarks.benchmark_ragdoll_push_recovery import PUSH_TICK, run_trial
from scenes.physics_geometry import collider_bounds
from scenes.ragdoll_control import STANDING_MOTOR_CAPS
from tools.ragdoll_project import default_project
from tools.ragdoll_standing import CONTROL_HZ, MAX_OFFSET, MAX_RESIDUAL_TORQUE, StandingTask, _support_margin, evaluate


@pytest.fixture
def project():
	pytest.importorskip("pyjoltphysics")
	return default_project()


def test_standing_reset_is_seeded_and_motor_actions_leave_pelvis_free(project):
	project.settings.root_force = 500
	project.settings.root_torque = 180
	task = StandingTask(project, episode_steps=1, initial_angular_speed=0.1)
	try:
		initial, _ = task.reset(seed=12)
		assert task.settings.root_force == task.settings.root_torque == 0
		assert project.settings.root_force == 500 and project.settings.root_torque == 180
		action = np.zeros(task.action_shape, dtype=np.float32)
		action[0], action[3] = 1, 1
		observation, reward, fallen, timed_out, info = task.step(action)
		assert initial.shape == observation.shape and np.isfinite(observation).all() and np.isfinite(reward)
		assert not fallen and timed_out and info["success"]
		assert info["elapsed_seconds"] == pytest.approx(1 / 60)
		np.testing.assert_array_equal(task.state.previous_action[0, 0], [0, 0, 0, 0])
		np.testing.assert_allclose(task.state.previous_action[0, 1], [MAX_OFFSET, 0, 0, 1])
		with pytest.raises(RuntimeError, match="reset"):
			task.step(action)
		restarted, _ = task.reset(seed=12)
		np.testing.assert_array_equal(restarted, initial)
		np.testing.assert_array_equal(task.step(action)[0], observation)
		other, _ = task.reset(seed=23)
		assert not np.array_equal(other, initial)
	finally:
		task.close()


def test_standing_reports_a_fall_even_with_foot_contact(project):
	task = StandingTask(project)
	try:
		action = np.zeros(task.action_shape, dtype=np.float32)
		action[3::4] = -1
		report = evaluate(task, lambda _: action, seed=0, episodes=1)
		assert report["success"] == 0
		assert 0 < report["elapsed_seconds"] < 3
		assert task.state.contacts[0, task.feet].any()
	finally:
		task.close()


def test_resetting_one_standing_world_preserves_the_other_episode(project):
	first = StandingTask(project, episode_steps=2, initial_angular_speed=0.1)
	second = StandingTask(project, episode_steps=2, initial_angular_speed=0.1)
	try:
		first.reset(seed=12)
		second.reset(seed=12)
		action = np.zeros(first.action_shape, dtype=np.float32)
		np.testing.assert_array_equal(first.step(action)[0], second.step(action)[0])
		want, *_ = first.step(action)
		first.reset(seed=99)
		assert first.world.tick == 0 and second.world.tick == 2
		actual, _, fallen, timed_out, info = second.step(action)
		np.testing.assert_array_equal(actual, want)
		assert not fallen and timed_out and info["elapsed_seconds"] == pytest.approx(2 / 60)
	finally:
		first.close()
		second.close()


@pytest.mark.parametrize("vary_start", [False, True])
def test_tuned_stance_and_compact_action_match_fixed_pose_motors(project, vary_start):
	options = dict(tuned_stance=True, vary_start=vary_start, initial_angular_speed=0.1 if vary_start else 0, episode_steps=1)
	compact = StandingTask(project, compact=True, **options)
	full = StandingTask(project, **options)
	try:
		actual, _ = compact.reset(seed=0)
		expected, _ = full.reset(seed=0)
		np.testing.assert_array_equal(actual[:-len(full.action)], expected[:-len(full.action)])
		np.testing.assert_array_equal(full.neutral, project.neutral)
		lower, upper = collider_bounds(full.settings.bones, full.state.position[0], full.state.orientation[0])
		assert lower[full.feet, 1].min() == pytest.approx(0.001, abs=1e-6)
		if not vary_start:
			np.testing.assert_allclose(lower[full.feet, 1], 0.001, atol=1e-6)
		center = full.state.center_of_mass[0, [0, 2]]
		assert np.all(center > lower[full.feet].min(axis=0)[[0, 2]])
		assert np.all(center < upper[full.feet].max(axis=0)[[0, 2]])
		assert 0 < compact.action_shape[0] < 3 * (len(project.settings) - 1)
		assert np.all(compact.action_bones > 0)
		action = np.zeros(compact.action_shape, dtype=np.float32)
		action[0] = 1
		mapped = np.zeros(full.action_shape, dtype=np.float32).reshape(-1, 4)
		mapped[:, 3] = 1
		mapped[compact.action_bones[0] - 1, :3] = compact.action_axes[0]
		np.testing.assert_array_equal(compact.step(action)[0], full.step(mapped.ravel())[0])
	finally:
		compact.close()
		full.close()


def test_support_margin_handles_single_foot_and_bridge_between_feet():
	left = np.array([[-3, -1], [-3, 1], [-1, -1], [-1, 1]])
	right = left + [4, 0]
	assert _support_margin(left, [-2, 0]) == pytest.approx(1)
	assert _support_margin(left, [0, 0]) == pytest.approx(-1)
	assert _support_margin(np.concatenate((left, right)), [0, 0]) == pytest.approx(1)
	assert _support_margin(left, [0, 2]) == pytest.approx(-np.sqrt(2))
	assert _support_margin(np.array([[0, 0], [0, 0], [4, 0], [4, 0]]), [2, 1]) == pytest.approx(-1)


def test_support_area_observation_masks_airborne_spawn_and_reports_contact(project):
	task = StandingTask(project, tuned_stance=True, compact=True, support_area=True)
	try:
		features, _ = task.reset(seed=0)
		assert task.support_margin is None and features[-1] == 0
		for _ in range(10):
			features, _, _, _, info = task.step(np.zeros(task.action_shape))
			if task.support_margin is not None:
				break
		assert task.support_margin is not None and info["minimum_support_margin_m"] is not None
		assert features[-1] == pytest.approx(task.support_margin / task.support_scale)
	finally:
		task.close()


def test_start_tilt_can_change_without_changing_seeded_spin(project):
	options = dict(tuned_stance=True, vary_start=True, initial_angular_speed=0.1)
	flat = StandingTask(project, max_start_tilt=0, **options)
	tilted = StandingTask(project, max_start_tilt=0.015, **options)
	try:
		flat.reset(seed=12)
		tilted.reset(seed=12)
		np.testing.assert_array_equal(flat.state.angular_velocity, tilted.state.angular_velocity)
		np.testing.assert_allclose(flat.state.orientation[0], flat.orientations, atol=1e-7)
		assert not np.array_equal(flat.state.orientation, tilted.state.orientation)
		lower, _ = collider_bounds(tilted.settings.bones, tilted.state.position[0], tilted.state.orientation[0])
		assert lower[:, 1].min() == pytest.approx(0.001, abs=1e-6)
	finally:
		flat.close()
		tilted.close()


def test_ankle_balance_uses_bounded_paired_torques_and_stands(project):
	task = StandingTask(project, episode_steps=5 * CONTROL_HZ, tuned_stance=True, vary_start=True,
		initial_angular_speed=0.1, ankle_balance=True, support_area=True)
	action = np.tile([0, 0, 0, 1], len(task.settings) - 1)
	shins = np.asarray(task.parents)[task.feet]
	caps = np.asarray(STANDING_MOTOR_CAPS)[task.feet]
	try:
		task.reset(seed=41001)
		np.testing.assert_array_equal(task.balance_torques(task.state), np.zeros((len(task.settings), 3)))
		for _ in range(30):
			task.step(action)
		torque = task.balance_torques(task.state)
		np.testing.assert_array_equal(np.flatnonzero(np.linalg.norm(torque, axis=1)), np.sort(np.r_[shins, task.feet]))
		np.testing.assert_array_equal(torque[task.feet], -torque[shins])
		assert np.all(np.linalg.norm(torque[task.feet], axis=1) <= caps + 1e-8)
		np.testing.assert_array_equal(task.state.previous_action[0, np.r_[0, task.feet]], np.zeros((3, 4)))
		while not task.done:
			_, _, _, _, info = task.step(action)
		assert info["settled_success"] and info["first_support_exit_seconds"] is None
	finally:
		task.close()


def test_zero_residual_repeats_classical_ankle_balance(project):
	options = dict(episode_steps=30, tuned_stance=True, vary_start=True, initial_angular_speed=0.1, support_area=True)
	baseline = StandingTask(project, ankle_balance=True, **options)
	learned = StandingTask(project, residual=True, **options)
	fixed = np.tile([0, 0, 0, 1], len(project.settings) - 1)
	try:
		learned.reset(seed=23)
		baseline.reset(seed=23)
		assert learned.action_shape == (4,)
		while not baseline.done:
			_, base_reward, base_fallen, base_completed, base_info = baseline.step(fixed)
			_, residual_reward, residual_fallen, residual_completed, residual_info = learned.step(np.zeros(learned.action_shape))
			np.testing.assert_array_equal(learned.state.position, baseline.state.position)
			np.testing.assert_array_equal(learned.state.orientation, baseline.state.orientation)
			assert residual_info.pop("mean_residual_command_nm") == 0
			assert (residual_reward, residual_fallen, residual_completed, residual_info) == (base_reward, base_fallen, base_completed, base_info)
	finally:
		baseline.close()
		learned.close()


def test_residual_commands_are_clipped_and_only_act_on_supporting_ankles(project):
	task = StandingTask(project, tuned_stance=True, residual=True)
	try:
		task.reset(seed=0)
		command = np.array([2, -2, -2, 2], dtype=np.float32)
		for _ in range(30):
			observation, _, _, _, info = task.step(command)
		np.testing.assert_array_equal(command, [2, -2, -2, 2])
		np.testing.assert_allclose(observation[-4:], np.sign(command) / np.sqrt(2))
		assert info["mean_residual_command_nm"] == pytest.approx(MAX_RESIDUAL_TORQUE)
		contacts = np.zeros_like(task.state.contacts)
		foot, raised = task.feet
		shin, raised_shin = np.asarray(task.parents)[task.feet]
		contacts[0, foot] = True
		state = replace(task.state, contacts=contacts)
		correction = observation[-4:].reshape(2, 2) * MAX_RESIDUAL_TORQUE
		torque = task.balance_torques(state, correction)
		np.testing.assert_array_equal(torque[[0, raised, raised_shin]], np.zeros((3, 3)))
		np.testing.assert_array_equal(torque[foot], -torque[shin])
		assert np.linalg.norm(torque[foot]) <= STANDING_MOTOR_CAPS[foot]
		base = task.balance_torques(state)
		assert 0 < np.linalg.norm(torque[foot] - base[foot]) <= MAX_RESIDUAL_TORQUE + 1e-6
	finally:
		task.close()


def test_push_waits_for_settled_stance_and_restarts_the_hold(project, monkeypatch):
	import pyjoltphysics as pj

	queue = pj.World.add_impulses
	queued = []
	def add_impulses(world, uids, impulses):
		queued.append((world.tick, int(uids[0]), impulses))
		return queue(world, uids, impulses)

	monkeypatch.setattr(pj.World, "add_impulses", add_impulses)
	task = StandingTask(project, episode_steps=7 * CONTROL_HZ, tuned_stance=True, ankle_balance=True)
	action = np.tile([0, 0, 0, 1], len(task.settings) - 1)
	impulse = [0.5, 0, 0]
	try:
		result = run_trial(task, lambda _: action, seed=0, bone="chest", impulse=impulse)
		assert queued == [(PUSH_TICK, int(task.uids[0, task.chest]), [impulse])]
		assert result["outcome"] == "recovered" and result["step_count"] == 0
		assert result["recovery_seconds"] == pytest.approx(1)
		assert result["elapsed_seconds"] == pytest.approx(3)
	finally:
		task.close()

	passive = StandingTask(project, episode_steps=7 * CONTROL_HZ)
	action[3::4] = -1
	try:
		result = run_trial(passive, lambda _: action, seed=0, bone="pelvis", impulse=impulse)
		assert result["outcome"] == "pre_push_fall" and result["recovery_seconds"] is None
		assert len(queued) == 1
	finally:
		passive.close()
