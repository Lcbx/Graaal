from dataclasses import replace
import json
import os
import sys

import numpy as np
import pytest

from scenes.benchmarks.benchmark_ragdoll_push_recovery import HARD_BACKWARD_CELLS, PUSH_TICK, TICK_HZ, benchmark, make_trial_controller, run_quiet_trial, run_trial
from scenes.physics_geometry import collider_bounds
from scenes.ragdoll_control import STANDING_MOTOR_CAPS
from scenes.ragdoll_recovery_viewer import discover_reports, filtered_trials, paired_selection, read_report, record_trial, suite_reports, view_reports
from tools import ragdoll_standing
from tools.ragdoll_project import default_project
from tools.ragdoll_standing import ARM_OFFSET, CONTROL_HZ, HIP_OFFSET, KNEE_BACKWARD_OFFSET, KNEE_FORWARD_OFFSET, MAX_OFFSET, MAX_RESIDUAL_TORQUE, POSE_FALL_TICKS, StandingTask, _support_margin, evaluate


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


def test_joint_action_spaces_preserve_observations_and_pair_pushes(project, monkeypatch):
	import pyjoltphysics as pj

	queue = pj.World.add_impulses
	queued = []

	def record(world, uids, impulses):
		queued.append((world.tick, np.asarray(impulses).tolist()))
		return queue(world, uids, impulses)

	monkeypatch.setattr(pj.World, "add_impulses", record)
	options = dict(episode_steps=7 * CONTROL_HZ, tuned_stance=True, vary_start=True, support_area=True, push_impulses=(8,))
	stepping = StandingTask(project, stepping=True, **options)
	whole_body = StandingTask(project, whole_body=True, **options)
	residual = StandingTask(project, residual=True, **options)
	try:
		initial = residual.reset(seed=22)[0]
		np.testing.assert_array_equal(stepping.reset(seed=22)[0], initial)
		np.testing.assert_array_equal(whole_body.reset(seed=22)[0], initial)
		assert stepping.action_shape == (10,)
		assert whole_body.action_shape == (26,)
		for _ in range(2 * CONTROL_HZ + 1):
			step_obs, step_reward, *_ = stepping.step(np.zeros(10))
			ankle_obs, ankle_reward, *_ = residual.step(np.zeros(4))
			whole_obs, whole_reward, *_ = whole_body.step(np.zeros(26))
			np.testing.assert_array_equal(step_obs, ankle_obs)
			np.testing.assert_array_equal(whole_obs, ankle_obs)
			assert step_reward == ankle_reward == whole_reward
		assert len(queued) == 3 and queued[0] == queued[1] == queued[2]
		assert queued[0][0] == PUSH_TICK
		assert np.linalg.norm(queued[0][1]) == pytest.approx(8)

		leg_action = [0, 0, 0, 0, 1, -1, -1, 1, 1, -1]
		stepping.step(leg_action)
		whole_body.step(leg_action + [1] * 16)
		controls = stepping.motor_action[0]
		np.testing.assert_allclose(controls[stepping.hips, :3], HIP_OFFSET / np.sqrt(2) * np.array([[1, 0, -1], [-1, 0, 1]]))
		np.testing.assert_allclose(controls[stepping.knees, 0], [KNEE_FORWARD_OFFSET, -KNEE_BACKWARD_OFFSET])
		np.testing.assert_array_equal(controls[np.r_[0, stepping.feet]], np.zeros((3, 4)))
		whole_controls = whole_body.motor_action[0]
		np.testing.assert_array_equal(whole_controls[7:], controls[7:])
		np.testing.assert_array_equal(whole_controls[0], controls[0])
		np.testing.assert_allclose(np.linalg.norm(whole_controls[1:7, :3], axis=1), [MAX_OFFSET] * 2 + [ARM_OFFSET] * 4)
	finally:
		stepping.close()
		whole_body.close()
		residual.close()


def test_scripted_backstep_lifts_and_loads_the_landing_foot(project):
	task = StandingTask(project, whole_body=True, tuned_stance=True, vary_start=True, initial_angular_speed=0.1, support_area=True, episode_steps=7 * CONTROL_HZ)
	zero = np.zeros(task.action_shape)
	try:
		result = run_trial(task, lambda _: zero, seed=58000, bone="chest", impulse=(0, 0, -12), scripted_step=True)
		landing = next(landing for landing in result["landings"] if landing["foot"] == "foot_r")
		assert landing["landing_xz_m"][1] < landing["takeoff_xz_m"][1] - 0.02
		assert landing["contact_ticks"] >= 12
		assert landing["peak_estimated_floor_impulse_ns"] > 0
		assert task.settings.root_force == task.settings.root_torque == 0
		options = dict(seed=58000, bone="chest", impulse=(0, 0, -2))
		assert run_trial(task, lambda _: zero, scripted_step=True, **options) == run_trial(task, lambda _: zero, **options)
	finally:
		task.close()


@pytest.mark.parametrize(("seed", "magnitude"), [(58000, 12), (60401, 16)])
def test_backstep_reference_recovers_and_swing_ablation_falls(seed, magnitude):
	task, policy, _ = make_trial_controller("backstep_reference", seed=seed, strong_backstep_reference=magnitude >= 14)
	try:
		options = dict(seed=seed, bone="chest", impulse=(0, 0, -magnitude))
		result = run_trial(task, policy, **options)
		assert result["outcome"] == "recovered"
		landing = result["landings"][0]
		assert landing["foot"] == "foot_r" and landing["displacement_along_push_m"] >= 0.05
		assert landing["airborne_seconds"] >= 0.05
		assert landing["contact_ticks"] >= 6
		assert landing["com_speed_at_touchdown_m_s"] - landing["com_speed_after_0_2s_m_s"] >= 0.02
		task.backstep_swing = False
		result = run_trial(task, policy, **options)
		assert result["outcome"] == "post_push_fall" and result["step_count"] == 0
	finally:
		task.close()


def test_backstep_training_reset_replays_the_full_reference_swing(project):
	task = StandingTask(project, backstep=True, tuned_stance=True, vary_start=True, initial_angular_speed=0.1,
		push_impulses=(16,), episode_steps=7 * CONTROL_HZ, strong_backstep_reference=True)
	try:
		replayed, _ = task.reset(seed=60401, replay=True)
		handoff = task.world.tick
		assert PUSH_TICK < handoff < PUSH_TICK + TICK_HZ // 2
		assert task.backstep_reference.residual_active()
		task.reset(seed=60401)
		while task.world.tick < handoff:
			observed, _, _, _, _ = task.step(task.backstep_zero)
		np.testing.assert_array_equal(observed, replayed)
	finally:
		task.close()


@pytest.mark.parametrize("impulses", [(), (0, 8, 9.99), (float("inf"),)])
def test_backstep_training_requires_an_activating_push(impulses):
	with pytest.raises(ValueError, match="backstep training requires --recovery-curriculum or a push of at least 10 N s"):
		ragdoll_standing.train_policy(None, steps=1, seed=0, backstep=True, push_impulses=impulses)


@pytest.mark.parametrize("curriculum", [False, True])
def test_backstep_cli_pairs_baseline_pushes_and_accepts_pose_curriculum(project, monkeypatch, curriculum):
	impulses = (11, 12, 13, 14, 14, 15, 15, 16)
	args = ["ragdoll_standing", "--stance", "tuned", "--action-space", "backstep", "--pose-recovery", "--seconds", "7",
		"--vary-start", "--initial-angular-speed", "0.1", "--eval-seed", "10098", "--episodes", "1"]
	args += ["--recovery-curriculum"] if curriculum else ["--push-impulses", *map(str, impulses)]
	task = StandingTask(project, episode_steps=7 * CONTROL_HZ, tuned_stance=True, vary_start=True, initial_angular_speed=0.1,
		backstep=True, recovery_curriculum=curriculum, push_impulses=() if curriculum else impulses)
	try:
		task.reset(seed=10098)
		bone, impulse = task.push
	finally:
		task.close()
	seen = []

	def evaluate_baseline(baseline, policy, *, seed, episodes):
		assert not baseline.backstep and baseline.pose_recovery and episodes == 1
		observation, _ = baseline.reset(seed=seed)
		assert baseline.push[0] == bone
		np.testing.assert_array_equal(baseline.push[1], impulse)
		assert policy(observation).shape == baseline.action_shape
		seen.append(baseline.ankle_balance)
		return {}

	monkeypatch.setattr(sys, "argv", args)
	monkeypatch.setattr(ragdoll_standing, "evaluate", evaluate_baseline)
	ragdoll_standing.main()
	assert seen == [False, False, True]


def test_backstep_impact_uses_velocity_change_and_active_floor_contacts(project):
	import pyjoltphysics as pj

	task = StandingTask(project, backstep=True, tuned_stance=True)
	try:
		task.reset(seed=0)
		_, reward, *_ = task.step(task.backstep_zero)
		task.reset(seed=0)
		posture = task._posture

		def fast_posture():
			height, upright, _, fallen = posture()
			return height, upright, 2.0, fallen

		task._posture = fast_posture
		assert task.step(task.backstep_zero)[1] == reward

		task.apply_push(task.chest, (0, 0, -12))
		task.world.run_ticks(1)
		velocity = np.array([3.0, 0, 0])
		task.state = replace(task.state, center_of_mass_velocity=velocity[None])
		task.events = np.zeros(3, dtype=task.events.dtype)
		task.events["a"] = task.uids[0, task.feet[0]]
		task.events["b"] = [task.floor[0], task.floor[0], task.uids[0, task.chest]]
		task.events["kind"] = pj.EventKind.CONTACT
		task.events["phase"] = [pj.EventPhase.BEGIN, pj.EventPhase.END, pj.EventPhase.PERSIST]
		task.events["impulse"] = [2, 5, 8]
		assert task._backstep_impact(velocity - 1) == (0, 0)
		task.world.run_ticks(1)
		assert task._backstep_impact(velocity) == (0, 2)
		assert task._backstep_impact(velocity - [0.1, 0, 0]) == pytest.approx((12, 2))
	finally:
		task.close()


def test_touchdown_support_reward_preserves_motion_and_scores_support_improvement(project):
	options = dict(backstep=True, tuned_stance=True, pose_recovery=True, episode_steps=7 * CONTROL_HZ)
	plain = StandingTask(project, **options)
	shaped = StandingTask(project, touchdown_support_reward=True, **options)
	try:
		for task in (plain, shaped):
			task.reset(seed=60700)
			while task.world.tick < PUSH_TICK:
				task.step(task.backstep_zero)
			task.apply_push(task.chest, [0, 0, -12])
		reward_difference = 0.0
		while not plain.done and not shaped.done:
			_, reward, fallen, completed, _ = plain.step(plain.backstep_zero)
			_, shaped_reward, shaped_fallen, shaped_completed, info = shaped.step(shaped.backstep_zero)
			assert (fallen, completed) == (shaped_fallen, shaped_completed)
			np.testing.assert_allclose(plain.state.position, shaped.state.position)
			reward_difference += shaped_reward - reward
		assert reward_difference == pytest.approx(info["touchdown_support_reward"])
		assert abs(reward_difference) > 1e-5
	finally:
		plain.close()
		shaped.close()


def test_recovery_pose_hold_checks_joint_angles_motion_and_sustained_fall(project):
	task = StandingTask(project, tuned_stance=True, pose_recovery=True)
	task.reset(seed=0)
	try:
		contacts = np.zeros_like(task.state.contacts)
		contacts[0, task.feet[0]] = True
		rotations = task.reference[None].copy()
		still = np.zeros_like(task.state.angular_velocity)
		velocity = np.zeros_like(task.state.center_of_mass_velocity)
		settled = replace(task.state, contacts=contacts, joint_rotation=rotations, angular_velocity=still, center_of_mass_velocity=velocity)
		task.state = settled
		assert not task._recovery_posture(task.target_up, 1.0) and task.pose_hold_ticks == 1
		spin = still.copy()
		spin[0, 0, 1] = 0.3
		bad_pose = rotations.copy()
		bad_pose[0, 1] = [1, 0, 0, 0]
		for state in (replace(settled, angular_velocity=spin), replace(settled, center_of_mass_velocity=velocity + [0, 0.06, 0]),
				replace(settled, joint_rotation=bad_pose), replace(settled, contacts=np.zeros_like(contacts))):
			task.pose_hold_ticks = 20
			task.state = state
			assert not task._recovery_posture(task.target_up, 1.0) and task.pose_hold_ticks == 0
		task.state = settled
		for _ in range(POSE_FALL_TICKS - 1):
			assert not task._recovery_posture(task.target_up, 0.6)
		assert task._recovery_posture(task.target_up, 0.6)
		assert not task._recovery_posture(task.target_up, 1.0) and task.toppled_ticks == 0
	finally:
		task.close()


@pytest.mark.parametrize(("break_at", "expected_tick", "recovered"), [(1, 4, False), (5, 5, False), (None, 5, True)])
def test_recovery_pose_grace_only_finishes_an_active_hold(project, monkeypatch, break_at, expected_tick, recovered):
	task = StandingTask(project, episode_steps=2, tuned_stance=True, pose_recovery=True)
	task.reset(seed=0)
	try:
		task.apply_push(task.chest, [0, 0, 0])
		task.pose_hold_ticks = TICK_HZ - 5

		def hold(up, height):
			task.pose_hold_ticks = task.pose_hold_ticks + 1 if break_at is None or task.world.tick < break_at else 0
			return False

		monkeypatch.setattr(task, "_recovery_posture", hold)
		samples = []
		action = np.zeros(task.action_shape, dtype=np.float32)
		while not task.done:
			_, _, fallen, completed, info = task.step(action, after_tick=lambda: samples.append(task.world.tick))
		assert not fallen and completed and info["pose_recovered"] is recovered
		assert info["settled_success"] is recovered
		assert samples == list(range(1, expected_tick + 1))
	finally:
		task.close()


def test_backstep_releases_hip_pitch_after_loading(project, monkeypatch):
	task = StandingTask(project, tuned_stance=True, backstep=True, release_backstep_hips=True)
	task.reset(seed=0)
	try:
		task.apply_push(task.chest, [0, 0, -14])
		reference = task.backstep_reference
		end = reference.swing_seconds + reference.load_seconds
		for offset, expected in ((0, [-0.2, 0.4]), (0.125, [0.2, 0.5]), (0.25, [0.6, 0.6])):
			monkeypatch.setattr(reference, "_seconds", lambda: end + offset)
			action = np.full_like(task.motor_action, 0.6)
			reference.edit_motors(action)
			np.testing.assert_allclose(action[0, task.hips, 0], expected)
	finally:
		task.close()


def test_direct_targets_run_before_push_and_without_foot_contact(project):
	task = StandingTask(project, direct=True, tuned_stance=True)
	try:
		initial, _ = task.reset(seed=0)
		assert initial.shape == (187,) and task.action_shape == (30,)
		assert not task.state.contacts.any()
		action = np.full(task.action_shape, 2.0)
		observation, reward, _, _, info = task.step(action)
		assert np.isfinite(observation).all() and np.isfinite(reward)
		np.testing.assert_array_equal(action, 2)
		np.testing.assert_array_equal(observation[-len(action):], 1)
		np.testing.assert_array_equal(task.motor_action[0, 0], 0)
		np.testing.assert_array_equal(task.motor_action[0, 1:, 3], 1)
		assert task.settings.root_force == task.settings.root_torque == 0
		assert not task.ankle_balance and task.recovery is None
		assert np.linalg.norm(task.motor_action[0, task.feet, :3], axis=1).min() > 0
		assert np.all(np.linalg.norm(task.motor_action[0, :, :3], axis=1) <= task.motor_offset_caps + 1e-7)
		assert info["target_saturation_fraction"] == 1 and info["effort_reward"] < 0
		assert info["braking_reward"] == info["support_reward"] == info["compliance_reward"] == 0
		np.testing.assert_array_equal(task.reset(seed=0)[0], initial)
	finally:
		task.close()


def test_direct_curriculum_keeps_quiet_light_and_hard_pushes(project):
	task = StandingTask(project, direct=True, tuned_stance=True, recovery_curriculum=True)
	rng = np.random.default_rng(87)
	samples = [task._sample_recovery_push(rng) for _ in range(256)]
	assert None in samples
	assert any(sample is not None and np.linalg.norm(sample[1]) < 2 for sample in samples)
	hard = {(task.settings.bones[bone].name, -impulse[2], "-Z") for bone, impulse in (value for value in samples if value is not None)
		if impulse[2] <= -8 and not impulse[:2].any()}
	assert HARD_BACKWARD_CELLS <= hard
	report = benchmark(seed=58200, episodes=1, impulses=(), hard_backward_only=True, recovery_seconds=10)
	assert {(cell["bone"], cell["magnitude_ns"], cell["direction"]) for cell in report["cells"]} == HARD_BACKWARD_CELLS
	assert report["hard_backward_summary"] == report["summary"]
	assert report["episode_seconds"] == 12 and report["recovery_window_seconds"] == 10


def test_direct_reward_ablation_changes_reward_without_changing_motion(project):
	shaped = StandingTask(project, direct=True, tuned_stance=True)
	ablation = StandingTask(project, direct=True, tuned_stance=True, without_direct_reward=("braking", "support", "compliance"))
	try:
		action = np.zeros(shaped.action_shape)
		for task in (shaped, ablation):
			task.reset(seed=1)
			for _ in range(10):
				task.step(action)
			task.apply_push(task.chest, (0, 0, -12))
		for _ in range(10):
			shaped_obs, _, _, _, shaped_info = shaped.step(action)
			basic_obs, _, _, _, basic_info = ablation.step(action)
			np.testing.assert_array_equal(shaped_obs, basic_obs)
		assert shaped_info["braking_reward"] != 0
		assert basic_info["braking_reward"] == basic_info["support_reward"] == basic_info["compliance_reward"] == 0
	finally:
		shaped.close()
		ablation.close()


def test_benchmark_counts_continuous_landing_contact_without_impulse_diagnostics(project):
	task, policy, _ = make_trial_controller("ankle_balance", seed=60700, pose_recovery=True)
	contacts = []
	try:
		result = run_trial(task, policy, seed=60700, bone="chest", impulse=[0, 0, -11],
			on_sample=lambda task: contacts.append(task.state.contacts[0].copy()))
		assert len(result["landings"]) == 2
		for landing in result["landings"]:
			foot = task.settings.bone_index(landing["foot"])
			tick = PUSH_TICK + round(landing["seconds_after_push"] * TICK_HZ)
			planted = np.array(contacts)[tick:, foot]
			breaks = np.flatnonzero(~planted)
			expected = int(breaks[0]) if len(breaks) else len(planted)
			assert landing["contact_ticks"] == expected > 0
			assert "peak_estimated_floor_impulse_ns" not in landing
	finally:
		task.close()


@pytest.mark.parametrize(("magnitude", "outcome"), [(0.5, "recovered"), (12, "post_push_fall")])
def test_recovery_viewer_recording_matches_benchmark(project, magnitude, outcome):
	task, policy, _ = make_trial_controller("ankle_balance", seed=58300)
	cell = dict(bone="chest", impulse_ns=(0, 0, -magnitude), magnitude_ns=magnitude)
	try:
		expected = run_trial(task, policy, seed=58300, bone=cell["bone"], impulse=cell["impulse_ns"])
		actual, (centers, rotations), mass_centers, speeds, bounds, push = record_trial(task, policy, cell, expected, "ankle_balance")
		assert actual == expected and actual["outcome"] == outcome
		assert len(centers) == len(rotations) == len(mass_centers) == len(speeds) == round(expected["elapsed_seconds"] * TICK_HZ) + 1
		assert task.world is None
		assert np.isfinite((centers.min(), rotations.min(), mass_centers.min(), speeds.min())).all()
		assert np.all(bounds[1] >= bounds[0])
		np.testing.assert_array_equal(push[0], centers[PUSH_TICK, task.settings.bone_index("chest")])
		np.testing.assert_array_equal(push[1], [0, 0, -1])
	finally:
		task.close()


def test_adaptive_step_uses_observed_motion_and_replays_touchdown():
	seed = 62700
	task, policy, _ = make_trial_controller("adaptive_step", seed=seed, pose_recovery=True)
	cell = dict(bone="chest", impulse_ns=[0, 0, -16], magnitude_ns=16)
	try:
		quiet = run_quiet_trial(task, policy, seed=58305)
		assert quiet["settled_success"] and task.recovery.planned_goal is None
		light = run_trial(task, policy, seed=seed, bone="chest", impulse=[0, 0, -0.5])
		assert light["outcome"] == "recovered" and not light["step_plans"]
		expected = run_trial(task, policy, seed=seed, bone=cell["bone"], impulse=cell["impulse_ns"])
		first, *followups = expected["step_plans"]
		assert 0 < first["start_tick_after_push"] < first["touchdown_tick_after_push"]
		assert first["peak_swing_torque_ratio"] > 0
		assert all(p["start_tick_after_push"] > first["touchdown_tick_after_push"] for p in followups)
		samples = []
		actual, *_ = record_trial(task, policy, cell, expected, "adaptive_step", balance_samples=samples)
		assert actual == expected
		tick = PUSH_TICK + first["touchdown_tick_after_push"]
		point, support, margin, velocity, load, slip, _ = samples[tick]
		assert np.isfinite(point).all() and np.isfinite(margin) and np.isfinite(velocity)
		assert len(support) >= 4 and len(load) == len(slip) == 2
	finally:
		task.close()


def test_recovery_browser_keeps_matching_case_and_seed():
	fell = dict(seed=41, outcome="post_push_fall", step_count=2)
	recovered = dict(seed=42, outcome="recovered", step_count=0)
	settled = dict(seed=42, outcome="settled")
	controllers = {"ppo_direct": {"episodes": [fell, recovered]}, "scripted_step": {"episodes": [recovered, fell]}}
	cases = [dict(bone="chest", direction="-Z", magnitude_ns=impulse, controllers=controllers) for impulse in (8, 12)]
	quiet = dict(bone=None, controllers={"ppo_direct": {"episodes": [settled]}})
	reports = [dict(cells=cases), dict(cells=list(reversed(cases))), dict(cells=[quiet])]
	current = (0, 0, "ppo_direct", 1)
	assert paired_selection(reports, current, 1) == (1, 1, "ppo_direct", 1)
	assert paired_selection(reports, current, 0, name="scripted_step") == (0, 0, "scripted_step", 0)
	assert paired_selection(reports, current, 0, case_index=1, result_filter="With steps") == (0, 1, "ppo_direct", 0)
	assert paired_selection(reports, current, 2) == (2, 0, "ppo_direct", 0)
	assert filtered_trials([fell, recovered], "With steps") == [(0, fell)]
	stepped = dict(recovered, step_count=1)
	assert filtered_trials([fell, recovered, stepped], "Recovered with steps") == [(2, stepped)]
	assert filtered_trials([fell, recovered], "Failed") == [(0, fell)]
	assert filtered_trials([fell, settled], "Passed") == [(1, settled)]


def test_recovery_viewer_opens_quiet_standing_report(project, tmp_path):
	quiet = benchmark(seed=58300, episodes=1, impulses=(), quiet=True)
	trial, = quiet["cells"][0]["controllers"]["ankle_balance"]["episodes"]
	task, policy, _ = make_trial_controller("ankle_balance", seed=58300, episode_seconds=quiet["episode_seconds"])
	try:
		expected = evaluate(task, policy, seed=58300, episodes=1)
		assert quiet["episode_seconds"] == 10
		assert quiet["summary"]["ankle_balance"]["settled"] == expected["settled_success"]
		assert all(trial[key] == value for key, value in expected.items())
		_, (centers, _), _, _, _, push = record_trial(task, policy, quiet["cells"][0], trial, "ankle_balance")
		assert len(centers) == 10 * TICK_HZ + 1 and push is None
	finally:
		task.close()
	path = tmp_path / "checkpoint-quiet-replay.json"
	path.write_text(json.dumps(quiet), encoding="utf-8")
	(tmp_path / "checkpoint-quiet.json").write_text(json.dumps(expected), encoding="utf-8")
	assert suite_reports(tmp_path / "checkpoint") == [(path, "Quiet standing")]
	assert view_reports([path], [read_report(path)], headless=True) == trial


def test_recovery_viewer_discovers_reports_with_available_assets(tmp_path):
	session = tmp_path / "session"
	session.mkdir()
	(session / "trained.zip").touch()
	trial = dict(episodes=[dict(seed=42, outcome="settled")])
	setup = dict(tick_hz=TICK_HZ, first_seed=42, max_start_spin_rad_s=0.1, max_start_tilt_rad=0.025)

	def save(path, timestamp, controllers, checkpoints=None):
		report = setup | dict(cells=[dict(bone=None, controllers=controllers)], checkpoints=checkpoints or {})
		path.write_text(json.dumps(report), encoding="utf-8")
		os.utime(path, (timestamp, timestamp))

	trained = tmp_path / "trained.json"
	quiet = tmp_path / "quiet.json"
	save(trained, 20, dict(ankle_balance=trial, ppo_backstep=trial, ppo_direct=trial),
		dict(ppo_backstep=dict(path="session/trained"), ppo_direct=dict(path="session/trained.zip", tuning="missing-rig.json")))
	save(quiet, 10, dict(ankle_balance=trial))
	save(session / "scratch.json", 40, dict(ankle_balance=trial))
	save(tmp_path / "missing.json", 30, dict(ppo_backstep=trial), dict(ppo_backstep=dict(path="missing.zip")))
	(tmp_path / "summary.json").write_text(json.dumps(setup), encoding="utf-8")
	(tmp_path / "incomplete.json").write_text("{", encoding="utf-8")

	assert [(path, list(report["cells"][0]["controllers"])) for path, report in discover_reports(tmp_path)] == [
		(trained, ["ankle_balance", "ppo_backstep"]), (quiet, ["ankle_balance"])]


@pytest.mark.parametrize(("seed", "window", "phase"), [(1, "before_landing", 1), (0, "after_landing", 2)])
def test_recovery_replays_real_swing_and_touchdown(project, seed, window, phase):
	task = StandingTask(project, recovery=True, tuned_stance=True, vary_start=True, initial_angular_speed=0.1,
		recovery_curriculum=True, replay_start_fraction=1)
	try:
		observation, info = task.reset(seed=seed)
		assert info == {} and task.world.tick == 0
		assert observation.shape == (165,) and task.action_shape == (32,)
		replayed, info = task.reset(seed=seed, replay=True)
		assert info["replay_start"] == window
		assert task.recovery.phase == phase and task.world.tick > PUSH_TICK
		foot = task.feet[task.recovery.side]
		assert bool(task.state.contacts[0, foot]) == (phase == 2)
		assert np.isfinite(replayed).all()
		np.testing.assert_array_equal(task.reset(seed=seed, replay=True)[0], replayed)
		torque = task.recovery.feedback(task.state, None)
		np.testing.assert_allclose(torque.sum(axis=0), 0, atol=1e-12)
		updated, reward, fallen, completed, _ = task.step(np.zeros(task.action_shape))
		assert np.isfinite(updated).all() and np.isfinite(reward)
		assert not fallen and not completed
		np.testing.assert_array_equal(task.motor_action[0, 0], 0)
	finally:
		task.close()


def test_recovery_quiet_step_request_and_single_settle_bonus(project):
	task = StandingTask(project, recovery=True, tuned_stance=True, episode_steps=7 * CONTROL_HZ)
	action = np.zeros(task.action_shape)
	action[task.motor_channels] = 1
	try:
		task.reset(seed=0)
		while task.world.tick < PUSH_TICK:
			task.step(action)
			assert task.recovery.phase == 0
		step = task.recovery
		assert task.settled_ticks >= 120 and not step.settle_awarded
		task.apply_push(task.chest, (0, 0, -0.5))
		while task.settled_ticks < 118:
			task.step(action)
		_, reward, *_ = task.step(action)
		_, next_reward, *_ = task.step(action)
		assert step.settle_awarded and step.phase == 0
		assert reward - next_reward == pytest.approx(8, abs=0.01)
	finally:
		task.close()
