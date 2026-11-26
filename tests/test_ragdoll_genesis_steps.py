import pytest

torch = pytest.importorskip("torch")

from tools.ragdoll_genesis_steps import StepTracker, motion_gated_actions, rescale_base_actions
from scenes.benchmarks.benchmark_genesis_steps import candidates
from tools.ragdoll_genesis_imitation import load_dataset
from tools.ragdoll_genesis_train import StandingEnv, main as train_main


def test_counts_relocated_braking_landing_but_not_wrong_direction_or_low_lift():
	tracker = StepTracker(3, "cpu")
	positions = torch.zeros((3, 2, 3))
	contacts = torch.ones((3, 2), dtype=torch.bool)
	active = torch.ones(3, dtype=torch.bool)
	push = torch.ones(3, dtype=torch.bool)
	direction = torch.full((3,), 2, dtype=torch.int64)
	tracker.update(positions, contacts, torch.full((3,), 0.4), active, push, direction)
	push.zero_()
	contacts[:, 1] = False
	positions[:, 1, 1] = torch.tensor([0.04, 0.04, 0.01])
	positions[:, 1, 2] = torch.tensor([0.06, -0.06, 0.06])
	for _ in range(4):
		tracker.update(positions, contacts, torch.full((3,), 0.4), active, push, direction)
	contacts[:, 1] = True
	positions[:, 1, 1] = 0
	tracker.update(positions, contacts, torch.full((3,), 0.4), active, push, direction)
	for _ in range(12):
		tracker.update(positions, contacts, torch.full((3,), 0.35), active, push, direction)
	assert tracker.landed[:, 1].tolist() == [True, False, False]
	assert tracker.braked[:, 1].tolist() == [True, False, False]
	tracker.reset(torch.tensor([False, True, True]))
	assert tracker.braked[:, 1].tolist() == [True, False, False]
	assert tracker.push_seen.tolist() == [True, False, False]


def test_does_not_count_a_brief_contact_as_landing():
	tracker = StepTracker(1, "cpu")
	positions = torch.zeros((1, 2, 3))
	contacts = torch.ones((1, 2), dtype=torch.bool)
	active = torch.ones(1, dtype=torch.bool)
	direction = torch.tensor([2])
	tracker.update(positions, contacts, torch.tensor([0.4]), active, active, direction)
	contacts[:, 1] = False
	positions[:, 1, 1:] = torch.tensor([0.04, 0.06])
	for _ in range(4):
		tracker.update(positions, contacts, torch.tensor([0.4]), active, ~active, direction)
	contacts[:, 1] = True
	positions[:, 1, 1] = 0
	tracker.update(positions, contacts, torch.tensor([0.4]), active, ~active, direction)
	contacts[:, 1] = False
	for _ in range(12):
		tracker.update(positions, contacts, torch.tensor([0.35]), active, ~active, direction)
	assert not tracker.landed.any()
	assert not tracker.braked.any()


def test_search_candidates_are_seeded_and_bounded():
	import numpy as np

	first = candidates(16, 7)
	np.testing.assert_array_equal(first, candidates(16, 7))
	refined = candidates(16, 9, first[:2])
	assert refined.shape == (16, 9)
	assert np.all((refined[:, :6] >= -1) & (refined[:, :6] <= 1))
	assert np.all((refined[:, 6] >= 4) & (refined[:, 6] <= 40))


def test_motion_gate_triggers_holds_and_preserves_base_target_scale():
	features = torch.zeros((3, 8))
	features[0, -2] = -0.2
	base = torch.ones((3, 2))
	step = torch.full((3, 2), -0.7)
	ticks = torch.tensor([0, 2, 0], dtype=torch.int32)
	non_ankle = torch.tensor([True, False])
	result, next_ticks = motion_gated_actions(features, base, step, ticks, non_ankle, trigger_speed=0.12, hold_ticks=4, base_scale=0.22, step_scale=0.5)
	torch.testing.assert_close(result, torch.tensor([[-0.7, -0.7], [-0.7, -0.7], [0.44, 1.0]]))
	assert next_ticks.tolist() == [4, 1, 0]
	torch.testing.assert_close(base, torch.ones((3, 2)))


def test_reference_actions_preserve_standing_targets_and_ankles():
	base = torch.tensor([[0.75, -0.5, 1.2]])
	non_ankle = torch.tensor([True, False, True])
	converted = rescale_base_actions(base, non_ankle, base_scale=0.22, step_scale=0.5)
	torch.testing.assert_close(converted, torch.tensor([[0.33, -0.5, 0.44]]))
	torch.testing.assert_close(base, torch.tensor([[0.75, -0.5, 1.2]]))


def test_step_reward_rejects_zero_impulse_before_initializing_genesis(monkeypatch, capsys):
	import sys

	with pytest.raises(ValueError, match="positive push impulse"):
		StandingEnv(None, [], None, num_envs=1, episode_seconds=5, push_impulse=0, step_reward=1)
	monkeypatch.setattr(sys, "argv", ["trainer", "--output", "unused", "--step-reward", "1", "--random-pushes", "--hard-push-probability", "0.5", "--hard-min-impulse", "0"])
	with pytest.raises(SystemExit) as exc:
		train_main()
	assert exc.value.code == 2
	assert "positive push impulse" in capsys.readouterr().err


def test_load_imitation_dataset_retains_episode_boundaries():
	import io
	import json
	import numpy as np

	buffer = io.BytesIO()
	np.savez(buffer, observations=np.zeros((8, 147)), actions=np.zeros((8, 30)), swing=np.array([True, False] * 4), metadata=json.dumps({"successful_episodes": 2, "oracle_timing": True}))
	buffer.seek(0)
	_, _, _, episode, metadata = load_dataset(buffer)
	assert episode.tolist() == [0, 1] * 4
	assert metadata["oracle_timing"] is True


def test_load_student_states_allows_episodes_ending_early():
	import io
	import json
	import numpy as np

	buffer = io.BytesIO()
	np.savez(buffer, observations=np.zeros((5, 147)), actions=np.zeros((5, 30)), swing=np.array([True, False, True, False, True]), episode=np.array([0, 1, 0, 1, 0]), metadata=json.dumps({"episodes": 2, "oracle_timing": True}))
	buffer.seek(0)
	_, _, _, episode, metadata = load_dataset(buffer)
	assert episode.tolist() == [0, 1, 0, 1, 0]
	assert metadata["episodes"] == 2
