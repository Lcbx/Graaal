import numpy as np
import pytest

from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.AnimationGraphPlayback import AnimationGraphPlayback
from graaal.Animation.BakedClipPool import BakedClipPool
from graaal.Animation.CrowdGraphSamples import CrowdGraphRouter
from graaal.Animation.CrowdGraphValidation import validate_crowd_graph
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices
from graaal.Math import Quaternion
from tests.test_animation_graph_playback import synchronized_graph


def test_crowd_validator_compares_exact_graph_poses_and_validates_shadowed_edges():
	translation = [[[0, 0, 0], [1, 0, 0]]]
	rotation = np.tile([0.0, 0, 0, 1], (1, 2, 1))
	scale = np.ones((1, 2, 3))
	rig = Skeleton([-1, 0], translation[0], rotation[0], scale[0], np.tile(np.eye(4), (2, 1, 1)))
	idle = Clip(translation, rotation, scale)
	rotation[0, 0] = Quaternion.from_z_rotation(np.pi / 2)
	turn = Clip(translation, rotation, scale)
	conditions = [{"parameter": "go", "comparison": "==", "threshold": 1}]
	graph = AnimationGraphDefinition({
		"version": 1, "parameters": {"go": 0, "lean": 0}, "initial_state": "idle",
		"states": {"idle": {"clip": "idle"}, "turn": {"clip": "turn"}, "blend": {"blend_space": "lean"}},
		"blend_spaces": {"lean": {"parameter": "lean", "points": [{"position": 0, "clip": "idle"}, {"position": 1, "clip": "turn"}]}},
		"transitions": [{"source": "idle", "target": "turn", "duration": duration, "conditions": conditions} for duration in (0.5, 1)],
		"sample_slots": ["base"], "tasks": [{"name": "output", "op": "sample", "slot": "base"}],
	}).compile(rig, {"idle": idle, "turn": turn})
	pool = BakedClipPool(rig, [idle, turn])
	report = validate_crowd_graph(graph, pool, error_budget=0.02, blend_samples=3, fade_samples=3)
	assert report.states["idle"].eligible and report.states["turn"].eligible
	assert report.states["blend"].samples == 3
	assert not report.states["blend"].eligible
	assert report.states["blend"].max_position_error == pytest.approx(1 - np.sqrt(0.5), abs=1e-6)
	assert [(edge.source, edge.target, edge.duration) for edge in report.transitions] == [("idle", "turn", 0.5), ("idle", "turn", 1)]
	assert all(edge.result == report.states["blend"] for edge in report.transitions)
	with pytest.raises(ValueError, match="compiled ID order"):
		validate_crowd_graph(graph, BakedClipPool(rig, [turn, idle]), error_budget=0.02)


@pytest.mark.parametrize("share_clips", [False, True])
def test_crowd_validator_accounts_for_sample_limits_and_identical_inputs(share_clips):
	rotation, scale = [[[0, 0, 0, 1]]], [[[1, 1, 1]]]
	rig = Skeleton([-1], [[0, 0, 0]], rotation[0], scale[0], [np.eye(4)])
	clips = {str(i): Clip([[[i, 0, 0]]], rotation, scale) for i in range(8)}
	target = [["0", "1"], ["2", "3"]] if share_clips else [["4", "5"], ["6", "7"]]
	graph = AnimationGraphDefinition({
		"version": 1, "parameters": {"x": 0, "y": 0}, "initial_state": "source",
		"states": {name: {"blend_space_2d": name} for name in ("source", "target")},
		"blend_spaces_2d": {name: {"x_parameter": "x", "y_parameter": "y", "x_positions": [0, 1], "y_positions": [0, 1], "clips": grid}
			for name, grid in (("source", [["0", "1"], ["2", "3"]]), ("target", target))},
		"transitions": [{"source": "source", "target": "target", "duration": 0.5, "conditions": [{"parameter": "x", "comparison": "==", "threshold": 1}]}],
		"sample_slots": ["base"], "tasks": [{"name": "output", "op": "sample", "slot": "base"}],
	}).compile(rig, clips)
	report = validate_crowd_graph(graph, BakedClipPool(rig, tuple(clips.values())), error_budget=0, blend_samples=3, fade_samples=3)
	assert all(state.eligible and state.samples == 9 for state in report.states.values())
	result = report.transitions[0].result
	assert result.samples == 27
	assert result.unsupported_samples == (0 if share_clips else 1)
	assert result.eligible == share_clips
	assert result.max_position_error == 0
	playback = AnimationGraphPlayback(graph, 2)
	playback.advance(0.25, [[0.5, 0.5], [1, 0.5]])
	parameters = np.full((2, 2), 0.5)
	router = CrowdGraphRouter(report)
	cpu, batch = router.split(playback, parameters)
	assert cpu.tolist() == ([] if share_clips else [1])
	assert batch.rows.tolist() == ([0, 1] if share_clips else [0])
	approx = router.pool.model(batch.clip_ids, batch.times, batch.weights)
	np.testing.assert_allclose(approx, model_matrices(rig, *playback.pose(parameters))[batch.rows], atol=1e-6)


def test_crowd_validator_samples_synchronized_fade_clocks_at_requested_rates():
	graph = synchronized_graph(1)
	pool = BakedClipPool(graph.plan.skeleton, tuple(graph.clips.values()))
	report = validate_crowd_graph(graph, pool, error_budget=0.02, phase_samples=9, blend_samples=3, fade_samples=5, playback_rates=(0, 0.5, 2))
	assert all(state.eligible and state.samples == 36 for state in report.states.values())
	result = report.transitions[0].result
	assert result.eligible and result.samples == 540
	assert 0 < result.max_position_error < 0.02
