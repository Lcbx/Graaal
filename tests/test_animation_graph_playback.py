"""Automatic authoring playback without Tk, ECS or a graphics device."""

import numpy as np
import pytest

from graaal.Animation.AnimationEvents import EventTrack
from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, blend_poses, sample_clip_pairs
from tools.animation_graph_playback import GraphPlayback, state_pose, state_sample


def playback_snapshot(player):
	return (player.state, player.time, player.speed, player.target_state,
		player.target_time, player.target_speed, player.elapsed, player.duration)


def graph(*, sync=False, duration=0.25, static=False):
	rotation, scale = np.array([[[0, 0, 0, 1]]] * 2), np.ones((2, 1, 3))
	rig = Skeleton([-1], [[0, 0, 0]], rotation[0], scale[0], [np.eye(4)])
	clips = {name: Clip(np.array([[[start, 0, 0]], [[start + 2, 0, 0]]]), rotation, scale, fps=fps) for name, start, fps in (("idle", 0, 1), ("walk", 4, 0.5), ("other", 8, 1))}
	if static: clips = {name: Clip(clip.translation[:1], clip.rotation[:1], clip.scale[:1]) for name, clip in clips.items()}
	data = {
		"version": 1,
		"parameters": {"move": 0},
		"states": {name: {"clip": name} for name in clips},
		"initial_state": "idle",
		"transitions": [
			{"source": "idle", "target": "walk", "conditions": [{"parameter": "move", "comparison": ">", "threshold": 0}], "duration": duration},
			{"source": "idle", "target": "other", "conditions": [{"parameter": "move", "comparison": ">", "threshold": 0}], "duration": duration},
			{"source": "walk", "target": "other", "conditions": [{"parameter": "move", "comparison": "<", "threshold": 0}], "duration": duration},
		],
		"sample_slots": ["base"],
		"tasks": [{"name": "output", "op": "sample", "slot": "base"}],
	}
	tracks = {"idle": EventTrack(1, [0, 0.5], ["left", "right"]), "walk": EventTrack(2, [0, 0.5], ["left", "right"])}
	if sync: data["sync_mappings"] = [{"source": "idle", "target": "walk"}]
	return AnimationGraphDefinition(data).compile(rig, clips, sync_tracks=tracks)


def test_ordered_fade_has_independent_clocks_and_promotes_without_restarting():
	compiled = graph()
	player = GraphPlayback(compiled)
	player.reset(time=0.7)
	player.advance(0.125, [1])
	assert player.state == 0 and player.target_state == 1
	ids, times, weights = player.sample()
	np.testing.assert_array_equal(ids, [[0, 1]])
	np.testing.assert_allclose(times, [[0.825, 0.125]])
	np.testing.assert_allclose(weights, [[0.5, 0.5]])
	pose = compiled.plan.evaluate(compiled.clips, [(ids, times, weights)], np.empty((1, 0)))
	np.testing.assert_allclose(pose[0][0, 0], [2.8875, 0, 0])
	# Changing conditions cannot interrupt the active fade or select a second edge.
	player.advance(0.375, [-1])
	assert player.state == 1 and player.target_state is None
	np.testing.assert_allclose(player.time, 0.5)
	np.testing.assert_allclose(player.sample()[1], [[0.5, 0.5]])
	np.testing.assert_allclose(player.sample()[2], [[1, 0]])
	player.advance(0.1, [-1])
	assert player.target_state == 2
	np.testing.assert_allclose(player.sample()[1], [[0.6, 0.1]])
	# Earlier snapshots remain usable after later updates.
	np.testing.assert_allclose(times, [[0.825, 0.125]])


def test_sync_crosses_markers_and_loops_and_retains_endpoint_speed():
	compiled = graph(sync=True, duration=2.25)
	player = GraphPlayback(compiled)
	player.reset(time=0.75)
	player.advance(2, [1])
	assert player.target_state == 1
	np.testing.assert_allclose(player.sample()[1], [[0.75, 1.25]])
	np.testing.assert_allclose(player.sample()[2], [[1 / 9, 8 / 9]])
	# The fade ends at source time zero, where the target's new interval rate is 1.
	player.advance(0.5, [1])
	assert player.state == 1 and player.target_state is None and player.speed == 1
	np.testing.assert_allclose(player.time, 0.25)
	player.advance(0.1, [0])
	np.testing.assert_allclose(player.time, 0.35)


def test_sync_promotion_inside_interval_retains_mapped_speed():
	player = GraphPlayback(graph(sync=True))
	player.reset(time=0.5)
	player.advance(0.5, [1])
	assert player.state == 1 and player.target_state is None and player.speed == 3
	# Source 0.75 maps to target 1.25; the remaining 0.25 seconds use speed 3.
	np.testing.assert_allclose(player.time, 0)
	player.advance(0.1, [0])
	np.testing.assert_allclose(player.time, 0.3)


def test_static_clips_still_complete_transitions():
	player = GraphPlayback(graph(static=True))
	player.advance(0.125, [1])
	np.testing.assert_allclose(player.sample()[1], [[0, 0]])
	np.testing.assert_allclose(player.sample()[2], [[0.5, 0.5]])
	player.advance(0.125, [1])
	assert player.state == 1 and player.target_state is None and player.time == 0


def test_reset_cancels_transition_and_restores_speed_and_wrapped_clock():
	player = GraphPlayback(graph(sync=True))
	player.advance(0.1, [1])
	player.reset(1, 2.75)
	assert player.state == 1 and player.target_state is None and player.speed == 1
	np.testing.assert_allclose(player.sample()[1], [[0.75, 0.75]])
	player.reset()
	assert player.state == 0 and player.time == 0


@pytest.mark.parametrize("sync", [False, True])
def test_preview_poses_and_clocks_match_existing_ecs_playback(sync):
	from graaal.ECS import ECS
	from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
	from graaal.EcsAdapters.EcsAnimationStateMachine import AnimationState, EcsAnimationStateAdapter
	from graaal.Animation.SkeletalAnimation import model_matrices, skin_matrices

	compiled = graph(sync=sync, duration=0.75)
	rig = compiled.plan.skeleton
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, AnimationState)
	entities = world.create(1)
	world.add(entities, SkeletonRef(0), ClipPlayer(0, time=0.5), AnimationState(0))
	control = EcsAnimationStateAdapter(world, compiled.machine, compiled.state_clips, sync_mappings=compiled.sync_mappings)
	animation = EcsAnimationAdapter(world, {0: rig}, compiled.clips, sync_mappings=compiled.sync_mappings)
	player = GraphPlayback(compiled)
	player.reset(time=0.5)
	for dt, parameter in ((0.125, 1), (0.25, -1), (0.5, -1), (0.125, -1), (1.25, 0)):
		control.update([[parameter]])
		batch, = animation.update(dt)
		player.advance(dt, [parameter])
		pose = compiled.plan.evaluate(compiled.clips, [player.sample()], np.empty((1, 0)))
		matrices = skin_matrices(rig, model_matrices(rig, *pose))
		np.testing.assert_allclose(matrices, batch.matrices, atol=1e-6)
		np.testing.assert_allclose(player.time, world.get(ClipPlayer)[entities].time[0], atol=1e-6)
		np.testing.assert_allclose(player.speed, world.get(ClipPlayer)[entities].speed[0], atol=1e-6)


@pytest.mark.parametrize("dt,parameters", [(-1, [0]), (np.nan, [0]), (np.inf, [0]), ([1], [0]), (0.1, [np.nan]), (0.1, []), (0.1, [[0]])])
def test_invalid_updates_preserve_active_transition(dt, parameters):
	player = GraphPlayback(graph())
	player.advance(0.1, [1])
	before = playback_snapshot(player)
	with pytest.raises(ValueError): player.advance(dt, parameters)
	assert playback_snapshot(player) == before


def blend_graph(initial="idle", *, static=False, sync=False):
	base = graph()
	clips = {}
	for name, clip, axis in zip(("idle", "walk", "other"), base.clips.values(), np.eye(3)):
		rotation = np.tile(np.r_[axis * np.sin(0.9), np.cos(0.9)], (2, 1, 1))
		clips[name] = Clip(clip.translation, rotation, clip.scale, fps=2 / 3 if name == "idle" else clip.fps)
	if static: clips = {name: Clip(clip.translation[:1], clip.rotation[:1], clip.scale[:1]) for name, clip in clips.items()}
	data = {
		"version": 1, "parameters": {"move": 0, "amount": 0.5},
		"states": {"idle": {"clip": "idle"}, "motion": {"blend_space": "motion"}, "other": {"blend_space": "other"}},
		"blend_spaces": {name: {"parameter": "amount", "points": [{"position": 0, "clip": a}, {"position": 1, "clip": b}]} for name, a, b in (("motion", "idle", "walk"), ("other", "walk", "other"))},
		"initial_state": initial,
		"transitions": [
			{"source": "idle", "target": "motion", "conditions": [{"parameter": "move", "comparison": ">", "threshold": 0}], "duration": 0.25},
			{"source": "motion", "target": "other", "conditions": [{"parameter": "move", "comparison": "<", "threshold": 0}], "duration": 0.5},
			{"source": "other", "target": "idle", "conditions": [{"parameter": "move", "comparison": "==", "threshold": 0}], "duration": 0.25},
		],
		"sample_slots": ["base"], "tasks": [{"name": "output", "op": "sample", "slot": "base"}],
	}
	tracks = None
	if sync:
		data["sync_mappings"] = [{"source": "idle", "target": "walk"}]
		tracks = {name: EventTrack(clips[name].duration, [0, right], ["left", "right"]) for name, right in (("idle", 0.5), ("walk", 1.25))}
	return AnimationGraphDefinition(data).compile(base.plan.skeleton, clips, sync_tracks=tracks)


def test_clip_to_blend_fade_evaluates_nested_local_poses_and_live_parameters():
	compiled = blend_graph()
	player = GraphPlayback(compiled)
	player.reset(time=0.7)
	player.advance(0.125, [1, 0.25])
	base = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[0, 0]], [[0.825, 0.825]], [[1, 0]])
	target = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[0, 1]], [[0.125, 0.125]], [[0.75, 0.25]])
	expected = blend_poses(compiled.plan.skeleton, base, target, [0.5])
	before = player.pose([1, 0.25])
	for actual, reference in zip(before, expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	# Parameter changes affect the blend while preserving the selected transition.
	changed = player.pose([-1, 1])
	assert player.state == 0 and player.target_state == 1 and not np.allclose(changed[1], before[1])
	player.advance(0.375, [-1, 1])
	assert player.state == 1 and player.target_state is None and player.time == 0.5
	np.testing.assert_allclose(player.pose([-1, 1])[0], [[[4.5, 0, 0]]])
	for actual, reference in zip(before, expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	with pytest.raises(ValueError, match="pose"): player.sample()


def test_blend_to_blend_and_clip_fades_preserve_independent_clocks():
	compiled = blend_graph("motion")
	player = GraphPlayback(compiled)
	player.reset(time=2.75)
	player.advance(0.25, [-1, 0.5])
	assert player.state == 1 and player.target_state == 2 and player.time == 3 and player.target_time == 0.25
	base = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[0, 1]], [[0, 1]], [[0.5, 0.5]])
	target = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[1, 2]], [[0.25, 0.25]], [[0.5, 0.5]])
	expected = blend_poses(compiled.plan.skeleton, base, target, [0.5])
	for actual, reference in zip(player.pose([-1, 0.5]), expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	player.advance(0.5, [0, 0.5])
	assert player.state == 2 and player.target_state is None and player.time == 0.75
	player.advance(0.125, [0, 0.5])
	assert player.target_state == 0 and player.target_time == 0.125
	player.advance(0.375, [0, 0.5])
	assert player.state == 0 and player.target_state is None and player.time == 0.5


def test_blend_clock_does_not_restart_other_clips_at_a_loop_or_parameter_change():
	compiled = blend_graph("motion")
	player = GraphPlayback(compiled)
	player.reset(time=1.9)
	player.advance(0.2, [0, 0.5])
	np.testing.assert_allclose(player.time, 2.1)
	np.testing.assert_allclose(state_sample(compiled, 1, player.time, [0, 0.5])[1], [[0.6, 0.1]])
	player.advance(0.25, [0, 1])
	np.testing.assert_allclose(state_sample(compiled, 1, player.time, [0, 1])[1], [[0.85, 0.35]])
	player.reset(1, 5.5)
	assert player.time == 5.5 and player.speed == 1
	np.testing.assert_allclose(state_sample(compiled, 1, player.time, [0, -5])[2], [[1, 0]])
	np.testing.assert_allclose(state_sample(compiled, 1, player.time, [0, 5])[2], [[0, 1]])


def test_static_blend_states_complete_fades_and_reject_invalid_updates_atomically():
	player = GraphPlayback(blend_graph(static=True))
	player.advance(0.125, [1, 0.5])
	np.testing.assert_allclose(player.pose([1, 0.5])[0], [[[1, 0, 0]]])
	before = playback_snapshot(player)
	for dt, parameters in ((-1, [1, 0.5]), (0.1, [1]), (0.1, [np.nan, 0])):
		with pytest.raises(ValueError): player.advance(dt, parameters)
		assert playback_snapshot(player) == before
	player.advance(0.125, [1, 0.5])
	assert player.state == 1 and player.target_state is None and player.time == 0.25
	np.testing.assert_allclose(player.pose([1, 0.5])[0], [[[2, 0, 0]]])


def test_clip_sync_mapping_does_not_retime_a_blend_state():
	player = GraphPlayback(blend_graph(sync=True))
	player.reset(time=0.5)
	player.advance(0.125, [1, 1])
	assert player.target_time == 0.125 and player.target_speed == 1
	player.advance(0.375, [1, 1])
	assert player.state == 1 and player.time == 0.5 and player.speed == 1


def grid_graph(initial="grid", *, static=False, x_positions=(0, 1), y_positions=(-1, 1)):
	base = blend_graph(static=static)
	clips = {name: clip for name, clip in zip(("idle", "walk", "other"), base.clips.values())}
	clip = clips["other"]
	clips["extra"] = Clip(clip.translation + [4, 0, 0], np.tile([0, 0, 0, -1], (len(clip.translation), 1, 1)), clip.scale, fps=1 / 3)
	cells = [["idle", "walk"], ["other", "extra"]]
	data = {
		"version": 1, "parameters": {"move": 0, "x": 0.25, "y": 0},
		"states": {"idle": {"clip": "idle"}, "line": {"blend_space": "motion"}, "grid": {"blend_space_2d": "motion"}, "other": {"blend_space_2d": "reverse"}},
		"blend_spaces": {"motion": {"parameter": "x", "points": [{"position": 0, "clip": "idle"}, {"position": 1, "clip": "walk"}]}},
		"blend_spaces_2d": {name: {"x_parameter": "x", "y_parameter": "y", "x_positions": list(x_positions), "y_positions": list(y_positions), "clips": [row[:len(x_positions)] for row in rows[:len(y_positions)]]} for name, rows in (("motion", cells), ("reverse", cells[::-1]))},
		"initial_state": initial,
		"transitions": [
			{"source": a, "target": b, "conditions": [{"parameter": "move", "comparison": comparison, "threshold": 0}], "duration": duration}
			for a, b, comparison, duration in (("idle", "grid", ">", 0.5), ("grid", "line", ">", 0.25), ("grid", "other", "<", 0.5), ("other", "idle", "==", 0.25), ("line", "grid", "<", 0.25))
		],
		"sample_slots": ["base"], "tasks": [{"name": "output", "op": "sample", "slot": "base"}],
	}
	return AnimationGraphDefinition(data).compile(base.plan.skeleton, clips)


def test_2d_state_preserves_nested_quaternion_blending_and_each_clip_loop():
	compiled = grid_graph()
	player = GraphPlayback(compiled)
	player.reset(time=1.75)
	player.advance(0.5, [0, 0.25, 0])
	assert player.time == 2.25
	lower = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[0, 1]], [[0.75, 0.25]], [[0.75, 0.25]])
	upper = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[2, 3]], [[0.25, 2.25]], [[0.75, 0.25]])
	expected = blend_poses(compiled.plan.skeleton, lower, upper, [0.5])
	before = player.pose([0, 0.25, 0])
	for actual, reference in zip(before, expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	# Clamp to a different corner without resetting any clocks.
	player.advance(0, [0, 5, 5])
	np.testing.assert_allclose(player.pose([0, 5, 5])[0], [[[13.5, 0, 0]]])
	for actual, reference in zip(before, expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	with pytest.raises(ValueError, match="pose"): player.sample()
	with pytest.raises(ValueError, match="state_pose"): state_sample(compiled, 2, 0, [0, 0, 0])


@pytest.mark.parametrize("source,move,target,step", [("idle", 1, "grid", 0.25), ("line", -1, "grid", 0.125), ("grid", 1, "line", 0.125), ("grid", -1, "other", 0.25), ("other", 0, "idle", 0.125)])
def test_2d_fades_with_clip_1d_and_2d_states_promote_without_restarting(source, move, target, step):
	compiled = grid_graph(source)
	player = GraphPlayback(compiled)
	player.reset(time=2.75)
	start = player.time
	parameters = [move, 0.25, 0]
	player.advance(step, parameters)
	assert player.target_state == compiled.state_ids[target] and player.target_time == step
	expected = blend_poses(compiled.plan.skeleton, state_pose(compiled, player.state, start + step, parameters), state_pose(compiled, player.target_state, step, parameters), [0.5])
	for actual, reference in zip(player.pose(parameters), expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	# Changed conditions do not interrupt the fade or select a second edge.
	player.advance(step + 0.125, [-move, 0.75, 1])
	assert player.state == compiled.state_ids[target] and player.target_state is None
	np.testing.assert_allclose(player.time, 2 * step + 0.125)
	for actual, reference in zip(player.pose([-move, 0.75, 1]), state_pose(compiled, player.state, player.time, [-move, 0.75, 1])):
		np.testing.assert_allclose(actual, reference, atol=1e-6)


@pytest.mark.parametrize("x,y,expected", [((0, 1), (-1, 1), 5), ((0, 1), (-1,), 1), ((0,), (-1, 1), 4), ((0,), (-1,), 0)])
def test_static_and_degenerate_2d_states(x, y, expected):
	player = GraphPlayback(grid_graph(static=True, x_positions=x, y_positions=y))
	player.advance(2.1, [0, 0.25, 0])
	np.testing.assert_allclose(player.pose([0, 0.25, 0])[0], [[[expected, 0, 0]]])
	assert player.time == 2.1
	player.advance(0.125, [1, 0.25, 0])
	before = playback_snapshot(player)
	for dt, parameters in ((-1, [1, 0, 0]), (0.1, [1, 0]), (0.1, [1, np.nan, 0])):
		with pytest.raises(ValueError): player.advance(dt, parameters)
		assert playback_snapshot(player) == before
	player.reset(2, 5.5)
	assert player.time == 5.5 and player.target_state is None and player.speed == 1


def synchronized_graph(dimension, *, wide=False):
	base = grid_graph()
	clips = {name: clip for name, clip in zip(("idle", "walk", "other", "extra"), base.clips.values())}
	tracks = {name: EventTrack(clips[name].duration, times, names) for name, times, names in (("idle", [0, 0.5], ["left", "right"]), ("walk", [0.25, 1.25], ["left", "right"]), ("other", [0, 0.5], ["left", "right"]), ("extra", [0.5, 2], ["right", "left"]))}
	data = {
		"version": 1, "parameters": {"move": 0, "x": 0.25, "y": 0},
		"initial_state": "source", "sample_slots": ["base"], "tasks": [{"name": "output", "op": "sample", "slot": "base"}],
		"transitions": [{"source": "source", "target": "target", "conditions": [{"parameter": "move", "comparison": "<", "threshold": 0}], "duration": 0.5}],
		"sync_mappings": [{"source": "idle", "target": "walk"}],
	}
	field, state_field = ("blend_spaces", "blend_space") if dimension == 1 else ("blend_spaces_2d", "blend_space_2d")
	data["states"] = {"source": {state_field: "a"}, "target": {state_field: "b"}}
	if dimension == 1:
		data[field] = {name: {"parameter": "x", "points": [{"position": 0, "clip": a}, {"position": 1, "clip": b}], "sync_reference": a} for name, a, b in (("a", "idle", "walk"), ("b", "walk", "other"))}
	else:
		data[field] = {name: {"x_parameter": "x", "y_parameter": "y", "x_positions": [0, 1], "y_positions": [-1, 1], "clips": cells, "sync_reference": reference} for name, reference, cells in (("a", "idle", [["idle", "walk"], ["other", "extra"]]), ("b", "walk", [["other", "extra"], ["idle", "walk"]]))}
	if wide:
		if dimension == 1: data[field]["a"]["points"].append({"position": 2, "clip": "other"})
		else:
			data[field]["a"]["x_positions"].append(2)
			data[field]["a"]["clips"][0].append("other")
			data[field]["a"]["clips"][1].append("walk")
	return AnimationGraphDefinition(data).compile(base.plan.skeleton, clips, sync_tracks=tracks)


@pytest.mark.parametrize("dimension", [1, 2])
def test_synchronized_blends_keep_reference_clock_across_loops_and_parameter_changes(dimension):
	compiled = synchronized_graph(dimension)
	player = GraphPlayback(compiled)
	player.reset(time=1.75)
	player.advance(0.5, [0, 0.25, 0])
	assert player.time == 2.25
	lower = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[0, 1]], [[0.75, 1.5]], [[0.75, 0.25]])
	if dimension == 1: expected = lower
	else:
		upper = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[2, 3]], [[0.625, 0.875]], [[0.75, 0.25]])
		expected = blend_poses(compiled.plan.skeleton, lower, upper, [0.5])
	for actual, reference in zip(player.pose([0, 0.25, 0]), expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	player.advance(0, [0, 1, 1])
	assert player.time == 2.25
	if dimension == 1:
		np.testing.assert_allclose(state_sample(compiled, 0, player.time, [0, 1, 1])[1], [[0.75, 1.5]])
	else:
		# The extra clip is at reference marker phase, rather than time 2.25.
		np.testing.assert_allclose(player.pose([0, 1, 1])[0], [[[12 + 2 * 0.875 / 3, 0, 0]]])
	player.reset(time=1000 * 1.5 + 0.75)
	for actual, reference in zip(player.pose([0, 0.25, 0]), expected): np.testing.assert_allclose(actual, reference, atol=1e-6)


@pytest.mark.parametrize("dimension", [1, 2])
def test_state_fades_sync_reference_clocks_and_retain_target_speed(dimension):
	compiled = synchronized_graph(dimension)
	player = GraphPlayback(compiled)
	player.reset(time=0.5)
	player.advance(0.25, [-1, 0.25, 0])
	assert player.target_time == 1.5 and player.target_speed == 1
	expected = blend_poses(compiled.plan.skeleton, state_pose(compiled, 0, 0.75, [-1, 0.25, 0]), state_pose(compiled, 1, 1.5, [-1, 0.25, 0]), [0.5])
	for actual, reference in zip(player.pose([-1, 0.25, 0]), expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	player.advance(0.5, [0, 1, 1])
	assert player.state == 1 and player.target_state is None and player.time == 2 and player.speed == 1
	player.advance(0.25, [0, 1, 1])
	assert player.time == 2.25
	player.reset(1, 5.5)
	assert player.time == 5.5 and player.speed == 1


@pytest.mark.parametrize("dimension", [1, 2])
def test_reference_keeps_leading_when_it_is_outside_the_selected_cell(dimension):
	compiled = synchronized_graph(dimension, wide=True)
	player = GraphPlayback(compiled)
	player.reset(time=2.25)
	player.advance(0, [0, 1.5, 0])
	lower = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[1, 2]], [[1.5, 0.625]], [[0.5, 0.5]])
	if dimension == 1: expected = lower
	else:
		upper = sample_clip_pairs(compiled.plan.skeleton, compiled.clips, [[3, 1]], [[0.875, 1.5]], [[0.5, 0.5]])
		expected = blend_poses(compiled.plan.skeleton, lower, upper, [0.5])
	for actual, reference in zip(player.pose([0, 1.5, 0]), expected): np.testing.assert_allclose(actual, reference, atol=1e-6)
	assert player.time == 2.25 and player.speed == 1
