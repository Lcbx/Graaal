"""Batched runtime graph playback and event delivery."""

from dataclasses import replace

import numpy as np
import pytest

from graaal.Animation.AnimationGraphPlayback import AnimationGraphPlayback
from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.AnimationEvents import EventTrack
from graaal.Animation.AnimationSync import SyncMapping
from graaal.Animation.RootMotion import RootMotion
from graaal.Animation.SkeletalAnimation import Clip, model_matrices, skin_matrices
from graaal.ECS import ECS
from graaal.EcsAdapters.EcsAnimationGraph import EcsAnimationGraphAdapter, GraphPlayer
from tests.test_animation_graph_playback import graph, grid_graph, synchronized_graph


FIELDS = ("states", "times", "speeds", "target_states", "target_times", "target_speeds", "elapsed", "durations")


def motion_tracks(compiled):
	return {key: RootMotion(compiled.plan.skeleton, clip) for key, clip in compiled.clips.items()}


@pytest.mark.parametrize("dimension", [1, 2])
def test_root_motion_blends_synced_clips_across_a_reference_loop(dimension):
	compiled = synchronized_graph(dimension)
	compiled = replace(compiled, clips={key: Clip(clip.translation, np.broadcast_to([0, 0, 0, 1], clip.rotation.shape), clip.scale, fps=clip.fps) for key, clip in compiled.clips.items()})
	player = AnimationGraphPlayback(compiled, 1, root_motions=motion_tracks(compiled))
	player.reset(times=1.25)
	parameters = [[0, 0.25, 0]]
	player.advance(0.5, parameters)
	# Each endpoint is in its own clip time, unwrapped through the seam.
	lower = 0.75 * (2 * 0.5 / 1.5) + 0.25 * 0.75
	upper = 0.75 * (2 * 0.375) + 0.25 * (2 * 1.125 / 3)
	expected = lower if dimension == 1 else (lower + upper) / 2
	np.testing.assert_allclose(player.root_motion[:, :3, 3], [[expected, 0, 0]], atol=1e-6)
	np.testing.assert_allclose(player.pose(parameters, in_place=True)[0], [[[0, 0, 0]]])
	player.advance(0, [[0, 1, -1]])
	np.testing.assert_allclose(player.root_motion[0], np.eye(4))


def test_synced_root_motion_blends_fade_and_composes_remaining_target_motion():
	compiled = graph(sync=True)
	player = AnimationGraphPlayback(compiled, 2, root_motions=motion_tracks(compiled))
	player.reset(times=0.75, speeds=[1, 0])
	player.advance(0.5, [[1], [1]])
	np.testing.assert_allclose(player.root_motion[:, :3, 3], [[0.875, 0, 0], [0, 0, 0]])
	np.testing.assert_allclose(player.times, [0.25, 1.25])
	assert not player.root_motion.flags.writeable
	player.reset(rows=[0])
	np.testing.assert_allclose(player.root_motion[0], np.eye(4))


def test_ecs_graph_remaps_fading_entities_and_event_cursors_by_full_handle():
	compiled = graph(duration=0.75)
	world = ECS()
	world.register(GraphPlayer)
	removed, kept, other_graph = world.create(3)
	world.add(removed, GraphPlayer(0))
	world.add(kept, GraphPlayer(0))
	world.add(other_graph, GraphPlayer(1))
	adapter = EcsAnimationGraphAdapter(world, 0, compiled, event_tracks={1: EventTrack(2, [0.5], ["step"])}, root_motions=motion_tracks(compiled))
	adapter.update(0.25, [[0], [1]])
	world.delete(removed)
	replacement, = world.create(1)
	world.add(replacement, GraphPlayer(0))
	assert replacement != removed
	with pytest.raises(ValueError, match="parameters"): adapter.update(0.25, [[0]])
	assert adapter.events == ()
	pose = adapter.update(0.25, [[0], [0]], in_place=True)
	rows = {owner: row for row, owner in enumerate(adapter.entities())}
	assert set(rows) == {kept, replacement}
	assert adapter.playback.target_times[rows[kept]] == 0.5
	assert adapter.playback.times[rows[replacement]] == 0.25
	assert [(event.entity, event.name, event.role) for event in adapter.events] == [(kept, "step", "target")]
	np.testing.assert_allclose(pose[0], 0)
	adapter.update(0)
	assert adapter.events == ()


def test_state_events_cross_multiple_loops_in_tick_order_without_replaying_start():
	compiled = graph()
	track = EventTrack(1, [0.5, 0, 0.5], ["left", "loop", "right"])
	tracks = {0: track}
	batch = AnimationGraphPlayback(compiled, 2, event_tracks=tracks)
	tracks.clear()
	track.names = ("changed",) * 3
	batch.reset(times=[0.5, 0.75], speeds=[2, 0])
	events = batch.advance([0.75, 10.0])
	assert [(event.row, event.state, event.name, event.role) for event in events] == [
		(0, 0, "loop", "source"), (0, 0, "left", "source"), (0, 0, "right", "source"), (0, 0, "loop", "source"),
	]
	np.testing.assert_allclose([event.offset for event in events], [0.25, 0.5, 0.5, 0.75])
	assert all(event.weight == 1 for event in events)
	assert batch.advance(0) == ()
	assert batch.advance(0.125) == ()


def test_clip_event_at_wrapped_tick_endpoint_fires_once():
	player = AnimationGraphPlayback(graph(), 1, event_tracks={0: EventTrack(1, [0.2], ["step"])})
	player.reset(times=0.9)
	assert [event.name for event in player.advance(0.3)] == ["step"]
	assert player.advance(0.01) == ()
	assert [event.name for event in player.advance(1)] == ["step"]


@pytest.mark.parametrize("fade_duration", [2.4, 3])
def test_target_event_cursor_survives_wrap_and_promotion(fade_duration):
	player = AnimationGraphPlayback(graph(duration=fade_duration), 1, event_tracks={
		1: EventTrack(2, [0.4], ["step"]),
	})
	events = player.advance(2.4, [[1]])
	assert [event.name for event in events] == ["step", "step"]
	np.testing.assert_allclose([event.offset for event in events], [0.4, 2.4])
	assert player.advance(0.01, [[1]]) == ()


def test_synchronized_loop_markers_share_exact_offset_with_source_first():
	compiled = graph(sync=True, duration=1)
	player = AnimationGraphPlayback(compiled, 1, event_tracks={
		0: EventTrack(1, [0, 0.5], ["source-wrap", "source-half"]),
		1: EventTrack(2, [0, 0.4999, 0.5, 0.5001], ["target-wrap", "before", "target-half", "after"]),
	})
	player.reset(times=0.9)
	events = player.advance(0.9, [[1]])
	assert [event.name for event in events] == ["source-wrap", "target-wrap", "before", "source-half", "target-half", "after"]
	assert events[0].offset == events[1].offset
	assert events[2].offset < events[3].offset == events[4].offset < events[5].offset


def test_coincident_sync_events_at_decimal_loop_endpoint_stay_in_same_tick():
	source = EventTrack(1, [0, 0.2], ["left", "right"])
	target = EventTrack(2, [0, 0.25], ["left", "right"])
	compiled = replace(graph(duration=1), sync_mappings={(0, 1): SyncMapping(source, target)})
	player = AnimationGraphPlayback(compiled, 1, event_tracks={
		0: EventTrack(1, [0.2], ["source"]),
		1: EventTrack(2, [0.25], ["target"]),
	})
	player.reset(times=0.9)
	events = player.advance(0.3, [[1]])
	assert [(event.name, event.offset) for event in events] == [("source", 0.3), ("target", 0.3)]
	assert player.advance(0.01, [[1]]) == ()


def test_fade_events_include_both_states_and_continue_target_past_promotion():
	compiled = graph(duration=0.5)
	batch = AnimationGraphPlayback(compiled, 1, event_tracks={
		0: EventTrack(1, [0.25, 0.5, 0.75], ["start", "middle", "end"]),
		1: EventTrack(2, [0, 0.25, 0.5, 0.625], ["entry", "middle", "promote", "after"]),
	})
	batch.reset(times=0.25)
	assert batch.advance(0, [[1]]) == ()
	events = batch.advance(0.625, [[0]])
	assert [(event.state, event.name, event.role) for event in events] == [
		(0, "middle", "source"), (1, "middle", "target"), (0, "end", "source"), (1, "promote", "target"), (1, "after", "target"),
	]
	np.testing.assert_allclose([event.offset for event in events], [0.25, 0.25, 0.5, 0.5, 0.625])
	np.testing.assert_allclose([event.weight for event in events], [0.5, 0.5, 0, 1, 1])
	assert batch.advance(0.125, [[0]]) == ()
	events = batch.advance(1.5, [[0]])
	assert [(event.name, event.role, event.offset) for event in events] == [("entry", "source", 1.25), ("middle", "source", 1.5)]


def test_synchronized_target_events_use_marker_segments_and_endpoint_rate_after_promotion():
	compiled = graph(sync=True, duration=0.5)
	batch = AnimationGraphPlayback(compiled, 1, event_tracks={
		1: EventTrack(2, [0.25, 0.5, 0.875, 1.25, 1.625], ["untraversed", "marker", "fast", "promote", "after"]),
	})
	batch.reset(times=0.25)
	events = batch.advance(0.625, [[1]])
	assert [event.name for event in events] == ["marker", "fast", "promote", "after"]
	np.testing.assert_allclose([event.offset for event in events], [0.25, 0.375, 0.5, 0.625])
	np.testing.assert_allclose([event.weight for event in events], [0.5, 0.75, 1, 1])
	assert batch.states[0] == 1
	assert batch.speeds[0] == 3
	assert batch.advance(0.125, [[0]]) == ()


@pytest.mark.parametrize("duration", [0.001, 0.007])
def test_sync_events_at_rounded_fade_endpoint_tie_and_are_delivered_once(duration):
	base = graph(duration=duration)
	# The inverse time at an exact forward sample can round in either direction.
	source = EventTrack(1, [0.123, 0.537], ["a", "b"])
	target = EventTrack(2, [0.361, 1.376], ["a", "b"])
	mapping = SyncMapping(source, target)
	compiled = replace(base, sync_mappings={(0, 1): mapping})
	event_time = mapping.map([duration])[0][0]
	tracks = {0: EventTrack(1, [duration], ["end"]), 1: EventTrack(2, [event_time], ["handoff"])}
	player = AnimationGraphPlayback(compiled, 1, event_tracks=tracks)
	combined = player.advance(2 * duration, [[1]])
	player.reset()
	split = player.advance(duration, [[1]])
	assert player.advance(duration, [[0]]) == ()
	assert combined == split
	assert [(event.name, event.offset, event.weight) for event in combined] == [("end", duration, 0), ("handoff", duration, 1)]


def test_synchronized_fade_reports_every_target_loop_before_promotion():
	compiled = graph(sync=True, duration=2.25)
	player = AnimationGraphPlayback(compiled, 1, event_tracks={1: EventTrack(2, [0, 0.25], ["wrap", "step"])})
	player.reset(times=0.875)
	events = player.advance(2.375, [[1]])
	assert [event.name for event in events] == ["wrap", "step"] * 3
	np.testing.assert_allclose([event.offset for event in events], [0.125, 0.375, 1.125, 1.375, 2.125, 2.375])
	assert events[-1].weight == 1


@pytest.mark.parametrize("dimension", [1, 2])
def test_synchronized_blend_state_tracks_follow_reference_off_cell_across_a_loop(dimension):
	compiled = synchronized_graph(dimension, wide=True)
	batch = AnimationGraphPlayback(compiled, 1, event_tracks={
		0: EventTrack(1.5, [0, 1.375], ["source-loop", "source-before"]),
		1: EventTrack(2, [0.25, 0.375], ["target-loop-phase", "target-after"]),
	})
	batch.reset(times=1.25)
	events = batch.advance(0.5, [[-1, 1.5, 0]])
	assert [(event.name, event.role) for event in events] == [
		("source-before", "source"), ("source-loop", "source"), ("target-loop-phase", "target"), ("target-after", "target"),
	]
	np.testing.assert_allclose([event.offset for event in events], [0.125, 0.25, 0.25, 0.3125])
	assert batch.advance(0.125, [[0, 0, -1]]) == ()


def test_unsynchronized_blend_events_use_authored_period_and_failed_updates_keep_clocks():
	compiled = grid_graph()
	batch = AnimationGraphPlayback(compiled, 1, event_tracks={2: EventTrack(0.75, [0], ["pulse"])})
	batch.reset(times=0.5)
	assert batch.advance(0.25, [[0, 1, 1]])[0].offset == 0.25
	batch.reset(speeds=np.finfo(np.float64).max)
	before = batch.times
	with pytest.raises(ValueError, match="overflow"): batch.advance(2)
	assert batch.times is before
	batch.reset(times=1.4)
	assert batch.advance(0.1, [[0, -1, -1]])[0].name == "pulse"


@pytest.mark.parametrize("tracks", [
	[(0, EventTrack(1, [], []))], {True: EventTrack(1, [], [])}, {"idle": EventTrack(1, [], [])},
	{99: EventTrack(1, [], [])}, {0: None}, {0: EventTrack(2, [], [])},
])
def test_graph_event_tracks_validate_state_ids_types_and_clock_duration(tracks):
	with pytest.raises(ValueError, match=r"event[ _]track"): AnimationGraphPlayback(graph(), 1, event_tracks=tracks)


def test_mixed_batch_advances_independent_clocks_and_blends_expected_poses():
	batch = AnimationGraphPlayback(grid_graph(static=True), 4)
	batch.reset([0, 1, 2, 3], [0.25, 1.75, 2, 2.75])
	parameters = [[1, 0.25, 0], [-1, 0.75, 1], [1, 0.5, -1], [0, 0, 0]]
	before = batch.pose(parameters)
	snapshots = {name: getattr(batch, name) for name in FIELDS}

	batch.advance([0.25, 0.125, 0.375, 0.125], parameters)

	np.testing.assert_array_equal(batch.states, [0, 1, 1, 3])
	np.testing.assert_array_equal(batch.target_states, [2, 2, -1, 0])
	np.testing.assert_allclose(batch.times, [0, 1.875, 0.375, 2.875])
	np.testing.assert_allclose(batch.target_times, [0.25, 0.125, 0, 0])
	np.testing.assert_allclose(batch.elapsed, [0.25, 0.125, 0, 0.125])
	actual = batch.pose(parameters)
	np.testing.assert_allclose(actual[0][:, 0, 0], [2.5, 7, 2, 2])
	assert all(not np.shares_memory(old, new) for old, new in zip(before, actual))
	assert all(not value.flags.writeable and not np.shares_memory(value, getattr(batch, name)) for name, value in snapshots.items())


def test_subset_reset_cancels_only_selected_fades_and_owns_its_inputs():
	compiled = grid_graph("idle")
	batch = AnimationGraphPlayback(compiled, 4)
	batch.advance(0.125, np.tile([1, 0.25, 0], (4, 1)))
	snapshots = {name: getattr(batch, name).copy() for name in FIELDS}
	states, times, speeds = np.array([2, 0]), np.array([5.5, 5.5]), np.array([0, 2.0])
	batch.reset(states, times, speeds=speeds, rows=[3, 1])
	states[:], times[:], speeds[:] = 0, 0, 1
	for name in FIELDS: np.testing.assert_array_equal(getattr(batch, name)[[0, 2]], snapshots[name][[0, 2]])
	np.testing.assert_array_equal(batch.states, [0, 0, 0, 2])
	np.testing.assert_allclose(batch.times[[1, 3]], [1, 5.5])
	np.testing.assert_allclose(batch.speeds[[1, 3]], [2, 0])
	np.testing.assert_array_equal(batch.target_states, [2, -1, 2, -1])
	batch.advance([0, 0.25, 0, 0.25])
	assert batch.times[1] == 0 and batch.times[3] == 5.5
	assert batch.elapsed[0] == 0.125 and batch.elapsed[2] == 0.125
	batch.reset()
	np.testing.assert_array_equal(batch.states, 0)
	np.testing.assert_array_equal(batch.speeds, 1)
	np.testing.assert_array_equal(batch.target_states, -1)


@pytest.mark.parametrize("count", [0, 1, 32])
def test_default_parameters_and_empty_batches_compose_with_pose_plan_and_skinning(count):
	compiled = grid_graph()
	batch = AnimationGraphPlayback(compiled, count)
	batch.advance(2.25)
	local = batch.pose()
	weights = np.empty((count, 0))
	output = compiled.input_plan.evaluate(compiled.clips, (), weights, poses=[local])
	for actual, reference in zip(output, local): np.testing.assert_allclose(actual, reference, atol=1e-6)
	matrices = skin_matrices(compiled.plan.skeleton, model_matrices(compiled.plan.skeleton, *output))
	assert matrices.shape == (count, 1, 3, 4)


def test_nonunit_speeds_sync_promotion_and_overshoot_use_per_row_clock_rates():
	compiled = graph(sync=True, duration=0.25)
	batch = AnimationGraphPlayback(compiled, 3)
	batch.reset(times=0.5, speeds=[0, 1, 2])
	batch.advance([0.5, 0.5, 0.125], [[1], [1], [1]])
	np.testing.assert_array_equal(batch.states, [1, 1, 0])
	np.testing.assert_array_equal(batch.target_states, [-1, -1, 1])
	np.testing.assert_allclose(batch.times, [0.5, 0, 0.75])
	np.testing.assert_allclose(batch.speeds, [0, 3, 2])
	assert batch.target_speeds[2] == 6 and batch.target_times[2] == 1.25


@pytest.mark.parametrize("dt,parameters", [(-1, None), (np.inf, None), ([0.1], None), ([0.1, np.nan], None), (0.1, [[0, 0, 0]]), (0.1, np.full((2, 3), np.inf))])
def test_invalid_advances_preserve_entire_batch(dt, parameters):
	batch = AnimationGraphPlayback(grid_graph("idle"), 2)
	batch.advance(0.125, [[1, 0, 0], [1, 0, 0]])
	before = {name: getattr(batch, name) for name in FIELDS}
	with pytest.raises(ValueError): batch.advance(dt, parameters)
	assert all(getattr(batch, name) is value for name, value in before.items())


@pytest.mark.parametrize("kwargs", [
	{"states": -1}, {"states": 99}, {"states": True}, {"states": 1.0}, {"states": [0]},
	{"times": -1}, {"times": [0, np.inf]}, {"speeds": -1}, {"speeds": np.nan},
	{"rows": [0, 0]}, {"rows": [-1]}, {"rows": [2]}, {"rows": [True]}, {"rows": [[0]]},
])
def test_invalid_resets_preserve_entire_batch(kwargs):
	batch = AnimationGraphPlayback(grid_graph(), 2)
	before = {name: getattr(batch, name) for name in FIELDS}
	with pytest.raises(ValueError): batch.reset(**kwargs)
	assert all(getattr(batch, name) is value for name, value in before.items())


@pytest.mark.parametrize("count", [-1, 1.0, True, [1]])
def test_invalid_batch_counts(count):
	with pytest.raises(ValueError): AnimationGraphPlayback(grid_graph(), count)


def test_overflow_during_one_instance_update_preserves_all_clocks():
	batch = AnimationGraphPlayback(grid_graph(), 2)
	batch.reset(speeds=[1, np.finfo(np.float64).max])
	before = {name: getattr(batch, name) for name in FIELDS}
	with pytest.raises(ValueError, match="overflow"): batch.advance(2)
	assert all(getattr(batch, name) is value for name, value in before.items())
	with pytest.raises(ValueError): batch.pose([[0, np.nan, 0], [0, 0, 0]])


def test_layered_batch_example_keeps_root_blending_separate_from_upper_body_mask():
	from scenes.animation_graph_batch_example import create_scene, step

	player, parameters = create_scene()
	first = step(player, parameters, 0.25)
	np.testing.assert_allclose(first[:, 0, 0, 3], [0.0625, 0.1875, 0])
	np.testing.assert_allclose(first[:, 1, 1, 3], [1, 3, 2])
	second = step(player, parameters, 0.25)
	np.testing.assert_array_equal(player.states, [1, 1, 0])
	np.testing.assert_allclose(second[:, 0, 0, 3], [0.25, 0.75, 0])
	np.testing.assert_allclose(first[:, 0, 0, 3], [0.0625, 0.1875, 0])


def test_empty_subset_reset_is_a_noop_and_empty_instance_rows_are_accepted():
	batch = AnimationGraphPlayback(grid_graph(), 2)
	batch.advance(0.25)
	before = {name: getattr(batch, name).copy() for name in FIELDS}
	batch.reset(states=[], rows=[])
	for name, values in before.items(): np.testing.assert_array_equal(getattr(batch, name), values)
	empty = AnimationGraphPlayback(grid_graph(), 0)
	empty.reset(states=[], times=[], speeds=[])
	empty.advance([])
	assert empty.pose()[0].shape == (0, 1, 3)
	with pytest.raises(ValueError): empty.advance(np.inf)
	with pytest.raises(ValueError): empty.reset(states=99)
	with pytest.raises(ValueError): empty.reset(times=-1)
	with pytest.raises(ValueError): empty.reset(speeds=np.nan)


def test_parameterless_graph_uses_empty_parameter_rows_without_special_handling():
	base = grid_graph(static=True)
	compiled = AnimationGraphDefinition({
		"version": 1, "states": {"idle": {"clip": "idle"}}, "initial_state": "idle",
		"sample_slots": ["base"], "tasks": [{"name": "output", "op": "sample", "slot": "base"}],
	}).compile(base.plan.skeleton, {"idle": base.clips[0]})
	batch = AnimationGraphPlayback(compiled, 2)
	batch.advance(1000)
	np.testing.assert_array_equal(batch.times, 0)
	np.testing.assert_allclose(batch.pose()[0], np.zeros((2, 1, 3)))
