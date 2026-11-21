import numpy as np
import pytest

from graaal.Animation.AnimationEvents import EventTrack
from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.AnimationStateMachine import AnimationStateMachine
from graaal.Animation.AnimationSync import SyncMapping
from graaal.ECS import ECS
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.EcsAdapters.EcsAnimationStateMachine import AnimationState, EcsAnimationStateAdapter
from graaal.Animation.RootMotion import RootMotion
from graaal.Animation.SkeletalAnimation import Clip, Skeleton


def setup():
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, AnimationState)
	entities = world.create(2)
	world.add(entities, SkeletonRef(0), ClipPlayer(0, time=0.5, extract_root_motion=True), AnimationState(0))
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clips = {i: Clip([[[0, 0, 0]], [[duration, 0, 0]]], [[[0, 0, 0, 1]]] * 2, np.ones((2, 1, 3)), fps=1 / duration) for i, duration in ((0, 2), (1, 4))}
	tracks = {0: EventTrack(2, [0, 1], ["left", "right"]), 1: EventTrack(4, [0, 3], ["left", "right"])}
	mappings = {(0, 1): SyncMapping(tracks[0], tracks[1])}
	control = EcsAnimationStateAdapter(world, AnimationStateMachine([(0, 1, 0, ">", 0, 1)]), {0: 0, 1: 1}, sync_mappings=mappings)
	playback = EcsAnimationAdapter(world, {0: rig}, clips, event_tracks=tracks, root_motions={(0, i): RootMotion(rig, clip) for i, clip in clips.items()}, sync_mappings=mappings)
	return world, entities, control, playback


def test_sync_fade_across_markers_events_and_root_motion_remainder():
	world, entities, control, playback = setup()
	control.update([[1], [0]])
	players = world.get(ClipPlayer)[entities]
	assert players.target_time[0] == pytest.approx(1.5)
	batch, = playback.update(1.5)
	assert players.clip_id.tolist() == [1, 0]
	assert players.time[0] == pytest.approx(0)
	assert players.speed[0] == pytest.approx(1)
	target_events = [event.name for event in playback.events if event.entity == entities[0] and event.role == "target"]
	assert target_events == ["right", "left"]
	requests = {int(entity): matrix for group in playback.root_motion_batches for entity, matrix in zip(group.entities, group.matrices)}
	assert requests[int(entities[0])][0, 3] == pytest.approx(2)
	np.testing.assert_allclose(batch.matrices[:, 0, 0, 3], 0)


def test_compiled_graph_mappings_feed_existing_state_and_playback_adapters():
	world, entities, _, original = setup()
	data = {
		"version": 1,
		"parameters": {"move": 0},
		"states": {"idle": {"clip": "idle"}, "walk": {"clip": "walk"}},
		"initial_state": "idle",
		"transitions": [{"source": "idle", "target": "walk", "conditions": [{"parameter": "move", "comparison": ">", "threshold": 0}], "duration": 1}],
		"sample_slots": ["base"],
		"tasks": [{"name": "base", "op": "sample", "slot": "base"}],
		"sync_mappings": [{"source": "idle", "target": "walk"}],
	}
	graph = AnimationGraphDefinition(data).compile(original.skeletons[0], {"idle": original.clips[0], "walk": original.clips[1]}, sync_tracks={"idle": original.event_tracks[0], "walk": original.event_tracks[1]})
	control = EcsAnimationStateAdapter(world, graph.machine, graph.state_clips, sync_mappings=graph.sync_mappings)
	playback = EcsAnimationAdapter(world, original.skeletons, graph.clips, event_tracks=original.event_tracks, root_motions=original.root_motions, sync_mappings=graph.sync_mappings)
	control.update([[1], [0]])
	players = world.get(ClipPlayer)[entities]
	assert players.target_time[0] == pytest.approx(1.5)
	playback.update(0.75)
	assert players.target_time[0] == pytest.approx(3.25)
	assert [(event.role, event.name) for event in playback.events if event.entity == entities[0]] == [("source", "right"), ("target", "right")]
	playback.update(0.25)
	control.update([[0], [0]])
	assert players.clip_id[0] == graph.clip_ids["walk"]
	assert world.get(AnimationState)[entities].state[0] == graph.state_ids["walk"]


def test_sync_reverse_pause_and_continuation():
	world, entities, control, playback = setup()
	players = world.get(ClipPlayer)[entities]
	players.assign(speed=[-1, 1], playing=[True, False])
	control.update([[1], [1]])
	playback.update(0.25)
	assert players.target_time[0] == pytest.approx(0.75)
	assert players.target_time[1] == pytest.approx(1.5)
	assert players.target_speed[0] == pytest.approx(-3)
	playback.update(0.5)
	assert players.target_time[0] == pytest.approx(3.75)
	assert players.target_speed[0] == pytest.approx(-1)


def test_synchronized_fade_stays_aligned_across_multiple_updates():
	world, entities, control, playback = setup()
	control.update([[1], [0]])
	players = world.get(ClipPlayer)[entities]
	playback.update(0.25)
	assert players.target_time[0] == pytest.approx(2.25)
	playback.update(0.5)
	assert players.target_time[0] == pytest.approx(3.25)
	playback.update(0.25)
	assert players.clip_id[0] == 1
	assert players.time[0] == pytest.approx(3.5)
	assert players.speed[0] == pytest.approx(1)


def test_invalid_sync_configuration_does_not_advance_any_player():
	world, entities, control, playback = setup()
	control.update([[1], [0]])
	players = world.get(ClipPlayer)[entities]
	players.take([0]).assign(target_time=0)
	with pytest.raises(ValueError, match="not aligned"): playback.update(0.5)
	np.testing.assert_allclose(players.time, [0.5, 0.5])
	players.take([0]).assign(target_time=1.5, target_loop=False)
	with pytest.raises(ValueError, match="looping"): playback.update(0.5)
	np.testing.assert_allclose(players.time, [0.5, 0.5])


def test_state_adapter_rejects_nonlooping_sync_before_starting():
	world, entities, control, _ = setup()
	players = world.get(ClipPlayer)[entities]
	players.take([0]).assign(loop=False)
	with pytest.raises(ValueError, match="looping"): control.update([[1], [0]])
	assert not players.transitioning.any()
	assert not world.get(AnimationState)[entities].pending.any()


def test_long_synchronized_fade_does_not_accumulate_rounding_drift():
	world, entities, control, playback = setup()
	control.update([[1], [0]])
	players = world.get(ClipPlayer)[entities]
	players.take([0]).assign(transition_duration=1000)
	players.take([1]).assign(playing=False)
	counts = {"source": [], "target": []}
	for _ in range(5500):
		playback.update(1 / 60, evaluate_poses=False)
		for event in playback.events:
			if event.entity == int(entities[0]): counts[event.role].append(event.name)
	assert players.transitioning[0]
	mapping = playback.sync_mappings[(0, 1)]
	expected = mapping.map(players.time[:1])[0][0]
	assert players.target_time[0] == pytest.approx(expected, abs=5e-7)
	assert counts["source"] == counts["target"]


@pytest.mark.parametrize("speed,direction,step", [(1, -np.inf, 4e-8), (-1, np.inf, 8e-8)])
def test_sync_events_cross_marker_once_when_source_rounds_to_it(speed, direction, step):
	world, entities, control, playback = setup()
	players = world.get(ClipPlayer)[entities]
	players.take([0]).assign(time=np.nextafter(np.float32(1), np.float32(direction)), speed=speed)
	control.update([[1], [0]])
	playback.update(step, evaluate_poses=False)
	events = [(event.role, event.name) for event in playback.events if event.entity == int(entities[0])]
	assert events == [("source", "right"), ("target", "right")]
	playback.update(0, evaluate_poses=False)
	assert not playback.events
