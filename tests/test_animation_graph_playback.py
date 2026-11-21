"""Automatic authoring playback without Tk, ECS or a graphics device."""

import numpy as np
import pytest

from graaal.AnimationEvents import EventTrack
from graaal.AnimationGraph import AnimationGraphDefinition
from graaal.SkeletalAnimation import Clip, Skeleton
from tools.animation_graph_playback import GraphPlayback


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
	from graaal.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
	from graaal.EcsAnimationStateMachine import AnimationState, EcsAnimationStateAdapter
	from graaal.SkeletalAnimation import model_matrices, skin_matrices

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
	before = player.__dict__.copy()
	with pytest.raises(ValueError): player.advance(dt, parameters)
	assert player.__dict__ == before
