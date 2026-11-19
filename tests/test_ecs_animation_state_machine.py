import numpy as np
import pytest

from graaal.AnimationStateMachine import AnimationStateMachine
from graaal.ECS import ECS
from graaal.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.EcsAnimationStateMachine import AnimationState, EcsAnimationStateAdapter
from graaal.SkeletalAnimation import Clip, Skeleton


def setup():
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, AnimationState)
	entities = world.create(2)
	world.add(entities, SkeletonRef(0), ClipPlayer(10, loop=False), AnimationState(0))
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clips = {i: Clip([[[x, 0, 0]], [[x, 0, 0]]], np.tile([0, 0, 0, 1], (2, 1, 1)), np.ones((2, 1, 3)), fps=1) for i, x in ((10, 0), (11, 2))}
	machine = AnimationStateMachine([(0, 1, 0, ">", 0.5, 1), (1, 0, 0, "<", 0.2, 1)])
	return world, entities, EcsAnimationStateAdapter(world, machine, {0: 10, 1: 11}), EcsAnimationAdapter(world, {0: rig}, clips)


def test_transitions_use_existing_player_and_promote_after_completion():
	world, entities, control, playback = setup()
	np.testing.assert_array_equal(control.update([[1], [0]]), entities)
	players, states = world.get(ClipPlayer)[entities], world.get(AnimationState)[entities]
	assert players.transitioning.tolist() == [True, False]
	assert states.pending.tolist() == [True, False]
	batch, = playback.update(0.5)
	np.testing.assert_allclose(batch.matrices[:, 0, 0, 3], [1, 0])
	control.update([[0], [0]])
	assert states.state.tolist() == [0, 0]
	playback.update(0.5)
	assert players.clip_id.tolist() == [11, 10]
	control.update([[1], [0]])
	assert states.state.tolist() == [1, 0]
	assert states.pending.tolist() == [False, False]
	control.update([[0], [0]])
	assert states.pending.tolist() == [True, False]
	assert players.target_clip_id.tolist()[0] == 10


def test_completion_can_start_next_fade_and_pause_preserves_state():
	world, entities, control, playback = setup()
	world.get(ClipPlayer)[entities[:1]].assign(playing=False)
	control.update([[1], [0]])
	playback.update(2)
	control.update([[0], [0]])
	assert world.get(AnimationState)[entities[:1]].state.tolist() == [0]
	world.get(ClipPlayer)[entities[:1]].assign(playing=True)
	playback.update(1)
	control.update([[0], [0]])
	assert world.get(AnimationState)[entities[:1]].state.tolist() == [1]
	assert world.get(ClipPlayer)[entities[:1]].target_clip_id.tolist() == [10]


def test_preflight_rejects_invalid_decisions_and_inconsistent_players():
	world, entities, control, _ = setup()
	control.state_clips.pop(1)
	with pytest.raises(ValueError, match="No clip"): control.update([[1], [0]])
	assert not world.get(ClipPlayer)[entities].transitioning.any()
	assert not world.get(AnimationState)[entities].pending.any()
	control.state_clips[1] = 11
	world.get(ClipPlayer)[entities[:1]].assign(transitioning=True)
	with pytest.raises(ValueError, match="not owned"): control.update([[1], [0]])


def test_multiple_parameters_control_existing_ecs_crossfade():
	world, entities, control, playback = setup()
	control.machine = AnimationStateMachine([(0, 1, [(0, ">", 0.5), (1, "==", 1)], 1)])
	control.update([[1, 0], [1, 1]])
	assert world.get(ClipPlayer)[entities].transitioning.tolist() == [False, True]
	batch, = playback.update(0.5)
	np.testing.assert_allclose(batch.matrices[:, 0, 0, 3], [0, 1])


def test_entity_removal_and_handle_recycling():
	world, entities, control, playback = setup()
	control.update([[1], [0]])
	world.delete(entities[:1])
	new = world.create()[0]
	world.add(new, SkeletonRef(0), ClipPlayer(10), AnimationState(0))
	fetched = control.query.fetch().entities
	params = np.array([[int(entity == new)] for entity in fetched])
	control.update(params)
	assert world.get(AnimationState)[[new]].pending.tolist() == [True]
	assert int(entities[0]) not in np.concatenate([batch.entities for batch in playback.update(0)])
