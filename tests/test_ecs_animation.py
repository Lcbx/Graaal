import numpy as np
import pytest

from graaal.ECS import ECS, Scheduler, columns
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.Animation.SkeletalAnimation import Clip, Skeleton


def resources():
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[2, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	return rig, clip


def test_ecs_batched_playback_seek_reverse_pause_and_clamp():
	rig, clip = resources()
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	entities = world.create(4)
	world.add(entities, SkeletonRef(7), ClipPlayer(3))
	world.get(ClipPlayer)[entities].assign(time=[0.8, 0.2, 0.9, 0.7], speed=[1, -1, 1, 1], loop=[True, True, False, False], playing=[True, True, True, False])
	adapter = EcsAnimationAdapter(world, {7: rig}, {3: clip})
	batch, = adapter.update(0.5)
	np.testing.assert_array_equal(batch.entities, entities)
	np.testing.assert_allclose(world.get(ClipPlayer)[entities].time, [0.3, 0.7, 1, 0.7], atol=1e-6)
	np.testing.assert_allclose(batch.matrices[:, 0, 0, 3], [0.6, 1.4, 2, 1.4], atol=1e-6)
	world.get(ClipPlayer)[entities[:1]].assign(time=0.5, speed=0)
	assert adapter.update(0)[0].matrices[0, 0, 0, 3] == pytest.approx(1)


def test_ecs_lifecycle_recycling_and_scheduler():
	rig, clip = resources()
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	ids = world.create(2)
	world.add(ids, SkeletonRef(1), ClipPlayer(2))
	adapter = EcsAnimationAdapter(world, {1: rig}, {2: clip})
	schedule = Scheduler(world)
	schedule.add("animation", adapter.evaluate, phase="fixed_update")
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	assert set(adapter.batches[0].entities) == set(ids)
	world.delete(ids[:1])
	new = world.create()[0]
	world.add(new, SkeletonRef(1), ClipPlayer(2, time=0.5))
	batch, = adapter.update(0)
	assert int(ids[0]) not in batch.entities
	assert int(new) in batch.entities
	assert batch.matrices[np.flatnonzero(batch.entities == new)[0], 0, 0, 3] == pytest.approx(1)
	world.remove(ids[1:], ClipPlayer)
	assert adapter.update(0)[0].entities.tolist() == [int(new)]
	world.delete(new)
	assert adapter.update(0) == []


def test_ecs_invalid_resources_do_not_advance_any_player():
	rig, clip = resources()
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	ids = world.create(2)
	world.add(ids, SkeletonRef(1), ClipPlayer(2))
	world.get(ClipPlayer)[ids].assign(clip_id=[2, 99], time=[0.1, 0.2])
	adapter = EcsAnimationAdapter(world, {1: rig}, {2: clip})
	with pytest.raises(ValueError, match="Unknown animation resources"):
		adapter.update(0.5)
	np.testing.assert_allclose(world.get(ClipPlayer)[ids].time, [0.1, 0.2])
	with pytest.raises(ValueError, match="dt"):
		adapter.update(-1)


def test_ecs_groups_resources_without_losing_entity_mapping():
	rig, clip = resources()
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	ids = world.create(3)
	world.add(ids, columns(SkeletonRef, id=[1, 1, 2]), ClipPlayer(2))
	world.get(ClipPlayer)[ids].assign(clip_id=[2, 3, 2], time=[0.2, 0.4, 0.6])
	adapter = EcsAnimationAdapter(world, {1: rig, 2: rig}, {2: clip, 3: clip})
	batches = adapter.update(0)
	assert [(batch.skeleton_id, batch.clip_id) for batch in batches] == [(1, 2), (1, 3), (2, 2)]
	assert [batch.entities.tolist() for batch in batches] == [[int(i)] for i in ids]
	np.testing.assert_allclose([batch.matrices[0, 0, 0, 3] for batch in batches], [0.4, 0.8, 1.2], atol=1e-6)


def test_ecs_crossfade_play_pause_and_complete():
	rig, source = resources()
	target = Clip([[[4, 0, 0]], [[8, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	ids = world.create(2)
	world.add(ids, SkeletonRef(1), ClipPlayer(2, loop=False, transitioning=True, target_clip_id=3, target_loop=False, transition_duration=1))
	world.get(ClipPlayer)[ids[1:]].assign(playing=False)
	adapter = EcsAnimationAdapter(world, {1: rig}, {2: source, 3: target})
	batch, = adapter.update(0.5)
	np.testing.assert_allclose(batch.matrices[:, 0, 0, 3], [3.5, 0], atol=1e-6)
	players = world.get(ClipPlayer)[ids]
	np.testing.assert_allclose(players.transition_time, [0.5, 0])
	np.testing.assert_allclose(players.target_time, [0.5, 0])
	batch, = adapter.update(0.75)
	np.testing.assert_allclose(batch.matrices[:, 0, 0, 3], [8, 0], atol=1e-6)
	assert batch.clip_id == 2
	assert players.clip_id.tolist() == [3, 2]
	assert players.transitioning.tolist() == [False, True]
	np.testing.assert_allclose(players.time, [1, 0])
	assert players.loop.tolist() == [False, False]


def test_ecs_invalid_transition_does_not_advance_players():
	rig, clip = resources()
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	ids = world.create(2)
	world.add(ids, SkeletonRef(1), ClipPlayer(2, transitioning=True, target_clip_id=2, transition_duration=1))
	players = world.get(ClipPlayer)[ids]
	players.assign(target_clip_id=[2, 99])
	adapter = EcsAnimationAdapter(world, {1: rig}, {2: clip})
	with pytest.raises(ValueError, match="Unknown target clip"):
		adapter.update(0.5)
	np.testing.assert_allclose(players.time, 0)
	np.testing.assert_allclose(players.transition_time, 0)
	players.assign(target_clip_id=2, transition_duration=[1, 0])
	with pytest.raises(ValueError, match="duration positive"):
		adapter.update(0.5)
	np.testing.assert_allclose(players.time, 0)
