import numpy as np
import pytest
from unittest.mock import patch

from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip, sample_clip_pairs, skin_matrices


def test_controller_sample_batches_reproduce_crossfaded_poses():
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clips = {i: Clip([[[i, 0, 0]], [[i + 1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1) for i in range(3)}
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	ids = world.create(4)
	world.add(ids, SkeletonRef(0), ClipPlayer(0, time=0.25, loop=False))
	world.get(ClipPlayer)[ids].assign(transitioning=[False, True, True, True], target_clip_id=[0, 1, 2, 1], target_time=[0, 0.2, 0.3, 0.4], target_loop=False, transition_time=[0, 0, 0.25, 0.9], transition_duration=1, playing=[True, True, False, True])
	adapter = EcsAnimationAdapter(world, {0: rig}, clips)
	pose, = adapter.update(0.25)
	samples, = adapter.sample_batches
	np.testing.assert_array_equal(samples.entities, ids)
	np.testing.assert_array_equal(samples.clip_ids, [[0, 0], [0, 1], [0, 2], [0, 1]])
	np.testing.assert_allclose(samples.times, [[0.5, 0.5], [0.5, 0.45], [0.25, 0.3], [0.5, 0.65]], atol=1e-6)
	np.testing.assert_allclose(samples.weights, [[1, 0], [0.75, 0.25], [0.75, 0.25], [0, 1]], atol=1e-6)
	assert not samples.in_place.any()
	local = sample_clip_pairs(rig, clips, samples.clip_ids, samples.times, samples.weights)
	expected = skin_matrices(rig, model_matrices(rig, *local))
	np.testing.assert_allclose(pose.matrices, expected, atol=1e-6)
	assert world.get(ClipPlayer)[ids].clip_id.tolist() == [0, 0, 0, 1]


def test_samples_group_full_handles_indicate_in_place_and_survive_failed_updates():
	from graaal.Animation.RootMotion import RootMotion
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	ids = world.create(2)
	world.add(ids, SkeletonRef(0), ClipPlayer(0))
	world.get(ClipPlayer)[ids[1:]].assign(extract_root_motion=True)
	adapter = EcsAnimationAdapter(world, {0: rig}, {0: clip}, root_motions={(0, 0): RootMotion(rig, clip)})
	adapter.update(0.25)
	samples, = adapter.sample_batches
	assert samples.in_place.tolist() == [False, True]
	world.get(ClipPlayer)[ids[1:]].assign(clip_id=99)
	with pytest.raises(ValueError, match="Unknown animation resources"):
		adapter.update(0.25)
	assert adapter.sample_batches[0] is samples
	world.delete(ids[0])
	new = world.create()[0]
	world.add(new, SkeletonRef(0), ClipPlayer(0))
	world.delete(ids[1])
	adapter.update(0)
	assert adapter.sample_batches[0].entities.tolist() == [int(new)]
	assert int(new) != int(ids[0])


def test_standalone_clip_pairs_validate_inputs_and_clamp_times():
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	result = sample_clip_pairs(rig, {0: clip}, [[0, 0]], [[-1, 2]], [[0.25, 0.75]])
	np.testing.assert_allclose(result[0][0, 0], [0.75, 0, 0])
	assert sample_clip_pairs(rig, {}, np.empty((0, 2), dtype=np.uint32), np.empty((0, 2)), np.empty((0, 2)))[0].shape == (0, 1, 3)
	for ids, times, weights, message in (([[0., 0.]], [[0, 0]], [[1, 0]], "clip_ids"), ([[0, 1]], [[0, 0]], [[1, 0]], "Unknown clip"), ([[0, 0]], [[0, np.nan]], [[1, 0]], "times"), ([[0, 0]], [[0, 0]], [[0.2, 0.2]], "weights")):
		with pytest.raises(ValueError, match=message): sample_clip_pairs(rig, {0: clip}, ids, times, weights)
	bad = Clip(np.zeros((1, 2, 3)), [[[0, 0, 0, 1], [0, 0, 0, 1]]], np.ones((1, 2, 3)))
	with pytest.raises(ValueError, match="bone counts"):
		sample_clip_pairs(rig, {0: clip, 1: bad}, [[0, 1]], [[0, 0]], [[1, 0]])


def test_controller_only_update_can_be_evaluated_later_with_root_motion_and_events():
	from graaal.Animation.AnimationEvents import EventTrack
	from graaal.Animation.RootMotion import RootMotion
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	entity = world.create()[0]
	world.add(entity, SkeletonRef(0), ClipPlayer(0, transitioning=True, target_clip_id=1, target_loop=False, transition_duration=1, extract_root_motion=True))
	clips = {0: clip, 1: clip}
	adapter = EcsAnimationAdapter(world, {0: rig}, clips, event_tracks={0: EventTrack(1, [0.2], ["step"])}, root_motions={(0, 0): RootMotion(rig, clip), (0, 1): RootMotion(rig, clip)})
	with patch("graaal.EcsAdapters.EcsAnimation.sample_clip_pairs", side_effect=AssertionError("unexpected CPU pose evaluation")):
		assert adapter.update(0.25, evaluate_poses=False) == []
	assert world.get(ClipPlayer)[[entity]].time[0] == pytest.approx(0.25)
	assert [event.name for event in adapter.events] == ["step"]
	assert adapter.root_motion_batches[0].matrices[0, 0, 3] == pytest.approx(0.25)
	samples = adapter.sample_batches
	pose, = adapter.evaluate_samples(samples)
	expected = pose.matrices.copy()
	assert expected[0, 0, 0, 3] == 0
	assert world.get(ClipPlayer)[[entity]].time[0] == pytest.approx(0.25)
	combined, = adapter.update(0)
	np.testing.assert_allclose(combined.matrices, expected, atol=1e-6)
	assert samples[0].weights[0, 1] == pytest.approx(0.25)


def test_sample_pair_output_buffers_are_reused_and_validated():
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	out = (np.empty((1, 1, 3), np.float32), np.empty((1, 1, 4), np.float32), np.empty((1, 1, 3), np.float32))
	pose = sample_clip_pairs(rig, {0: clip}, [[0, 0]], [[0, 1]], [[0.5, 0.5]], out=out)
	assert all(a is b for a, b in zip(pose, out))
	np.testing.assert_allclose(pose[0][0, 0], [0.5, 0, 0])
	with pytest.raises(ValueError, match="overlap"):
		sample_clip_pairs(rig, {0: clip}, [[0, 0]], [[0, 1]], [[0.5, 0.5]], out=(out[0], out[1], out[0]))


def test_scheduler_separates_control_and_pose_finalization():
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	entity = world.create()[0]
	world.add(entity, SkeletonRef(0), ClipPlayer(0))
	adapter = EcsAnimationAdapter(world, {0: rig}, {0: clip})
	seen = []
	def physics(world, commands, dt):
		seen.append((adapter.sample_batches[0].times[0, 0], len(adapter.batches)))
	schedule = Scheduler(world)
	schedule.add("animation_control", adapter.advance, phase="fixed_update")
	schedule.add("physics", physics, phase="fixed_update", after="animation_control")
	schedule.add("animation_pose", adapter.finalize, phase="fixed_update", after="physics")
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	assert seen == [(0.25, 0)]
	assert adapter.batches[0].matrices[0, 0, 0, 3] == pytest.approx(0.25)


def test_clip_pair_endpoint_weights_sample_only_the_contributing_clip():
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[1, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	with patch("graaal.Animation.SkeletalAnimation.sample_clip", wraps=sample_clip) as sample:
		local = sample_clip_pairs(rig, {0: clip}, [[0, 0], [0, 0]], [[0.25, 0.9], [0.75, 0.1]], [[1, 0], [1, 0]])
		assert sample.call_count == 1
	np.testing.assert_allclose(local[0][:, 0, 0], [0.25, 0.75])
	with patch("graaal.Animation.SkeletalAnimation.sample_clip", wraps=sample_clip) as sample:
		local = sample_clip_pairs(rig, {0: clip}, [[0, 0]], [[0.25, 0.75]], [[0, 1]])
		assert sample.call_count == 1
	assert local[0][0, 0, 0] == pytest.approx(0.75)
