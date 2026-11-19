import numpy as np
import pytest

from graaal.AnimationEvents import EventTrack
from graaal.AnimationSync import SyncMapping
from graaal.RootMotion import RootMotion
from graaal.SkeletalAnimation import Clip, Skeleton, sample_clip_pairs


def test_mapping_marker_phase_looping_reverse_and_rate():
	source = EventTrack(2, [0, 0.5, 1.5], ["left", "right", "lift"])
	target = EventTrack(4, [0.5, 1, 3], ["right", "lift", "left"])
	mapping = SyncMapping(source, target)
	times, rates = mapping.map([0, 0.25, 0.5, 1, 1.5, 1.75, 2, -0.25, 4.5])
	np.testing.assert_allclose(times, [3, 3.75, 0.5, 0.75, 1, 2, 3, 2, 0.5])
	np.testing.assert_allclose(rates, [3, 3, 0.5, 0.5, 4, 4, 3, 4, 0.5])
	reverse_times, reverse_rates = mapping.map([0, 0.5, 1.5], reverse=True)
	np.testing.assert_allclose(reverse_times, [3, 0.5, 1])
	np.testing.assert_allclose(reverse_rates, [4, 3, 0.5])
	assert mapping.map([])[0].shape == (0,)


def test_mapping_times_feed_existing_clip_pair_sampler():
	source = EventTrack(2, [0, 1], ["left", "right"])
	target = EventTrack(4, [0, 3], ["left", "right"])
	target_time, _ = SyncMapping(source, target).map([0.5, 1])
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clips = {0: Clip([[[0, 0, 0]], [[2, 0, 0]]], [[[0, 0, 0, 1]]] * 2, np.ones((2, 1, 3)), fps=0.5), 1: Clip([[[0, 0, 0]], [[4, 0, 0]]], [[[0, 0, 0, 1]]] * 2, np.ones((2, 1, 3)), fps=0.25)}
	pose = sample_clip_pairs(rig, clips, np.array([[0, 1], [0, 1]]), np.column_stack(([0.5, 1], target_time)), np.tile([0.5, 0.5], (2, 1)))
	np.testing.assert_allclose(pose[0][:, 0, 0], [1, 2])


def test_advance_preserves_phase_across_markers_loops_and_reverse():
	source = EventTrack(2, [0.2, 0.7, 1.7], ["left", "right", "lift"])
	target = EventTrack(4, [0.5, 1, 3], ["right", "lift", "left"])
	mapping = SyncMapping(source, target)
	start = np.array([0.2, 0.5, 1.9, 3.8, 0.5, -2.1, 0.2])
	end = np.array([2.2, 1.2, 2.4, -0.3, -3.5, 4.1, 0.2])
	target_start, _ = mapping.map(start)
	target_end = mapping.advance(start, end, target_start)
	expected = mapping.map(end)[0]
	np.testing.assert_allclose(np.mod(target_end, target.duration), expected, atol=1e-12)
	assert target_end[0] - target_start[0] == pytest.approx(4)
	assert target_end[4] - target_start[4] == pytest.approx(-8)
	np.testing.assert_allclose(mapping.advance(end, start, target_end), target_start, atol=1e-12)


def test_advance_unwrapped_intervals_work_with_events_and_root_motion():
	source = EventTrack(2, [0, 1], ["left", "right"])
	target = EventTrack(4, [0, 3], ["left", "right"])
	mapping = SyncMapping(source, target)
	start, _ = mapping.map([0.5])
	end = mapping.advance([0.5], [4.5], start)
	assert end[0] - start[0] == pytest.approx(8)
	assert [item.name for item in target.between(start[0], end[0])] == ["right", "left", "right", "left"]
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[4, 0, 0]]], [[[0, 0, 0, 1]]] * 2, np.ones((2, 1, 3)), fps=0.25)
	motion = RootMotion(rig, clip)
	np.testing.assert_allclose(motion.character_delta(start, end)[:, 0, 3], [8])


def test_advance_validates_intervals_and_loop_range():
	track = EventTrack(1, [0, 0.5], ["a", "b"])
	mapping = SyncMapping(track, track)
	for start, end, target in (([0], [1, 2], [0]), ([0], [np.nan], [0]), ([0], [1], [np.inf]), ([0], [2**52], [0])):
		with pytest.raises(ValueError): mapping.advance(start, end, target)
	assert mapping.advance([], [], []).shape == (0,)


def test_advance_matches_mapped_phase_over_varied_intervals():
	mapping = SyncMapping(EventTrack(3, [0.4, 1.1, 2.5], ["a", "b", "c"]), EventTrack(5, [0.3, 1.9, 3.7], ["b", "c", "a"]))
	rng = np.random.default_rng(314)
	start, end = rng.uniform(-30, 30, size=(2, 200))
	target_start = mapping.map(start)[0]
	target_end = mapping.advance(start, end, target_start)
	difference = np.mod(target_end - mapping.map(end)[0] + 2.5, 5) - 2.5
	np.testing.assert_allclose(difference, 0, atol=1e-12)
	np.testing.assert_allclose(mapping.advance(end, start, target_end), target_start, atol=1e-12)


@pytest.mark.parametrize("source,target", [
	(EventTrack(1, [], []), EventTrack(1, [], [])),
	(EventTrack(1, [0], ["a"]), EventTrack(1, [0], ["a"])),
	(EventTrack(1, [0, 0.5], ["a", "a"]), EventTrack(1, [0, 0.5], ["a", "b"])),
	(EventTrack(1, [0, 0], ["a", "b"]), EventTrack(1, [0, 0.5], ["a", "b"])),
	(EventTrack(1, [0, 0.5], ["a", "b"]), EventTrack(1, [0, 0.5], ["a", "c"])),
	(EventTrack(1, [0, 0.3, 0.6], ["a", "b", "c"]), EventTrack(1, [0, 0.3, 0.6], ["a", "c", "b"])),
	(None, EventTrack(1, [0, 0.5], ["a", "b"])),
])
def test_rejects_incompatible_tracks(source, target):
	with pytest.raises(ValueError): SyncMapping(source, target)


def test_input_validation_and_large_track_durations():
	track = EventTrack(1e308, [0, 8e307], ["a", "b"])
	mapping = SyncMapping(track, track)
	values = np.array([0, 4e307, 8e307, 9e307])
	np.testing.assert_allclose(mapping.map(values)[0], values)
	for values in (0, [np.nan], [np.inf], [[0]]):
		with pytest.raises(ValueError): mapping.map(values)
