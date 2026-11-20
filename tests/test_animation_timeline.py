import pytest

from graaal.AnimationEvents import EventTimeline, EventTrack


def test_edit_markers_keep_ids_and_runtime_event_order():
	timeline = EventTimeline(EventTrack(2, [1, 0.5, 0.5], ["late", "left", "right"]))
	ids, original = timeline.snapshot()
	assert ids == (0, 1, 2)
	assert original.names == ("left", "right", "late")
	added = timeline.add(0.5, "step")
	timeline.update(ids[2], time=0.5, name="land")
	ids_after, edited = timeline.snapshot()
	assert ids_after == (0, 1, 2, added)
	assert edited.names == ("left", "right", "land", "step")
	assert [event.name for event in edited.between(0, 0.5)] == list(edited.names)
	assert original.names == ("left", "right", "late")
	timeline.remove(ids[1])
	assert timeline.snapshot()[0] == (0, 2, added)
	assert timeline.add(1.5, "next") > added


def test_invalid_edits_leave_timeline_and_ids_unchanged():
	timeline = EventTimeline(EventTrack(1, [0.2], ["step"]))
	for time, name in ((-1, "bad"), (1, "bad"), (float("nan"), "bad"), (0.5, "")):
		with pytest.raises(ValueError): timeline.add(time, name)
	with pytest.raises(ValueError): timeline.update(0, time=2, name="changed")
	with pytest.raises(KeyError): timeline.update(42, name="missing")
	with pytest.raises(KeyError): timeline.remove(42)
	with pytest.raises(ValueError): timeline.remove(True)
	ids, track = timeline.snapshot()
	assert ids == (0,)
	assert track.names == ("step",)
	assert track.times.tolist() == [0.2]
	assert timeline.add(0, "start") == 1
	with pytest.raises(AttributeError): timeline.duration = 3


def test_empty_timeline_snapshots_are_independent():
	timeline = EventTimeline(EventTrack(1, [], []))
	ids, track = timeline.snapshot()
	assert ids == () and track.names == ()
	timeline.add(0, "start")
	assert track.names == ()
	assert [event.name for event in timeline.snapshot()[1].between(0.9, 1.0)] == ["start"]
	with pytest.raises(ValueError): EventTimeline(None)
