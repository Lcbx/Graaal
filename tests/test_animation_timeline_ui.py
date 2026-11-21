"""Hidden desktop checks for the timeline's event authoring controls."""
from types import SimpleNamespace

import numpy as np
import pytest

from graaal.Animation.AnimationEvents import EventTimeline, EventTrack
from tools.animation_preview import Preview
from tools.animation_asset import Animation
from tools.animation_timeline import MarkerKind, TimelineWindow
from tools.animation_project import read_project


@pytest.fixture
def window():
	tk = pytest.importorskip("tkinter")
	try: root = tk.Tk()
	except tk.TclError as exc: pytest.skip(f"Tk display unavailable: {exc}")
	root.withdraw()
	callback_errors = []
	root.report_callback_exception = lambda *error: callback_errors.append(error)
	try:
		window = TimelineWindow(root)
		preview = Preview([], np.array([], dtype=np.intp), (), np.array([], dtype=np.intp))
		window.animations = [Animation(name, 2, (), EventTimeline(EventTrack(2, [], [])), preview) for name in ("walk", "run")]
		window.clip_box.configure(values=[animation.name for animation in window.animations])
		window.clip_box.current(0)
		window.select_animation()
		root.update_idletasks()
		window.zoom.set(100)
		window.draw()
		yield window
		root.update()
		assert not callback_errors
	finally:
		root.destroy()


def add_marker(window, time, name):
	window.event_time.set(str(time))
	window.event_name.set(name)
	window.add_event()
	window.root.update()
	return int(window.event_list.selection()[0])


def pointer(window, time, y=None):
	x = window.LABEL_WIDTH + time * window.zoom.get() - window.canvas.canvasx(0)
	if y is None: y = window.HEADER_HEIGHT + window.ROW_HEIGHT / 2
	return SimpleNamespace(x=x, y=y - window.canvas.canvasy(0))


def test_marker_project_restores_both_tracks_and_clears_stale_selection(window, tmp_path):
	add_marker(window, 0.5, "step")
	add_marker(window, 0.5, "sound")
	window.animation.sync_markers.add(0.25, "left")
	window.animations[1].events.add(1, "run step")
	path = tmp_path / "markers.animation.json"
	window.save_project(path)
	assert "Saved markers" in window.status.get() and window.project_path == path
	assert window.event_list.selection()
	window.animation.events.update(1, name="unsaved")
	window.drag_event_id, window.playing = 1, True
	window.event_preview.set("old occurrences")
	window.load_project(path)
	window.root.update()
	assert "Loaded markers" in window.status.get()
	assert not window.event_list.selection() and window.drag_event_id is None
	assert not window.playing and window.event_preview.get() == "Preview events: none"
	assert window.animation.events.snapshot()[1].names == ("step", "sound")
	assert window.animation.sync_markers.snapshot()[1].names == ("left",)
	assert window.animations[1].events.snapshot()[1].names == ("run step",)
	window.event_name.set("stale editor value")
	window.update_event()
	assert window.animation.events.snapshot()[1].names == ("step", "sound")


def test_failed_project_load_keeps_marker_selection_and_drag(window, tmp_path):
	marker = add_marker(window, 0.5, "step")
	window.drag_event_id = marker
	window.playing = True
	events = window.animation.events
	path = tmp_path / "bad.json"
	path.write_text('{"version": 2, "animations": []}', encoding="utf-8")
	window.load_project(path)
	assert "Cannot load project" in window.status.get()
	assert window.animation.events is events and window.event_list.selection() == (str(marker),)
	assert window.drag_event_id == marker and window.playing
	assert window.project_path is None
	window.playing = False


def test_timeline_round_trips_graph_from_other_tool_and_keeps_source_asset(window, tmp_path):
	from pathlib import Path
	from tools.animation_asset import load_animations
	from tools.animation_graph import default_definition
	from tools.animation_project import write_project

	pytest.importorskip("pygltflib")
	asset = Path(__file__).resolve().parents[1] / "scenes/resources/greenman.glb"
	before = asset.read_bytes()
	window.load(asset)
	data = default_definition(window.animations)
	path = tmp_path / "graph.json"
	write_project(path, window.animations, data)
	window.load_project(path)
	assert window.project_graph == data
	window.clip_box.current(1)
	window.select_animation()
	add_marker(window, 0.1, "footstep")
	window.save_project(path)
	draft, graph, timelines = read_project(path, load_animations(asset))
	assert draft.data == data and graph.initial_state == 0
	assert timelines[1][0].snapshot()[1].names == ("footstep",)
	assert asset.read_bytes() == before


def test_update_coincident_events_preserves_selection_and_rejects_invalid_edits(window):
	left = add_marker(window, 0.5, "left")
	right = add_marker(window, 0.5, "right")
	_, original = window.animation.events.snapshot()
	window.event_list.selection_set(str(left))
	window.select_event()
	assert window.event_name.get() == "left"
	window.event_time.set("1.25")
	window.event_name.set("land")
	window.update_event()
	window.root.update()
	ids, track = window.animation.events.snapshot()
	assert ids == (right, left) and track.names == ("right", "land")
	assert window.event_list.selection() == (str(left),)
	assert window.position == 1.25
	assert original.names == ("left", "right")
	for time, name in (("2", "outside"), ("NaN", "invalid"), ("1", "")):
		window.event_time.set(time)
		window.event_name.set(name)
		window.update_event()
		assert window.animation.events.snapshot()[1].names == track.names
		np.testing.assert_array_equal(window.animation.events.snapshot()[1].times, track.times)
		assert window.event_list.selection() == (str(left),)
	window.remove_event()
	assert window.animation.events.snapshot()[0] == (right,)
	assert window.event_list.selection() == ()


def test_drag_uses_selected_coincident_marker_and_clamps_event_time(window):
	left = add_marker(window, 0.5, "left")
	right = add_marker(window, 0.5, "right")
	window.playing = True
	window.click_timeline(pointer(window, 0.5))
	assert window.drag_event_id == right and not window.playing
	window.drag_timeline(pointer(window, 1.25))
	ids, track = window.animation.events.snapshot()
	assert ids == (left, right)
	np.testing.assert_allclose(track.times, [0.5, 1.25])
	assert window.event_list.selection() == (str(right),)
	window.drag_timeline(pointer(window, -10))
	assert window.position == 0
	window.drag_timeline(pointer(window, 10))
	assert 0 < window.position < window.animation.duration
	assert float(window.event_time.get()) == window.position
	window.event_name.set("renamed at endpoint")
	window.update_event()
	assert window.animation.events.snapshot()[1].names[-1] == "renamed at endpoint"
	window.end_drag()
	window.root.update()
	assert window.drag_event_id is None
	window.click_timeline(pointer(window, 0.5, y=10))
	window.drag_timeline(pointer(window, 1.0))
	assert window.animation.events.snapshot()[1].times[-1] < 2
	assert window.event_preview.get() == "Preview events: none"


def test_clip_switch_drops_selection_and_drag_without_changing_markers(window):
	first = add_marker(window, 0.5, "first clip")
	window.animations[1].events.add(1.5, "second clip")
	window.click_timeline(pointer(window, 0.5))
	assert window.drag_event_id == first
	window.clip_box.current(1)
	window.select_animation()
	window.root.update_idletasks()
	assert window.event_list.selection() == () and window.drag_event_id is None
	assert window.position == 0 and not window.playing
	window.event_name.set("accidental edit")
	window.update_event()
	assert window.animation.events.snapshot()[1].names == ("second clip",)
	assert window.animations[0].events.snapshot()[1].names == ("first clip",)


def test_playback_previews_runtime_event_boundaries_and_ignores_scrubbing(window):
	for time, name in ((0, "start"), (0.5, "left"), (0.5, "right"), (1.5, "finish")):
		add_marker(window, time, name)
	window.position, window.playing = 1.4, True
	window.advance_playback(0.6)
	assert window.position == pytest.approx(0)
	assert window.event_preview.get() == "Preview events: finish @ 1.500s; start @ 0.000s"
	window.advance_playback(0.5)
	assert window.event_preview.get() == "Preview events: left @ 0.500s; right @ 0.500s"
	before = window.event_preview.get()
	window.advance_playback(0)
	assert window.event_preview.get() == before
	window.scrub_timeline(pointer(window, 1.4, y=10))
	assert not window.playing and window.event_preview.get() == before
	window.advance_playback(1)
	assert window.position == pytest.approx(1.4)
	window.animation.events.update(3, name="edited finish")
	window.playing = True
	window.advance_playback(0.2)
	assert window.event_preview.get() == "Preview events: edited finish @ 1.500s"


def test_sync_marker_editing_keeps_events_and_selection_namespaces_separate(window):
	event_id = add_marker(window, 0.5, "footstep")
	window.marker_kind.set(MarkerKind.SYNC)
	window.change_marker_kind()
	assert window.event_list.selection() == ()
	sync_id = add_marker(window, 0.5, "left")
	assert sync_id == event_id
	window.event_name.set("right")
	window.event_time.set("1.25")
	window.update_event()
	assert window.animation.events.snapshot()[1].names == ("footstep",)
	assert window.animation.sync_markers.snapshot()[1].names == ("right",)
	assert window.canvas.find_withtag(f"event:{event_id}")
	assert window.canvas.find_withtag(f"sync:{sync_id}")
	window.click_timeline(pointer(window, 0.5))
	assert window.marker_kind.get() == MarkerKind.EVENTS
	assert window.event_name.get() == "footstep"
	window.end_drag()
	window.click_timeline(pointer(window, 1.25, y=window.HEADER_HEIGHT + window.ROW_HEIGHT * 1.5))
	assert window.marker_kind.get() == MarkerKind.SYNC
	assert window.event_name.get() == "right"
	window.drag_timeline(pointer(window, 1.5))
	window.end_drag()
	assert window.animation.sync_markers.snapshot()[1].times.tolist() == [1.5]
	assert window.animation.events.snapshot()[1].times.tolist() == [0.5]
	window.position, window.playing = 0, True
	window.advance_playback(1.75)
	assert window.event_preview.get() == "Preview events: footstep @ 0.500s"
	window.remove_event()
	assert window.animation.sync_markers.snapshot()[0] == ()
	assert window.animation.events.snapshot()[0] == (event_id,)
