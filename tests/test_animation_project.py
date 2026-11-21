"""Portable authoring projects validate before replacing files or sessions."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from graaal.Animation.AnimationEvents import EventTimeline, EventTrack
from graaal.Animation.SkeletalAnimation import Clip, Skeleton
from tools.animation_asset import Animation
from tools.animation_project import read_project, write_project


def animations():
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	preview = SimpleNamespace(asset=SimpleNamespace(skeleton=rig))
	result = []
	for name, source_duration, fps in (("marche", 1.25, 1), ("course", 2, 0.5), ("pose", 0, 1)):
		count = 2 if source_duration > 0 else 1
		clip = Clip(np.zeros((count, 1, 3)), np.tile([0, 0, 0, 1], (count, 1, 1)), np.ones((count, 1, 3)), fps=fps)
		events = EventTimeline(EventTrack(source_duration, [], [])) if source_duration > 0 else None
		result.append(Animation(name, source_duration, (), events, preview, clip))
	return result


def definition():
	return {
		"version": 1,
		"states": {"idle": {"clip": "0: marche"}, "run": {"clip": "1: course"}},
		"initial_state": "idle",
		"sample_slots": ["base"],
		"tasks": [{"name": "output", "op": "sample", "slot": "base"}],
		"sync_mappings": [{"source": "0: marche", "target": "1: course"}],
	}


def add_sync(resources):
	for animation, right in zip(resources[:2], (0.5, 1.25)):
		animation.sync_markers.add(0, "left")
		animation.sync_markers.add(right, "right")


def test_markers_round_trip_precise_times_unicode_coincident_order_and_durations(tmp_path):
	resources = animations()
	resources[0].events.add(0.12500000000000003, "pas gauche 🦶")
	resources[0].events.add(0.12500000000000003, "sound")
	resources[0].events.add(1.125, "after baked endpoint")
	add_sync(resources)
	before = resources[0].events.snapshot()
	path = tmp_path / "markers.animation.json"
	write_project(path, resources)
	data = json.loads(path.read_text(encoding="utf-8"))
	assert "graph" not in data and "pas gauche 🦶" in path.read_text(encoding="utf-8")
	draft, graph, timelines = read_project(path, animations())
	assert draft is graph is None
	assert timelines[2] == (None, None)
	events, sync = timelines[0]
	assert events.duration == 1.25 and sync.duration == 1
	np.testing.assert_array_equal(events.snapshot()[1].times, before[1].times)
	assert events.snapshot()[1].names == before[1].names
	assert events is not resources[0].events
	events.add(0.5, "new session")
	assert resources[0].events.snapshot()[0] == before[0]
	assert resources[0].events.snapshot()[1].names == before[1].names


def test_graph_round_trip_compiles_with_saved_markers_without_modifying_live_tracks(tmp_path):
	resources = animations()
	add_sync(resources)
	data = definition()
	path = tmp_path / "graph.animation.json"
	write_project(path, resources, data)
	fresh = animations()
	draft, graph, timelines = read_project(path, fresh)
	assert draft.data == data
	assert all(not animation.sync_markers.snapshot()[0] for animation in fresh[:2])
	np.testing.assert_allclose(graph.sync_mappings[(0, 1)].map([0.25])[0], [0.625])
	# Marker edits after loading cannot change the compiled mapping.
	timelines[1][1].update(1, time=1)
	np.testing.assert_allclose(graph.sync_mappings[(0, 1)].map([0.25])[0], [0.625])
	draft.data["initial_state"] = "run"
	assert json.loads(path.read_text(encoding="utf-8"))["graph"]["initial_state"] == "idle"


@pytest.mark.parametrize("field,value,match", [
	("version", True, "project.version"),
	("version", 2, "project.version"),
	("animations", [], "animation count"),
	("unknown", 1, "project.unknown"),
	("graph", None, "graph"),
])
def test_invalid_project_schema_is_rejected(tmp_path, field, value, match):
	path = tmp_path / "invalid.json"
	resources = animations()
	write_project(path, resources)
	data = json.loads(path.read_text(encoding="utf-8"))
	data[field] = value
	path.write_text(json.dumps(data), encoding="utf-8")
	with pytest.raises(ValueError, match=match): read_project(path, resources)


@pytest.mark.parametrize("field,value,match", [
	("name", "other", "animations\\[0\\].name"),
	("duration", 1, "animations\\[0\\].duration"),
	("sync_duration", 1.25, "animations\\[0\\].sync_duration"),
	("duration", True, "duration"),
	("events", [{"time": True, "name": "step"}], r"events\[0\].time"),
	("events", [{"time": 1.25, "name": "step"}], r"events\[0\].time"),
	("events", [{"time": -1, "name": "step"}], r"events\[0\].time"),
	("events", [{"time": 0, "name": ""}], r"events\[0\].name"),
	("events", [{"time": 0, "name": "step", "extra": 1}], r"events\[0\].extra"),
	("events", None, "events"),
	("sync_markers", [{"time": 1, "name": "step"}], r"sync_markers\[0\].time"),
])
def test_incompatible_assets_and_invalid_markers_report_item_location(tmp_path, field, value, match):
	path = tmp_path / "invalid.json"
	resources = animations()
	write_project(path, resources)
	data = json.loads(path.read_text(encoding="utf-8"))
	data["animations"][0][field] = value
	path.write_text(json.dumps(data), encoding="utf-8")
	with pytest.raises(ValueError, match=match): read_project(path, resources)
	assert not resources[0].events.snapshot()[0]


@pytest.mark.parametrize("text", ['{"version": 1, "version": 1}', '{"version": NaN}', '{"version": Infinity}', '{"version": -Infinity}', '{invalid'])
def test_nonportable_or_ambiguous_json_is_rejected(tmp_path, text):
	path = tmp_path / "invalid.json"
	path.write_text(text, encoding="utf-8")
	with pytest.raises(ValueError): read_project(path, animations())


def test_invalid_graph_or_sync_draft_cannot_overwrite_saved_project(tmp_path):
	resources = animations()
	add_sync(resources)
	path = tmp_path / "graph.json"
	write_project(path, resources, definition())
	before = path.read_bytes()
	data = deepcopy(definition())
	data["initial_state"] = "missing"
	with pytest.raises(ValueError, match="initial_state"): write_project(path, resources, data)
	assert path.read_bytes() == before
	resources[1].sync_markers.update(1, name="left")
	with pytest.raises(ValueError, match="unique"): write_project(path, resources, definition())
	assert path.read_bytes() == before
	assert not list(tmp_path.glob("*.tmp"))


def test_failed_atomic_replacement_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch):
	path = tmp_path / "graph.json"
	resources = animations()
	write_project(path, resources)
	before = path.read_bytes()
	resources[0].events.add(0.5, "step")
	def fail_replace(source, destination): raise OSError("replacement failed")
	monkeypatch.setattr("tools.animation_project.os.replace", fail_replace)
	with pytest.raises(OSError, match="replacement failed"): write_project(path, resources)
	assert path.read_bytes() == before
	assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize("extension", [".gltf", ".glb"])
def test_save_rejects_source_asset_extensions(tmp_path, extension):
	path = tmp_path / ("asset" + extension)
	path.write_bytes(b"original asset")
	with pytest.raises(ValueError, match="sidecar"): write_project(path, animations())
	assert path.read_bytes() == b"original asset"


def test_graph_requires_runtime_rig_but_marker_projects_do_not(tmp_path):
	resources = animations()
	for animation in resources:
		animation.clip = None
		animation.preview = None
		animation.sync_markers = EventTimeline(EventTrack(animation.duration, [], [])) if animation.duration else None
	path = tmp_path / "markers.json"
	write_project(path, resources)
	assert read_project(path, resources)[0] is None
	with pytest.raises(ValueError, match="skinned asset"): write_project(path, resources, definition())
