"""Explicit JSON persistence shared by the standalone animation tools."""

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from graaal.Animation.AnimationEvents import EventTimeline, EventTrack
from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from .animation_asset import clip_resources


def compile_graph(data, animations, timelines=None):
	"""Compile a graph against current clips and optional replacement markers."""
	asset = getattr(animations[0].preview, "asset", None) if animations else None
	if asset is None or any(animation.clip is None for animation in animations): raise ValueError("graph requires a supported skinned asset")
	definition = AnimationGraphDefinition(data)
	markers = [animation.sync_markers for animation in animations] if timelines is None else [sync for events, sync in timelines]
	tracks = {name: timeline.snapshot()[1] for name, timeline in zip(clip_resources(animations), markers) if timeline is not None}
	return definition, definition.compile(asset.skeleton, clip_resources(animations), sync_tracks=tracks)


def _record(value, fields, path, *, optional=()):
	if not isinstance(value, dict): raise ValueError(f"{path}: expected an object")
	for key in value:
		if key not in fields: raise ValueError(f"{path}.{key}: unsupported field")
	for key in fields:
		if key not in value and key not in optional: raise ValueError(f"{path}.{key}: missing field")
	return value


def _duration(value, path):
	message = f"{path}: expected a finite nonnegative duration"
	if isinstance(value, bool) or not isinstance(value, (int, float)): raise ValueError(message)
	try: value = float(value)
	except OverflowError as exc: raise ValueError(message) from exc
	if not np.isfinite(value) or value < 0: raise ValueError(message)
	return value


def _sync_duration(animation):
	return float(animation.clip.duration if animation.clip is not None else animation.duration)


def _markers(values, duration, path):
	if not isinstance(values, list): raise ValueError(f"{path}: expected a list")
	times, names = [], []
	for i, value in enumerate(values):
		item = _record(value, ("time", "name"), f"{path}[{i}]")
		time = _duration(item["time"], f"{path}[{i}].time")
		if time >= duration: raise ValueError(f"{path}[{i}].time: must be in [0, {duration:g})")
		if not isinstance(item["name"], str) or not item["name"]: raise ValueError(f"{path}[{i}].name: expected a nonempty string")
		times.append(time)
		names.append(item["name"])
	return EventTimeline(EventTrack(duration, times, names)) if duration > 0 else None


def _decode(data, animations):
	data = _record(data, ("version", "animations", "graph"), "project", optional=("graph",))
	if type(data["version"]) is not int or data["version"] != 1: raise ValueError("project.version: supported version is integer 1")
	if not animations: raise ValueError("project requires an asset with animations")
	if not isinstance(data["animations"], list) or len(data["animations"]) != len(animations): raise ValueError("project.animations: must match the loaded animation count")
	timelines = []
	for i, (value, animation) in enumerate(zip(data["animations"], animations)):
		path = f"project.animations[{i}]"
		item = _record(value, ("name", "duration", "sync_duration", "events", "sync_markers"), path)
		if item["name"] != animation.name: raise ValueError(f"{path}.name: must match loaded animation {animation.name!r}")
		for key, expected in (("duration", animation.duration), ("sync_duration", _sync_duration(animation))):
			if _duration(item[key], f"{path}.{key}") != expected: raise ValueError(f"{path}.{key}: must match loaded animation duration {expected:g}")
		timelines.append((_markers(item["events"], animation.duration, path + ".events"), _markers(item["sync_markers"], _sync_duration(animation), path + ".sync_markers")))
	definition, graph = compile_graph(data["graph"], animations, timelines) if "graph" in data else (None, None)
	return definition, graph, timelines


def _json_object(pairs):
	result = {}
	for key, value in pairs:
		if key in result: raise ValueError(f"duplicate JSON field: {key}")
		result[key] = value
	return result


def _invalid_constant(value):
	raise ValueError(f"non-finite JSON number: {value}")


def read_project(path, animations):
	"""Validate without modifying the session; return definition, graph, timelines.

	The graph is optional. Marker IDs belong to the loaded session; marker order,
	including coincident events, survives a round trip.
	"""
	data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_json_object, parse_constant=_invalid_constant)
	return _decode(data, animations)


def _snapshot(timeline):
	if timeline is None: return []
	track = timeline.snapshot()[1]
	return [{"time": float(time), "name": name} for time, name in zip(track.times, track.names)]


def write_project(path, animations, graph=None):
	"""Round-trip and validate JSON before atomically replacing a sidecar file."""
	path = Path(path)
	if path.suffix.lower() != ".json": raise ValueError("authoring projects must use a .json sidecar file")
	data = {
		"version": 1,
		"animations": [{"name": animation.name, "duration": float(animation.duration), "sync_duration": _sync_duration(animation), "events": _snapshot(animation.events), "sync_markers": _snapshot(animation.sync_markers)} for animation in animations],
	}
	if graph is not None: data["graph"] = graph
	text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
	_decode(json.loads(text, object_pairs_hook=_json_object, parse_constant=_invalid_constant), animations)
	temporary = None
	try:
		with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
			temporary = Path(output.name)
			output.write(text)
		os.replace(temporary, path)
	finally:
		if temporary is not None: temporary.unlink(missing_ok=True)
