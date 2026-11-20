from copy import deepcopy

import numpy as np
import pytest

from graaal.AnimationGraph import AnimationGraphDefinition
from graaal.AnimationEvents import EventTimeline, EventTrack
from graaal.AnimationSync import SyncMapping
from graaal.AnimationStateMachine import AnimationStateMachine
from graaal.PoseTasks import PoseOp, PosePlan
from graaal.SkeletalAnimation import Clip, Skeleton, blend_poses, model_matrices, sample_clip_pairs, skin_matrices, subtree_mask


def resources():
	translation = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
	rotation, scale = np.tile([0, 0, 0, 1], (2, 1)), np.ones((2, 3))
	rig = Skeleton([-1, 0], translation, rotation, scale, np.tile(np.eye(4), (2, 1, 1)))
	idle, walk, upper = (translation.copy() for _ in range(3))
	walk[0, 0], upper[1, 1] = 2, 3
	clips = {name: Clip(pose[None], rotation[None], scale[None]) for name, pose in zip(("idle", "walk", "upper"), (idle, walk, upper))}
	return rig, clips


def definition():
	return {
		"version": 1,
		"parameters": {"speed": 0, "grounded": 1, "upper_weight": 0.5},
		"states": {"idle": {"clip": "idle"}, "walk": {"clip": "walk"}},
		"initial_state": "idle",
		"transitions": [
			{"source": "idle", "target": "walk", "conditions": [{"parameter": "speed", "comparison": ">", "threshold": 0.5}, {"parameter": "grounded", "comparison": "==", "threshold": 1}], "duration": 0.25},
			{"source": "walk", "target": "idle", "conditions": [{"parameter": "speed", "comparison": "<", "threshold": 0.2}], "duration": 0.5},
		],
		"sample_slots": ["locomotion", "upper"],
		"weight_slots": ["upper_weight"],
		"root_motion_slots": ["locomotion"],
		"masks": {"upper_body": {"root": 1}},
		"tasks": [
			{"name": "base", "op": PoseOp.SAMPLE, "slot": "locomotion"},
			{"name": "in_place", "op": PoseOp.IN_PLACE, "base": "base", "slot": "locomotion"},
			{"name": "upper", "op": PoseOp.SAMPLE, "slot": "upper"},
			{"name": "output", "op": PoseOp.BLEND, "base": "in_place", "overlay": "upper", "weight": "upper_weight", "mask": "upper_body"},
		],
	}


def samples(graph):
	ids = graph.clip_ids
	return [(np.array([[ids["idle"], ids["walk"]]]), np.zeros((1, 2)), np.array([[0.5, 0.5]])), (np.array([[ids["upper"], ids["upper"]]]), np.zeros((1, 2)), np.array([[1, 0]]))]


def evaluate(graph):
	return graph.plan.evaluate(graph.clips, samples(graph), [[0.5]], root_motions=[(None, np.array([False]))])


def test_definition_compiles_existing_runtime_resources_and_named_bindings():
	rig, clips = resources()
	graph = AnimationGraphDefinition(definition()).compile(rig, clips)
	assert isinstance(graph.machine, AnimationStateMachine) and isinstance(graph.plan, PosePlan)
	assert graph.clips[graph.clip_ids["walk"]] is clips["walk"]
	assert graph.state_clips[graph.state_ids["walk"]] == graph.clip_ids["walk"]
	assert graph.initial_state == graph.state_ids["idle"]
	assert graph.parameter_columns == {"speed": 0, "grounded": 1, "upper_weight": 2}
	assert graph.sample_slots == {"locomotion": 0, "upper": 1}
	assert graph.weight_columns == (2,) and graph.root_motion_slots == {"locomotion": 0}
	np.testing.assert_array_equal(graph.parameter_defaults, [0, 1, 0.5])
	selected, targets, durations = graph.machine.select(np.array([0, 0, 1]), [[1, 1, 0], [1, 0, 0], [0, 1, 0]], np.zeros(3, dtype=bool))
	np.testing.assert_array_equal(selected, [True, False, True])
	np.testing.assert_array_equal(targets, [1, 0, 0])
	np.testing.assert_allclose(durations, [0.25, 0, 0.5])
	base, overlay = (sample_clip_pairs(rig, graph.clips, *sample) for sample in samples(graph))
	expected = blend_poses(rig, base, overlay, [0.5], mask=subtree_mask(rig, 1))
	actual = evaluate(graph)
	for value, reference in zip(actual, expected): np.testing.assert_allclose(value, reference)
	np.testing.assert_allclose(skin_matrices(rig, model_matrices(rig, *actual))[0, :, :, 3], [[1, 0, 0], [1, 2, 0]])
	with pytest.raises(TypeError): graph.state_clips[0] = 99
	with pytest.raises(ValueError): graph.parameter_defaults[0] = 99


def test_draft_edits_and_failed_compiles_leave_existing_resources_unchanged():
	rig, clips = resources()
	data = definition()
	data["masks"]["upper_body"] = [0, 1]
	draft = AnimationGraphDefinition(data)
	data["masks"]["upper_body"][1] = 0
	graph = draft.compile(rig, clips)
	before = tuple(value.copy() for value in evaluate(graph))
	draft.data["tasks"][-1]["overlay"] = "missing"
	with pytest.raises(ValueError, match=r"tasks\[3\].*output.*overlay"):
		draft.compile(rig, clips)
	draft.data["tasks"][-1]["overlay"] = "upper"
	draft.data["masks"]["upper_body"][1] = 0
	draft.data["parameters"]["upper_weight"] = 0.75
	rebuilt = draft.compile(rig, clips)
	assert rebuilt.parameter_defaults[-1] == 0.75 and graph.parameter_defaults[-1] == 0.5
	changed = evaluate(rebuilt)
	assert not np.allclose(changed[0], before[0])
	for value, reference in zip(evaluate(graph), before): np.testing.assert_array_equal(value, reference)
	assert not np.shares_memory(changed[0], evaluate(graph)[0])


def test_additive_and_minimal_parameterless_plans():
	rig, clips = resources()
	data = definition()
	data["tasks"][-1]["op"] = PoseOp.ADDITIVE
	graph = AnimationGraphDefinition(data).compile(rig, clips)
	np.testing.assert_allclose(evaluate(graph)[0][0], [[1, 0, 0], [0, 2, 0]])
	minimal = {"version": 1, "states": {"idle": {"clip": "idle"}}, "initial_state": "idle", "sample_slots": ["base"], "tasks": [{"name": "base", "op": "sample", "slot": "base"}]}
	graph = AnimationGraphDefinition(minimal).compile(rig, clips)
	assert graph.parameter_defaults.shape == (0,) and graph.weight_columns == ()
	assert not graph.machine.select(np.array([0]), np.empty((1, 0)), np.array([False]))[0][0]
	assert graph.plan.evaluate(graph.clips, samples(graph)[:1], np.empty((1, 0)))[0].shape == (1, 2, 3)


@pytest.mark.parametrize("path,value,error", [
	(("version",), 2, r"graph.version"),
	(("version",), True, r"graph.version"),
	(("sync",), {}, r"graph.sync"),
	(("parameters", "speed"), np.nan, r"parameters.speed"),
	(("parameters", "upper_weight"), 2, r"parameters.upper_weight"),
	(("states",), {}, r"states"),
	(("states", "idle", "clip"), "missing", r"states.idle.clip"),
	(("states", "idle", "blend_space"), {}, r"states.idle.blend_space"),
	(("initial_state",), "missing", r"initial_state"),
	(("sample_slots",), ["base", "base"], r"sample_slots\[1\]"),
	(("weight_slots",), ["unknown"], r"weight_slots\[0\]"),
	(("masks", "upper_body"), [0, 2], r"masks.upper_body"),
	(("masks", "unused"), [0], r"masks.unused"),
	(("masks", "upper_body", "root"), True, r"masks.upper_body"),
	(("transitions", 0, "target"), "missing", r"transitions\[0\].target"),
	(("transitions", 0, "duration"), 0, r"transitions\[0\]"),
	(("transitions", 0, "duration"), 1e300, r"transitions\[0\].duration"),
	(("transitions", 0, "duration"), 1e-300, r"transitions\[0\].duration"),
	(("transitions", 0, "conditions"), [], r"transitions\[0\]"),
	(("transitions", 0, "conditions", 0, "comparison"), "?", r"transitions\[0\]"),
	(("transitions", 0, "conditions", 0, "parameter"), "missing", r"transitions\[0\].conditions\[0\].parameter"),
	(("tasks",), [], r"tasks"),
	(("tasks", 0, "slot"), "missing", r"tasks\[0\].*base.*slot"),
	(("tasks", 0, "op"), "other", r"tasks\[0\].*op"),
	(("tasks", 0, "op"), PoseOp.INPUT, r"tasks\[0\].*op"),
	(("tasks", 1, "base"), "output", r"tasks\[1\].*earlier task"),
	(("tasks", 2, "name"), "base", r"tasks\[2\].name"),
	(("tasks", 3, "base"), "output", r"tasks\[3\].*earlier task"),
	(("tasks", 3, "mask"), "missing", r"tasks\[3\].*mask"),
])
def test_invalid_items_report_editable_location(path, value, error):
	data = deepcopy(definition())
	item = data
	for key in path[:-1]: item = item[key]
	item[path[-1]] = value
	with pytest.raises(ValueError, match=error): AnimationGraphDefinition(data).compile(*resources())


def test_rejects_missing_fields_and_incompatible_clip_resources():
	data = definition()
	del data["tasks"][0]["slot"]
	with pytest.raises(ValueError, match=r"tasks\[0\].*slot"):
		AnimationGraphDefinition(data).compile(*resources())
	rig, clips = resources()
	clips["idle"] = Clip(np.zeros((1, 1, 3)), [[[0, 0, 0, 1]]], np.ones((1, 1, 3)))
	with pytest.raises(ValueError, match=r"clips.idle"):
		AnimationGraphDefinition(definition()).compile(rig, clips)


def sync_resources():
	rig, original = resources()
	clips = {name: Clip(np.repeat(clip.translation, 2, axis=0), np.repeat(clip.rotation, 2, axis=0), np.repeat(clip.scale, 2, axis=0), fps=0.5 if name == "walk" else 1) for name, clip in original.items()}
	tracks = {"idle": EventTrack(1, [0, 0.5], ["left", "right"]), "walk": EventTrack(2, [0, 0.5], ["left", "right"])}
	data = definition()
	data["sync_mappings"] = [{"source": "idle", "target": "walk"}]
	return rig, clips, tracks, data


def test_compiled_sync_mapping_uses_existing_runtime_and_independent_snapshots():
	rig, clips, tracks, data = sync_resources()
	timeline = EventTimeline(tracks["walk"])
	tracks["walk"] = timeline.snapshot()[1]
	graph = AnimationGraphDefinition(data).compile(rig, clips, sync_tracks=tracks)
	pair = graph.clip_ids["idle"], graph.clip_ids["walk"]
	mapping = graph.sync_mappings[pair]
	assert isinstance(mapping, SyncMapping)
	np.testing.assert_allclose(mapping.map([0.25, 0.75])[0], [0.25, 1.25])
	np.testing.assert_allclose(mapping.advance([0.75], [1.25], [1.25]), [2.25])
	timeline.update(1, time=1)
	tracks["walk"] = timeline.snapshot()[1]
	rebuilt = AnimationGraphDefinition(data).compile(rig, clips, sync_tracks=tracks)
	np.testing.assert_allclose(rebuilt.sync_mappings[pair].map([0.75])[0], [1.5])
	np.testing.assert_allclose(mapping.map([0.75])[0], [1.25])
	with pytest.raises(TypeError): graph.sync_mappings[pair] = None


@pytest.mark.parametrize("change,error", [
	("missing", r"sync_mappings\[0\].target"),
	("duration", r"sync_mappings\[0\].target.*duration"),
	("names", r"sync_mappings\[0\].*unique"),
	("times", r"sync_mappings\[0\].*distinct"),
	("pair", r"sync_mappings\[1\].*unique"),
	("self", r"sync_mappings\[0\].*distinct"),
	("reference", r"sync_mappings\[0\].target"),
])
def test_invalid_sync_authoring_reports_mapping_location(change, error):
	rig, clips, tracks, data = sync_resources()
	if change == "missing": del tracks["walk"]
	elif change == "duration": tracks["walk"] = EventTrack(3, [0, 1], ["left", "right"])
	elif change == "names": tracks["walk"] = EventTrack(2, [0, 0.5], ["left", "left"])
	elif change == "times": tracks["walk"] = EventTrack(2, [0.5, 0.5], ["left", "right"])
	elif change == "pair": data["sync_mappings"] *= 2
	elif change == "self": data["sync_mappings"][0]["target"] = "idle"
	elif change == "reference": data["sync_mappings"][0]["target"] = "missing"
	with pytest.raises(ValueError, match=error): AnimationGraphDefinition(data).compile(rig, clips, sync_tracks=tracks)
