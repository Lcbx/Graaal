"""Graph draft editing and viewport integration with hidden Tk windows."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from graaal.AnimationEvents import EventTimeline, EventTrack
from graaal.SkeletalAnimation import Clip, Skeleton
from tool.animation_preview import Preview
from tool.animation_timeline import Animation, TimelineWindow


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
		translation = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
		rotation, scale = np.tile([0, 0, 0, 1], (2, 1)), np.ones((2, 3))
		rig = Skeleton([-1, 0], translation, rotation, scale, np.tile(np.eye(4), (2, 1, 1)))
		idle, walk, upper = (translation.copy() for _ in range(3))
		walk[0, 0], upper[1, 1] = 2, 3
		clips = [Clip(np.tile(pose, (2, 1, 1)), np.tile(rotation, (2, 1, 1)), np.tile(scale, (2, 1, 1)), fps=1) for pose in (idle, walk, upper)]
		nodes = [SimpleNamespace(matrix=None, translation=position.tolist(), rotation=None, scale=None) for position in translation]
		asset = SimpleNamespace(skeleton=rig, joint_nodes=(0, 1), primitives=())
		preview = Preview(nodes, rig.parents, (0, 1), np.array([0, 1]), asset)
		window.animations = [Animation(name, 1, (), EventTimeline(EventTrack(1, [], [])), preview, clip) for name, clip in zip(("idle", "walk", "upper"), clips)]
		window.clip_box.configure(values=[animation.name for animation in window.animations])
		window.clip_box.current(0)
		window.select_animation()
		window.open_graph()
		window.graph_window.window.withdraw()
		root.update()
		yield window
		root.update()
		assert not callback_errors
	finally:
		root.destroy()


def set_draft(editor, data):
	editor.editor.delete("1.0", "end")
	editor.editor.insert("1.0", json.dumps(data))
	editor.compile_draft()


def test_state_selection_conditions_and_manual_crossfade_use_existing_viewport(window):
	editor = window.graph_window
	assert editor.graph is not None and editor.preview_enabled.get()
	assert editor.target_state is None
	editor.parameter_entries["advance"].set("1")
	editor.apply_parameters()
	assert editor.target_state == 1
	assert "0.25s" in editor.decision.get()
	editor.fade.set(0.5)
	editor.change_fade(None)
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[1, 0, 0], [1, 1, 0]])
	assert len(window.viewport.find_all()) >= 4
	editor.pick_state("1: walk")
	assert window.animation_id == 1 and window.position == 0
	assert editor.fade.get() == 0
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[2, 0, 0], [2, 1, 0]])


def test_layer_bindings_parameters_and_validation_keep_last_good_preview(window):
	editor = window.graph_window
	data = json.loads(editor.editor.get("1.0", "end-1c"))
	data["parameters"]["upper_weight"] = 0.5
	data["weight_slots"] = ["upper_weight"]
	data["sample_slots"].append("upper")
	data["masks"] = {"upper_body": {"root": 1}}
	data["tasks"] += [
		{"name": "upper", "op": "sample", "slot": "upper"},
		{"name": "layered", "op": "blend", "base": "output", "overlay": "upper", "weight": "upper_weight", "mask": "upper_body"},
	]
	set_draft(editor, data)
	editor.bindings["upper"].set("2: upper")
	editor.pick_state("1: walk")
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[2, 0, 0], [2, 2, 0]])
	editor.parameter_entries["upper_weight"].set("1")
	editor.apply_parameters()
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[2, 0, 0], [2, 3, 0]])
	graph, before = editor.graph, editor.joint_model(0).copy()
	editor.parameter_entries["upper_weight"].set("NaN")
	editor.apply_parameters()
	assert "Cannot apply parameters" in editor.status.get()
	np.testing.assert_array_equal(editor.joint_model(0), before)
	editor.parameter_entries["upper_weight"].set("2")
	editor.apply_parameters()
	assert "[0, 1]" in editor.status.get()
	data["tasks"][-1]["base"] = "missing"
	set_draft(editor, data)
	assert "tasks[2]" in editor.status.get() and editor.graph is graph
	np.testing.assert_array_equal(editor.joint_model(0), before)
	editor.editor.delete("1.0", "end")
	editor.editor.insert("1.0", "{invalid JSON")
	editor.compile_draft()
	assert editor.graph is graph and "Cannot compile draft" in editor.status.get()


def test_graph_preview_is_explicit_and_reopening_keeps_draft(window):
	editor = window.graph_window
	editor.editor.insert("end", "\n")
	draft = editor.editor.get("1.0", "end-1c")
	window.open_graph()
	assert window.graph_window is editor
	assert editor.editor.get("1.0", "end-1c") == draft
	window.clip_box.current(1)
	window.select_animation()
	assert not editor.preview_enabled.get()
	editor.preview_enabled.set(True)
	editor.change_preview()
	assert window.animation_id == 0
	editor.close()
	assert window.graph_window is None
	assert len(window.viewport.find_all()) >= 4


def test_greenman_graph_preview_renders_mesh_and_failed_load_preserves_draft(window):
	pytest.importorskip("pygltflib")
	path = Path(__file__).resolve().parents[1] / "scenes/resources/greenman.glb"
	before = path.read_bytes()
	window.load(path)
	assert window.graph_window is None
	window.open_graph()
	editor = window.graph_window
	editor.window.withdraw()
	editor.pick_state("1: 1_idle")
	editor.parameter_entries["advance"].set("1")
	editor.apply_parameters()
	editor.fade.set(0.5)
	window.position = 0.25
	window.draw_viewport()
	assert np.isfinite(editor.joint_model(window.position)).all()
	assert any(window.viewport.type(item) == "polygon" for item in window.viewport.find_all())
	assert not any("Preview unavailable" in window.viewport.itemcget(item, "text") for item in window.viewport.find_all() if window.viewport.type(item) == "text")
	draft = editor.editor.get("1.0", "end-1c")
	window.load(path.with_name("missing-animation.glb"))
	assert window.graph_window is editor
	assert editor.editor.get("1.0", "end-1c") == draft
	assert path.read_bytes() == before


def test_authored_sync_markers_align_preview_only_after_successful_compile(window):
	editor = window.graph_window
	walk = window.animations[1].clip
	translation = walk.translation.copy()
	translation[1, 0, 0] = 6
	window.animations[1].clip = Clip(translation, walk.rotation, walk.scale, fps=1)
	for animation, right in zip(window.animations[:2], (0.5, 0.25)):
		animation.sync_markers.add(0, "left")
		animation.sync_markers.add(right, "right")
	data = json.loads(editor.editor.get("1.0", "end-1c"))
	data["sync_mappings"] = [{"source": "0: idle", "target": "1: walk"}]
	set_draft(editor, data)
	editor.parameter_entries["advance"].set("1")
	editor.apply_parameters()
	editor.fade.set(1)
	np.testing.assert_allclose(editor.joint_model(0.75)[0, 0, 3], 4.5)
	compiled = editor.graph
	window.animations[1].sync_markers.update(1, time=0.5)
	np.testing.assert_allclose(editor.joint_model(0.75)[0, 0, 3], 4.5)
	window.animations[1].sync_markers.update(1, name="left")
	editor.compile_draft()
	assert editor.graph is compiled and "unique" in editor.status.get()
	window.animations[1].sync_markers.update(1, name="right")
	editor.compile_draft()
	editor.parameter_entries["advance"].set("1")
	editor.apply_parameters()
	editor.fade.set(1)
	np.testing.assert_allclose(editor.joint_model(0.75)[0, 0, 3], 5)


def transition_form(editor):
	editor.open_transitions()
	form = editor.transition_window
	form.window.withdraw()
	form.window.update()
	return form


def test_transition_form_multi_conditions_priority_and_explicit_application(window):
	editor = window.graph_window
	data = json.loads(editor.editor.get("1.0", "end-1c"))
	data["parameters"]["grounded"] = 0
	set_draft(editor, data)
	graph = editor.graph
	form = transition_form(editor)
	form.source.set("0: idle")
	form.target.set("2: upper")
	form.duration.set("0.75")
	for parameter, comparison, threshold in (("advance", ">", "0.5"), ("grounded", "==", "1")):
		form.parameter.set(parameter)
		form.comparison.set(comparison)
		form.threshold.set(threshold)
		form.store_condition(False)
	form.store_rule(False)
	assert len(form.data["transitions"][-1]["conditions"]) == 2
	for _ in range(3): form.move_rule(-1)
	form.window.update()
	assert form.selected(form.rules) == 0
	assert form.data["transitions"][0]["target"] == "2: upper"
	form.move_rule(-1)
	form.apply()
	assert editor.graph is graph
	assert editor.definition.data == data
	assert "Compile draft" in editor.status.get()
	editor.compile_draft()
	editor.parameter_entries["advance"].set("1")
	editor.apply_parameters()
	assert editor.target_state == 1
	editor.parameter_entries["grounded"].set("1")
	editor.apply_parameters()
	assert editor.target_state == 2 and "0.75s" in editor.decision.get()
	editor.fade.set(1)
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[0, 0, 0], [0, 3, 0]])
	form.window.update()
	assert form.window.winfo_exists()


def test_transition_form_updates_removes_and_preserves_unrelated_draft_data(window):
	editor = window.graph_window
	form = transition_form(editor)
	original = json.loads(editor.editor.get("1.0", "end-1c"))
	form.rules.selection_set("0")
	form.select_rule()
	form.condition_list.selection_set("0")
	form.select_condition()
	form.threshold.set("0.875")
	form.store_condition(True)
	form.duration.set("0.125")
	form.store_rule(True)
	assert form.data["transitions"][0]["conditions"][0]["threshold"] == 0.875
	form.move_rule(1)
	form.move_rule(1)
	form.move_rule(1)
	assert form.selected(form.rules) == 2
	form.remove_rule()
	form.apply()
	draft = json.loads(editor.editor.get("1.0", "end-1c"))
	assert draft["transitions"] == original["transitions"][1:]
	assert {key: value for key, value in draft.items() if key != "transitions"} == {key: value for key, value in original.items() if key != "transitions"}
	while form.rules.get_children():
		form.rules.selection_set("0")
		form.remove_rule()
	form.apply()
	editor.compile_draft()
	editor.parameter_entries["advance"].set("1")
	editor.apply_parameters()
	assert editor.target_state is None


def test_transition_form_rejects_invalid_fields_without_changing_draft_or_preview(window):
	editor = window.graph_window
	form = transition_form(editor)
	before = editor.editor.get("1.0", "end-1c")
	graph = editor.graph
	form.rules.selection_set("0")
	form.select_rule()
	rules = json.dumps(form.data)
	form.condition_list.selection_set("0")
	form.select_condition()
	form.threshold.set("NaN")
	form.store_condition(True)
	assert "finite" in form.status.get()
	assert form.conditions[0]["threshold"] == 0.5
	for duration in ("0", "NaN", "1e100", "1e-100", "bad"):
		form.duration.set(duration)
		form.store_rule(True)
		assert "Cannot stage rule" in form.status.get()
		assert json.dumps(form.data) == rules
	form.duration.set("0.25")
	form.target.set(form.source.get())
	form.store_rule(True)
	assert "states must differ" in form.status.get()
	form.target.set("1: walk")
	form.remove_condition()
	form.store_rule(True)
	assert "at least one condition" in form.status.get()
	assert json.dumps(form.data) == rules
	assert editor.editor.get("1.0", "end-1c") == before and editor.graph is graph
	form.close()
	assert editor.transition_window is None


def test_transition_form_detects_concurrent_json_edits_and_reopens_from_current_draft(window):
	editor = window.graph_window
	form = transition_form(editor)
	form.rules.selection_set("0")
	form.remove_rule()
	editor.editor.insert("end", "\n")
	before = editor.editor.get("1.0", "end-1c")
	form.apply()
	assert "JSON draft changed" in form.status.get()
	assert editor.editor.get("1.0", "end-1c") == before
	editor.open_transitions()
	assert editor.transition_window is form
	form.close()
	form = transition_form(editor)
	assert len(form.data["transitions"]) == 3
	form.close()
	editor.editor.delete("1.0", "end")
	editor.editor.insert("1.0", "{invalid")
	editor.open_transitions()
	assert editor.transition_window is None
	assert "Cannot edit transitions" in editor.status.get()


def test_transition_form_handles_numeric_strings_and_undo(window):
	editor = window.graph_window
	data = json.loads(editor.editor.get("1.0", "end-1c"))
	data["transitions"][0]["duration"] = "0.25"
	data["transitions"][0]["conditions"][0]["threshold"] = "0.5"
	set_draft(editor, data)
	before = editor.editor.get("1.0", "end-1c")
	form = transition_form(editor)
	form.rules.selection_set("0")
	form.select_rule()
	form.condition_list.selection_set("0")
	form.select_condition()
	assert form.threshold.get() == "0.5"
	form.duration.set("0.5")
	form.store_rule(True)
	form.apply()
	assert json.loads(editor.editor.get("1.0", "end-1c"))["transitions"][0]["duration"] == 0.5
	editor.editor.edit_undo()
	assert editor.editor.get("1.0", "end-1c") == before


def test_transition_form_cancel_and_parameterless_graph(window):
	editor = window.graph_window
	before = editor.editor.get("1.0", "end-1c")
	form = transition_form(editor)
	form.rules.selection_set("0")
	form.remove_rule()
	form.close()
	assert editor.editor.get("1.0", "end-1c") == before
	data = json.loads(before)
	data["parameters"], data["transitions"] = {}, []
	data["states"] = {"only": {"clip": "0: idle"}}
	data["initial_state"] = "only"
	set_draft(editor, data)
	form = transition_form(editor)
	assert form.source.get() == form.target.get() == "only"
	form.store_condition(False)
	assert "choose a draft parameter" in form.status.get()
	form.store_rule(False)
	assert "Cannot stage rule" in form.status.get()
	form.apply()
	editor.compile_draft()
	assert len(editor.graph.state_ids) == 1 and editor.target_state is None


def test_named_blend_space_binding_previews_parameter_driven_pose(window):
	editor = window.graph_window
	data = json.loads(editor.editor.get("1.0", "end-1c"))
	data["parameters"]["speed"] = 0
	data["blend_spaces"] = {"locomotion": {"parameter": "speed", "points": [{"position": 0, "clip": "0: idle"}, {"position": 2, "clip": "1: walk"}]}}
	set_draft(editor, data)
	editor.bindings["base"].set("<blend: locomotion>")
	editor.parameter_entries["speed"].set("1")
	editor.apply_parameters()
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[1, 0, 0], [1, 1, 0]])
	editor.parameter_entries["speed"].set("9")
	editor.apply_parameters()
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[2, 0, 0], [2, 1, 0]])
	graph = editor.graph
	data["blend_spaces"]["locomotion"]["points"][1]["clip"] = "missing"
	set_draft(editor, data)
	assert editor.graph is graph and "blend_spaces.locomotion.points[1].clip" in editor.status.get()
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[2, 0, 0], [2, 1, 0]])


def test_2d_grid_binds_two_samples_and_vertical_pose_weight(window):
	editor = window.graph_window
	data = json.loads(editor.editor.get("1.0", "end-1c"))
	data["parameters"].update({"speed": 0, "turn": 0, "grid_mix": 0})
	data["sample_slots"] = ["lower", "upper"]
	data["weight_slots"] = ["grid_mix"]
	data["tasks"] = [
		{"name": "lower", "op": "sample", "slot": "lower"},
		{"name": "upper", "op": "sample", "slot": "upper"},
		{"name": "output", "op": "blend", "base": "lower", "overlay": "upper", "weight": "grid_mix"},
	]
	data["blend_spaces_2d"] = {"motion": {"x_parameter": "speed", "y_parameter": "turn", "x_positions": [0, 2], "y_positions": [-1, 1], "clips": [["0: idle", "1: walk"], ["2: upper", "1: walk"]], "sample_slots": ["lower", "upper"], "weight_slot": "grid_mix"}}
	set_draft(editor, data)
	editor.bindings["lower"].set("<grid: motion / lower>")
	editor.change_binding("lower")
	assert editor.bindings["upper"].get() == "<grid: motion / upper>"
	for key, value in (("speed", "1"), ("turn", "0")):
		editor.parameter_entries[key].set(value)
	editor.apply_parameters()
	np.testing.assert_allclose(editor.joint_model(0)[:, :3, 3], [[1, 0, 0], [1, 1.5, 0]])
	editor.bindings["upper"].set("2: upper")
	editor.change_binding("upper")
	assert editor.bindings["lower"].get() == editor.STATE_SAMPLE
