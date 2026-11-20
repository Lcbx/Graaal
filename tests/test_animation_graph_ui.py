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
