import base64
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pygltflib")

from tools.animation_asset import Animation, Track, load_animations
from tools.animation_preview import Preview, animation_bounds, camera_from_bounds, mesh_triangles, pose_matrices, project
from graaal.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip


def write_gltf(path, *, times=(0, 1, 2)):
	key_times = np.array(times, dtype="<f4")
	values = np.arange(27, dtype="<f4").reshape(9, 3)
	data = key_times.tobytes() + values.tobytes()
	document = {
		"asset": {"version": "2.0"},
		"buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
		"bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": key_times.nbytes}, {"buffer": 0, "byteOffset": key_times.nbytes, "byteLength": values.nbytes}],
		"accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "SCALAR"}, {"bufferView": 1, "componentType": 5126, "count": 9, "type": "VEC3"}],
		"nodes": [{"name": "hip"}],
		"animations": [{"name": "walk", "samplers": [{"input": 0, "output": 1, "interpolation": "CUBICSPLINE"}], "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}]}],
	}
	path.write_text(json.dumps(document), encoding="utf-8")
	return path


def test_gltf_track_timeline_and_temporary_events(tmp_path):
	path = write_gltf(tmp_path / "animation.gltf")
	before = path.read_bytes()
	animation, = load_animations(path)
	assert animation.name == "walk" and animation.duration == 2
	track, = animation.tracks
	assert track.label == "hip [0] / translation" and track.interpolation == "CUBICSPLINE"
	np.testing.assert_array_equal(track.times, [0, 1, 2])
	np.testing.assert_array_equal(track.values, [[3, 4, 5], [12, 13, 14], [21, 22, 23]])
	event = animation.events.add(0.5, "step")
	assert animation.events.snapshot()[1].names == ("step",)
	animation.events.remove(event)
	assert path.read_bytes() == before
	assert load_animations(path)[0].events.snapshot()[0] == ()


def test_real_glb_animations_can_be_inspected():
	path = Path(__file__).resolve().parents[1] / "scenes/resources/greenman.glb"
	animations = load_animations(path)
	assert [animation.name for animation in animations] == ["0_pose", "1_idle", "2_move", "3_attack"]
	assert any(len(animation.tracks) for animation in animations)
	assert all(len(track.times) == len(track.values) for animation in animations for track in animation.tracks)


def test_invalid_key_times_report_error(tmp_path):
	with pytest.raises(ValueError, match="strictly increasing"):
		load_animations(write_gltf(tmp_path / "invalid.gltf", times=(0, 1, 1)))


def test_animations_keep_separate_temporary_events(tmp_path):
	path = write_gltf(tmp_path / "multiple.gltf")
	document = json.loads(path.read_text(encoding="utf-8"))
	document["animations"].append({**document["animations"][0], "name": "run"})
	path.write_text(json.dumps(document), encoding="utf-8")
	first, second = load_animations(path)
	first.events.add(0.5, "step")
	assert first.events.snapshot()[1].names == ("step",)
	assert second.events.snapshot()[0] == ()


def test_file_without_animations_returns_empty_list(tmp_path):
	path = tmp_path / "empty.gltf"
	path.write_text(json.dumps({"asset": {"version": "2.0"}}), encoding="utf-8")
	assert load_animations(path) == []


def test_preview_samples_cubic_values_and_frames_bone_bounds(tmp_path):
	animation, = load_animations(write_gltf(tmp_path / "cubic.gltf"))
	model = pose_matrices(animation, 0.5)
	np.testing.assert_allclose(model[0, :3, 3], [7.125, 8.125, 9.125])
	bounds = animation_bounds(animation)
	for aspect in (0.4, 2):
		position, target = camera_from_bounds(bounds, aspect)
		np.testing.assert_allclose(target, (bounds[0] + bounds[1]) / 2)
		corners = np.array([[bounds[i][0], bounds[j][1], bounds[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
		screen, depth = project(corners, position, target, 600 * aspect, 600)
		assert (depth > 0).all()
		assert (screen >= 0).all() and (screen <= [600 * aspect, 600]).all()


def test_preview_deforms_greenman_mesh_and_moves_bones():
	animation = load_animations(Path(__file__).resolve().parents[1] / "scenes/resources/greenman.glb")[2]
	assert animation.preview.asset is not None
	first, later = pose_matrices(animation, 0), pose_matrices(animation, animation.duration * 0.35)
	assert not np.allclose(first, later)
	first_mesh, later_mesh = mesh_triangles(animation.preview, first), mesh_triangles(animation.preview, later)
	assert first_mesh.shape[1:] == (3, 3) and np.isfinite(later_mesh).all()
	assert not np.allclose(first_mesh, later_mesh)


def test_preview_matches_runtime_clips_in_source_node_order():
	animations = load_animations(Path(__file__).resolve().parents[1] / "scenes/resources/greenman.glb")
	asset = animations[0].preview.asset
	for animation_id, animation in enumerate(animations):
		assert animation.clip is asset.clips[animation_id][1]
		for time in (-0.1, 0, animation.duration * 0.351, animation.duration, animation.duration + 0.1):
			expected = model_matrices(asset.skeleton, *sample_clip(animation.clip, [time], loop=False))[0]
			np.testing.assert_allclose(pose_matrices(animation, time)[list(asset.joint_nodes)], expected)


def test_viewport_projection_orientation_and_depth():
	points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 6]])
	screen, depth = project(points, np.array([0., 0., 5.]), np.zeros(3), 800, 600)
	np.testing.assert_allclose(screen[0], [400, 300])
	assert screen[1, 0] > screen[0, 0]
	assert screen[2, 1] < screen[0, 1]
	np.testing.assert_allclose(depth, [5, 5, 5, -1])


def test_raw_node_channels_follow_runtime_joint_parent():
	from pygltflib import Node
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], np.eye(4)[None])
	clip = Clip([[[0, 0, 0]], [[2, 0, 0]]], np.tile([0, 0, 0, 1], (2, 1, 1)), np.ones((2, 1, 3)), fps=1)
	preview = Preview([Node(), Node(children=[0])], np.array([1, -1]), (1, 0), np.array([1]), SimpleNamespace(skeleton=rig, joint_nodes=(1,)))
	track = Track("attachment", "STEP", np.array([0, 1]), np.array([[0, 2, 0], [0, 4, 0]]), 0, "translation")
	animation = Animation("test", 1, (track,), None, preview, clip)
	np.testing.assert_allclose(pose_matrices(animation, 0.5)[:, :3, 3], [[1, 2, 0], [1, 0, 0]])
	np.testing.assert_allclose(pose_matrices(animation, 1)[:, :3, 3], [[2, 4, 0], [2, 0, 0]])
