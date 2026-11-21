import base64
import json

import numpy as np
import pytest

pytest.importorskip("pygltflib")

from graaal.Animation.SkeletalGltf import load_skinned_gltf
from graaal.Animation.SkeletalAnimation import model_matrices, sample_clip, skin_matrices


def write_asset(path, *, interpolation="LINEAR", ancestor_translation=None, rotation=False):
	data, views, accessors = bytearray(), [], []
	def accessor(values, component_type, kind):
		array = np.asarray(values)
		while len(data) % 4: data.append(0)
		offset = len(data)
		data.extend(array.tobytes())
		views.append({"buffer": 0, "byteOffset": offset, "byteLength": array.nbytes})
		accessors.append({"bufferView": len(views) - 1, "componentType": component_type, "count": len(array), "type": kind})
		return len(accessors) - 1
	position = accessor(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4"), 5126, "VEC3")
	normal = accessor(np.array([[0, 0, 1]] * 3, dtype="<f4"), 5126, "VEC3")
	joints = accessor(np.array([[0, 1, 0, 0]] * 3, dtype="u1"), 5121, "VEC4")
	weights = accessor(np.array([[0.25, 0.75, 0, 0]] * 3, dtype="<f4"), 5126, "VEC4")
	indices = accessor(np.array([0, 1, 2], dtype="<u2"), 5123, "SCALAR")
	inverse_bind = np.tile(np.eye(4, dtype="<f4"), (2, 1, 1))
	inverse_bind[0, 3, 0] = -2
	bind = accessor(inverse_bind, 5126, "MAT4")
	time = accessor(np.array([0, 1], dtype="<f4"), 5126, "SCALAR")
	track_values = ([[0, 0, 0, 1], [0, 0, -1, 0]] if rotation else [[0, 0, 0], [2, 0, 0]])
	track = accessor(np.array(track_values, dtype="<f4"), 5126, "VEC4" if rotation else "VEC3")
	nodes = [{"name": "child", "translation": [2, 0, 0]}, {"name": "root", "children": [0]}, {"mesh": 0, "skin": 0}]
	if ancestor_translation is not None:
		nodes.append({"children": [1, 2], "translation": ancestor_translation})
	document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
		"bufferViews": views, "accessors": accessors, "nodes": nodes,
		"skins": [{"joints": [0, 1], "inverseBindMatrices": bind}],
		"meshes": [{"primitives": [{"attributes": {"POSITION": position, "NORMAL": normal, "JOINTS_0": joints, "WEIGHTS_0": weights}, "indices": indices, "material": 0}]}],
		"materials": [{}], "animations": [{"name": "move", "samplers": [{"input": time, "output": track, "interpolation": interpolation}],
			"channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation" if rotation else "translation"}}]}]}
	path.write_text(json.dumps(document), encoding="utf-8")
	return path


def test_skinned_gltf_reorders_joint_weights_and_bind_matrices(tmp_path):
	asset = load_skinned_gltf(write_asset(tmp_path / "skin.gltf"), fps=2)
	assert asset.joint_nodes == (1, 0)
	assert asset.skeleton.parents.tolist() == [-1, 0]
	primitive, = asset.primitives
	assert primitive.material_id == 0
	np.testing.assert_array_equal(primitive.vertices["joints"][:, :2], [[1, 0]] * 3)
	np.testing.assert_allclose(primitive.vertices["weights"][:, :2], [[0.25, 0.75]] * 3)
	np.testing.assert_allclose(asset.skeleton.inverse_bind[:, 0, 3], [0, -2])
	name, clip = asset.clips[0]
	assert name == "move" and clip.fps == 2
	pose = sample_clip(clip, [0, 0.5], loop=False)
	matrices = skin_matrices(asset.skeleton, model_matrices(asset.skeleton, *pose))
	np.testing.assert_allclose(matrices[:, :, 0, 3], [[0, 0], [1, 1]])


def test_skinned_gltf_rejects_unsupported_tracks_and_ancestor_transforms(tmp_path):
	with pytest.raises(ValueError, match="Only LINEAR"):
		load_skinned_gltf(write_asset(tmp_path / "step.gltf", interpolation="STEP"))
	with pytest.raises(ValueError, match="transforms must be identity"):
		load_skinned_gltf(write_asset(tmp_path / "offset.gltf", ancestor_translation=[1, 0, 0]))


def test_skinned_gltf_resamples_quaternion_tracks(tmp_path):
	asset = load_skinned_gltf(write_asset(tmp_path / "rotation.gltf", rotation=True), fps=4)
	clip = asset.clips[0][1]
	np.testing.assert_allclose(clip.rotation[1, 0], [0, 0, -np.sin(np.pi / 8), np.cos(np.pi / 8)], atol=1e-6)
	np.testing.assert_allclose(np.linalg.norm(clip.rotation, axis=-1), 1, atol=1e-6)
