import numpy as np
import pytest

from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.CrowdGraphValidation import validate_reduced_crowd_graph
from graaal.Animation.CrowdSkeletonLod import reduce_crowd_skeleton
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, skin_matrices, skin_vertices
from graaal.Animation.SkeletalGltf import SKINNED_VERTEX_DTYPE, SkinnedPrimitive


def test_reduction_merges_weights_and_preserves_retained_animation():
	rotation = np.tile([0, 0, 0, 1], (4, 1))
	scale = np.ones((4, 3))
	bind = np.tile(np.eye(4), (4, 1, 1))
	rig = Skeleton([-1, 0, 1, 0], np.zeros((4, 3)), rotation, scale, bind)
	translation = np.zeros((2, 4, 3))
	translation[1, :, 0] = (0, 2, 0.2, -7)
	clip = Clip(translation, [rotation, rotation], [scale, scale], fps=2)
	vertices = np.zeros(2, dtype=SKINNED_VERTEX_DTYPE)
	vertices["joints"] = [(1, 2, 0, 3), (2, 1, 3, 0)]
	vertices["weights"] = [(0.75, 0.25, 0, 0), (0, 0.5, 0.5, 0)]
	primitive = SkinnedPrimitive(vertices, np.array([0, 1, 0]), 0)

	assert reduce_crowd_skeleton(rig, (clip,), (primitive,), error_budget=0.02) is None
	lod = reduce_crowd_skeleton(rig, (clip,), (primitive,), error_budget=0.06)
	assert lod is not None
	np.testing.assert_array_equal(lod.skeleton.parents, [-1, 0, 0])
	np.testing.assert_array_equal(lod.joint_map, [0, 1, 1, 2])
	assert lod.clips[0].duration == clip.duration
	assert lod.max_position_error_bound == pytest.approx(0.05)
	reduced = lod.primitives[0].vertices
	np.testing.assert_array_equal(reduced["joints"], [[1, 0, 0, 0], [1, 0, 2, 0]])
	np.testing.assert_array_equal(reduced["weights"], [[1, 0, 0, 0], [0.5, 0, 0.5, 0]])
	np.testing.assert_array_equal(vertices["joints"], [[1, 2, 0, 3], [2, 1, 3, 0]])
	full_pose = skin_matrices(rig, model_matrices(rig, clip.translation, clip.rotation, clip.scale))[1]
	short = lod.clips[0]
	reduced_pose = skin_matrices(lod.skeleton, model_matrices(lod.skeleton, short.translation, short.rotation, short.scale))[1]
	full_points, reduced_points = [
		skin_vertices(v["position"], None, v["joints"], v["weights"], pose)[0]
		for v, pose in ((vertices, full_pose), (reduced, reduced_pose))
	]
	np.testing.assert_allclose(full_points - reduced_points, [[0.05, 0, 0], [0, 0, 0]], atol=1e-6)


def test_graph_validation_reports_reduced_vertex_error():
	rotation = np.tile([0, 0, 0, 1], (2, 1))
	scale = np.ones((2, 3))
	rig = Skeleton([-1, 0], np.zeros((2, 3)), rotation, scale, np.tile(np.eye(4), (2, 1, 1)))
	clip = Clip([[[0, 0, 0], [1, 0, 0]]], [rotation], [scale])
	vertices = np.zeros(1, dtype=SKINNED_VERTEX_DTYPE)
	vertices["joints"][0, 0] = 1
	vertices["weights"][0, 0] = 1
	primitive = SkinnedPrimitive(vertices, np.zeros(3, dtype=np.intp), None)
	graph = AnimationGraphDefinition({
		"version": 1, "initial_state": "idle", "states": {"idle": {"clip": "idle"}},
		"sample_slots": ["base"], "tasks": [{"name": "base", "op": "sample", "slot": "base"}],
	}).compile(rig, {"idle": clip})
	lod = reduce_crowd_skeleton(rig, [clip], [primitive], error_budget=1)
	assert lod is not None
	report = validate_reduced_crowd_graph(graph, lod, error_budget=0.9)
	assert report.pool.bones == 1
	assert report.states["idle"].max_position_error == pytest.approx(1)
	assert not report.states["idle"].eligible
