import numpy as np
import pytest

pytest.importorskip("pygltflib")

from graaal.Animation.RootMotion import RootMotion
from graaal.Animation.SkeletalAnimation import model_matrices, sample_clip, skin_matrices, skin_vertices
from scenes.animation_dummy import load_dummy
from scenes.locomotion_example import make_locomotion_clips


def test_dummy_clips_drive_valid_skin_matrices():
	rig, primitive, clips = load_dummy()
	vertices, indices = primitive.vertices, primitive.indices
	assert len(indices) and np.max(indices) < len(vertices)
	np.testing.assert_allclose(vertices["weights"].sum(axis=1), 1, atol=1e-6)
	assert [name for name, _ in clips] == ["Rest pose", "Walking", "Running"]
	for name, clip in clips:
		pose = sample_clip(clip, [0, clip.duration * 0.35])
		matrices = skin_matrices(rig, model_matrices(rig, *pose))
		assert matrices.shape == (2, len(rig.parents), 3, 4)
		assert np.isfinite(matrices).all()
		if name == "Rest pose":
			positions, _ = skin_vertices(vertices["position"], None, vertices["joints"], vertices["weights"], matrices[0])
			np.testing.assert_allclose(positions, vertices["position"], atol=1e-6)
		else:
			assert not np.allclose(matrices[0], matrices[1])


def test_city_locomotion_extracts_trajectory_and_preserves_the_dummy_pose():
	rig, primitive, named_clips = load_dummy()
	city_rig, city_primitive, clips = make_locomotion_clips(rig, primitive, named_clips)
	np.testing.assert_array_equal(city_primitive.vertices["joints"], primitive.vertices["joints"] + 1)
	for clip_id, speed in enumerate((0, 1.5, 4)):
		motion = RootMotion(city_rig, clips[clip_id])
		np.testing.assert_allclose(motion.character_delta([0], [2])[0, :3, 3], (0, 0, 2 * speed), atol=1e-6)
		time = [0.2]
		pose = motion.in_place(*sample_clip(clips[clip_id], time))
		original = sample_clip(named_clips[clip_id][1], time)
		for value, expected in zip(pose, original):
			np.testing.assert_allclose(value[:, 1:], expected)
