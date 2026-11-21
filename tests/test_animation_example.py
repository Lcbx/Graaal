import numpy as np
import pytest

pytest.importorskip("pygltflib")

from graaal.Animation.SkeletalAnimation import model_matrices, sample_clip, skin_matrices
from scenes.simple_animation_example import ASSET, load_greenman


def test_greenman_clips_drive_valid_skin_matrices():
	if not ASSET.exists(): pytest.skip("Greenman asset is unavailable")
	rig, vertices, indices, clips = load_greenman()
	assert len(indices) and np.max(indices) < len(vertices)
	np.testing.assert_allclose(vertices["weights"].sum(axis=1), 1, atol=1e-6)
	assert [name for name, _ in clips] == ["0_pose", "1_idle", "2_move", "3_attack"]
	for _, clip in clips:
		pose = sample_clip(clip, [0, clip.duration * 0.35])
		matrices = skin_matrices(rig, model_matrices(rig, *pose))
		assert matrices.shape == (2, len(rig.parents), 3, 4)
		assert np.isfinite(matrices).all()
	move = clips[2][1]
	pose = sample_clip(move, [0, move.duration * 0.35])
	matrices = skin_matrices(rig, model_matrices(rig, *pose))
	assert not np.allclose(matrices[0], matrices[1])
