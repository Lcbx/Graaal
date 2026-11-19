import numpy as np
import pytest
from graaal.Math import Quaternion


def test_quaternion_construction_and_operations():
	np.testing.assert_allclose(np.asarray(Quaternion()), [0, 0, 0, 1])
	q = Quaternion.from_axis_rotation((0.0, 1.0, 0.0), 0.7)
	np.testing.assert_allclose(np.asarray(q), [0, np.sin(0.35), 0, np.cos(0.35)])
	np.testing.assert_allclose(np.asarray(q.cross(q)), [0, np.sin(0.7), 0, np.cos(0.7)])
	np.testing.assert_allclose(np.asarray(q.inverse), [0, -np.sin(0.35), 0, np.cos(0.35)])
	np.testing.assert_allclose(q.matrix33 @ [0, 0, 1], [np.sin(0.7), 0, np.cos(0.7)])


def test_batch_matrix_conversion_and_broadcast_interpolation():
	quaternions = np.asarray([Quaternion(), Quaternion.from_x_rotation(np.pi), Quaternion.from_y_rotation(np.pi), Quaternion.from_z_rotation(np.pi)])
	matrices = Quaternion.matrix_batch(quaternions)
	for q, matrix in zip(quaternions, matrices):
		np.testing.assert_allclose(matrix, Quaternion(q).matrix33, atol=1e-6)
	converted = Quaternion.from_matrices_batch(matrices)
	np.testing.assert_allclose(np.abs(np.sum(converted * quaternions, axis=1)), 1, atol=1e-6)
	weights = np.array([[0], [0.5], [1]], dtype=np.float32)
	mixed = Quaternion.nlerp_batch([0, 0, 0, 1], [0, 0, 1, 0], weights)
	np.testing.assert_allclose(mixed[:, 2], [0, np.sqrt(0.5), 1], atol=1e-6)
	spherical = Quaternion.slerp_batch([0, 0, 0, 1], [0, 0, -1, 0], weights)
	np.testing.assert_allclose(spherical[:, 2], [0, -np.sqrt(0.5), -1], atol=1e-6)
	q = Quaternion.from_y_rotation(0.7)
	np.testing.assert_allclose(Quaternion.slerp_batch(np.asarray(q), -np.asarray(q), 0.5), np.asarray(q), atol=1e-6)


def test_quaternion_compatibility_methods_and_rotation_conventions():
	q = Quaternion.from_z_rotation(np.pi / 2)
	np.testing.assert_allclose(np.asarray(q * [0, 0, 1, 0]), [0, 0, np.sqrt(0.5), -np.sqrt(0.5)], atol=1e-6)
	np.testing.assert_allclose(q * [1, 0, 0], [0, 1, 0], atol=1e-6)
	np.testing.assert_allclose(np.asarray(q.cross(q.inverse)), [0, 0, 0, 1], atol=1e-6)
	assert q == Quaternion(-np.asarray(q))
	assert q != Quaternion()
	np.testing.assert_allclose(np.asarray(Quaternion.from_matrix(q.matrix33)), np.asarray(q), atol=1e-6)
	np.testing.assert_allclose(q.matrix44[:3, :3], q.matrix33)
	np.testing.assert_allclose(np.asarray(Quaternion.from_eulers([0, 0, np.pi / 2])), np.asarray(q), atol=1e-6)
	np.testing.assert_allclose(np.asarray(Quaternion.from_axis([0, 0, np.pi / 2])), np.asarray(q), atol=1e-6)
	editable = Quaternion()
	editable.xy = [0.1, 0.2]
	editable.xzw = [0.3, 0.4, 0.5]
	np.testing.assert_allclose(np.asarray(editable), [0.3, 0.2, 0.4, 0.5])
	with pytest.raises(ValueError, match="zero quaternion"): Quaternion([0, 0, 0, 0]).normalized
