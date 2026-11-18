from types import SimpleNamespace

import numpy as np
import pytest

from graaal.MaterialSkinning import MaterialSkinningAdapter


def test_owner_mapping_uses_bind_pose_and_rejects_stale_handles():
	dtype = np.dtype([("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2), ("joints", "u4", 1), ("weights", "f4", 1)])
	vertices = np.zeros(3, dtype=dtype)
	vertices["normal"] = (0, 1, 0)
	vertices["weights"] = 1
	class Mesh:
		bounds = ((0, 0, 0), (5, 5, 5))
		def __init__(self): self.vertices = vertices
	first, second = Mesh(), Mesh()
	writes = []
	renderer = SimpleNamespace(mesh_ids={first: 0, second: 1}, update_mesh_vertices=lambda mesh, data: writes.append((mesh, data)))
	adapter = MaterialSkinningAdapter(renderer)
	old, other = np.uint64(0), np.uint64(1)
	recycled = np.uint64(1 << 32)
	adapter.bind(old, first, vertices)
	adapter.bind(other, second, vertices)
	with pytest.raises(ValueError, match="distinct material meshes"):
		adapter.bind(recycled, first, vertices)
	matrices = np.tile(np.eye(4, dtype=np.float32)[:3], (2, 1, 1, 1))
	matrices[0, 0, 0, 3], matrices[1, 0, 0, 3] = 2, 3
	adapter.update(np.array([other, old]), matrices)
	assert [mesh for mesh, _ in writes] == [second, first]
	np.testing.assert_allclose(writes[0][1]["position"][:, 0], 2)
	np.testing.assert_allclose(writes[1][1]["position"][:, 0], 3)
	writes.clear()
	adapter.update(np.array([other, old]), matrices)
	np.testing.assert_allclose(writes[0][1]["position"][:, 0], 2)
	writes.clear()
	adapter.unbind(old)
	adapter.bind(recycled, first, vertices)
	with pytest.raises(ValueError, match="No skinned mesh bound"):
		adapter.update(np.array([other, old]), matrices)
	assert not writes
	adapter.update(np.array([other, recycled]), matrices)
	assert len(writes) == 2
	with pytest.raises(ValueError, match="distinct"):
		adapter.update(np.array([other, other]), matrices)


def test_material_skinning_preflights_all_poses():
	dtype = np.dtype([("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2), ("joints", "u4", 1), ("weights", "f4", 1)])
	vertices = np.zeros(3, dtype=dtype)
	vertices["normal"], vertices["weights"] = (0, 1, 0), 1
	class Mesh:
		bounds = ((0, 0, 0), (5, 5, 5))
		def __init__(self): self.vertices = vertices
	meshes = [Mesh(), Mesh()]
	writes = []
	renderer = SimpleNamespace(mesh_ids={mesh: i for i, mesh in enumerate(meshes)}, update_mesh_vertices=lambda *args: writes.append(args))
	adapter = MaterialSkinningAdapter(renderer)
	for i, mesh in enumerate(meshes): adapter.bind(i, mesh, vertices)
	matrices = np.tile(np.eye(4, dtype=np.float32)[:3], (2, 1, 1, 1))
	matrices[1, 0, :3, :3] = 0
	with pytest.raises(ValueError, match="singular"):
		adapter.update(np.array([0, 1]), matrices)
	assert not writes
