"""CPU geometry shared by the physics example viewers."""

import numpy as np


def cube_vertices():
	from graaal.RenderPasses.TransparencyData import VERTEX_DTYPE

	# Each face has its own normals and UVs; collision geometry is a separate asset.
	normals = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.float32)
	up = np.array([(0, 1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (0, 1, 0)], dtype=np.float32)
	right = np.cross(up, normals)
	uv = np.array([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)], dtype=np.float32)
	vertices = np.zeros((6, 6), dtype=VERTEX_DTYPE)
	vertices["position"] = normals[:, None] * 0.5 + right[:, None] * (uv[None, :, :1] - 0.5) + up[:, None] * (uv[None, :, 1:] - 0.5)
	vertices["normal"] = normals[:, None]
	vertices["uv"] = uv
	return vertices.reshape(-1)
