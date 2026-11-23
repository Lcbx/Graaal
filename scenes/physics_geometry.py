"""CPU geometry shared by the physics viewers and ragdoll tuner."""

import numpy as np


def collider_poses(positions, orientations, offsets, shape_rotations):
	"""Compose xyzw bone poses with their bone-local collider transforms."""
	from graaal.Math import Quaternion

	centers = positions + np.einsum("...ij,...j->...i", Quaternion.matrix_batch(orientations), offsets)
	return centers, Quaternion.multiply_batch(orientations, shape_rotations)


def collider_bounds(bones, positions, orientations):
	"""Return the world-space AABB of each collider for one pose or a batch."""
	from graaal.Math import Quaternion

	centers, rotations = collider_poses(positions, orientations, [bone.shape_offset for bone in bones], [bone.shape_rotation for bone in bones])
	axes = Quaternion.matrix_batch(rotations)
	extents = []
	for i, bone in enumerate(bones):
		axis = axes[..., i, :, :]
		kind, size = bone.shape.kind.name, bone.shape.params
		if kind == "BOX":
			extent = np.abs(axis) @ (np.asarray(size[:3]) / 2)
		elif kind == "CAPSULE":
			extent = size[0] + np.abs(axis[..., :, 1]) * size[1] / 2
		elif kind == "CYLINDER":
			extent = np.linalg.norm(axis[..., :, (0, 2)], axis=-1) * size[0] + np.abs(axis[..., :, 1]) * size[1] / 2
		else:  # Sphere
			extent = np.full(axis.shape[:-1], size[0])
		extents.append(extent)
	half_sizes = np.stack(extents, axis=-2)
	return centers - half_sizes, centers + half_sizes


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


def rounded_vertices(template, radius, cylinder_length, *, segments=24, rings=8):
	u = np.linspace(0, 2 * np.pi, segments + 1)
	v = np.concatenate((np.linspace(0, np.pi / 2, rings + 1), np.linspace(np.pi / 2, np.pi, rings + 1)))
	u, v = np.meshgrid(u, v)
	normals = np.stack((np.sin(v) * np.cos(u), np.cos(v), np.sin(v) * np.sin(u)), axis=-1)
	grid = np.zeros(u.shape, dtype=template.dtype)
	grid["position"], grid["normal"] = normals * radius, normals
	# Duplicate the equator to connect the two hemispheres with a cylinder.
	grid["position"][:rings + 1, :, 1] += cylinder_length / 2
	grid["position"][rings + 1:, :, 1] -= cylinder_length / 2
	grid["uv"] = np.stack((u / (2 * np.pi), v / np.pi), axis=-1)
	return np.stack((grid[:-1, :-1], grid[1:, 1:], grid[1:, :-1],
		grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:]), axis=2).reshape(-1)


def cylinder_vertices(radius, length, *, segments=24):
	angles = np.arange(segments) * (2 * np.pi / segments)
	top = np.column_stack((radius * np.cos(angles), np.full(segments, length / 2), radius * np.sin(angles)))
	points = np.concatenate((top, top * [1, -1, 1], [[0, length / 2, 0], [0, -length / 2, 0]]))
	a = np.arange(segments)
	b = (a + 1) % segments
	triangles = np.concatenate((
		np.column_stack((a, b, b + segments)),
		np.column_stack((a, b + segments, a + segments)),
		np.column_stack((np.full(segments, 2 * segments), b, a)),
		np.column_stack((np.full(segments, 2 * segments + 1), a + segments, b + segments)),
	))
	return triangle_vertices(points, triangles)


def skinned_boxes(asset, cube):
	"""One skinned mesh of boxes showing the prefab's offset capsule volumes."""
	from graaal.Animation.SkeletalAnimation import model_matrices
	from graaal.Animation.SkeletalGltf import SKINNED_VERTEX_DTYPE
	from graaal.Math import Quaternion

	rig = asset.skeleton
	model = model_matrices(rig, rig.translation[None], rig.rotation[None], rig.scale[None])[0]
	vertices = np.zeros((len(asset.bones), len(cube)), dtype=SKINNED_VERTEX_DTYPE)
	for i, bone in enumerate(asset.settings.bones):
		radius, cylinder_length = bone.shape.params[:2]
		size = np.array([radius * 2, radius * 2 + cylinder_length, radius * 2])
		shape_rotation = Quaternion(bone.shape_rotation).matrix33
		position = (cube["position"] * size) @ shape_rotation.T + bone.shape_offset
		vertices[i]["position"] = position @ model[asset.bones[i], :3, :3].T + model[asset.bones[i], :3, 3]
		vertices[i]["normal"] = cube["normal"] @ shape_rotation.T @ model[asset.bones[i], :3, :3].T
		vertices[i]["uv"] = cube["uv"]
		vertices[i]["joints"][:, 0] = asset.bones[i]
		vertices[i]["weights"][:, 0] = 1
	return vertices.reshape(-1)


def triangle_vertices(points, triangles):
	"""Expand an indexed triangle mesh with one normal per face."""
	from graaal.RenderPasses.TransparencyData import VERTEX_DTYPE

	faces = points[triangles]
	normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
	normals /= np.linalg.norm(normals, axis=1, keepdims=True)
	vertices = np.zeros(faces.shape[:2], dtype=VERTEX_DTYPE)
	vertices["position"] = faces
	vertices["normal"] = normals[:, None]
	return vertices.reshape(-1)


def segment_transform(start, end, width):
	"""Place a unit box along a world-space segment using its local Y axis."""
	from graaal.Math import Quaternion

	delta = end - start
	x, y, z = delta
	angle = np.arctan2(np.hypot(x, z), y)
	rotation = Quaternion.from_axis_rotation((z, 0, -x), angle) if x or z else Quaternion.from_x_rotation(angle)
	return (start + end) / 2, rotation, (width, np.linalg.norm(delta), width)
