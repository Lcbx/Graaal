"""CPU pose and camera helpers for the timeline viewport."""
from dataclasses import dataclass

import numpy as np

from graaal.Assets import gltf_node_matrix
from graaal.Math import Camera, Quaternion
from graaal.SkeletalAnimation import model_matrices, sample_clip, skin_matrices, skin_vertices


@dataclass
class Preview:
	nodes: list
	parents: np.ndarray
	order: tuple
	bones: np.ndarray
	asset: object = None
	message: str = "Skeleton preview"


def load_preview(gltf, path):
	parents = np.full(len(gltf.nodes), -1, dtype=np.intp)
	for parent, node in enumerate(gltf.nodes):
		for child in node.children or ():
			if not 0 <= child < len(parents) or parents[child] != -1: raise ValueError("Invalid node hierarchy")
			parents[child] = parent
	order, pending = [], set(range(len(parents)))
	while pending:
		ready = sorted(node for node in pending if parents[node] == -1 or parents[node] not in pending)
		if not ready: raise ValueError("Node hierarchy contains a cycle")
		order.extend(ready)
		pending.difference_update(ready)
	bones = np.array(sorted({joint for skin in gltf.skins or () for joint in skin.joints}), dtype=np.intp)
	if not len(bones): bones = np.arange(len(parents))
	if np.any(bones < 0) or np.any(bones >= len(parents)): raise ValueError("Invalid skin joints")
	preview = Preview(gltf.nodes, parents, tuple(order), bones)
	if gltf.skins:
		from graaal.SkeletalGltf import load_skinned_gltf
		try:
			preview.asset = load_skinned_gltf(path)
			preview.message = "Skinned mesh + bones"
		except ValueError as exc: preview.message = f"Skeleton preview: {exc}"
	return preview


def sample_track(track, time):
	times, values = track.times, track.values
	if time <= times[0]: return values[0]
	if time >= times[-1]: return values[-1]
	right = int(np.searchsorted(times, time, side="right"))
	left = right - 1
	span = times[right] - times[left]
	u = (time - times[left]) / span
	if track.interpolation == "STEP": return values[left]
	if track.interpolation == "CUBICSPLINE":
		result = (2*u**3 - 3*u**2 + 1)*values[left] + (u**3 - 2*u**2 + u)*span*track.tangents[left, 1] + (-2*u**3 + 3*u**2)*values[right] + (u**3 - u**2)*span*track.tangents[right, 0]
	elif track.path == "rotation": result = Quaternion.slerp_batch(values[left], values[right], u)
	else: result = values[left] * (1-u) + values[right] * u
	if track.path == "rotation":
		length = np.linalg.norm(result)
		if length == 0: raise ValueError("Animation produced a zero quaternion")
		result = result / length
	return result


def pose_matrices(animation, time):
	preview = animation.preview
	if not preview.nodes: return np.empty((0, 4, 4))
	joints = {}
	if animation.clip is not None:
		# Imported clips use the same sampling and hierarchy evaluation as playback.
		asset = preview.asset
		joints = dict(zip(asset.joint_nodes, model_matrices(asset.skeleton, *sample_clip(animation.clip, [time], loop=False))[0]))
	local = np.array([gltf_node_matrix(node) for node in preview.nodes])
	channels = {}
	for track in animation.tracks:
		if track.node is not None and track.node not in joints and track.path in ("translation", "rotation", "scale"):
			channels.setdefault(track.node, {})[track.path] = sample_track(track, time)
	for node_id, fields in channels.items():
		node = preview.nodes[node_id]
		translation = fields.get("translation", node.translation or [0, 0, 0])
		rotation = np.asarray(fields.get("rotation", node.rotation or [0, 0, 0, 1]))
		scale = fields.get("scale", node.scale or [1, 1, 1])
		local[node_id] = np.eye(4)
		local[node_id, :3, :3] = Quaternion.matrix_batch(rotation / np.linalg.norm(rotation)) * np.asarray(scale)[None, :]
		local[node_id, :3, 3] = translation
	model = local.copy()
	for node in preview.order:
		if node in joints: model[node] = joints[node]
		elif preview.parents[node] != -1: model[node] = model[preview.parents[node]] @ local[node]
	return model


def animation_bounds(animation):
	points = [pose_matrices(animation, t)[animation.preview.bones, :3, 3] for t in np.linspace(0, animation.duration, 25)]
	points = np.concatenate(points)
	return (points.min(axis=0), points.max(axis=0)) if len(points) else (np.zeros(3), np.ones(3))


def camera_from_bounds(bounds, aspect):
	lower, upper = bounds
	target = (lower + upper) * 0.5
	radius = max(float(np.linalg.norm(upper - lower)) * 0.5, 0.1)
	half_fov = min(np.deg2rad(22.5), np.arctan(np.tan(np.deg2rad(22.5)) * aspect))
	distance = radius * 1.2 / np.sin(half_fov)
	direction = np.array([0.7, 0.35, 1.0])
	position = target + direction / np.linalg.norm(direction) * distance
	return position, target


def project(points, position, target, width, height):
	# Math.Camera matrices act on row vectors; canvas Y points down.
	local = np.column_stack((points, np.ones(len(points)))) @ Camera.calc_view(position, target, [0, 1, 0])
	clip = local @ Camera.perspective_projection(45, width / height, 0.1, 1000)
	depth = -local[:, 2]
	ndc = clip[:, :2] / np.maximum(depth[:, None], 1e-6)
	return (ndc * [1, -1] + 1) * [width * 0.5, height * 0.5], depth


def mesh_triangles(preview, model):
	if preview.asset is None: return np.empty((0, 3, 3))
	asset = preview.asset
	matrices = skin_matrices(asset.skeleton, model[None, list(asset.joint_nodes)])[0]
	triangles = []
	for primitive in asset.primitives:
		vertices = primitive.vertices
		positions, _ = skin_vertices(vertices["position"], None, vertices["joints"], vertices["weights"], matrices)
		triangles.append(positions[primitive.indices.reshape(-1, 3)])
	return np.concatenate(triangles) if triangles else np.empty((0, 3, 3))
