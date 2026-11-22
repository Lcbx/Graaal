"""CPU import of a supported subset of skinned glTF, independent of rendering."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..Assets import gltf_node_matrix, read_gltf_accessor
from ..Math import Quaternion
from .SkeletalAnimation import Clip, Skeleton


SKINNED_VERTEX_DTYPE = np.dtype([
	("position", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2),
	("joints", "<u4", 4), ("weights", "<f4", 4),
])


@dataclass(frozen=True)
class SkinnedPrimitive:
	vertices: np.ndarray
	indices: np.ndarray
	material_id: int | None


@dataclass(frozen=True)
class SkinnedAsset:
	skeleton: Skeleton
	primitives: tuple[SkinnedPrimitive, ...]
	clips: tuple[tuple[str, Clip], ...]
	joint_nodes: tuple[int, ...]


def _sample_track(times: np.ndarray, values: np.ndarray, target: np.ndarray, rotation: bool) -> np.ndarray:
	if len(times) == 1: return np.broadcast_to(values[0], (len(target), values.shape[1])).copy()
	right = np.clip(np.searchsorted(times, target, side="right"), 1, len(times) - 1)
	left = right - 1
	fraction = np.clip((target - times[left]) / (times[right] - times[left]), 0, 1)[:, None]
	a, b = values[left], values[right]
	if not rotation: return a * (1 - fraction) + b * fraction
	return Quaternion.slerp_batch(a, b, fraction)


def load_skinned_gltf(path: str | Path, *, node: int | None=None, fps: float=30) -> SkinnedAsset:
	"""Load a skinned mesh node, depth-first skeleton and uniform-rate clips.

	Supports triangle lists, four influences and LINEAR joint TRS tracks. Mesh
	and non-joint ancestor transforms must be identity; non-joint nodes between
	joints, morph targets and other interpolation modes are not supported.
	Material IDs are returned, but material and image loading is caller-owned.
	"""
	from pygltflib import GLTF2

	if not np.isscalar(fps) or not np.isfinite(fps) or fps <= 0: raise ValueError("fps must be positive and finite")
	path = Path(path)
	gltf = GLTF2().load(str(path))
	read = lambda accessor: read_gltf_accessor(gltf, accessor, path.parent)
	nodes = gltf.nodes
	parents = [-1] * len(nodes)
	for i, parent in enumerate(nodes):
		for child in parent.children or ():
			if child < 0 or child >= len(nodes) or parents[child] != -1: raise ValueError("glTF nodes must form a tree")
			parents[child] = i
	for i in range(len(nodes)):
		seen, current = set(), i
		while current != -1:
			if current in seen: raise ValueError("glTF nodes must not contain cycles")
			seen.add(current)
			current = parents[current]
	if node is None:
		candidates = [i for i, entry in enumerate(nodes) if entry.skin is not None and entry.mesh is not None]
		if len(candidates) != 1: raise ValueError("Specify a skinned mesh node when the file does not have exactly one")
		node = candidates[0]
	if not isinstance(node, (int, np.integer)) or node < 0 or node >= len(nodes): raise ValueError("Invalid skinned mesh node")
	mesh_node = nodes[node]
	if mesh_node.skin is None or mesh_node.mesh is None: raise ValueError("Node must have both mesh and skin")
	skin = gltf.skins[mesh_node.skin]
	joint_list = list(skin.joints)
	if not joint_list or len(set(joint_list)) != len(joint_list) or any(j < 0 or j >= len(nodes) for j in joint_list):
		raise ValueError("Skin joints must be distinct valid nodes")
	joint_set = set(joint_list)
	protected = set()
	for joint in joint_list:
		parent = parents[joint]
		if parent not in joint_set:
			while parent != -1:
				if parent in joint_set: raise ValueError("Non-joint nodes between joints are not supported")
				protected.add(parent)
				parent = parents[parent]
	parent = node
	while parent != -1:
		if parent in joint_set: raise ValueError("Skinned mesh nodes must not descend from joints")
		protected.add(parent)
		parent = parents[parent]
	if any(not np.allclose(gltf_node_matrix(nodes[i]), np.eye(4), atol=1e-6) for i in protected):
		raise ValueError("Mesh and non-joint ancestor transforms must be identity")
	order = []
	def visit(joint: int) -> None:
		order.append(joint)
		for child in nodes[joint].children or ():
			if child in joint_set: visit(child)
	for joint in joint_list:
		if parents[joint] not in joint_set: visit(joint)
	joint_index = {joint: i for i, joint in enumerate(order)}
	old_index = {joint: i for i, joint in enumerate(joint_list)}
	remap = np.array([joint_index[joint] for joint in joint_list], dtype=np.uint32)
	ordered_nodes = [nodes[i] for i in order]
	if any(entry.matrix is not None for entry in ordered_nodes): raise ValueError("Joint nodes must use TRS rather than matrices")
	translations = np.array([entry.translation or [0, 0, 0] for entry in ordered_nodes], dtype=np.float32)
	rotations = np.array([entry.rotation or [0, 0, 0, 1] for entry in ordered_nodes], dtype=np.float32)
	scales = np.array([entry.scale or [1, 1, 1] for entry in ordered_nodes], dtype=np.float32)
	if skin.inverseBindMatrices is None: inverse_bind = np.broadcast_to(np.eye(4), (len(order), 4, 4)).copy()
	else:
		raw = read(skin.inverseBindMatrices)
		if raw.shape != (len(order), 16): raise ValueError("Invalid inverse bind matrices")
		inverse_bind = raw.reshape(-1, 4, 4).transpose(0, 2, 1)[[old_index[joint] for joint in order]]
	rig = Skeleton([joint_index.get(parents[joint], -1) for joint in order], translations, rotations, scales, inverse_bind)

	primitives = []
	for primitive in gltf.meshes[mesh_node.mesh].primitives:
		attributes = primitive.attributes
		if primitive.mode not in (None, 4) or primitive.targets:
			raise ValueError("Skinned meshes require triangle lists without morph targets")
		if any(getattr(attributes, name, None) is None for name in ("POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0")):
			raise ValueError("Skinned mesh needs POSITION, NORMAL, JOINTS_0 and WEIGHTS_0")
		if getattr(attributes, "JOINTS_1", None) is not None or getattr(attributes, "WEIGHTS_1", None) is not None:
			raise ValueError("More than four skin influences are not supported")
		positions, normals = read(attributes.POSITION), read(attributes.NORMAL)
		joints, weights = read(attributes.JOINTS_0), read(attributes.WEIGHTS_0)
		count = len(positions)
		if positions.shape != (count, 3) or normals.shape != (count, 3) or joints.shape != (count, 4) or weights.shape != (count, 4):
			raise ValueError("Invalid skinned vertex attribute shapes")
		if joints.dtype.kind not in "iu" or np.any(joints < 0) or np.any(joints >= len(order)) or not np.isfinite(weights).all() or np.any(weights < 0):
			raise ValueError("Invalid skin weights or joint indices")
		weight_sum = weights.sum(axis=1, keepdims=True)
		if not np.isfinite(weight_sum).all() or np.any(weight_sum <= 0) or not np.isfinite(positions).all() or not np.isfinite(normals).all(): raise ValueError("Invalid skinned vertex data")
		vertices = np.zeros(count, dtype=SKINNED_VERTEX_DTYPE)
		vertices["position"], vertices["normal"] = positions, normals
		vertices["joints"], vertices["weights"] = remap[joints], weights / weight_sum
		if attributes.TEXCOORD_0 is not None:
			uv = read(attributes.TEXCOORD_0)
			if uv.shape != (count, 2) or not np.isfinite(uv).all(): raise ValueError("Invalid texture coordinates")
			vertices["uv"] = uv
		indices = read(primitive.indices).reshape(-1) if primitive.indices is not None else np.arange(count, dtype=np.uint32)
		if indices.dtype.kind not in "iu" or not len(indices) or len(indices) % 3 or np.any(indices < 0) or np.any(indices >= count):
			raise ValueError("Invalid triangle indices")
		primitives.append(SkinnedPrimitive(vertices, indices.astype(np.uint32), primitive.material))
	if not primitives: raise ValueError("Skinned mesh has no primitives")

	clips = []
	for animation in gltf.animations:
		tracks, used = [], set()
		for channel in animation.channels:
			target = channel.target
			if target.node in protected: raise ValueError("Animated mesh or non-joint ancestors are not supported")
			if target.node not in joint_index: continue
			if target.path not in ("translation", "rotation", "scale"): raise ValueError("Unsupported joint animation path")
			key = target.node, target.path
			if key in used: raise ValueError("Duplicate animation channel")
			used.add(key)
			sampler = animation.samplers[channel.sampler]
			if sampler.interpolation not in (None, "LINEAR"): raise ValueError("Only LINEAR animation tracks are supported")
			times, values = read(sampler.input), read(sampler.output)
			if times.ndim != 2 or times.shape[1] != 1 or not len(times): raise ValueError("Invalid animation times")
			times = times[:, 0]
			width = 4 if target.path == "rotation" else 3
			if values.shape != (len(times), width) or not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(times < 0) or np.any(np.diff(times) <= 0):
				raise ValueError("Invalid animation track")
			if target.path == "rotation":
				length = np.linalg.norm(values, axis=1, keepdims=True)
				if np.any(length < 1e-8): raise ValueError("Invalid animation quaternion")
				values = values / length
			tracks.append((joint_index[target.node], target.path, times, values))
		duration = max((float(times[-1]) for _, _, times, _ in tracks), default=0)
		count = max(2, int(np.ceil(duration * fps)) + 1) if duration else 1
		rate = (count - 1) / duration if duration else fps
		targets = np.linspace(0, duration, count)
		data = {"translation": np.broadcast_to(translations, (count, *translations.shape)).copy(),
			"rotation": np.broadcast_to(rotations, (count, *rotations.shape)).copy(),
			"scale": np.broadcast_to(scales, (count, *scales.shape)).copy()}
		for joint, field, times, values in tracks:
			data[field][:, joint] = _sample_track(times, values, targets, field == "rotation")
		clips.append((animation.name or "clip", Clip(data["translation"], data["rotation"], data["scale"], rate)))
	return SkinnedAsset(rig, tuple(primitives), tuple(clips), tuple(order))
