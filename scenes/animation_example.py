"""Greenman skeletal animation: 1 pose, 2 idle, 3 move, 4 attack; Space pauses.

Run from the repository root: python main.py scenes/animation_example.py
The small glTF loader is specific to this asset; it is not a general importer.
"""
from pathlib import Path

import numpy as np

from graaal.Assets import read_gltf_accessor
from graaal.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip, skin_matrices, skin_vertices


ASSET = Path(__file__).parent / "resources" / "greenman.glb"
VERTEX_DTYPE = np.dtype([("position", "<f4", 3), ("normal", "<f4", 3), ("uv", "<f4", 2), ("joints", "<u4", 4), ("weights", "<f4", 4)])


def _interpolate(times, values, target, rotation=False):
	"""Resample a glTF LINEAR track; rotations use shortest-path slerp."""
	if len(times) == 1: return np.broadcast_to(values[0], (len(target), values.shape[1])).copy()
	right = np.clip(np.searchsorted(times, target, side="right"), 1, len(times) - 1)
	left = right - 1
	fraction = np.clip((target - times[left]) / (times[right] - times[left]), 0, 1)[:, None]
	a, b = values[left], values[right]
	if not rotation: return a * (1 - fraction) + b * fraction
	dot = np.sum(a * b, axis=1, keepdims=True)
	b = np.where(dot < 0, -b, b)
	dot = np.clip(np.abs(dot), 0, 1)
	angle = np.arccos(dot)
	sine = np.sin(angle)
	denominator = np.where(sine > 1e-6, sine, 1)
	result = np.where(sine > 1e-6, np.sin((1 - fraction) * angle) / denominator * a + np.sin(fraction * angle) / denominator * b, a * (1 - fraction) + b * fraction)
	return result / np.linalg.norm(result, axis=1, keepdims=True)


def load_greenman(path=ASSET):
	"""Load this asset's joint-ordered rig, indexed mesh, and sampled clips."""
	from pygltflib import GLTF2

	path = Path(path)
	gltf = GLTF2().load(str(path))
	read = lambda index: read_gltf_accessor(gltf, index, path.parent)
	skin = gltf.skins[0]
	joints = skin.joints
	joint_index = {node: i for i, node in enumerate(joints)}
	parents = [-1] * len(joints)
	for node_id in joints:
		for child in gltf.nodes[node_id].children or ():
			if child in joint_index: parents[joint_index[child]] = joint_index[node_id]
	nodes = [gltf.nodes[i] for i in joints]
	if any(node.matrix is not None for node in nodes): raise ValueError("Greenman example expects TRS joint nodes")
	translations = np.array([node.translation or [0, 0, 0] for node in nodes], dtype=np.float32)
	rotations = np.array([node.rotation or [0, 0, 0, 1] for node in nodes], dtype=np.float32)
	scales = np.array([node.scale or [1, 1, 1] for node in nodes], dtype=np.float32)
	inverse_bind = read(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
	rig = Skeleton(parents, translations, rotations, scales, inverse_bind)

	primitive = gltf.meshes[0].primitives[0]
	attributes = primitive.attributes
	positions = read(attributes.POSITION)
	vertices = np.zeros(len(positions), dtype=VERTEX_DTYPE)
	vertices["position"] = positions
	vertices["normal"] = read(attributes.NORMAL)
	vertices["uv"] = read(attributes.TEXCOORD_0)
	vertices["joints"] = read(attributes.JOINTS_0)
	vertices["weights"] = read(attributes.WEIGHTS_0)
	weights = vertices["weights"].sum(axis=1, keepdims=True)
	if np.any(weights <= 0) or np.any(vertices["joints"] >= len(joints)): raise ValueError("Invalid skin weights or joint indices")
	vertices["weights"] /= weights
	indices = read(primitive.indices).reshape(-1).astype(np.uint32)

	clips = []
	for animation in gltf.animations:
		tracks = []
		for channel in animation.channels:
			if channel.target.node not in joint_index: continue
			sampler = animation.samplers[channel.sampler]
			if sampler.interpolation not in (None, "LINEAR"): raise ValueError("Greenman example expects LINEAR tracks")
			times = read(sampler.input).reshape(-1)
			tracks.append((joint_index[channel.target.node], channel.target.path, times, read(sampler.output)))
		duration = max((float(times[-1]) for _, _, times, _ in tracks), default=0)
		count = max(2, int(np.ceil(duration * 30)) + 1) if duration else 1
		fps = (count - 1) / duration if duration else 30
		target = np.linspace(0, duration, count)
		data = {"translation": np.broadcast_to(translations, (count, *translations.shape)).copy(),
			"rotation": np.broadcast_to(rotations, (count, *rotations.shape)).copy(),
			"scale": np.broadcast_to(scales, (count, *scales.shape)).copy()}
		for joint, field, times, values in tracks:
			if field in data: data[field][:, joint] = _interpolate(times, values, target, field == "rotation")
		clips.append((animation.name or "clip", Clip(data["translation"], data["rotation"], data["scale"], fps)))
	return rig, vertices, indices, clips



def run_scene():
	from graaal.DefaultMaterial import DefaultMaterialShader
	from graaal.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext

	rig, vertices, indices, clips = load_greenman()
	RenderContext.init_window(960, 720, "Greenman | 1-4: animation | Space: pause", required_gpu_features=["indirect-first-instance"])
	shader = DefaultMaterialShader(default_material={"base_color": (0.22, 0.65, 0.25, 1), "roughness": 0.65})
	mesh = MaterialMesh(vertices, shader, indices=indices, bounds=((0, 1, 0), (4, 4, 4)))
	renderer = MaterialRenderer([DrawInstance(mesh)])
	camera = Camera((3, 2.2, 5), (0, 1, 0), (0, 1, 0), 45, near=0.05, far=100)
	clip_index, time, playing = min(1, len(clips) - 1), 0.0, True
	print("Animations:", ", ".join(f"{i + 1}: {name}" for i, (name, _) in enumerate(clips)))

	def on_char(code):
		nonlocal clip_index, time, playing
		key = chr(code)
		if key == " ": playing = not playing
		elif key in "123456789" and int(key) <= len(clips):
			clip_index, time = int(key) - 1, 0.0
			print(f"Animation: {clips[clip_index][0]}")
		else: return False
		return True

	RenderContext.subscribe_event("char", on_char)
	while RenderContext.window_loop():
		if playing: time += RenderContext.frame_time
		pose = sample_clip(clips[clip_index][1], [time])
		model = model_matrices(rig, *pose)
		matrices = skin_matrices(rig, model)[0]
		deformed = vertices.copy()
		deformed["position"], deformed["normal"] = skin_vertices(vertices["position"], vertices["normal"], vertices["joints"], vertices["weights"], matrices)
		renderer.update_mesh_vertices(mesh, deformed)
		commands = RenderContext.commands("greenman")
		renderer.record(commands, RenderContext.window_dimensions, camera, RenderContext.screen())
		RenderContext.submit(commands.finish())


if __name__ == "__main__":
	run_scene()
