import numpy as np
import pytest

from graaal.Materials.DefaultMaterial import DefaultMaterialShader
from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from graaal.Math import Camera
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
from graaal.Animation.SkeletalAnimation import skin_vertices


@pytest.mark.parametrize("transparent", [False, True])
def test_shared_mesh_uses_independent_gpu_palettes(transparent):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	dtype = np.dtype([("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2), ("joints", "u4", 4), ("weights", "f4", 4)])
	vertices = np.zeros(3, dtype=dtype)
	vertices["position"] = [[-0.3, -0.4, 0], [0.3, -0.4, 0], [0, 0.4, 0]]
	vertices["normal"] = (0, 0, 1)
	vertices["weights"][:, 0] = 1
	shader = DefaultMaterialShader(transparent=transparent, default_material={"base_color": (1, 0.3, 0.1, 0.8 if transparent else 1), "alpha_mode": 2 if transparent else 0})
	mesh = MaterialMesh(vertices, shader, bounds=((0, 0, 0), (3, 3, 1)))
	palette = SkinMatrixBuffer(2)
	palette.acquire(1, 1)
	palette.acquire(2, 1)
	poses = np.tile(np.eye(4, dtype=np.float32)[:3], (2, 1, 1, 1))
	poses[0, 0, 0, 3], poses[1, 0, 0, 3] = -0.8, 0.8
	palette.write(np.array([1, 2], dtype=np.uint64), poses)
	renderer = MaterialRenderer([DrawInstance(mesh, surface_id=1), DrawInstance(mesh, surface_id=2)], skin_palette=palette, target_format="rgba8unorm")
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 50)
	output = Texture((96, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def render():
		commands = RenderContext.commands()
		renderer.record(commands, (96, 64), camera, output.color_attachment())
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 384}, (96, 64, 1))
		return np.frombuffer(data, dtype=np.uint8).reshape(64, 96, 4).copy()
	image = render()
	assert np.max(image[:, :48, :3]) > 20 and np.max(image[:, 48:, :3]) > 20
	poses[1, 0, 0, 3] = -0.8
	palette.write(np.array([1, 2], dtype=np.uint64), poses)
	changed = render()
	assert np.count_nonzero(image[:, 48:, :3] != changed[:, 48:, :3]) > 50


@pytest.mark.parametrize("transparent", [False, True])
@pytest.mark.parametrize("sample_count", [1, 4])
def test_gpu_skinning_matches_cpu_material_shading(transparent, sample_count):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	dtype = np.dtype([("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2), ("joints", "u4", 4), ("weights", "f4", 4)])
	vertices = np.zeros(3, dtype=dtype)
	vertices["position"] = [[-0.5, -0.4, 0], [0.5, -0.4, 0], [0, 0.5, 0]]
	vertices["normal"] = (0, 0, 1)
	vertices["weights"][:, 0] = 1
	vertices["joints"][1, 0] = 1
	poses = np.tile(np.eye(4, dtype=np.float32)[:3], (1, 2, 1, 1))
	poses[0, 0, 0, 3] = -0.2
	poses[0, 1, 1, 3] = 0.25
	palette = SkinMatrixBuffer(2)
	palette.acquire(1, 2)
	palette.write(np.array([1], dtype=np.uint64), poses)
	deformed = vertices.copy()
	deformed["position"], deformed["normal"] = skin_vertices(vertices["position"], vertices["normal"], vertices["joints"], vertices["weights"], poses[0])
	source = '''
#extends "default_material.shader"
#block lighting
	return vec3f(0.5+position.x*0.2, 0.5+position.y*0.2, 0.3+normal.z*0.2);
#endblock
'''
	shader = DefaultMaterialShader(transparent=transparent, source=source,
		default_material={"base_color": (1, 1, 1, 0.8 if transparent else 1), "alpha_mode": 2 if transparent else 0})
	bounds = ((0, 0, 0), (2, 2, 1))
	gpu_mesh = MaterialMesh(vertices, shader, bounds=bounds)
	cpu_mesh = MaterialMesh(deformed, shader, bounds=bounds)
	gpu = MaterialRenderer([DrawInstance(gpu_mesh, surface_id=1)], skin_palette=palette, target_format="rgba8unorm", sample_count=sample_count)
	cpu = MaterialRenderer([DrawInstance(cpu_mesh, surface_id=1)], target_format="rgba8unorm", sample_count=sample_count)
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 50)
	output = Texture((96, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def render(renderer):
		commands = RenderContext.commands()
		renderer.record(commands, (96, 64), camera, output.color_attachment())
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 384}, (96, 64, 1))
		return np.frombuffer(data, dtype=np.uint8).reshape(64, 96, 4).copy()
	actual, expected = render(gpu), render(cpu)
	assert np.max(actual[..., :3]) > 20
	np.testing.assert_allclose(actual.astype(np.int16), expected.astype(np.int16), atol=1)


def test_ecs_palette_reallocation_and_entity_recycling():
	from graaal.ECS import ECS
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.EcsAdapters.MaterialTransform import MaterialTransform

	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	dtype = np.dtype([("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2), ("joints", "u4", 4), ("weights", "f4", 4)])
	vertices = np.zeros(3, dtype=dtype)
	vertices["position"] = [[-0.3, -0.4, 0], [0.3, -0.4, 0], [0, 0.4, 0]]
	vertices["normal"] = (0, 0, 1)
	vertices["weights"][:, 0] = 1
	mesh = MaterialMesh(vertices, DefaultMaterialShader(), bounds=((0, 0, 0), (2, 2, 1)))
	palette = SkinMatrixBuffer(2)
	renderer = MaterialRenderer(meshes=[mesh], skin_palette=palette, target_format="rgba8unorm")
	world = ECS()
	world.register(MaterialInstance, MaterialTransform)
	owners = world.create(2)
	world.add(owners, MaterialInstance(mesh), MaterialTransform())
	for owner in owners: palette.acquire(owner, 1)
	adapter = EcsMaterialAdapter(renderer, world)
	assert adapter.sync()
	palette.release(owners[0])
	palette.release(owners[1])
	palette.acquire(owners[1], 1)
	palette.acquire(owners[0], 1)
	renderer.sync_skinning()
	packed = renderer.draws.buffers["instances"].content[:2]
	for owner, record in zip(renderer.draws.entities, packed):
		assert record["iTint"][0] == palette.range(owner)[0] + 1
	world.get(MaterialTransform)[owners[:1]].assign(position=[1, 0, 0])
	assert adapter.sync()
	packed = renderer.draws.buffers["instances"].content[:2]
	for owner, record in zip(renderer.draws.entities, packed):
		assert record["iTint"][0] == palette.range(owner)[0] + 1
	world.delete(owners[:1])
	assert adapter.sync()
	palette.release(owners[0])
	new = world.create()[0]
	assert new != owners[0]
	palette.acquire(new, 1)
	world.add(new, MaterialInstance(mesh), MaterialTransform())
	assert adapter.sync()
	packed = renderer.draws.buffers["instances"].content[:2]
	for owner, record in zip(renderer.draws.entities, packed):
		assert record["iTint"][0] == palette.range(owner)[0] + 1


def test_gpu_skinning_accepts_static_meshes_and_validates_palette_ranges():
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	dtype = np.dtype([("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2), ("joints", "u4", 4), ("weights", "f4", 4)])
	vertices = np.zeros(3, dtype=dtype)
	vertices["position"] = [[-0.3, -0.4, 0], [0.3, -0.4, 0], [0, 0.4, 0]]
	vertices["normal"] = (0, 0, 1)
	vertices["weights"][:, 0] = 1
	shader = DefaultMaterialShader()
	skin = MaterialMesh(vertices, shader, bounds=((0, 0, 0), (2, 2, 1)))
	static_dtype = np.dtype([("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2)])
	static_vertices = np.zeros(3, dtype=static_dtype)
	for name in static_dtype.names: static_vertices[name] = vertices[name]
	static = MaterialMesh(static_vertices, shader)
	palette = SkinMatrixBuffer(2)
	palette.acquire(5, 1)
	renderer = MaterialRenderer([DrawInstance(skin, surface_id=5), DrawInstance(static, surface_id=6)], skin_palette=palette, target_format="rgba8unorm")
	for owner, record in zip(renderer.draws.entities, renderer.draws.buffers["instances"].content[:2]):
		assert record["iTint"][0] == (1 if owner == 5 else 0)
	palette.release(5)
	with pytest.raises(ValueError, match="No skin palette range"):
		renderer.sync_skinning()
	vertices["joints"][0, 0] = 1
	palette.acquire(5, 1)
	with pytest.raises(ValueError, match="does not cover"):
		MaterialRenderer([DrawInstance(skin, surface_id=5)], skin_palette=palette, target_format="rgba8unorm")
