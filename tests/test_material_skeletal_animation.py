import numpy as np
import pytest

from graaal.DefaultMaterial import DefaultMaterialShader
from graaal.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from graaal.Math import Camera
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.TransparencyData import VERTEX_DTYPE


def test_deformed_material_mesh_updates_raster_and_transparent_reconstruction():
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	vertices = np.zeros(3, dtype=VERTEX_DTYPE)
	vertices["position"] = [[-0.8, -0.5, 0], [-0.2, -0.5, 0], [-0.5, 0.5, 0]]
	vertices["normal"] = (0, 0, 1)
	shader = DefaultMaterialShader(transparent=True, default_material={"base_color": (1, 0.4, 0.1, 0.7), "alpha_mode": 2})
	mesh = MaterialMesh(vertices, shader, bounds=((0, 0, 0), (2, 2, 1)))
	renderer = MaterialRenderer([DrawInstance(mesh)], target_format="rgba8unorm")
	camera = Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 50)
	output = Texture((96, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def render():
		commands = RenderContext.commands()
		renderer.record(commands, (96, 64), camera, output.color_attachment())
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 384}, (96, 64, 1))
		return np.frombuffer(data, dtype=np.uint8).copy()
	first = render()
	updated = vertices.copy()
	updated["position"][:, 0] += 1
	renderer.update_mesh_vertices(mesh, updated)
	np.testing.assert_allclose(renderer._mesh_gpu[0].vertices["position"], updated["position"])
	np.testing.assert_allclose(renderer.transparency.vertices.content["position"][:, :3], updated["position"])
	second = render()
	assert np.count_nonzero(first != second) > 100
	with pytest.raises(ValueError, match="original vertex"):
		renderer.update_mesh_vertices(mesh, updated[:2])


def test_ecs_pose_batches_update_independent_indexed_material_meshes():
	from graaal.ECS import ECS
	from graaal.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
	from graaal.MaterialSkinning import MaterialSkinningAdapter
	from graaal.SkeletalAnimation import Clip, Skeleton

	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	dtype = np.dtype([("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2), ("joints", "u4", 1), ("weights", "f4", 1)])
	vertices = np.zeros(3, dtype=dtype)
	vertices["position"] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
	vertices["normal"], vertices["weights"] = (0, 0, 1), 1
	shader = DefaultMaterialShader()
	indices = np.array([2, 1, 0], dtype=np.uint32)
	meshes = [MaterialMesh(vertices, shader, indices=indices, bounds=((0, 0, 0), (5, 5, 5))) for _ in range(2)]
	renderer = MaterialRenderer(meshes=meshes, target_format="rgba8unorm")
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	owners = world.create(2)
	world.add(owners, SkeletonRef(0), ClipPlayer(0, speed=0, loop=False))
	world.get(ClipPlayer)[owners].assign(time=[0.25, 0.75])
	rig = Skeleton([-1], [[0, 0, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	clip = Clip([[[0, 0, 0]], [[2, 0, 0]]], [[[0, 0, 0, 1]], [[0, 0, 0, 1]]], np.ones((2, 1, 3)), fps=1)
	animation = EcsAnimationAdapter(world, {0: rig}, {0: clip})
	skinning = MaterialSkinningAdapter(renderer)
	for owner, mesh in zip(owners, meshes): skinning.bind(owner, mesh, vertices)
	for batch in animation.update(0): skinning.update(batch.entities, batch.matrices)
	for i, offset in enumerate((0.5, 1.5)):
		expected = vertices["position"][indices] + (offset, 0, 0)
		np.testing.assert_allclose(renderer._mesh_gpu[i].vertices["position"], expected)
		start = renderer.offsets[i] * 3
		np.testing.assert_allclose(renderer.transparency.vertices.content["position"][start:start + 3, :3], expected)
