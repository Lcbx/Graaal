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
