"""Offscreen geometry and readback helpers for banded transparency tests."""

import numpy as np
import pytest

from graaal.RenderPasses import TransparencyPass as transparency
from graaal.Materials.DefaultMaterial import DefaultMaterialShader
from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
from graaal.Math import Camera
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.RenderPasses.TransparencyData import VERTEX_DTYPE


def read(texture, dtype=np.float16):
	width, height, layers = texture.size
	data = RenderContext.device.queue.read_texture({"texture": texture.handle},
		{"bytes_per_row": width*4*np.dtype(dtype).itemsize, "rows_per_image": height}, texture.size)
	return np.frombuffer(data, dtype=dtype).reshape(layers, height, width, 4).astype(np.float32)


@pytest.fixture
def factory(monkeypatch):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	def readable(*args, usage, **kwargs):
		if kwargs.get("sample_count", 1) == 1: usage |= TextureUsage.COPY_SRC
		return Texture(*args, usage=usage, **kwargs)
	monkeypatch.setattr(transparency, "Texture", readable)
	def make(*, sample_count=1, alpha=0.5, layers=1, **options):
		options.setdefault("banded_tail_msaa", True)
		shader = DefaultMaterialShader(transparent=True, source='''
#extends "default_material.shader"
#block lighting
	return p.emission;
#endblock
''')
		vertices = np.zeros(6, dtype=VERTEX_DTYPE)
		vertices["position"] = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, -10, 0), (10, 10, 0), (-10, 10, 0)]
		vertices["normal"] = (0, 0, 1)
		vertices["uv"] = [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)]
		mesh = MaterialMesh(vertices, shader)
		items = [DrawInstance(mesh, material={"base_color": (1, 1, 1, alpha),
			"emission": (1, 0, 0) if i == 0 else (0, 0, 1)}, position=(0, 0, -i*0.3), surface_id=17+i) for i in range(layers)]
		renderer = MaterialRenderer(items, target_format="rgba8unorm", sample_count=sample_count, **options)
		return renderer
	return make


def render(renderer, size=(96, 96)):
	output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("banded_test")
	renderer.record(commands, size, Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 55, near=0.01, far=100),
		output.color_attachment())
	RenderContext.submit(commands.finish())
	return read(output, np.uint8)


def encoded(hdr):
	linear = np.asarray(hdr)/(1+np.asarray(hdr))
	return np.rint(255*np.where(linear <= 0.0031308, 12.92*linear, 1.055*linear**(1/2.4)-0.055))
