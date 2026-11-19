from dataclasses import replace

import numpy as np
import pytest

from tests.postprocessing_harness import AntialiasingHarness, AntialiasingSettings
from graaal.RenderContext import RenderContext, Texture, TextureUsage


@pytest.mark.parametrize("mode", ["lxaa", "faaa", "ddaa2"])
@pytest.mark.parametrize("format", ["rgba8unorm", "rgba8unorm-srgb"])
def test_antialiasing_preserves_flat_color_and_processes_edges(mode, format):
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingHarness(target_format=format)
	size = (32, 32)
	aa.resize(size)
	aa.source = Texture(size, format=format, usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST | TextureUsage.COPY_SRC)
	image = np.zeros((32, 32, 4), dtype=np.uint8)
	image[:, :, 3] = 137
	image[:, :, :3] = (np.arange(32)[None, :] <= np.arange(32)[:, None])[:, :, None]*255
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 128}, (*size, 1))
	output = Texture(size, format=format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("antialiasing_test")
	aa.record(commands, output.color_attachment(), mode)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128, "rows_per_image": 32}, (*size, 1))
	result = np.frombuffer(data, dtype=np.uint8).reshape(32, 32, 4)
	assert np.all(result[2, 20, :3] == 0)
	assert np.all(result[28, 2, :3] == 255)
	assert np.all(result[:, :, 3] == 137)
	assert np.any((result[:, :, 0] > 0) & (result[:, :, 0] < 255))


@pytest.mark.parametrize("mode", ["lxaa", "faaa", "ddaa2"])
def test_antialiasing_unfilterable_float_target(mode):
	RenderContext.setup_graphics_backend(True)
	if "float32-filterable" in RenderContext.device.features: pytest.skip("Device supports float32 filtering")
	aa = AntialiasingHarness(target_format="rgba32float")
	size = (32, 32)
	aa.resize(size)
	aa.source = Texture(size, format="rgba32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.zeros((32, 32, 4), dtype=np.float32)
	image[:, :, 3] = 1
	image[:, :, :3] = (np.arange(32)[None, :] <= np.arange(32)[:, None])[:, :, None]
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 512}, (*size, 1))
	output = Texture(size, format="rgba32float", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("antialiasing_float_test")
	aa.record(commands, output.color_attachment(), mode)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 512, "rows_per_image": 32}, (*size, 1))
	result = np.frombuffer(data, dtype=np.float32).reshape(32, 32, 4)
	assert np.allclose(result[:, :, 3], 1)
	assert np.any((result[:, :, 0] > 0) & (result[:, :, 0] < 1))


def test_antialiasing_settings_validation():
	settings = AntialiasingSettings()
	for values in ({"lxaa_edge_threshold": -1}, {"faaa_edge_threshold": 10}, {"faaa_quality": 0}, {"show_edges": 1}, {"hybrid_luma_chroma": 1}, {"chroma_weight": -1}, {"chroma_weight": 5.01}, {"chroma_weight": 1e308}, {"chroma_weight": float("nan")}, {"ddaa2_strength": -1}, {"ddaa2_threshold_min": float("nan")}, {"ddaa2_edge_steps": 6}):
		with pytest.raises(ValueError): replace(settings, **values)
	assert replace(settings, chroma_weight=5).chroma_weight == 5


@pytest.mark.parametrize("mode", ["lxaa", "faaa", "ddaa2"])
def test_antialiasing_hybrid_luma_chroma(mode):
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingHarness(target_format="rgba8unorm", settings=AntialiasingSettings(show_edges=True))
	aa.resize((16, 16))
	aa.source = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.empty((16, 16, 4), dtype=np.uint8)
	image[:] = (255, 0, 0, 137)
	image[np.arange(16)[None, :] > np.arange(16)[:, None]] = (0, 80, 255, 137)
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 64}, (16, 16, 1))
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	results = []
	for enabled, weight in ((False, 0.5), (True, 0.0), (True, 0.5), (True, 1.0)):
		aa.set_settings(replace(aa.settings, hybrid_luma_chroma=enabled, chroma_weight=weight))
		commands = RenderContext.commands("aa_chroma")
		aa.record(commands, output.color_attachment(), mode)
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
		results.append(np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4))
	assert np.array_equal(results[0], results[1])
	assert not np.any(np.all(results[0][:, :, :3] == (0, 255, 0), axis=2))
	for pixels in results[2:]:
		assert np.any(np.all(pixels[:, :, :3] == (0, 255, 0), axis=2))
		assert np.all(pixels[:, :, 3] == 137)
	aa.set_settings(replace(aa.settings, show_edges=False))
	commands = RenderContext.commands("aa_chroma_blend")
	aa.record(commands, output.color_attachment(), mode)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
	pixels = np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4)
	assert np.any(np.all(pixels[:, :, :3] != image[:, :, :3], axis=2))
	assert np.all(pixels[:, :, 3] == 137)


@pytest.mark.parametrize("mode", ["lxaa", "faaa"])
def test_antialiasing_ignores_faint_blue_chroma_variation(mode):
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingHarness(target_format="rgba8unorm", settings=AntialiasingSettings(hybrid_luma_chroma=True, chroma_weight=5, show_edges=True))
	size = (64, 64)
	aa.resize(size)
	aa.source = Texture(size, format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.zeros((64, 64, 4), dtype=np.uint8)
	image[:, :, 0] = np.random.default_rng(7).integers(0, 3, size, dtype=np.uint8)
	image[:, :, 2:] = 255
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 256}, (*size, 1))
	output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("aa_faint_blue")
	aa.record(commands, output.color_attachment(), mode)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256, "rows_per_image": 64}, (*size, 1))
	pixels = np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 4)
	assert not np.any(np.all(pixels[4:-4, 4:-4, :3] == (0, 255, 0), axis=2))


@pytest.mark.parametrize("mode", ["lxaa", "faaa", "ddaa2"])
def test_antialiasing_edge_display(mode):
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingHarness(target_format="rgba8unorm", settings=AntialiasingSettings(show_edges=True))
	aa.resize((16, 16))
	aa.source = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	image[:, :, 3] = 137
	image[:, :, :3] = np.where((np.arange(16)[None, :] <= np.arange(16)[:, None])[:, :, None], 255, 64)
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 64}, (16, 16, 1))
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("aa_edges")
	aa.record(commands, output.color_attachment(), mode)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
	pixels = np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4)
	assert np.any(np.all(pixels[:, :, :3] == (0, 255, 0), axis=2))
	assert np.all(np.abs(pixels[1, 14, :3].astype(int)-115) <= 1)
	assert np.all(pixels[:, :, 3] == 137)
	aa.set_settings(replace(aa.settings, show_edges=False))
	commands = RenderContext.commands("aa_edges_disabled")
	aa.record(commands, output.color_attachment(), mode)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
	pixels = np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4)
	assert np.all(pixels[1, 14, :3] == 64)
	assert not np.any(np.all(pixels[:, :, :3] == (0, 255, 0), axis=2))


@pytest.mark.parametrize("mode", ["lxaa", "faaa", "ddaa2"])
def test_antialiasing_filters_encoded_rgb_for_both_output_formats(mode):
	RenderContext.setup_graphics_backend(True)
	image = np.full((32, 32, 4), 255, dtype=np.uint8)
	mask = np.arange(32)[None, :] <= np.arange(32)[:, None]
	image[:, :, :3] = np.where(mask[:, :, None], (220, 130, 60), (25, 45, 80))
	results = []
	for format in ("rgba8unorm", "rgba8unorm-srgb"):
		aa = AntialiasingHarness(target_format=format)
		aa.resize((32, 32))
		aa.source = Texture((32, 32), format=format, usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST | TextureUsage.COPY_SRC)
		RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 128}, (32, 32, 1))
		output = Texture((32, 32), format=format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
		commands = RenderContext.commands("aa_encoded_filtering")
		aa.record(commands, output.color_attachment(), mode)
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128, "rows_per_image": 32}, (32, 32, 1))
		results.append(np.frombuffer(data, dtype=np.uint8).astype(np.int16))
	assert np.max(np.abs(results[0]-results[1])) <= 1


@pytest.mark.parametrize("hybrid_luma_chroma", [False, True])
def test_specialized_edge_search_settings(hybrid_luma_chroma):
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingHarness(target_format="rgba8unorm", settings=AntialiasingSettings(hybrid_luma_chroma=hybrid_luma_chroma))
	aa.resize((16, 16))
	aa.source = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.full((16, 16, 4), 255, dtype=np.uint8)
	image[:, :, :3] = (np.arange(16)[None, :] <= np.arange(16)[:, None])[:, :, None]*255
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 64}, (16, 16, 1))
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	for mode, name, values in (("faaa", "faaa_quality", range(1, 10)), ("ddaa2", "ddaa2_edge_steps", range(6))):
		for value in values:
			aa.set_settings(replace(aa.settings, **{name: value}))
			commands = RenderContext.commands("aa_specialization")
			aa.record(commands, output.color_attachment(), mode)
			RenderContext.submit(commands.finish())
			data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
			pixels = np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4)
			assert np.any((pixels[:, :, 0] > 0) & (pixels[:, :, 0] < 255))


@pytest.mark.parametrize("hybrid_luma_chroma", [False, True])
@pytest.mark.parametrize("ddaa2_edge_steps", [0, 1, 5])
def test_ddaa2_manual_filter_matches_hardware_filter(hybrid_luma_chroma, ddaa2_edge_steps):
	RenderContext.setup_graphics_backend(True)
	size = (31, 23)
	x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
	palette = np.array(((0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 80, 255)), dtype=np.uint8)
	indices = np.where(x > y * size[0] / size[1], 2, 3)
	indices[:, size[0] // 2] = 1
	indices[:7, :10] = np.random.default_rng(7).integers(0, 4, (7, 10))
	image = np.empty((size[1], size[0], 4), dtype=np.uint8)
	image[:, :, :3] = palette[indices]
	image[:, :, 3] = np.random.default_rng(8).integers(0, 256, (size[1], size[0]), dtype=np.uint8)
	results = []
	for manual_filter in (False, True):
		aa = AntialiasingHarness(target_format="rgba8unorm", settings=AntialiasingSettings(hybrid_luma_chroma=hybrid_luma_chroma, ddaa2_edge_steps=ddaa2_edge_steps))
		# Exercise both filtering paths on the same texture format, regardless of device features.
		aa.manual_filter = manual_filter
		aa.resize(size)
		aa.source = Texture(size, format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
		RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": size[0]*4}, (*size, 1))
		output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
		commands = RenderContext.commands("ddaa2_filter_comparison")
		aa.record(commands, output.color_attachment(), "ddaa2")
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": size[0]*4, "rows_per_image": size[1]}, (*size, 1))
		pixels = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 4)
		assert np.array_equal(pixels[:, :, 3], image[:, :, 3])
		results.append(pixels.astype(np.int16))
	assert np.max(np.abs(results[0]-results[1])) <= 1


def test_ddaa2_disabled_diffusion_and_search_preserve_image():
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingHarness(target_format="rgba8unorm", settings=AntialiasingSettings(ddaa2_strength=0, ddaa2_edge_steps=0))
	aa.resize((16, 16))
	aa.source = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	image[:, :, 3] = 255
	image[:, :, :3] = (np.arange(16)[None, :] <= np.arange(16)[:, None])[:, :, None]*255
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 64}, (16, 16, 1))
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("ddaa2_settings")
	aa.record(commands, output.color_attachment(), "ddaa2")
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
	assert np.array_equal(np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4), image)


def test_faaa_edge_display_and_quality_settings():
	RenderContext.setup_graphics_backend(True)
	aa = AntialiasingHarness(target_format="rgba8unorm")
	aa.resize((16, 16))
	aa.source = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	image[:, :, 3] = 255
	image[:, :, :3] = (np.arange(16)[None, :] <= np.arange(16)[:, None])[:, :, None]*255
	RenderContext.device.queue.write_texture({"texture": aa.source.handle}, image, {"bytes_per_row": 64}, (16, 16, 1))
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	for quality in range(1, 10):
		aa.set_settings(replace(aa.settings, faaa_quality=quality, show_edges=True))
		commands = RenderContext.commands("faaa_edges")
		aa.record(commands, output.color_attachment(), "faaa")
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64, "rows_per_image": 16}, (16, 16, 1))
		pixels = np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4)
		assert np.any(np.all(pixels[:, :, :3] == (0, 255, 0), axis=2))
