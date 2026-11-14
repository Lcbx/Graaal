"""CPU loader checks and offscreen, read-back checks of the actual OIT passes."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.Assets import gltf_node_matrix, load_blue_noise
from graaal import TransparencyData as transparency_data
from graaal import TextureProcessing as texture_processing
from scenes import transparency_example as scene
from graaal import TransparencyPass as transparency


def test_bundled_blue_noise_loads_as_deterministic_linear_thresholds():
	noise = load_blue_noise(scene.BLUE_NOISE_PATH)
	np.testing.assert_array_equal(noise, load_blue_noise(scene.BLUE_NOISE_PATH))
	assert noise.shape == (64, 64) and noise.dtype == np.float32
	assert 0 < noise.min() < noise.max() < 1
	assert len(np.unique(noise)) == noise.size  # retain the full 16-bit ranks
	# No sRGB conversion: a uniform threshold distribution stays uniform.
	# The source stores 4096 discrete ranks in a 16-bit container; allow one
	# rank's quantization error rather than assuming it spans every 16-bit bin.
	assert abs(noise.mean()-0.5) < 1/noise.size
	np.testing.assert_array_equal(np.histogram(noise, bins=16, range=(0, 1))[0], 256)


def test_node_matrix_uses_gltf_column_major_matrix_and_trs():
	trs = SimpleNamespace(matrix=None, rotation=[0, 0, 0, 1], scale=[2, 3, 4], translation=[5, 6, 7])
	matrix = gltf_node_matrix(trs)
	np.testing.assert_allclose(matrix @ [1, 1, 1, 1], [7, 9, 11, 1])
	np.testing.assert_array_equal(gltf_node_matrix(SimpleNamespace(matrix=matrix.T.reshape(-1).tolist())), matrix)


def test_loader_bakes_hierarchy_reflection_and_default_material(tmp_path):
	positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
	# A non-indexed primitive with no normals or UVs also exercises fallbacks.
	gltf = {
		"asset": {"version": "2.0"},
		"buffers": [{"byteLength": positions.nbytes, "uri": "data:application/octet-stream;base64," + base64.b64encode(positions.tobytes()).decode()}],
		"bufferViews": [{"buffer": 0, "byteLength": positions.nbytes}],
		"accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
		"meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
		"nodes": [{"translation": [2, 3, 4], "children": [1]}, {"mesh": 0, "scale": [-2, 1, 1]}],
		"scenes": [{"nodes": [0]}], "scene": 0,
	}
	path = tmp_path / "triangle.gltf"
	path.write_text(json.dumps(gltf))
	primitives, materials, images, stored = transparency_data.load_transparency_scene(path)
	vertices = primitives[0].vertices
	np.testing.assert_allclose(vertices["position"], [[2, 3, 4], [2, 4, 4], [0, 3, 4]])
	np.testing.assert_allclose(vertices["normal"], [[0, 0, 1]]*3)
	np.testing.assert_array_equal(stored["position"][:, :3], vertices["position"])
	assert primitives[0].material_id == 0
	assert materials[0]["surface"][2] == 0
	assert len(images) == 1


def test_real_scene_has_opaque_mask_and_blended_geometry():
	primitives, materials, images, vertices = transparency_data.load_transparency_scene(scene.PATH)
	assert {int(materials[p.material_id]["surface"][2]) for p in primitives} == {0, 1, 2}
	assert len(images) == len(materials)
	for primitive in primitives:
		start = primitive.triangle_offset*3
		count = len(primitive.vertices)
		np.testing.assert_array_equal(vertices["position"][start:start+count, :3], primitive.vertices["position"])
	assert np.isfinite(vertices["normal"]).all()
	# The example overrides only the named canopy, keeping generic loading faithful.
	_, cutout_materials, _, _ = scene.load_transparency_scene(scene.PATH)
	changed = cutout_materials["surface"][:, 2] != materials["surface"][:, 2]
	assert changed.sum() == 1
	assert materials["surface"][changed, 2] == 2
	assert cutout_materials["surface"][changed, 2] == 1
	assert cutout_materials["flags"][changed, 2] == 0  # no geometry-specific anchoring flags


def mip_material(mode=2, alpha_factor=1, cutoff=0.5):
	material = np.zeros(1, dtype=transparency.MATERIAL_DTYPE)[0]
	material["base_color"] = (1, 1, 1, alpha_factor)
	material["surface"] = (1, 0, mode, cutoff)
	return material


def alpha_coverage(alpha, cutoff):
	"""Independent CPU measurement of GPU-produced repeat/bilinear coverage."""
	count = 0
	for y in (-1, 1):
		row = np.roll(alpha, y, axis=0)
		for x in (-1, 1):
			samples = alpha*0.5625 + row*0.1875 + np.roll(alpha, x, axis=1)*0.1875 + np.roll(row, x, axis=1)*0.0625
			count += np.count_nonzero(samples >= cutoff)
	return count/(4*alpha.size)


@pytest.fixture
def gpu_mips(monkeypatch):
	RenderContext.setup_graphics_backend(True)
	monkeypatch.setattr(texture_processing, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	def generate(image, material, size=None):
		size = size or (image.shape[1], image.shape[0])
		texture = texture_processing.build_material_texture_array([image], np.array([material], dtype=transparency.MATERIAL_DTYPE), size)
		mips = []
		for level in range(max(size).bit_length()):
			width, height = max(1, size[0] >> level), max(1, size[1] >> level)
			data = RenderContext.device.queue.read_texture({"texture": texture.handle, "mip_level": level},
				{"bytes_per_row": width*4, "rows_per_image": height}, (width, height, 1))
			mips.append(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4).copy())
		return mips
	return generate


def test_blend_mips_preserve_mean_opacity_and_ignore_transparent_rgb(gpu_mips):
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	visible = np.indices((16, 16)).sum(axis=0) % 2 == 0
	image[visible] = (255, 0, 0, 255)
	image[~visible] = (0, 255, 0, 0)
	mips = gpu_mips(image, mip_material())
	assert [mip.shape[:2] for mip in mips] == [(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
	for mip in mips[1:]:
		assert abs(mip[:, :, 3].mean()/255-0.5) <= 1/255
		np.testing.assert_array_equal(mip[:, :, :3], np.broadcast_to((255, 0, 0), mip[:, :, :3].shape))


def test_opaque_mips_filter_in_linear_space_and_ignore_alpha_for_rgb(gpu_mips):
	image = np.array([[(255, 255, 255, 0), (0, 0, 0, 255)]], dtype=np.uint8)
	last = gpu_mips(image, mip_material(mode=0))[-1]
	np.testing.assert_allclose(last[0, 0, :3], (188, 188, 188), atol=1)


@pytest.mark.parametrize("mode", [1, 2])
def test_cutout_mips_pad_transparent_rgb_without_changing_alpha(gpu_mips, mode):
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	color = (32, 220, 72)
	image[4:12, :4] = (*color, 255)
	mips = gpu_mips(image, mip_material(mode=mode, cutoff=0.2))
	for mip in mips:
		alpha = mip[:, :, 3]
		neighbor = np.zeros_like(alpha, dtype=bool)
		for y in (-1, 0, 1):
			for x in (-1, 0, 1):
				neighbor |= np.roll(alpha > 0, (y, x), axis=(0, 1))
		fringe = (alpha == 0) & neighbor
		if fringe.any():
			np.testing.assert_allclose(mip[:, :, :3][fringe], np.broadcast_to(color, mip[:, :, :3][fringe].shape), atol=1)
	# This donor crosses the repeat boundary. RGB may bleed; alpha must not.
	np.testing.assert_allclose(mips[0][8, -1, :3], color, atol=1)
	assert mips[0][8, -1, 3] == 0
	if mode == 2:
		for mip in mips:
			assert abs(mip[:, :, 3].mean()-image[:, :, 3].mean()) <= 1


@pytest.mark.parametrize("alpha_factor,cutoff", [(1, 0.5), (0.5, 0.35)])
def test_mask_mips_preserve_cutoff_coverage_including_material_factor(gpu_mips, alpha_factor, cutoff):
	rng = np.random.default_rng(13)
	image = np.full((64, 64, 4), 255, dtype=np.uint8)
	image[:, :, 3] = (rng.random((64, 64)) < 0.4)*255
	mask = mip_material(mode=1, alpha_factor=alpha_factor, cutoff=cutoff)
	plain = mip_material(mode=2, alpha_factor=alpha_factor, cutoff=cutoff)
	# Include the initial source-to-array resize in the coverage guarantee.
	corrected = gpu_mips(image, mask, (32, 32))
	unscaled = gpu_mips(image, plain, (32, 32))
	effective_cutoff = cutoff/alpha_factor
	target = alpha_coverage(image[:, :, 3]/255, effective_cutoff)
	best_improvement = 0
	for mip, raw in zip(corrected, unscaled):
		error = abs(alpha_coverage(mip[:, :, 3]/255, effective_cutoff)-target)
		raw_error = abs(alpha_coverage(raw[:, :, 3]/255, effective_cutoff)-target)
		assert error <= raw_error + 1e-6
		best_improvement = max(best_improvement, raw_error-error)
	assert best_improvement > 0.1
	assert abs(alpha_coverage(corrected[0][:, :, 3]/255, effective_cutoff)-target) < 0.02


@pytest.mark.parametrize("alpha", [0, 255])
def test_mips_handle_constant_and_degenerate_alpha(gpu_mips, alpha):
	image = np.full((8, 8, 4), 255, dtype=np.uint8)
	image[:, :, 3] = alpha
	for mode in (1, 2):
		for mip in gpu_mips(image, mip_material(mode)):
			np.testing.assert_array_equal(mip[:, :, 3], alpha)
	# An unreachable cutoff and zero material alpha need no coverage adjustment.
	for mip in gpu_mips(image, mip_material(mode=1, alpha_factor=0)):
		np.testing.assert_array_equal(mip[:, :, 3], alpha)


def test_gpu_mips_preserve_npot_mean_alpha_and_repeat_edges_when_upsampling(gpu_mips):
	image = np.full((5, 7, 4), 255, dtype=np.uint8)
	image[:, :, 3] = np.arange(35).reshape(5, 7)*7
	mips = gpu_mips(image, mip_material())
	assert [mip.shape[:2] for mip in mips] == [(5, 7), (2, 3), (1, 1)]
	for mip in mips:
		assert abs(mip[:, :, 3].mean()-image[:, :, 3].mean()) <= 1
	image = np.full((1, 2, 4), 255, dtype=np.uint8)
	image[0, :, 3] = (0, 255)
	base = gpu_mips(image, mip_material(), (4, 1))[0]
	# Linear repeat sampling blends the final texel into the first on upscale.
	np.testing.assert_allclose(base[0, :, 3], [64, 64, 191, 191], atol=1)


def make_layers():
	"""Two unsorted blended sheets, an occluded sheet, and an opaque backdrop."""
	materials = np.zeros(4, dtype=transparency.MATERIAL_DTYPE)
	materials["base_color"] = [(1, 0.1, 0.1, 0.25), (0.1, 1, 0.1, 0.5), (0.1, 0.1, 1, 1), (0.4, 0.4, 0.4, 1)]
	materials["opacity_rgb"] = 1
	materials["surface"][:, 0] = 1
	materials["surface"][:, 2] = [2, 2, 2, 0]
	materials["flags"][:, 0] = np.arange(4)
	materials["flags"][:, 1] = 1
	primitives, storage = [], []
	for i, z in enumerate((0.6, 0.0, -1.2, -0.6)):
		vertices = np.zeros(6, dtype=transparency.VERTEX_DTYPE)
		vertices["position"] = [(-10, -10, z), (10, -10, z), (10, 10, z), (-10, -10, z), (10, 10, z), (-10, 10, z)]
		vertices["normal"] = (0, 0, 1)
		vertices["barycentric"] = [(1, 0), (0, 1), (0, 0)]*2
		vertices["triangle"] = [0, 0, 0, 1, 1, 1]
		primitives.append(transparency_data.Primitive(vertices, i, i*2, i))
		stored = np.zeros(6, dtype=transparency.SHADING_VERTEX_DTYPE)
		stored["position"][:, :3] = vertices["position"]
		stored["normal"][:, :3] = vertices["normal"]
		storage.append(stored)
	images = [np.full((1, 1, 4), 255, dtype=np.uint8) for _ in materials]
	return primitives, materials, images, np.concatenate(storage)


@pytest.fixture(params=[0, 1, 3])
def gpu_scene(monkeypatch, request):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: make_layers())
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 1)
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "window_dimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	scene.init_scene(kawase_passes=request.param)
	scene.camera.position, scene.camera.target = scene.Vec3(0, 0, 3), scene.Vec3(0, 0, 0)
	return scene, output


def render_frame(render_scene):
	"""Exercise the example's complete culling and transparency pipeline."""
	render_scene.render_frame()


def test_postprocess_antialiasing_with_both_ao_paths(gpu_scene):
	renderer, output = gpu_scene
	for mode in ("lxaa", "faaa"):
		for ao_method in ("compose", "pass"):
			renderer.aa_pass.mode = mode
			renderer.set_ao_method(ao_method)
			renderer.set_ao_enabled(True)
			render_frame(renderer)
			assert read_texture(output, np.uint8, 4)[:, :, :3].max() > 0


def read_texture(texture, dtype, components):
	width, height, _ = texture.size
	row_bytes = width*np.dtype(dtype).itemsize*components
	data = RenderContext.device.queue.read_texture({"texture": texture.handle},
		{"bytes_per_row": row_bytes, "rows_per_image": height}, texture.size)
	return np.frombuffer(data, dtype=dtype).reshape(height, width, components).copy()


def test_transparent_draws_are_skipped_in_opaque_depth_prepass(gpu_scene):
	renderer, _ = gpu_scene
	render_frame(renderer)
	render = Mock()
	renderer.transparent_batches.draw(render, "prepass")
	render.draw_indexed_indirect.assert_not_called()
	render.set_pipeline.assert_not_called()
	# The separate transmittance stage still renders the transparent geometry.
	renderer.transparent_batches.draw(render, "transmittance")
	assert render.draw_indexed_indirect.call_count == len(renderer.transparent_batches.data.draw_batches)
	assert not renderer.transparent_batches.passes.render_shader["transmittance"].pipeline.depth_write


def test_gpu_resize_rebuilds_targets_and_bindings(gpu_scene, monkeypatch):
	renderer, _ = gpu_scene
	render_frame(renderer)
	previous = renderer.transparency.nearest_payload
	previous_hzb = renderer.hzb.view
	output = Texture((48, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "window_dimensions", (48, 32))
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	render_frame(renderer)
	assert renderer.transparency.nearest_payload is not previous
	assert renderer.hzb.view is not previous_hzb
	assert renderer.hzb.size == (48, 32)
	assert renderer.transparency.nearest_payload.size == (48, 32, 1)
	assert np.any(read_texture(renderer.transparency.nearest_payload, np.uint32, 4)[:, :, 0] != 0)
	assert np.any(read_texture(output, np.uint8, 4)[:, :, :3] != 0)
	# Minimized windows leave valid resources alone and submit no new work.
	previous = renderer.transparency.nearest_payload
	previous_hzb = renderer.hzb.view
	monkeypatch.setattr(RenderContext, "window_dimensions", (0, 0))
	render_frame(renderer)
	assert renderer.transparency.nearest_payload is previous
	assert renderer.hzb.view is previous_hzb


def test_mask_canopy_is_noise_and_tail_filter_independent(monkeypatch):
	"""The real MASK canopy is rendered entirely through opaque depth/cutoff."""
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "window_dimensions", (320, 240))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((320, 240), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	scene.init_scene()
	renderer = scene
	materials = renderer.transparency.shared["materials"]
	modes = materials.content["surface"][:, 2]
	mask_ids = set(np.flatnonzero(modes == 1))
	assert mask_ids and mask_ids <= set(renderer.opaque_batches.metadata.content[:, 1])
	assert mask_ids.isdisjoint(renderer.transparent_batches.metadata.content[:, 1])
	materials.content["base_color"][modes == 2, 3] = 0
	materials.upload()
	distance = renderer.radius*1.4
	renderer.camera.position = scene.Vec3(renderer.center+distance*np.array([np.cos(0.65), 0.5, np.sin(0.65)]))
	render_frame(renderer)
	image = read_texture(output, np.uint8, 4)
	assert len(np.unique(image[:, :, :3].reshape(-1, 3), axis=0)) > 10
	noise = load_blue_noise(scene.BLUE_NOISE_PATH)
	for pairs, shift in ((0, (9, 23)), (1, (31, 11)), (3, (4, 39))):
		renderer.transparency.set_kawase_passes(pairs)
		pixels = np.ascontiguousarray(np.roll(noise, shift, axis=(0, 1)))
		RenderContext.device.queue.write_texture({"texture": renderer.transparency.noise.handle}, pixels,
			{"bytes_per_row": noise.shape[1]*4}, (*noise.shape, 1))
		render_frame(renderer)
		np.testing.assert_array_equal(read_texture(output, np.uint8, 4), image)
		np.testing.assert_array_equal(read_texture(renderer.transparency.transmittance, np.float16, 4), 1)


def test_gpu_hiz_culls_opaque_occlusion_and_preserves_cutout_holes(gpu_scene):
	renderer, _ = gpu_scene
	# Keep the transparent bounds inside the opaque backdrop's screen coverage.
	renderer.transparent_batches.data.write_instance_fields(iScale=scene.pack_scale([0.05, 0.05, 1]))
	view = renderer.transparent_batches.visibility._view
	for name in ("prepass_draw_cmd", "main_draw_cmd"):
		buffer = view.buffers[name]
		view.buffers[name] = scene.GpuBuffer(buffer.content.copy(), buffer.usage | scene.BufferUsage.COPY_SRC)
	view.bindings.clear()

	def instance_counts(name):
		buffer = view.buffers[name]
		raw = RenderContext.device.queue.read_buffer(buffer.handle)
		return np.frombuffer(raw, dtype=buffer.dtype)["instance_count"][:len(renderer.transparent_batches.data.draw_batches)]

	render_frame(renderer)
	np.testing.assert_array_equal(instance_counts("prepass_draw_cmd"), [1, 1, 1])
	np.testing.assert_array_equal(instance_counts("main_draw_cmd"), [1, 1, 0])
	# Transparent layers do not occlude one another, even at full opacity.
	materials = renderer.transparency.shared["materials"]
	materials.content["surface"][3, 2:] = (1, 0.5)
	materials.content["base_color"][3, 3] = 0
	materials.upload()
	render_frame(renderer)
	np.testing.assert_array_equal(instance_counts("main_draw_cmd"), [1, 1, 1])


def create_textured_gpu_scene(monkeypatch, uv_scale=2):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	primitives, materials, images, vertices = make_layers()
	materials["base_color"][0] = (1, 1, 1, 1)
	materials["base_color"][1, 3] = 0  # keep one visible transparent layer
	visible = np.indices((64, 64)).sum(axis=0) % 2 == 0
	image = np.empty((64, 64, 4), dtype=np.uint8)
	image[visible] = (255, 0, 0, 255)
	image[~visible] = (0, 255, 0, 0)
	images[0] = image
	for primitive in primitives:
		primitive.vertices["uv"] = primitive.vertices["position"][:, :2]*uv_scale
		start = primitive.triangle_offset*3
		vertices["uv"][start:start+len(primitive.vertices), :2] = primitive.vertices["uv"]
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, vertices))
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 64)
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "window_dimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	scene.init_scene()
	renderer = scene
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	return renderer, output


def test_gpu_cutout_sampling_has_no_black_bilinear_fringe(monkeypatch):
	RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
	primitives, materials, images, vertices = make_layers()
	materials["base_color"][0] = (1, 1, 1, 1)
	materials["surface"][0, 2:] = (1, 0.2)
	materials["base_color"][1, 3] = 0
	image = np.zeros((16, 16, 4), dtype=np.uint8)
	image[:, :8] = (255, 255, 255, 255)
	images[0] = image
	for primitive in primitives:
		primitive.vertices["uv"] = primitive.vertices["position"][:, :2]*0.05
		start = primitive.triangle_offset*3
		vertices["uv"][start:start+len(primitive.vertices), :2] = primitive.vertices["uv"]
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives, materials, images, vertices))
	monkeypatch.setattr(scene, "TEXTURE_SIZE", 16)
	monkeypatch.setattr(transparency, "Texture", lambda *args, usage, **kwargs: Texture(*args, usage=usage | TextureUsage.COPY_SRC, **kwargs))
	monkeypatch.setattr(RenderContext, "window_dimensions", (64, 64))
	monkeypatch.setattr(RenderContext, "presentation_format", "rgba8unorm", raising=False)
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT)
	monkeypatch.setattr(RenderContext, "screen", lambda **kwargs: output.color_attachment())
	scene.init_scene()
	renderer = scene
	renderer.camera.position = scene.Vec3(0, 0, 3)
	renderer.camera.target = scene.Vec3(0, 0, 0)
	render_frame(renderer)
	depth = read_texture(renderer.transparency.depth, np.float32, 1)[:, :, 0]
	foreground = depth < depth.max()-0.0001
	assert 0.2 < foreground.mean() < 0.9
	color = read_texture(renderer.transparency.main_color, np.float16, 4)[:, :, :3][foreground]
	# With black transparent neighbors, accepted alpha-edge samples darken to
	# roughly 20% of this brightness. Padding keeps straight white color intact.
	assert color.min() > 0.47
	assert color.max() < 0.51
	# MASK now writes opaque depth with a fixed cutoff and contributes no
	# tail opacity or nearest transparent payload.
	np.testing.assert_array_equal(read_texture(renderer.transparency.transmittance, np.float16, 4), 1)
	np.testing.assert_array_equal(read_texture(renderer.transparency.nearest_payload, np.uint32, 4), 0)


def test_gpu_minified_alpha_and_deferred_color_use_matching_mips(monkeypatch):
	renderer, _ = create_textured_gpu_scene(monkeypatch)
	render_frame(renderer)
	t = 1-read_texture(renderer.transparency.nearest_alpha, np.float16, 4)
	np.testing.assert_allclose(t, 1-128/255, atol=0.001)
	payload = read_texture(renderer.transparency.nearest_payload, np.uint32, 4)
	occupied = payload[:, :, 0] != 0
	assert occupied.all()
	lod_face = payload[:, :, 3][occupied].copy().view(np.float16).reshape(-1, 2)
	assert np.all(lod_face[:, 0] > 1.5)
	assert np.all(lod_face[:, 1] == 1)
	color = read_texture(renderer.transparency.nearest_color, np.float16, 4)[occupied]
	# At these footprints all mips contain straight red. Sampling level zero in
	# compute would darken accepted edge samples despite correctly filtered alpha.
	assert color[:, 0].min() > 0.47
	# The lighting model adds neutral dielectric specular, equally to G and B.
	np.testing.assert_allclose(color[:, 1], color[:, 2], atol=0.0001)
	assert color[:, 1:3].max() < 0.011
	# Subpixel camera translation leaves the minified opacity stable.
	renderer.camera.position.x += 0.005
	renderer.camera.target.x += 0.005
	render_frame(renderer)
	np.testing.assert_array_equal(1-read_texture(renderer.transparency.nearest_alpha, np.float16, 4), t)
	# Increasing distance selects a coarser mip without thinning the sheet.
	renderer.camera.position.z = 6
	render_frame(renderer)
	np.testing.assert_array_equal(1-read_texture(renderer.transparency.nearest_alpha, np.float16, 4), t)
	distant = read_texture(renderer.transparency.nearest_payload, np.uint32, 4)
	distant_lod = distant[:, :, 3][distant[:, :, 0] != 0].copy().view(np.float16).reshape(-1, 2)[:, 0]
	assert distant_lod.mean() > lod_face[:, 0].mean()+0.5


def test_static_primitive_input_order_preserves_packed_metadata_and_shading(gpu_scene, monkeypatch):
	renderer, output = gpu_scene
	render_frame(renderer)
	before = read_texture(output, np.uint8, 4)
	payload = read_texture(renderer.transparency.nearest_payload, np.uint32, 4)
	pairs = renderer.transparency.kawase_passes
	primitives, materials, images, vertices = make_layers()
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: (primitives[::-1], materials, images, vertices))
	scene.init_scene(kawase_passes=pairs)
	renderer.camera.position, renderer.camera.target = scene.Vec3(0, 0, 3), scene.Vec3(0, 0, 0)
	render_frame(renderer)
	np.testing.assert_array_equal(read_texture(output, np.uint8, 4), before)
	np.testing.assert_array_equal(read_texture(renderer.transparency.nearest_payload, np.uint32, 4), payload)


def test_gpu_grazing_fade_angles_front_back_and_all_passes_agree(gpu_scene, monkeypatch):
	renderer, _ = gpu_scene
	primitives, materials, images, stored = make_layers()
	primitive = primitives[0]
	primitive.vertices["position"][:, 2] = 0
	# Deliberately incorrect shading normals prove the fade uses geometry.
	primitive.vertices["normal"] = (1, 0, 0)
	stored = stored[:6].copy()
	stored["position"][:, :3] = primitive.vertices["position"]
	stored["normal"][:, :3] = primitive.vertices["normal"]
	materials = materials[:1].copy()
	materials["base_color"] = (0, 0, 0, 0.8)
	materials["surface"][:, 1] = 1
	materials["emission"][:, :3] = 1
	monkeypatch.setattr(scene, "load_transparency_scene", lambda path: ([primitive], materials, images[:1], stored))
	scene.init_scene()
	renderer.camera = scene.Camera((0, 0, 3), (0, 0, 0), (0, 1, 0), 3,
		perspective=False, near=0.1, far=100)
	for angle, expected in ((0, 0.8), (65, 0.8), (77.5, 0.4), (85, 0), (88, 0), (102.5, 0.4), (180, 0.8)):
		radians = np.deg2rad(angle)
		renderer.transparent_batches.data.write_instance_fields(iPosition=(0, 0, 0.6),
			iRotation=scene.pack_quaternion([0, np.sin(radians/2), 0, np.cos(radians/2)]))
		renderer.transparency.set_grazing_fade(scene.GrazingFade(enabled=False, start_angle=70, end_angle=85))
		render_frame(renderer)
		baseline = 1-read_texture(renderer.transparency.nearest_alpha, np.float16, 4)
		visible = baseline[:, :, 3] < 1
		assert visible.mean() > 0.1
		np.testing.assert_allclose(1-baseline[:, :, 3][visible], 0.8, atol=0.001)
		target = renderer.transparency.nearest_alpha
		renderer.transparency.set_grazing_fade(scene.GrazingFade(enabled=True, start_angle=70, end_angle=85))
		render_frame(renderer)
		assert renderer.transparency.nearest_alpha is target
		t = 1-read_texture(target, np.float16, 4)
		np.testing.assert_allclose(1-t[visible], expected, atol=0.001)
		payload = read_texture(renderer.transparency.nearest_payload, np.uint32, 4)
		if expected > 0:
			assert np.all(payload[:, :, 0][visible] != 0)
		elif angle > 85:
			np.testing.assert_array_equal(payload, 0, err_msg=f"angle={angle}, maximum alpha={float((1-t).max())}")
			np.testing.assert_array_equal(t, 1)
		if angle > 85 and expected == 0:
			# Packed rotation can put nominally 85-degree geometry below the endpoint.
			np.testing.assert_array_equal(read_texture(renderer.transparency.nearest_depth, np.float32, 1), 1)
	# Perspective rays vary across a plane: its centre stays face-on while the
	# edges fade. Compare the complete probability image to independent angles.
	renderer.transparent_batches.data.write_instance_fields(iPosition=(0, 0, 0.6),
		iRotation=scene.pack_quaternion([0, 0, 0, 1]))
	renderer.camera.perspective = True
	renderer.camera.fovy_deg = 55
	renderer.transparency.set_grazing_fade(scene.GrazingFade(enabled=True, start_angle=10, end_angle=30))
	render_frame(renderer)
	y, x = np.indices((64, 64))
	slope_x = (2*(x+0.5)/64-1)*np.tan(np.deg2rad(55/2))
	slope_y = (2*(y+0.5)/64-1)*np.tan(np.deg2rad(55/2))
	angles = np.rad2deg(np.arctan(np.sqrt(slope_x*slope_x+slope_y*slope_y)))
	u = np.clip((angles-10)/20, 0, 1)
	expected = 0.8*(1-u*u*(3-2*u))
	t = 1-read_texture(renderer.transparency.nearest_alpha, np.float16, 4)
	np.testing.assert_allclose(1-t[:, :, 3], expected, atol=0.001)
