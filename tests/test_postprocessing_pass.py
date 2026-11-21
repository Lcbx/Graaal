import numpy as np
import pytest

from graaal.RenderPasses.PostprocessingPass import AntialiasingEffect, AmbientOcclusionEffect, PostprocessingEffect, PostprocessingPass
from scenes.postprocessing_example import PeripheralMotionBlurEffect
from tests.postprocessing_harness import AntialiasingHarness, PeripheralMotionBlurHarness
from graaal.RenderPasses.pass_settings import AmbientOcclusionSettings
from graaal.RenderContext import RenderContext, RenderPipeline, Sampler, Shader, Texture, TextureUsage


@pytest.mark.parametrize("effects", [[], [AntialiasingEffect("lxaa")], [AntialiasingEffect("faaa")], [AntialiasingEffect("ddaa2")], [PeripheralMotionBlurEffect()], [(PeripheralMotionBlurEffect(), AntialiasingEffect())], [PeripheralMotionBlurEffect(), AntialiasingEffect()]])
@pytest.mark.parametrize("format", ["rgba8unorm", "rgba8unorm-srgb"])
def test_postprocessing_flat_color(effects, format):
	RenderContext.setup_graphics_backend(True)
	post = PostprocessingPass(effects, target_format=format)
	post.resize((16, 16))
	output = Texture((16, 16), format=format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("postprocessing_test")
	with commands.render_pass(color=post.color_attachment(clear=(0.2, 0.4, 0.6, 0.7))): pass
	post.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64}, (16, 16, 1))
	image = np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4)
	assert np.all(image == image[0, 0])
	rgb = np.array((0.2, 0.4, 0.6))
	if format.endswith("-srgb"): rgb = np.where(rgb <= 0.0031308, rgb*12.92, 1.055*rgb**(1/2.4)-0.055)
	assert np.allclose(image[0, 0, :3], rgb*255, atol=1)
	assert abs(int(image[0, 0, 3])-178) <= 1


def test_postprocessing_groups_validate():
	with pytest.raises(ValueError): PostprocessingPass([()])
	with pytest.raises(TypeError): PostprocessingPass(())
	with pytest.raises(ValueError, match="same effect type twice"): PostprocessingPass([(AntialiasingEffect(), AntialiasingEffect())])


@pytest.mark.parametrize("sample_count", [1, 4])
def test_postprocessing_ambient_occlusion(sample_count):
	RenderContext.setup_graphics_backend(True)
	class Camera:
		near, far = 0.1, 100
		def projection(self, aspect): return np.eye(4, dtype=np.float32)
	ao = AmbientOcclusionEffect()
	ao.update_camera(Camera(), 1)
	post = PostprocessingPass([(PeripheralMotionBlurEffect(), ao)], target_format="rgba8unorm")
	post.resize((16, 16))
	depth = Texture((16, 16), format="depth32float", sample_count=sample_count, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING)
	output = Texture((16, 16), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("postprocessing_ao_test")
	with commands.render_pass(color=post.color_attachment(clear=(0.2, 0.4, 0.6, 0.7))): pass
	with commands.render_pass(depth=depth.depth_attachment(clear=1.0)): pass
	post.record(commands, output.color_attachment(), depth=depth)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64}, (16, 16, 1))
	image = np.frombuffer(data, dtype=np.uint8).reshape(16, 16, 4)
	assert np.allclose(image[0, 0], (51, 102, 153, 178), atol=1)


def test_postprocessing_sequential_matches_separate_passes():
	RenderContext.setup_graphics_backend(True)
	size = (32, 32)
	motion = dict(linear_velocity=(5, 0, 0), angular_velocity=(0, 0, 0), fovy_deg=60, exposure=0.2, max_blur=0.3)
	post = PostprocessingPass([PeripheralMotionBlurEffect(**motion), AntialiasingEffect("faaa")], target_format="rgba8unorm")
	post.resize(size)
	separate_blur = PeripheralMotionBlurHarness(target_format="rgba8unorm")
	separate_aa = AntialiasingHarness(target_format="rgba8unorm")
	separate_blur.resize(size)
	separate_aa.resize(size)
	post.targets[0] = Texture(size, format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	separate_blur.source = Texture(size, format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	image = np.full((32, 32, 4), 255, dtype=np.uint8)
	image[:, :, :3] = ((np.arange(32)[None, :] <= np.arange(32)[:, None])*255)[:, :, None]
	for texture in (post.targets[0], separate_blur.source):
		RenderContext.device.queue.write_texture({"texture": texture.handle}, image, {"bytes_per_row": 128}, (*size, 1))
	outputs = [Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC) for _ in range(2)]
	commands = RenderContext.commands("postprocessing_comparison")
	post.record(commands, outputs[0].color_attachment())
	separate_blur.record(commands, separate_aa.color_attachment(), **motion)
	separate_aa.record(commands, outputs[1].color_attachment(), "faaa")
	RenderContext.submit(commands.finish())
	results = [np.frombuffer(RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128}, (*size, 1)), dtype=np.uint8).astype(int) for output in outputs]
	assert np.max(np.abs(results[0]-results[1])) <= 1


@pytest.mark.parametrize("mode", ["lxaa", "faaa", "ddaa2"])
@pytest.mark.parametrize("format", ["rgba8unorm", "rgba8unorm-srgb"])
def test_antialiasing_effect_filters_edges(mode, format):
	RenderContext.setup_graphics_backend(True)
	size = (32, 32)
	post = PostprocessingPass([AntialiasingEffect(mode)], target_format=format)
	post.resize(size)
	image = np.full((32, 32, 4), 173, dtype=np.uint8)
	image[:, :, :3] = ((np.arange(32)[None, :] <= np.arange(32)[:, None])*255)[:, :, None]
	post.targets[0] = Texture(size, format=format, usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST | TextureUsage.COPY_SRC)
	RenderContext.device.queue.write_texture({"texture": post.targets[0].handle}, image, {"bytes_per_row": 128}, (*size, 1))
	output = Texture(size, format=format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("compare_antialiasing")
	post.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	result = np.frombuffer(RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128}, (*size, 1)), dtype=np.uint8).reshape(32, 32, 4)
	rgb = result[:, :, :3]
	assert np.any((rgb > 0) & (rgb < 255))
	assert np.array_equal(result[:, :, 3], image[:, :, 3])


def test_postprocessing_float_and_independent_group_uniforms():
	RenderContext.setup_graphics_backend(True)
	first = PeripheralMotionBlurEffect(linear_velocity=(2, 0, 0))
	second = PeripheralMotionBlurEffect(linear_velocity=(0, 3, 0))
	post = PostprocessingPass([first, second], target_format="rgba32float")
	post.resize((16, 16))
	output = Texture((16, 16), format="rgba32float", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("postprocessing_float_test")
	with commands.render_pass(color=post.color_attachment(clear=(0.2, 0.4, 0.6, 0.7))): pass
	post.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	buffers = [entry[2]["motion"] for entry in post.pipelines.values()]
	assert len(buffers) == 2 and buffers[0] is not buffers[1]
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256}, (16, 16, 1))
	image = np.frombuffer(data, dtype=np.float32).reshape(16, 16, 4)
	assert np.allclose(image, (0.2, 0.4, 0.6, 0.7), atol=1e-5)
	assert post.resize((8, 8))
	assert not post.resize((0, 0))
	output = Texture((8, 8), format="rgba32float", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("postprocessing_resize_test")
	with commands.render_pass(color=post.color_attachment(clear=(0.2, 0.4, 0.6, 0.7))): pass
	post.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128}, (8, 8, 1))
	assert np.allclose(np.frombuffer(data, dtype=np.float32).reshape(8, 8, 4), (0.2, 0.4, 0.6, 0.7), atol=1e-5)


def test_custom_effect_shader():
	class DimEffect(PostprocessingEffect):
		def shader_fragment(self, index):
			source = f"fn effect{index}(uv: vec2f) -> vec4f {{ let c = sample{index}(uv); return vec4f(c.rgb*0.5, c.a); }}"
			return source, {}
	RenderContext.setup_graphics_backend(True)
	class SecondDimEffect(DimEffect): pass
	post = PostprocessingPass([(DimEffect(), SecondDimEffect())], target_format="rgba8unorm")
	post.resize((8, 8))
	output = Texture((8, 8), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("custom_postprocessing_test")
	with commands.render_pass(color=post.color_attachment(clear=(0.8, 0.4, 0.2, 1))): pass
	post.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 32}, (8, 8, 1))
	assert np.allclose(np.frombuffer(data, dtype=np.uint8).reshape(8, 8, 4)[0, 0], (51, 26, 13, 255), atol=1)


def test_custom_effect_texture_binding():
	class TextureEffect(PostprocessingEffect):
		def __init__(self, texture): self.texture = texture
		def shader_fragment(self, index):
			source = f"texture<{index+1}, 0> effect{index}_extra: texture_2d<f32>;\nfn effect{index}(uv: vec2f) -> vec4f {{ return textureLoad(effect{index}_extra, vec2i(0), 0); }}"
			return source, {}
		def resources(self, index, depth): return {f"effect{index}_extra": self.texture}
	RenderContext.setup_graphics_backend(True)
	texture = Texture((1, 1), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": texture.handle}, bytes((64, 128, 192, 255)), {"bytes_per_row": 4}, (1, 1, 1))
	post = PostprocessingPass([TextureEffect(texture)], target_format="rgba8unorm")
	post.resize((8, 8))
	output = Texture((8, 8), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("custom_texture_test")
	post.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 32}, (8, 8, 1))
	assert np.all(np.frombuffer(data, dtype=np.uint8).reshape(8, 8, 4) == (64, 128, 192, 255))


@pytest.mark.parametrize("format", ["rgba8unorm", "rgba32float"])
def test_custom_unfilterable_texture_layout(format):
	class TextureEffect(PostprocessingEffect):
		def __init__(self, texture):
			self.texture = texture
			self.sampler = Sampler(min_filter="nearest", mag_filter="nearest")
		def shader_fragment(self, index):
			return f"texture<{index+1}, 0> extra: texture_2d<f32>;\ntexture<{index+1}, 1> extra_sampler: sampler;\nfn effect{index}(uv: vec2f) -> vec4f {{ return textureSampleLevel(extra, extra_sampler, uv, 0.0); }}", {}
		def resources(self, index, depth): return {"extra": self.texture, "extra_sampler": self.sampler}
		def binding_layouts(self, index):
			return {"extra": {"texture": {"sample_type": "unfilterable-float", "view_dimension": "2d", "multisampled": False}}, "extra_sampler": {"sampler": {"type": "non-filtering"}}}
	RenderContext.setup_graphics_backend(True)
	texture = Texture((1, 1), format="rgba32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": texture.handle}, np.array([0.2, 0.4, 0.6, 1], dtype=np.float32), {"bytes_per_row": 16}, (1, 1, 1))
	post = PostprocessingPass([TextureEffect(texture)], target_format=format)
	post.resize((8, 8))
	output = Texture((8, 8), format=format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("custom_layout_test")
	post.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	is_float = format == "rgba32float"
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128 if is_float else 32}, (8, 8, 1))
	image = np.frombuffer(data, dtype=np.float32 if is_float else np.uint8).reshape(8, 8, 4)
	assert np.allclose(image, (0.2, 0.4, 0.6, 1) if is_float else (51, 102, 153, 255), atol=1e-6 if is_float else 1)


def test_fused_effects_allocate_reflected_slots_and_multiple_uniforms():
	class ScaleEffect(PostprocessingEffect):
		def shader_fragment(self, index):
			return '''
{{ field("Scale" ~ INDEX, "value", "vec4f") }}
{{ binding("scale" ~ INDEX, "Scale" ~ INDEX, GROUP, SLOT, storage="uniform") }}
{{ binding("offset" ~ INDEX, "Scale" ~ INDEX, GROUP, SLOT+1, storage="uniform") }}
fn effect{{ INDEX }}(uv: vec2f) -> vec4f {
	let c = sample{{ INDEX }}(uv);
	return vec4f(c.rgb*scale{{ INDEX }}.value.xyz+offset{{ INDEX }}.value.xyz, c.a);
}
''', {}
		def uniform_values(self, index):
			return {f"scale{index}": {"value": (0.5, 0.5, 0.5, 0)}, f"offset{index}": {"value": (0, 0, 0, 0)}}
	RenderContext.setup_graphics_backend(True)
	class ScaleEffect2(ScaleEffect): pass
	class ScaleEffect3(ScaleEffect): pass
	class ScaleEffect4(ScaleEffect): pass
	post = PostprocessingPass([(ScaleEffect(), ScaleEffect2(), ScaleEffect3(), ScaleEffect4())], target_format="rgba32float")
	post.resize((8, 8))
	output = Texture((8, 8), format="rgba32float", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("reflected_uniform_test")
	with commands.render_pass(color=post.color_attachment(clear=(0.8, 0.4, 0.2, 1))): pass
	post.record(commands, output.color_attachment())
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128}, (8, 8, 1))
	assert np.allclose(np.frombuffer(data, dtype=np.float32).reshape(8, 8, 4), (0.05, 0.025, 0.0125, 1), atol=1e-6)


@pytest.mark.parametrize("sample_count", [1, 4])
def test_ambient_occlusion_darkens_depth_step(sample_count):
	RenderContext.setup_graphics_backend(True)
	class Camera:
		near, far = 0.1, 10
		def projection(self, aspect): return np.eye(4, dtype=np.float32)
	settings = AmbientOcclusionSettings(radius=0.6, sample_count=32, bias=0.001, strength=2, falloff=0.001)
	effect = AmbientOcclusionEffect(settings)
	effect.update_camera(Camera(), 1)
	post = PostprocessingPass([effect], target_format="rgba8unorm")
	post.resize((32, 32))
	depth = Texture((32, 32), format="depth32float", sample_count=sample_count, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING)
	shader = Shader(source="""
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(uv*2.0-1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @builtin(frag_depth) f32 { return select(0.3, 0.4, p.x >= 16.0); }
""")
	pipeline = RenderPipeline(shader, depth_test="always", depth_write=True, cull_mode="none", sample_count=sample_count)
	output = Texture((32, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("ao_depth_step")
	with commands.render_pass(depth=depth.depth_attachment(clear=1)) as render:
		render.set_pipeline(pipeline)
		render.draw(3)
	with commands.render_pass(color=post.color_attachment(clear=(0.8, 0.8, 0.8, 1))): pass
	post.record(commands, output.color_attachment(), depth=depth)
	RenderContext.submit(commands.finish())
	image = np.frombuffer(RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128}, (32, 32, 1)), dtype=np.uint8).reshape(32, 32, 4)
	assert image[:, :, :3].min() < 190
	assert image[:, :, :3].max() >= 203
