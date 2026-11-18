import numpy as np
import pytest
from graaal.PeripheralMotionBlurPass import PeripheralMotionBlurPass
from graaal.RenderContext import RenderContext, RenderPipeline, Shader, Texture, TextureUsage


@pytest.mark.parametrize("format", ["rgba8unorm", "rgba8unorm-srgb", "rgba32float"])
def test_peripheral_motion_blur(format):
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format=format)
	assert blur.resize((64, 64))
	assert not blur.resize((0, 0))
	shader = Shader(source="""
		@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
			let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
			return vec4f(uv*2.0-1.0, 0.0, 1.0);
		}
		@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
			let c = f32(u32(p.x / 2.0) % 2u);
			return vec4f(c, c, c, 0.7);
		}
	""")
	pipeline = RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none")
	output = Texture((64, 64), format=format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def image(linear, angular):
		commands = RenderContext.commands("motion_test")
		with commands.render_pass(color=blur.color_attachment(clear=(0, 0, 0, 1))) as render:
			render.set_pipeline(pipeline)
			render.draw(3)
		blur.record(commands, output.color_attachment(), linear_velocity=linear, angular_velocity=angular, fovy_deg=60, exposure=0.2, reference_distance=2, max_blur=0.3)
		RenderContext.submit(commands.finish())
		float_format = format == "rgba32float"
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 64*(16 if float_format else 4)}, (64, 64, 1))
		return np.frombuffer(data, dtype=np.float32 if float_format else np.uint8).reshape(64, 64, 4).astype(float)
	still = image((0, 0, 0), (0, 0, 0))
	for linear, angular in [((8, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 2, 0))]:
		moving = image(linear, angular)
		assert np.max(np.abs(moving[30:34, 30:34] - still[30:34, 30:34])) < (0.01 if format == "rgba32float" else 2)
		assert np.mean(np.abs(moving[2:10, :, :3] - still[2:10, :, :3])) > (0.05 if format == "rgba32float" else 10)
		assert np.max(np.abs(moving[:, :, 3] - still[:, :, 3])) < (0.01 if format == "rgba32float" else 2)



def test_motion_blur_tracks_projected_scene_direction():
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format="rgba8unorm")
	blur.resize((64, 64))
	shader = Shader(source="""
		@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
			let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
			return vec4f(uv*2.0-1.0, 0.0, 1.0);
		}
		@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
			let c = select(0.0, 1.0, (p.x >= 15.0 && p.x < 18.0) || (p.x >= 46.0 && p.x < 49.0));
			return vec4f(c, c, c, 1.0);
		}
	""")
	pipeline = RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none")
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def centers(velocity):
		commands = RenderContext.commands("motion_direction")
		with commands.render_pass(color=blur.color_attachment(clear=(0, 0, 0, 1))) as render:
			render.set_pipeline(pipeline)
			render.draw(3)
		blur.record(commands, output.color_attachment(), linear_velocity=velocity, angular_velocity=(0, 0, 0), fovy_deg=60, exposure=0.2, reference_distance=2, max_blur=0.2)
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256}, (64, 64, 1))
		row = np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 4)[4, :, 0].astype(float)
		return tuple(np.average(np.arange(start, end), weights=row[start:end]) for start, end in ((0, 32), (32, 64)))
	still = centers((0, 0, 0))
	left = centers((8, 0, 0))
	right = centers((-8, 0, 0))
	forward = centers((0, 0, -8))
	backward = centers((0, 0, 8))
	assert all(a < b for a, b in zip(left, still))
	assert all(a > b for a, b in zip(right, still))
	assert forward[0] < still[0] and forward[1] > still[1]
	assert backward[0] > still[0] and backward[1] < still[1]


def test_motion_blur_debug_hue_and_strength():
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format="rgba8unorm")
	blur.resize((64, 64))
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def image(velocity, debug, strength=1):
		commands = RenderContext.commands("motion_debug")
		with commands.render_pass(color=blur.color_attachment(clear=(0.1, 0.1, 0.1, 0.5))):
			pass
		blur.record(commands, output.color_attachment(), linear_velocity=velocity, angular_velocity=(0, 0, 0), fovy_deg=60, exposure=0.1, reference_distance=2, strength=strength, max_blur=0.2, debug=debug)
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256}, (64, 64, 1))
		return np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 4).copy()
	still = image((0, 0, 0), True)
	baseline = image((8, 0, 0), False)
	left = image((8, 0, 0), True)
	right = image((-8, 0, 0), True)
	weak = image((8, 0, 0), True, 0.25)
	assert np.array_equal(still, baseline)
	assert np.array_equal(left[32, 32], baseline[32, 32])
	assert left[4, 32, 1] > left[4, 32, 0] and left[4, 32, 2] > left[4, 32, 0]
	assert right[4, 32, 0] > right[4, 32, 1] and right[4, 32, 0] > right[4, 32, 2]
	assert np.linalg.norm(weak[4, 32, :3].astype(float)-baseline[4, 32, :3]) < np.linalg.norm(left[4, 32, :3].astype(float)-baseline[4, 32, :3])
	assert np.array_equal(left[:, :, 3], baseline[:, :, 3])


def test_minimum_velocity_offsets_translation_without_affecting_rotation():
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format="rgba8unorm")
	blur.resize((32, 32))
	output = Texture((32, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def image(linear, angular, minimum):
		commands = RenderContext.commands("minimum_velocity")
		with commands.render_pass(color=blur.color_attachment(clear=(0.1, 0.1, 0.1, 1))):
			pass
		blur.record(commands, output.color_attachment(), linear_velocity=linear, angular_velocity=angular, fovy_deg=60, exposure=0.2, reference_distance=2, max_blur=0.3, debug=True, minimum_velocity=minimum)
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 128}, (32, 32, 1))
		return np.frombuffer(data, dtype=np.uint8).copy()
	zero = (0, 0, 0)
	assert np.array_equal(image((4, 0, 0), zero, 5), image(zero, zero, 0))
	assert np.array_equal(image((8, 0, 0), zero, 3), image((5, 0, 0), zero, 0))
	assert np.array_equal(image((-8, 0, 0), zero, 3), image((-5, 0, 0), zero, 0))
	assert np.array_equal(image((4, 0, 0), (0, 2, 0), 5), image(zero, (0, 2, 0), 0))
	with pytest.raises(ValueError, match="Invalid motion blur"):
		blur.record(RenderContext.commands("invalid"), output.color_attachment(), linear_velocity=zero, angular_velocity=zero, fovy_deg=60, minimum_velocity=-1)


def test_blur_exponent_shapes_vignette():
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format="rgba8unorm")
	blur.resize((64, 64))
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def image(exponent):
		commands = RenderContext.commands("blur_exponent")
		with commands.render_pass(color=blur.color_attachment(clear=(0.1, 0.1, 0.1, 1))):
			pass
		blur.record(commands, output.color_attachment(), linear_velocity=(-4, 0, 0), angular_velocity=(0, 0, 0), fovy_deg=60, exposure=0.1, reference_distance=2, max_blur=0.3, debug=True, blur_exponent=exponent)
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256}, (64, 64, 1))
		return np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 4).copy()
	soft, steep = image(0.5), image(3.0)
	assert soft[32, 18, 0] > steep[32, 18, 0]
	assert np.array_equal(soft[32, 32], steep[32, 32])
	assert np.array_equal(soft[4, 32], steep[4, 32])


def test_long_blur_filters_fine_stripes():
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format="rgba8unorm")
	blur.resize((1200, 32))
	shader = Shader(source="""
		@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
			let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
			return vec4f(uv*2.0-1.0, 0.0, 1.0);
		}
		@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
			let c = f32(u32(p.x / 2.0) % 2u);
			return vec4f(c, c, c, 1.0);
		}
	""")
	pipeline = RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none")
	output = Texture((1200, 32), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("long_motion_blur")
	with commands.render_pass(color=blur.color_attachment(clear=(0, 0, 0, 1))) as render:
		render.set_pipeline(pipeline)
		render.draw(3)
	blur.record(commands, output.color_attachment(), linear_velocity=(100, 0, 0), angular_velocity=(0, 0, 0), fovy_deg=60, exposure=1, reference_distance=1, max_blur=0.08)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 4800}, (1200, 32, 1))
	row = np.frombuffer(data, dtype=np.uint8).reshape(32, 1200, 4)[0, 300:900, 0]
	assert row.min() > 100 and row.max() < 155


@pytest.mark.parametrize("format", ["rgba8unorm", "rgba8unorm-srgb", "rgba32float"])
def test_resize_releases_cached_target_bindings(format):
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format=format)
	for size in ((16, 16), (24, 24), (32, 32)):
		blur.resize(size)
		for variant in blur.pipeline._cache.values():
			assert len(variant._bind_groups) == 0
		if blur.manual_filter:
			assert len(blur.pipeline.layout.groups[0]._bind_groups) == 0
		output = Texture(size, format=format, usage=TextureUsage.RENDER_ATTACHMENT)
		commands = RenderContext.commands("resize_motion_blur")
		with commands.render_pass(color=blur.color_attachment(clear=(0, 0, 0, 1))):
			pass
		blur.record(commands, output.color_attachment(), linear_velocity=(1, 0, 0), angular_velocity=(0, 0, 0), fovy_deg=60)
		RenderContext.submit(commands.finish())
		cache = blur.pipeline.layout.groups[0]._bind_groups if blur.manual_filter else next(iter(blur.pipeline._cache.values()))._bind_groups
		assert len(cache) == 1


def test_blur_strength_changes_smoothly_across_sample_boundaries():
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format="rgba8unorm")
	blur.resize((64, 64))
	shader = Shader(source="""
		@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
			let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
			return vec4f(uv*2.0-1.0, 0.0, 1.0);
		}
		@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
			let c = f32(u32(p.x / 2.0) % 2u);
			return vec4f(c, c, c, 1.0);
		}
	""")
	pipeline = RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none")
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	def image(strength):
		commands = RenderContext.commands("continuous_blur")
		with commands.render_pass(color=blur.color_attachment(clear=(0, 0, 0, 1))) as render:
			render.set_pipeline(pipeline)
			render.draw(3)
		blur.record(commands, output.color_attachment(), linear_velocity=(1, 0, 0), angular_velocity=(0, 0, 0), fovy_deg=60, exposure=0.0360844, reference_distance=1, strength=strength, max_blur=0.3)
		RenderContext.submit(commands.finish())
		data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256}, (64, 64, 1))
		return np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 4).astype(np.int16)
	below, above = image(0.999), image(1.001)
	assert np.max(np.abs(below-above)) <= 2


def test_vignette_blends_saturated_blur_gradually():
	RenderContext.setup_graphics_backend(True)
	blur = PeripheralMotionBlurPass(target_format="rgba8unorm")
	blur.resize((64, 64))
	shader = Shader(source="""
		@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
			let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
			return vec4f(uv*2.0-1.0, 0.0, 1.0);
		}
		@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
			let c = f32(u32(p.x / 2.0) % 2u);
			return vec4f(c, c, c, 1.0);
		}
	""")
	pipeline = RenderPipeline(shader, depth_test=None, depth_write=False, cull_mode="none")
	output = Texture((64, 64), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
	commands = RenderContext.commands("vignette_blend")
	with commands.render_pass(color=blur.color_attachment(clear=(0, 0, 0, 1))) as render:
		render.set_pipeline(pipeline)
		render.draw(3)
	blur.record(commands, output.color_attachment(), linear_velocity=(10000, 0, 0), angular_velocity=(0, 0, 0), fovy_deg=60, exposure=0.1, reference_distance=1, max_blur=0.15)
	RenderContext.submit(commands.finish())
	data = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": 256}, (64, 64, 1))
	pixels = np.frombuffer(data, dtype=np.uint8).reshape(64, 64, 4)
	assert pixels[32, 18, 0] > 200
	assert 90 < pixels[32, 4, 0] < 165
