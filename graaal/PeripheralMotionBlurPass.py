from operator import index
from pathlib import Path
import math
import numpy as np
from .RenderContext import BindGroupLayout, PipelineLayout, RenderContext, RenderPipeline, Sampler, Shader, Texture, TextureUsage, wgpu


class PeripheralMotionBlurPass:
	"""Camera-motion blur with a sharp center and a depth-independent translation approximation.

	Velocities passed to record are camera-local: x right, y up, z backwards;
	angular velocity is in radians per second about those same axes.
	"""
	def __init__(self, *, target_format=None):
		self.target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.sample_format = self.target_format.removesuffix("-srgb")
		self.manual_filter = self.sample_format == "rgba32float" and "float32-filterable" not in RenderContext.device.features
		self.sampler = None if self.manual_filter else Sampler(min_filter="linear", mag_filter="linear", address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
		layout = "auto"
		if self.manual_filter:
			entries = [dict(binding=0, visibility=wgpu.ShaderStage.FRAGMENT, texture=dict(sample_type="unfilterable-float", view_dimension="2d", multisampled=False)), dict(binding=2, visibility=wgpu.ShaderStage.FRAGMENT, buffer=dict(type="uniform"))]
			layout = PipelineLayout([BindGroupLayout(entries)])
		self.shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "peripheral_motion_blur.shader"), params={"srgb_surface": self.target_format.endswith("-srgb"), "manual_filter": self.manual_filter})
		self.pipeline = RenderPipeline(self.shader, vertex_entry="fullscreen", fragment_entry="fragment", depth_test=None, depth_write=False, cull_mode="none", layout=layout, label="peripheral_motion_blur")
		self.uniforms = self.shader.uniform_buffer()
		self.source = self.sampled_color = self.size = None

	def resize(self, size):
		"""Allocate the single-sample scene color target."""
		if len(size) != 2: raise ValueError("Expected width and height")
		try: size = tuple(index(value) for value in size)
		except TypeError: raise ValueError("Expected integer dimensions") from None
		if min(size) <= 0 or size == self.size: return False
		self.source = Texture(size, format=self.target_format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_SRC, label="motion_blur_source")
		self.sampled_color = Texture(size, format=self.sample_format, usage=TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING, label="motion_blur_encoded") if self.sample_format != self.target_format else None
		self.size = size
		# Cached bind groups retain the old input texture views.
		self.pipeline.clear_bind_group_cache()
		return True

	def color_attachment(self, *, clear=None):
		if self.source is None: raise RuntimeError("Resize motion blur targets first")
		return self.source.color_attachment(clear=clear)

	def record(self, commands, output, *, linear_velocity, angular_velocity, fovy_deg, exposure=1/60, reference_distance=10.0, strength=1.0, inner_radius=0.25, outer_radius=0.85, max_blur=0.08, debug=False, minimum_velocity=0.0, blur_exponent=1.0):
		"""Record blur, subtracting minimum_velocity from linear speed before projection.

		inner_radius and outer_radius bound the smooth peripheral transition, with
		0 at screen center and 1 at screen corners. blur_exponent shapes its ramp.
		The mask scales blur length, not a second color blend.
		Debug blends direction hue with opacity based on blur length relative to max_blur.
		"""
		if self.source is None: raise RuntimeError("Resize motion blur targets first")
		if output.format != self.target_format or output.sample_count != 1: raise ValueError("Motion blur output must match the target format and be single-sample")
		linear, angular = (np.asarray(v, dtype=np.float32) for v in (linear_velocity, angular_velocity))
		if linear.shape != (3,) or angular.shape != (3,) or not np.all(np.isfinite(linear)) or not np.all(np.isfinite(angular)): raise ValueError("Expected finite three-component velocities")
		values = (fovy_deg, exposure, reference_distance, strength, inner_radius, outer_radius, max_blur, minimum_velocity, blur_exponent)
		if not all(math.isfinite(v) for v in values) or not (0 < fovy_deg < 180 and exposure >= 0 and reference_distance > 0 and strength >= 0 and 0 <= inner_radius < outer_radius and max_blur >= 0 and minimum_velocity >= 0 and blur_exponent > 0): raise ValueError("Invalid motion blur parameters")
		speed = np.linalg.norm(linear.astype(np.float64))
		if speed > 0: linear = linear * (max(0.0, speed-minimum_velocity)/speed)
		self.uniforms.content["linear"] = (*linear, 0)
		self.uniforms.content["angular"] = (*angular, 0)
		self.uniforms.content["params"] = (math.tan(math.radians(fovy_deg)/2), exposure * strength, reference_distance, max_blur)
		self.uniforms.content["radii"] = (inner_radius, outer_radius, float(debug), blur_exponent)
		self.uniforms.upload()
		color = self.source
		if self.sampled_color is not None:
			commands.handle.copy_texture_to_texture({"texture": self.source.handle}, {"texture": self.sampled_color.handle}, (*self.size, 1))
			color = self.sampled_color
		bindings = self.shader.bind_group(0, scene_color=color, motion=self.uniforms) if self.manual_filter else self.shader.bind_group(0, scene_color=color, scene_sampler=self.sampler, motion=self.uniforms)
		with commands.render_pass(color=output, label="peripheral_motion_blur") as render:
			render.set_pipeline(self.pipeline)
			render.set_bind_groups(bindings)
			render.draw(3)
