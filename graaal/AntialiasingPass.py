"""Display-space antialiasing as an independent postprocessing pass."""
from dataclasses import fields
from operator import index
from pathlib import Path

import numpy as np

from .RenderContext import BindGroupLayout, PipelineLayout, RenderContext, RenderPipeline, Sampler, Shader, Texture, TextureUsage, wgpu
from .settings_validation import checked_dataclass, float_range, int_range


@checked_dataclass(frozen=True)
class AntialiasingSettings:
	lxaa_edge_threshold: float_range(0, 1) = 0.375
	lxaa_line_check: bool = True
	faaa_edge_threshold: float_range(1, 9) = 5.0
	faaa_quality: int_range(1, 9) = 5
	ddaa2_strength: float_range(0, 10) = 3.0
	ddaa2_threshold_min: float_range(0, 1) = 0.0625
	ddaa2_threshold_max: float_range(0, 1) = 0.166
	ddaa2_edge_steps: int_range(0, 5) = 5
	show_edges: bool = False
	hybrid_luma_chroma: bool = False
	chroma_weight: float_range(0, 5) = 0.5


class AntialiasingPass:
	"""Apply LXAA, FAAA or DDAA2 to a final, opaque single-sample color target.

	The pass preserves each pixel's alpha and expects opaque input.
	"""
	def __init__(self, *, target_format=None, settings=None):
		self.target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.sample_format = self.target_format.removesuffix("-srgb")
		self.manual_filter = self.sample_format == "rgba32float" and "float32-filterable" not in RenderContext.device.features
		self.sampler = None if self.manual_filter else Sampler(min_filter="linear", mag_filter="linear", address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
		self.layout = "auto"
		if self.manual_filter:
			color_binding = dict(binding=0, visibility=wgpu.ShaderStage.FRAGMENT, texture=dict(sample_type="unfilterable-float", view_dimension="2d", multisampled=False))
			self.layout = PipelineLayout([BindGroupLayout([color_binding])])
		self.sampled_color = None
		self.pipelines = {}
		self.settings = None
		self.set_settings(settings if settings is not None else AntialiasingSettings())
		self.mode = "off"
		self.size = self.source = None

	def set_settings(self, settings):
		"""Set the antialiasing controls."""
		if not isinstance(settings, AntialiasingSettings): raise TypeError("Expected AntialiasingSettings")
		if settings == self.settings: return False
		self.settings = settings
		return True

	def resize(self, size):
		"""Allocate the input target for a width and height."""
		if len(size) != 2: raise ValueError("Expected antialiasing width and height")
		try: size = tuple(index(value) for value in size)
		except TypeError: raise ValueError("Expected integer antialiasing dimensions") from None
		if min(size) <= 0: return False
		if size == self.size: return False
		self.source = Texture(size, format=self.target_format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_SRC, label="antialiasing_source")
		if self.sample_format != self.target_format:
			self.sampled_color = Texture(size, format=self.sample_format, usage=TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING, label="antialiasing_encoded")
		self.size = size
		return True

	def color_attachment(self, *, clear=None):
		"""Return the input attachment for the scene color."""
		if self.source is None: raise RuntimeError("Resize antialiasing targets first")
		return self.source.color_attachment(clear=clear)

	def record(self, commands, output, mode):
		"""Record LXAA, FAAA or DDAA2 to a single-sample output attachment."""
		if mode not in ("lxaa", "faaa", "ddaa2"): raise ValueError("Antialiasing mode must be lxaa, faaa or ddaa2")
		if self.source is None: raise RuntimeError("Resize antialiasing targets first")
		if output.format != self.target_format or output.sample_count != 1: raise ValueError("Antialiasing output must match the target format and be single-sample")
		params = {field.name: getattr(self.settings, field.name) for field in fields(self.settings) if field.name.startswith(mode+"_")}
		params["show_edges"] = self.settings.show_edges
		params["hybrid_luma_chroma"] = self.settings.hybrid_luma_chroma
		params["chroma_weight"] = float(self.settings.chroma_weight)
		key = (mode, *params.values())
		if key not in self.pipelines:
			shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "antialiasing.shader"), params={"srgb_surface": self.target_format.endswith("-srgb"), "manual_filter": self.manual_filter, "mode": mode, **params})
			pipeline = RenderPipeline(shader, vertex_entry="fullscreen", fragment_entry=mode, depth_test=None, depth_write=False, cull_mode="none", layout=self.layout, label=mode)
			if len(self.pipelines) >= 32: self.pipelines.pop(next(iter(self.pipelines)))
			self.pipelines[key] = (shader, pipeline)
		shader, pipeline = self.pipelines[key]
		color = self.source
		if self.sampled_color is not None:
			# Copy encoded bytes: the installed wgpu binding cannot create alternate sRGB views.
			commands.handle.copy_texture_to_texture({"texture": self.source.handle}, {"texture": self.sampled_color.handle}, (*self.size, 1))
			color = self.sampled_color
		bindings = shader.bind_group(0, scene_color=color) if self.manual_filter else shader.bind_group(0, scene_color=color, scene_sampler=self.sampler)
		with commands.render_pass(color=output, label=mode) as render:
			render.set_pipeline(pipeline)
			render.set_bind_groups(bindings)
			render.draw(3)
