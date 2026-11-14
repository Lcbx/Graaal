"""Display-space antialiasing as an independent postprocessing pass."""
from dataclasses import dataclass, fields
from operator import index
from pathlib import Path

import numpy as np

from .RenderContext import RenderContext, RenderPipeline, Sampler, Shader, Texture, TextureUsage


@dataclass(frozen=True)
class AntialiasingSettings:
	lxaa_edge_threshold: float = 0.375
	lxaa_line_check: bool = True
	faaa_edge_threshold: float = 5.0
	faaa_quality: int = 5
	faaa_show_edges: bool = False
	ddaa2_strength: float = 3.0
	ddaa2_threshold_min: float = 0.0625
	ddaa2_threshold_max: float = 0.166
	ddaa2_edge_steps: int = 5

	def __post_init__(self):
		if not np.isfinite(self.lxaa_edge_threshold) or not 0 <= self.lxaa_edge_threshold <= 1: raise ValueError("LXAA threshold must be between 0 and 1")
		if not np.isfinite(self.faaa_edge_threshold) or not 1 <= self.faaa_edge_threshold <= 9: raise ValueError("FAAA threshold must be between 1 and 9")
		if type(self.faaa_quality) is not int or not 1 <= self.faaa_quality <= 9: raise ValueError("FAAA quality must be an integer between 1 and 9")
		if type(self.lxaa_line_check) is not bool or type(self.faaa_show_edges) is not bool: raise ValueError("AA switches must be boolean")
		if not np.isfinite(self.ddaa2_strength) or not 0 <= self.ddaa2_strength <= 10: raise ValueError("DDAA2 strength must be between 0 and 10")
		if not all(np.isfinite(value) and 0 <= value <= 1 for value in (self.ddaa2_threshold_min, self.ddaa2_threshold_max)): raise ValueError("DDAA2 thresholds must be between 0 and 1")
		if type(self.ddaa2_edge_steps) is not int or not 0 <= self.ddaa2_edge_steps <= 5: raise ValueError("DDAA2 edge steps must be an integer between 0 and 5")


class AntialiasingPass:
	"""Apply LXAA, FAAA or DDAA2 to a final, opaque single-sample color target.

	The pass preserves each pixel's alpha and expects opaque input.
	"""
	def __init__(self, *, target_format=None, settings=None):
		self.target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.sample_format = self.target_format.removesuffix("-srgb")
		self.sampler = Sampler(min_filter="linear", mag_filter="linear", address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
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
		key = (mode, *params.values())
		if key not in self.pipelines:
			shader = Shader(filepath=str(Path(__file__).with_name("shaders") / "antialiasing.shader"), params={"srgb_surface": self.target_format.endswith("-srgb"), "mode": mode, **params})
			pipeline = RenderPipeline(shader, vertex_entry="fullscreen", fragment_entry=mode, depth_test=None, depth_write=False, cull_mode="none", label=mode)
			if len(self.pipelines) >= 32: self.pipelines.pop(next(iter(self.pipelines)))
			self.pipelines[key] = (shader, pipeline)
		shader, pipeline = self.pipelines[key]
		color = self.source
		if self.sampled_color is not None:
			# Copy encoded bytes: the installed wgpu binding cannot create alternate sRGB views.
			commands.handle.copy_texture_to_texture({"texture": self.source.handle}, {"texture": self.sampled_color.handle}, (*self.size, 1))
			color = self.sampled_color
		bindings = shader.bind_group(0, scene_color=color, scene_sampler=self.sampler)
		with commands.render_pass(color=output, label=mode) as render:
			render.set_pipeline(pipeline)
			render.set_bind_groups(bindings)
			render.draw(3)
