"""Scene-sized color target sampled into a window-sized output."""
from pathlib import Path

from graaal.RenderContext import RenderContext, RenderPipeline, Sampler, Shader, Texture, TextureUsage


class ResolutionPresenter:
	def __init__(self, target_format=None):
		self.target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.shader = Shader(filepath=str(Path(__file__).resolve().parents[1] / "shaders/render_resolution.shader"))
		self.pipeline = RenderPipeline(self.shader, vertex_entry="fullscreen", fragment_entry="present",
			depth_test=None, depth_write=False, cull_mode="none", label="resolution_present")
		self.sampler = Sampler(min_filter="linear", mag_filter="linear", address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
		self.size = self.color = self.bindings = None

	def resize(self, size):
		if size == self.size: return False
		self.color = Texture(size, format=self.target_format,
			usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
			label="resolution_color")
		self.bindings = self.shader.bind_group(0, scene_color=self.color, scene_sampler=self.sampler)
		self.size = size
		return True

	def color_attachment(self):
		return self.color.color_attachment()

	def record(self, commands, output):
		"""Sample the scene color across the destination attachment."""
		with commands.render_pass(color=output, label="resolution_present") as render:
			render.set_pipeline(self.pipeline)
			render.set_bind_groups(self.bindings)
			render.draw(3)
