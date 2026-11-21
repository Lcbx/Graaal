"""Adapt existing image regressions to the unified postprocessing pass."""
from graaal.RenderPasses.PostprocessingPass import AntialiasingEffect, PostprocessingPass
from graaal.RenderPasses.pass_settings import AntialiasingSettings
from scenes.postprocessing_example import PeripheralMotionBlurEffect
from graaal.RenderContext import Sampler


class AntialiasingHarness:
	def __init__(self, *, target_format=None, settings=None):
		self.effect = AntialiasingEffect(settings=settings)
		self.post = PostprocessingPass([self.effect], target_format=target_format)

	@property
	def source(self): return self.post.targets[0]

	@source.setter
	def source(self, value): self.post.targets[0] = value

	@property
	def settings(self): return self.effect.settings

	@property
	def manual_filter(self): return self.post.manual_filter

	@manual_filter.setter
	def manual_filter(self, value):
		self.post.manual_filter = value
		self.post.sampler = None if value else Sampler(min_filter="linear", mag_filter="linear", address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")

	def set_settings(self, settings): return self.effect.set_settings(settings)
	def resize(self, size): return self.post.resize(size)
	def color_attachment(self, **kwargs): return self.post.color_attachment(**kwargs)
	def record(self, commands, output, mode):
		self.effect.mode = mode
		self.post.record(commands, output)


class PeripheralMotionBlurHarness:
	def __init__(self, *, target_format=None):
		self.effect = PeripheralMotionBlurEffect()
		self.post = PostprocessingPass([self.effect], target_format=target_format)

	@property
	def source(self): return self.post.targets[0]

	@source.setter
	def source(self, value): self.post.targets[0] = value

	@property
	def manual_filter(self): return self.post.manual_filter

	def resize(self, size): return self.post.resize(size)
	def color_attachment(self, **kwargs): return self.post.color_attachment(**kwargs)
	def record(self, commands, output, **kwargs):
		self.effect.set_motion(**kwargs)
		self.post.record(commands, output)
