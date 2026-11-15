"""Shared rendering controls and optional profiler for interactive examples.

Create alongside a scene's RenderingSettings, register handle_char/handle_key,
and call commands()/submit() for work shown in the profiler. Scene-specific
shortcuts remain with the scene.
"""

from graaal.Diagnostics import RenderMonitor
from graaal.diagnostics_visualizer import DiagnosticsVisualizer
from graaal.RenderContext import RenderContext


class ExampleOverlay:
	def __init__(self, settings, *, renderer=None, get_ao_method=None, set_ao_method=None, profiler_position=(370, 12)):
		self.settings = settings
		self.get_ao_method = get_ao_method
		self.set_ao_method = set_ao_method
		self.monitor = RenderMonitor(renderer=renderer)
		self.profiler = DiagnosticsVisualizer(visible=False, position=profiler_position)

	def handle_char(self, code):
		char = chr(code).lower()
		if char == "m":
			self.settings.set_msaa_enabled(self.settings.aa_mode != "msaa")
			print(f"MSAA: {self.settings.renderer.sample_count}x")
		elif char == "a":
			self.settings.set_ssao_enabled(not self.settings.ssao_enabled)
			print(f"SSAO: {self.settings.ssao_enabled}")
		elif char == "o" and self.get_ao_method is not None and self.set_ao_method is not None:
			method = "pass" if self.get_ao_method() == "compose" else "compose"
			self.set_ao_method(method)
			print(f"SSAO path: {method}")
		elif char == "p":
			self.profiler.visible = not self.profiler.visible
		else:
			return False
		return True

	def handle_key(self, key, scancode, action, mods):
		return self.settings.handle_key(key, scancode, action, mods)

	def commands(self, label):
		return self.monitor.commands(label) if self.profiler.visible else RenderContext.commands(label)

	def submit(self, commands):
		if self.profiler.visible:
			self.profiler.update(self.monitor.submit(commands.finish()))
		else:
			RenderContext.submit(commands.finish())

	def draw(self):
		self.settings.draw()
		self.profiler.draw()
