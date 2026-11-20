"""ImGui view of RenderMonitor submission reports.

The widget owns only presentation state. Feed it reports after submission and
call draw() inside an ImGui.render callback; it creates no GPU resources.
"""
from collections import deque

import numpy as np

try:
	from imgui_bundle import imgui
except ModuleNotFoundError as error:
	if error.name != "imgui_bundle": raise
	imgui = None


def _size(value):
	if value is None: return "unavailable"
	for unit in ("B", "KiB", "MiB", "GiB"):
		if abs(value) < 1024 or unit == "GiB": return f"{value:.1f} {unit}"
		value /= 1024


def _ms(value):
	return "--" if value is None else f"{value:.2f} ms"


class DiagnosticsVisualizer:
	"""Bounded performance history and expandable pass/resource details.

	Call update(monitor.submit(...)) once per measured submission, then draw()
	inside ImGui.render(). Set visible=False to hide it; the owner decides when
	to collect reports. Updating reports works without the optional UI package.
	"""
	def __init__(self, *, visible=True, history_size=120, position=(480, 12)):
		if type(history_size) is not int or history_size < 1:
			raise ValueError("history_size must be a positive integer")
		self.visible = visible
		self.position = position
		self.report = None
		self.paused = False
		self.history = deque(maxlen=history_size)
		self._history_source = None

	def update(self, report):
		"""Keep one report plus a bounded history of the measured command spans."""
		if self.paused: return
		self.report = report
		commands = report.get("commands", ())
		gpu = report.get("gpu_timing_available", False) and any(entry.get("gpu_span_ms") is not None for entry in commands)
		source = "GPU pass span" if gpu else "CPU recording"
		if source != self._history_source:
			self.history.clear()
			self._history_source = source
		value = sum(entry.get("gpu_span_ms") or 0 for entry in commands) if gpu else sum(entry.get("encode_ms") or 0 for entry in commands)
		self.history.append(value)

	def draw(self):
		"""Draw inside the active ImGui frame; requires the optional UI package."""
		if not self.visible: return
		if imgui is None:
			raise ModuleNotFoundError('Install UI with: python -m pip install -e ".[ui]"', name="imgui_bundle")

		imgui.set_next_window_pos(self.position, imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((500, 620), imgui.Cond_.first_use_ever)
		expanded, self.visible = imgui.begin("Performance profiler", self.visible)
		try:
			if not expanded: return
			_, self.paused = imgui.checkbox("Pause display", self.paused)
			if self.report is None:
				imgui.text_disabled("No report yet. Submit work through RenderMonitor.")
				return
			self._draw_summary(imgui)
			self._draw_passes(imgui)
			self._draw_commands(imgui)
			self._draw_resources(imgui)
		finally:
			imgui.end()

	def _draw_summary(self, imgui):
		report = self.report
		commands = report.get("commands", ())
		cpu_record = sum(entry.get("encode_ms") or 0 for entry in commands)
		gpu_spans = [entry["gpu_span_ms"] for entry in commands if entry.get("gpu_span_ms") is not None]
		imgui.separator_text("Frame")
		imgui.text(f"CPU record: {_ms(cpu_record)}   Submit: {_ms(report.get('cpu_submit_ms'))}")
		imgui.text(f"GPU pass span: {_ms(sum(gpu_spans) if gpu_spans else None)}")
		if not report.get("gpu_timing_available"):
			imgui.text_disabled("GPU times need the timestamp-query device feature.")
		imgui.text(f"Batches: {report.get('draw_batches') if report.get('draw_batches') is not None else '--'}"
			f"   Draws: {report.get('encoded_draw_calls', 0)}   Dispatches: {report.get('encoded_dispatches', 0)}")
		if self.history:
			values = np.asarray(self.history, dtype=np.float32)
			imgui.plot_lines(f"{self._history_source} (ms)", values, overlay_text=f"avg {float(values.mean()):.2f} ms",
				graph_size=(0, 60))
		memory = report.get("memory", {})
		imgui.text(f"CPU process: {_size(memory.get('cpu_rss_bytes'))}"
			f"   Buffer mirrors: {_size(memory.get('cpu_buffer_mirrors_bytes'))}")
		imgui.text(f"Tracked GPU allocation estimate: {_size(memory.get('gpu_tracked_bytes'))}")
		if memory.get("gpu_unestimated_count"):
			imgui.text_disabled(f"{memory['gpu_unestimated_count']} texture format(s) have no size estimate.")

	def _draw_passes(self, imgui):
		if not imgui.collapsing_header("Pass timings"): return
		passes = [(command.get("label") or "commands", scope) for command in self.report.get("commands", ())
			for scope in command.get("passes", ())]
		if not passes:
			imgui.text_disabled("No render or compute passes recorded.")
			return
		gpu = any(scope.get("gpu_ms") is not None for _, scope in passes)
		imgui.text_disabled("GPU execution" if gpu else "CPU encoding; GPU times unavailable")
		maximum = max((scope.get("gpu_ms") if gpu else scope.get("cpu_encode_ms")) or 0 for _, scope in passes)
		for index, (command_name, scope) in enumerate(passes):
			value = (scope.get("gpu_ms") if gpu else scope.get("cpu_encode_ms")) or 0
			label = scope.get("label") or f"{scope.get('kind', 'pass')}_{index}"
			imgui.text(f"{command_name} / {label}: {_ms(scope.get('gpu_ms'))} GPU, {_ms(scope.get('cpu_encode_ms'))} CPU")
			imgui.progress_bar(value / maximum if maximum > 0 else 0, (-1, 5), "")

	def _draw_commands(self, imgui):
		if not imgui.collapsing_header("Commands and used resources"): return
		for index, command in enumerate(self.report.get("commands", ())):
			label = command.get("label") or f"commands_{index}"
			if not imgui.tree_node(f"{label}##command_{index}"): continue
			try:
				imgui.text(f"CPU record {_ms(command.get('encode_ms'))}; GPU pass span {_ms(command.get('gpu_span_ms'))}")
				for call in command.get("calls", ()):
					resources = ", ".join(str(call[name]) for name in ("buffer", "source", "destination") if call.get(name))
					imgui.text(f"{call.get('operation')}({resources}): {_ms(call.get('cpu_encode_ms'))} CPU")
				for pass_index, scope in enumerate(command.get("passes", ())):
					name = scope.get("label") or f"{scope.get('kind', 'pass')}_{pass_index}"
					if not imgui.tree_node(f"{name}##pass_{index}_{pass_index}"): continue
					try:
						imgui.text(f"{_ms(scope.get('gpu_ms'))} GPU; {_ms(scope.get('cpu_encode_ms'))} CPU")
						for call in scope.get("calls", ()):
							arguments = ", ".join(call.get("arguments", ()))
							imgui.text(f"{call.get('operation')}({arguments})  {_ms(call.get('cpu_encode_ms'))} CPU")
						for resource in scope.get("resources", ()):
							imgui.text_disabled(resource)
					finally:
						imgui.tree_pop()
			finally:
				imgui.tree_pop()

	def _draw_resources(self, imgui):
		if not imgui.collapsing_header("Live buffers and textures"): return
		resources = self.report.get("memory", {}).get("resources", ())
		if not resources:
			imgui.text_disabled("No tracked engine resources.")
			return
		for resource in sorted(resources, key=lambda item: item.get("gpu_bytes") or 0, reverse=True):
			name = resource.get("name", "unnamed")
			gpu = _size(resource.get("gpu_bytes"))
			if resource.get("kind") == "texture":
				imgui.text(f"Texture {name}: {gpu} | {resource.get('format')} {resource.get('size')}")
			else:
				imgui.text(f"Buffer {name}: {gpu} GPU | {_size(resource.get('cpu_mirror_bytes'))} CPU")
