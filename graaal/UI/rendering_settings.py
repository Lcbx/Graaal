"""Optional ImGui rendering and transparency controls.

Pass a TransparencyPass or MaterialRenderer and optional camera to RenderingSettings, register
handle_key with the key event, and call draw inside the ImGui frame. Optional
callbacks connect the MSAA and SSAO toggles to scene-specific rendering paths.
F toggles visibility; visible=False starts with the settings hidden.
"""
from dataclasses import replace

import glfw
import numpy as np

try:
	from imgui_bundle import imgui
except ModuleNotFoundError as error:
	if error.name != "imgui_bundle": raise
	imgui = None

from graaal.RenderPasses.TransparencyPass import DEBUG_VIEWS


DEBUG_VIEW_HELP = {
	"final": "Final image: opaque color seen through the transparent surfaces, plus reconstructed transparent color.",
	"transmittance": "Scalar light transmission: white means fully transparent and black means fully opaque.",
	"winner": "Raw averaged tail color before filtering. Missing winners contribute zero.",
	"reconstruction": "Reconstructed transparent contribution without the opaque background.",
	"nearest": "Exact nearest-layer color multiplied by its RGB opacity, before tail and background composition.",
	"nearest_mask": "White where a nonzero-opacity visible fragment was selected as the exact nearest layer.",
	"tail_transmittance": "Scalar tail transmission after excluding the exact nearest fragment.",
}


def _item_tooltip(text):
	if imgui.is_item_hovered():
		imgui.begin_tooltip()
		imgui.push_text_wrap_pos(imgui.get_font_size()*30)
		try:
			imgui.text_unformatted(text)
		finally:
			imgui.pop_text_wrap_pos()
			imgui.end_tooltip()


class RenderingSettings:
	"""Live rendering controls for a TransparencyPass or MaterialRenderer."""

	def __init__(self, renderer, *, visible=True, position=(12, 12), on_msaa_changed=None,
		on_ssao_changed=None, ssao_enabled=False, ao_pass=None, aa_pass=None, camera=None):
		self.controls = renderer
		self.renderer = getattr(renderer, "transparency", renderer)
		self.camera = camera
		self.visible = visible
		self.position = position
		self.resolution_percent = 100
		self.on_msaa_changed = on_msaa_changed
		self.on_ssao_changed = on_ssao_changed
		self.ao_pass = ao_pass
		self.aa_pass = aa_pass
		self.aa_mode = "msaa" if getattr(self.renderer, "sample_count", 1) == 4 else "off"
		self.ssao_enabled = ssao_enabled
		if self.on_ssao_changed is not None:
			self.on_ssao_changed(ssao_enabled)
		elif hasattr(self.renderer, "set_ambient_occlusion"):
			self.renderer.set_ambient_occlusion(ssao_enabled)

	def set_msaa_enabled(self, enabled):
		if type(enabled) is not bool: raise ValueError("MSAA enabled must be boolean")
		self.set_aa_mode("msaa" if enabled else "off")

	def set_resolution_percent(self, percent):
		"""Set the scene render size as a percentage of the window framebuffer."""
		if type(percent) is not int or not 25 <= percent <= 100:
			raise ValueError("Rendering resolution must be an integer percentage from 25 to 100")
		self.resolution_percent = percent

	def render_size(self, window_size):
		return tuple(max(1, (size*self.resolution_percent+50)//100) for size in window_size)

	def set_aa_mode(self, mode):
		if mode not in ("off", "msaa", "lxaa", "faaa", "ddaa2"): raise ValueError("Unknown antialiasing mode")
		if mode in ("lxaa", "faaa", "ddaa2") and self.aa_pass is None: raise ValueError("Postprocess AA requires an antialiasing pass or effect")
		sample_count = 4 if mode == "msaa" else 1
		if self.renderer.sample_count != sample_count:
			if self.on_msaa_changed is not None:
				self.on_msaa_changed(mode == "msaa")
			else:
				self.controls.set_sample_count(sample_count)
		self.aa_mode = mode
		if self.aa_pass is not None: self.aa_pass.mode = mode

	def set_ssao_enabled(self, enabled):
		if type(enabled) is not bool: raise ValueError("SSAO enabled must be boolean")
		self.ssao_enabled = enabled
		if self.on_ssao_changed is not None:
			self.on_ssao_changed(enabled)
		elif hasattr(self.renderer, "set_ambient_occlusion"):
			self.renderer.set_ambient_occlusion(enabled)


	@staticmethod
	def available_debug_views(renderer):
		"""Diagnostic views supported by banded transparency."""
		return DEBUG_VIEWS


	@classmethod
	def sync_debug_view(cls, renderer):
		views = cls.available_debug_views(renderer)
		if renderer.debug_view not in views:
			renderer.set_debug_view("final")
		return views


	@staticmethod
	def debug_view_label(renderer, view):
		return "signed residual" if view == "winner" and renderer.banded_proxy else view

	def adjust(self, direction):
		self.renderer.set_kawase_passes(max(0, min(8, self.renderer.kawase_passes+direction)))

	def reset(self):
		self.renderer.set_kawase_passes(1)

	def handle_key(self, key, scancode, action, mods):
		if action not in (glfw.PRESS, glfw.REPEAT):
			return False
		if key == glfw.KEY_F:
			if action == glfw.PRESS:
				self.visible = not self.visible
			return True
		if not self.visible:
			return False
		if key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
			self.adjust(-1 if key == glfw.KEY_LEFT else 1)
		elif key == glfw.KEY_HOME:
			self.reset()
		else:
			return False
		return True

	def set_grazing_angle(self, name, value):
		"""Keep the editable fade interval ordered, including typed input."""
		if name not in ("start_angle", "end_angle"):
			raise ValueError("Expected start_angle or end_angle")
		if not np.isfinite(value):
			return
		options = self.renderer.grazing_fade
		if name == "start_angle":
			value = max(0, min(value, options.end_angle-0.01))
		else:
			value = min(90, max(value, options.start_angle+0.01))
		self.renderer.set_grazing_fade(replace(options, **{name: value}))

	def draw_grazing(self):
		if imgui is None:
			raise ModuleNotFoundError('Install UI with: python -m pip install -e ".[ui]"', name="imgui_bundle")
		imgui.separator_text("Grazing fade")
		fade = self.renderer.grazing_fade
		changed, enabled = imgui.checkbox("Fade grazing surfaces (G)", fade.enabled)
		_item_tooltip("Smooth opacity fade between angles from the geometric normal: 0 face-on, 90 edge-on.")
		if changed: self.renderer.set_grazing_fade(replace(fade, enabled=enabled))
		if self.renderer.grazing_fade.enabled:
			for name, label in (("start_angle", "Fade start (degrees)"), ("end_angle", "Fully faded (degrees)")):
				changed, value = imgui.drag_float(label, getattr(self.renderer.grazing_fade, name), 0.5, 0.0, 90.0, "%.2f")
				if changed: self.set_grazing_angle(name, value)

	def draw_general_rendering_options(self, imgui):
		changed, value = imgui.slider_int("Rendering resolution (%)", self.resolution_percent, 25, 100)
		_item_tooltip("Render the scene at a percentage of the window framebuffer, then sample it into the unchanged window size. The interface stays sharp.")
		if changed: self.set_resolution_percent(value)
		if self.camera is not None:
			changed, value = imgui.slider_float("FoV (degrees)", self.camera.fovy_deg, 20.0, 120.0, "%.0f")
			if changed: self.camera.fovy_deg = value
		modes = ("off", "msaa", "lxaa", "faaa", "ddaa2") if self.aa_pass is not None else ("off", "msaa")
		changed, index = imgui.combo("Antialiasing (M)", modes.index(self.aa_mode), ("Off", "MSAA 4x", "LXAA", "FAAA", "DDAA2")[:len(modes)])
		_item_tooltip("4x samples opaque/MASK geometry, transmission and the exact nearest layer. Tail estimation defaults to 1x. Linear HDR is resolved before tone mapping.")
		if changed: self.set_aa_mode(modes[index])
		if self.aa_pass is not None and self.aa_mode in ("lxaa", "faaa", "ddaa2"):
			options = self.aa_pass.settings
			changed, value = imgui.checkbox("Show AA edges", options.show_edges)
			if changed: options = replace(options, show_edges=value)
			changed, value = imgui.checkbox("Hybrid luma/chroma", options.hybrid_luma_chroma)
			if changed: options = replace(options, hybrid_luma_chroma=value)
			if options.hybrid_luma_chroma:
				changed, value = imgui.slider_float("Chroma weight", options.chroma_weight, 0.01, 5.0, "%.3f")
				if changed: options = replace(options, chroma_weight=value)
			if self.aa_mode == "lxaa":
				changed, value = imgui.slider_float("LXAA edge threshold", options.lxaa_edge_threshold, 0.01, 1.0, "%.3f")
				if changed: options = replace(options, lxaa_edge_threshold=value)
				changed, value = imgui.checkbox("LXAA line check", options.lxaa_line_check)
				if changed: options = replace(options, lxaa_line_check=value)
			elif self.aa_mode == "faaa":
				changed, value = imgui.slider_float("FAAA edge threshold", options.faaa_edge_threshold, 1.0, 9.0, "%.2f")
				if changed: options = replace(options, faaa_edge_threshold=value)
				changed, value = imgui.slider_int("FAAA quality", options.faaa_quality, 1, 9)
				if changed: options = replace(options, faaa_quality=value)
			else:
				changed, value = imgui.slider_float("DDAA2 diffusion strength", options.ddaa2_strength, 0.01, 10.0, "%.2f")
				if changed: options = replace(options, ddaa2_strength=value)
				changed, value = imgui.slider_float("DDAA2 minimum edge threshold", options.ddaa2_threshold_min, 0.01, 1.0, "%.4f")
				if changed: options = replace(options, ddaa2_threshold_min=value)
				changed, value = imgui.slider_float("DDAA2 relative edge threshold", options.ddaa2_threshold_max, 0.01, 1.0, "%.4f")
				if changed: options = replace(options, ddaa2_threshold_max=value)
				changed, value = imgui.slider_int("DDAA2 edge search steps", options.ddaa2_edge_steps, 0, 5)
				_item_tooltip("Each step searches three pixels in each direction; zero disables edge search.")
				if changed: options = replace(options, ddaa2_edge_steps=value)
			self.aa_pass.set_settings(options)
		changed, enabled = imgui.checkbox("SSAO (A)", self.ssao_enabled)
		_item_tooltip("Screen-space ambient occlusion from the main depth buffer. A toggles SSAO; O selects the compose or separate-pass implementation.")
		if changed: self.set_ssao_enabled(enabled)

	def draw_ambient_occlusion_options(self, imgui):
		if not self.ssao_enabled or not hasattr(self.renderer, "ambient_occlusion_settings"): return
		imgui.separator_text("Ambient occlusion")
		options = self.renderer.ambient_occlusion_settings
		for name, label, step in (("radius", "Radius", 0.01), ("turns", "Spiral turns", 0.1), ("bias", "Bias", 0.005), ("falloff", "Falloff", 0.1), ("strength", "Strength", 0.1)):
			changed, value = imgui.drag_float(label, getattr(options, name), step, 0.0, 0.0, "%.4g")
			if changed and np.isfinite(value):
				options = replace(options, **{name: max(0.0, value)})
				self.controls.set_ambient_occlusion_settings(options)
				if self.ao_pass is not None: self.ao_pass.set_settings(options)
		changed, count = imgui.slider_int("AO samples", options.sample_count, 1, 64)
		if changed:
			options = replace(options, sample_count=count)
			self.controls.set_ambient_occlusion_settings(options)
			if self.ao_pass is not None: self.ao_pass.set_settings(options)

	def draw_transparency_options(self, imgui):
		views = self.sync_debug_view(self.renderer)
		labels = tuple(self.debug_view_label(self.renderer, view) for view in views)
		changed, index = imgui.combo("Diagnostic view", views.index(self.renderer.debug_view), labels)
		help_text = DEBUG_VIEW_HELP[views[index]]
		if views[index] == "winner" and self.renderer.banded_proxy:
			help_text = "Signed unblurred residual, displayed as 0.5 + 0.5 * residual: neutral grey is zero. Display clips outside [-1, 1]; float filtering remains signed and unclamped."
		_item_tooltip(help_text)
		if changed: self.renderer.set_debug_view(views[index])

	def draw_banded_tail_options(self, imgui):
		imgui.separator_text("Tail estimation")
		changed, enabled = imgui.checkbox("Proxy residual", self.renderer.banded_proxy)
		_item_tooltip("Accumulate opacity-weighted texture albedo at full resolution and blur only the signed shading residual. The proxy is restored locally; thick-tail mixing remains approximate.")
		if changed: self.controls.set_banded_options(proxy=enabled)
		changed, enabled = imgui.checkbox("Shade one pixel per 2 x 2 block", self.renderer.banded_subset)
		_item_tooltip("Fixed spatial mask, compact payloads, and one lighting invocation per block. Increases noise; compare one additional blur pair. Alpha derivatives still run before the allocation discard.")
		if changed: self.controls.set_banded_options(subset=enabled)
		imgui.separator_text("Dual Kawase tail blur")
		changed, passes = imgui.slider_int("Down/up pairs", self.renderer.kawase_passes, 0, 8)
		_item_tooltip("Each pair adds one half-resolution downsample and one upsample. Zero shows the unblurred tail. Color and allocation confidence are blurred together; the nearest layer stays sharp. Blur is not depth-guided and can bleed across tail edges.")
		if changed: self.renderer.set_kawase_passes(passes)
		changed, value = imgui.drag_float("Screen noise cell size (px at 1x)", self.renderer.noise_scale, 0.05, 0.25, 4.0, "%.3g")
		if changed and np.isfinite(value): self.renderer.set_noise_scale(max(0.25, min(4.0, value)))
		self.draw_grazing()
		imgui.text_wrapped("Left/Right changes blur pairs. F hides/shows settings. Tail noise is shared in screen space.")

		if imgui.button("Reset blur (Home)"): self.reset()

	def draw(self):
		if imgui is None:
			raise ModuleNotFoundError('Install UI with: python -m pip install -e ".[ui]"', name="imgui_bundle")

		self.sync_debug_view(self.renderer)
		io = imgui.get_io()
		imgui.set_next_window_pos((io.display_size.x-12, 12), imgui.Cond_.always, (1, 0))
		imgui.set_next_window_bg_alpha(0.65)
		flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize
			| imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_focus_on_appearing
			| imgui.WindowFlags_.no_inputs)
		imgui.begin("Frame rate", flags=flags)
		try:
			fps = io.framerate  # ImGui's rolling average includes the whole frame.
			imgui.text(f"{fps:.0f} FPS | {1000/fps:.2f} ms" if fps > 0 else "FPS: --")
		finally:
			imgui.end()
		if not self.visible: return
		imgui.set_next_window_pos(self.position, imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((460, 560), imgui.Cond_.first_use_ever)
		expanded, self.visible = imgui.begin("Rendering settings", self.visible)
		try:
			if not expanded: return
			imgui.text_disabled("Hover a setting or diagnostic view for an explanation.")
			imgui.separator_text("Rendering")
			self.draw_general_rendering_options(imgui)
			self.draw_ambient_occlusion_options(imgui)
			imgui.separator_text("Transparency")
			self.draw_transparency_options(imgui)
			self.draw_banded_tail_options(imgui)
		finally:
			imgui.end()
			self.sync_debug_view(self.renderer)


FilterSettings = RenderingSettings
