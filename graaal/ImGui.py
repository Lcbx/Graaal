"""Dear ImGui plugin for RenderContext's GLFW window and WGPU surface.

Uses WGPU's ImguiWgpuBackend for draw lists, textures, clipping and blending.
Create after init_window(), render after the scene, and close before cleanup().
"""

from contextlib import contextmanager
from enum import Enum
import json

import glfw
from pathlib import Path

from .RenderContext import RenderContext

class ImGui:
	def __init__(self, context=RenderContext):
		if context.window is None or context.canvas is None:
			raise RuntimeError("Initialize a RenderContext window before creating ImGui")
		from imgui_bundle import imgui, hello_imgui
		from wgpu.utils.imgui import ImguiWgpuBackend

		self.imgui = imgui
		self._hello_imgui = hello_imgui
		self.context = context
		self._handlers = {}
		self._backend = None
		previous = imgui.get_current_context()
		self._imgui_context = imgui.create_context()
		self.font = None
		self._show_theme_editor = False
		self._style_path = "imgui_style.json"
		self._style_message = ""

		try:
			imgui.set_current_context(self._imgui_context)
			self._backend = ImguiWgpuBackend(context.device, context.presentation_format)

			self.set_font(Path(__file__).parent / "data/GlacialIndifference-Regular.otf")
			self.set_theme("black_is_black")
			
			self._keys = self._key_map()
			self._update_display()
			self.io.add_mouse_pos_event(*glfw.get_cursor_pos(context.window))
			self._handlers = {
				"key": self._on_key, "char": self._on_char,
				"mouse_click": self._on_mouse_button, "mouse_move": self._on_mouse_move,
				"mouse_scroll": self._on_scroll, "mouse_enter": self._on_mouse_enter,
				"focus": self._on_focus,
			}
			for channel, handler in self._handlers.items():
				context.event_handlers.setdefault(channel, []).insert(0, handler)
		except BaseException:
			self.close()
			raise
		finally:
			if previous is not None:
				imgui.set_current_context(previous)

	def set_theme(self, name: str):
		with self._activate():
			self._tweaked_theme = self._hello_imgui.ImGuiTweakedTheme()
			self._tweaked_theme.theme = self._hello_imgui.ImGuiTheme_[name]
			self._apply_theme()

	def _apply_theme(self):
		self._hello_imgui.apply_tweaked_theme(self._tweaked_theme)

	def save_style(self, path: Path):
		"""Save the active ImGuiStyle, including manual color edits, as JSON."""
		with self._activate():
			style = self.imgui.get_style()
			fields = {}
			for name in dir(style):
				if name.startswith("_"):
					continue
				value = getattr(style, name)
				if isinstance(value, Enum):
					fields[name] = int(value)
				elif isinstance(value, (bool, int, float)):
					fields[name] = value
				elif isinstance(value, self.imgui.ImVec2):
					fields[name] = [value.x, value.y]
			colors = {}
			for idx in range(int(self.imgui.Col_.count)):
				color = style.color_(idx)
				colors[str(idx)] = [color.x, color.y, color.z, color.w]
		Path(path).write_text(json.dumps({"fields": fields, "colors": colors}, indent=2) + "\n", encoding="utf-8")

	def load_style(self, path: Path):
		"""Restore an exported ImGuiStyle to this ImGui context."""
		data = json.loads(Path(path).read_text(encoding="utf-8"))
		with self._activate():
			style = self.imgui.get_style()
			for name, value in data["fields"].items():
				current = getattr(style, name)
				if isinstance(current, Enum):
					value = type(current)(value)
				elif isinstance(current, self.imgui.ImVec2):
					value = self.imgui.ImVec2(*value)
				setattr(style, name, value)
			for idx, color in data["colors"].items():
				style.set_color_(int(idx), color)

	def _draw_theme_editor(self):
		if not self._show_theme_editor:
			return
		visible, self._show_theme_editor = self.imgui.begin("Theme Editor", self._show_theme_editor)
		try:
			if visible:
				if self._hello_imgui.show_theme_tweak_gui(self._tweaked_theme):
					self._apply_theme()
				self.imgui.separator()
				_, self._style_path = self.imgui.input_text("Style path", self._style_path)
				if self.imgui.button("Save style"):
					try:
						self.save_style(Path(self._style_path))
						self._style_message = f"Saved {self._style_path}"
					except OSError as e:
						self._style_message = str(e)
				self.imgui.same_line()
				if self.imgui.button("Load style"):
					try:
						self.load_style(Path(self._style_path))
						self._style_message = f"Loaded {self._style_path}"
					except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
						self._style_message = str(e)
				self.imgui.text(self._style_message)
		finally:
			self.imgui.end()

	def set_font(self,font_path:Path):
		if font_path.exists():
			try:
				self.io.fonts.clear()
				self.font = self.io.fonts.add_font_from_file_ttf(str(font_path), 18.0)
			except Exception as e:
				print(f"Error reading font file: {e}")
		else:
			print(f"font not found : {font_path}")

	@property
	def io(self):
		return self._backend.io

	def _key_map(self):
		key = self.imgui.Key
		names = {
			"TAB": "tab", "LEFT": "left_arrow", "RIGHT": "right_arrow",
			"UP": "up_arrow", "DOWN": "down_arrow", "PAGE_UP": "page_up",
			"PAGE_DOWN": "page_down", "HOME": "home", "END": "end",
			"INSERT": "insert", "DELETE": "delete", "BACKSPACE": "backspace",
			"SPACE": "space", "ENTER": "enter", "ESCAPE": "escape",
			"APOSTROPHE": "apostrophe", "COMMA": "comma", "MINUS": "minus",
			"PERIOD": "period", "SLASH": "slash", "SEMICOLON": "semicolon",
			"EQUAL": "equal", "LEFT_BRACKET": "left_bracket",
			"BACKSLASH": "backslash", "RIGHT_BRACKET": "right_bracket",
			"GRAVE_ACCENT": "grave_accent", "CAPS_LOCK": "caps_lock",
			"SCROLL_LOCK": "scroll_lock", "NUM_LOCK": "num_lock",
			"PRINT_SCREEN": "print_screen", "PAUSE": "pause",
			"LEFT_SHIFT": "left_shift", "RIGHT_SHIFT": "right_shift",
			"LEFT_CONTROL": "left_ctrl", "RIGHT_CONTROL": "right_ctrl",
			"LEFT_ALT": "left_alt", "RIGHT_ALT": "right_alt",
			"LEFT_SUPER": "left_super", "RIGHT_SUPER": "right_super", "MENU": "menu",
			"KP_DECIMAL": "keypad_decimal", "KP_DIVIDE": "keypad_divide",
			"KP_MULTIPLY": "keypad_multiply", "KP_SUBTRACT": "keypad_subtract",
			"KP_ADD": "keypad_add", "KP_ENTER": "keypad_enter", "KP_EQUAL": "keypad_equal",
		}
		names.update({chr(i): chr(i).lower() for i in range(ord("A"), ord("Z")+1)})
		names.update({str(i): f"_{i}" for i in range(10)})
		names.update({f"KP_{i}": f"keypad{i}" for i in range(10)})
		names.update({f"F{i}": f"f{i}" for i in range(1, 25)})
		return {getattr(glfw, f"KEY_{name}"): getattr(key, target) for name, target in names.items()}

	@contextmanager
	def _activate(self):
		if self._imgui_context is None:
			raise RuntimeError("ImGui is closed")
		previous = self.imgui.get_current_context()
		self.imgui.set_current_context(self._imgui_context)
		try:
			yield
		finally:
			if previous is not None:
				self.imgui.set_current_context(previous)

	def _update_display(self):
		width, height = glfw.get_window_size(self.context.window)
		pixels_w, pixels_h = glfw.get_framebuffer_size(self.context.window)
		self.io.display_size = (width, height)
		self.io.display_framebuffer_scale = (pixels_w/max(width, 1), pixels_h/max(height, 1))
		self.io.delta_time = max(self.context.frame_time, 1e-6)
		return width > 0 and height > 0 and pixels_w > 0 and pixels_h > 0

	def render(self, draw, *, clear=None):
		"""Build widgets with draw() and overlay them on this frame's screen view.

		Pass a clear color for UI-only scenes. Presentation remains owned by the
		RenderContext loop; no extra canvas or swapchain texture is acquired.
		"""
		with self._activate():
			if not self._update_display():
				return
			color = self.context.screen(clear=clear)
			# A resize can arrive after the scene acquired this frame's surface.
			# Derive clipping/viewport scale from that exact view, not live GLFW size.
			pixels_w, pixels_h = color.view.size[:2]
			width, height = self.io.display_size
			self.io.display_framebuffer_scale = (pixels_w/width, pixels_h/height)
			self.imgui.new_frame()
			try:
				draw()
				self._draw_theme_editor()
			finally:
				self.imgui.render()
			commands = self.context.commands("imgui")
			with commands.render_pass(color=color, label="imgui") as render:
				self._backend.render(self.imgui.get_draw_data(), render.handle)
			self.context.submit(commands.finish())

	def _on_key(self, key, scancode, action, mods):
		with self._activate():
			if key == glfw.KEY_F10:
				if action == glfw.PRESS:
					self._show_theme_editor = not self._show_theme_editor
				return True
			for flag, name, left, right in (
				(glfw.MOD_CONTROL, "mod_ctrl", glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL),
				(glfw.MOD_SHIFT, "mod_shift", glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT),
				(glfw.MOD_ALT, "mod_alt", glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT),
				(glfw.MOD_SUPER, "mod_super", glfw.KEY_LEFT_SUPER, glfw.KEY_RIGHT_SUPER),
			):
				pressed = bool(mods & flag)
				if key in (left, right):
					other = right if key == left else left
					pressed = action != glfw.RELEASE or glfw.get_key(self.context.window, other) == glfw.PRESS
				self.io.add_key_event(getattr(self.imgui.Key, name), pressed)
			if key in self._keys and action != glfw.REPEAT:
				self.io.add_key_event(self._keys[key], action == glfw.PRESS)
			return self.io.want_capture_keyboard

	def _on_char(self, codepoint):
		with self._activate():
			self.io.add_input_character(codepoint)
			return self.io.want_capture_keyboard or self.io.want_text_input

	def _on_mouse_move(self, x, y):
		with self._activate():
			self.io.add_mouse_pos_event(x, y)
			return self.io.want_capture_mouse

	def _on_mouse_button(self, button, action, mods):
		with self._activate():
			self.io.add_mouse_pos_event(*glfw.get_cursor_pos(self.context.window))
			if 0 <= button < 5:
				self.io.add_mouse_button_event(button, action == glfw.PRESS)
			return self.io.want_capture_mouse

	def _on_scroll(self, x, y):
		with self._activate():
			self.io.add_mouse_wheel_event(x, y)
			return self.io.want_capture_mouse

	def _on_mouse_enter(self, entered):
		with self._activate():
			self.io.add_mouse_pos_event(*(glfw.get_cursor_pos(self.context.window) if entered else (-3.4e38, -3.4e38)))
			return self.io.want_capture_mouse

	def _on_focus(self, focused):
		with self._activate():
			self.io.add_focus_event(bool(focused))

	def close(self):
		"""Detach only this plugin's handlers and release its context/resources."""
		if self._imgui_context is None:
			return
		for channel, handler in self._handlers.items():
			handlers = self.context.event_handlers.get(channel, [])
			if handler in handlers:
				handlers.remove(handler)
		previous = self.imgui.get_current_context()
		self.imgui.set_current_context(self._imgui_context)
		self.imgui.destroy_context(self._imgui_context)
		if previous is not None and previous != self._imgui_context:
			self.imgui.set_current_context(previous)
		self._imgui_context = None
		self._backend = None
		self._handlers.clear()

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		self.close()
