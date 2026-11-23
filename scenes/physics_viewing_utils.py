"""Shared window and playback helpers for the physics examples."""

from contextlib import contextmanager

import numpy as np


def add_viewing_options(parser):
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("--view", action="store_true", help="open a window with looping playback (default)")
	mode.add_argument("--headless", action="store_true", help="run the simulation and print its results")
	mode.add_argument("--offscreen", action="store_true", help="check rendered samples without opening a window")


@contextmanager
def viewing_target(title, window_size=(1000, 760), *, offscreen=False, offscreen_size=(640, 480)):
	"""Own the graphics session and yield a readback texture in offscreen mode."""
	from graaal.RenderContext import RenderContext, Texture, TextureUsage

	features = ["indirect-first-instance"]
	try:
		if offscreen:
			RenderContext.setup_graphics_backend(True, required_features=features)
			yield Texture(offscreen_size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
		else:
			RenderContext.init_window(*window_size, title, required_gpu_features=features)
			yield None
	finally:
		RenderContext.cleanup()


def draw_frame(renderer, camera, output, label):
	"""Record and submit a synced material renderer to the current target."""
	from graaal.RenderContext import RenderContext

	size = output.size[:2] if output is not None else RenderContext.window_dimensions
	if min(size) <= 0:
		return
	commands = RenderContext.commands(label)
	renderer.record(commands, size, camera, output.color_attachment() if output is not None else RenderContext.screen())
	RenderContext.submit(commands.finish())


def read_pixels(output):
	"""Read the offscreen target as a height × width array of RGBA bytes."""
	from graaal.RenderContext import RenderContext

	width, height, _ = output.size
	pixels = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": width * 4}, output.size)
	return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)


def has_blue_pixels(pixels):
	rgb = pixels.astype(np.int16)
	return np.any((rgb[..., 2] > rgb[..., 0] + 20) & (rgb[..., 2] > 80))


def play_frames(renderer, camera, output, update, *, label, sample_count, sample_hz, offscreen_samples=None, check_frame=None,
		before_frame=None, draw_overlay=None):
	"""Loop samples at their source rate, or draw selected samples offscreen.

	In a window, before_frame can return True to restart at sample zero.
	draw_overlay runs after the scene is submitted.
	"""
	from graaal.RenderContext import RenderContext

	last = sample_count - 1
	if output is not None:
		samples = (0, last) if last > 0 else (0,)
		if offscreen_samples is not None:
			samples = offscreen_samples
		for index in samples:
			update(index)
			renderer.sync()
			draw_frame(renderer, camera, output, label)
			if check_frame is not None and not check_frame(read_pixels(output)):
				raise RuntimeError(f"{label} sample {index} rendered no expected pixels")
		print(f"{label} rendered {len(samples)} offscreen frames")
		return

	import glfw

	cursor, paused = 0.0, False

	def key_pressed(key, scancode, action, mods):
		nonlocal cursor, paused
		if action == glfw.PRESS:
			if key == glfw.KEY_SPACE:
				paused = not paused
			elif key == glfw.KEY_R:
				cursor = 0.0

	RenderContext.subscribe_event("key", key_pressed)
	while RenderContext.window_loop():
		if before_frame is not None and before_frame():
			cursor = 0.0
		update(min(int(cursor), last))
		renderer.sync()
		draw_frame(renderer, camera, output, label)
		if draw_overlay is not None:
			draw_overlay()
		if not paused:
			# Hold the last pose for one second before restarting.
			cursor = (cursor + min(RenderContext.frame_time, 0.1) * sample_hz) % (last + sample_hz)


def view_poses(title, camera, samples, moving, *, sample_hz, static=(), offscreen=False, check_frame=None):
	"""Replay positions and orientations for a fixed list of direct draw instances."""
	from graaal.Materials.MaterialRenderer import MaterialRenderer

	def update(index):
		positions, orientations = samples[index]
		for instance, position, rotation in zip(moving, positions, orientations, strict=True):
			instance.position, instance.rotation = position, rotation

	with viewing_target(f"{title} | Space: pause, R: restart", offscreen=offscreen) as output:
		renderer = MaterialRenderer([*moving, *static])
		play_frames(renderer, camera, output, update, label=title, sample_count=len(samples), sample_hz=sample_hz, check_frame=check_frame)
