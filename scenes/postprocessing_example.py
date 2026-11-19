"""Drive through a cube city. Space toggles peripheral motion blur."""
from math import cos, exp, pi, sin
import math
from pathlib import Path

import glfw
import numpy as np

from graaal.DrawBatches import DrawBatches, standard_render_shader
from graaal.Geometry import make_cube_mesh
from graaal.ImGui import ImGui
from graaal.InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_rgba8_srgb, pack_scale
from graaal.Math import Camera, Vec3
from graaal.PostprocessingPass import PostprocessingEffect, PostprocessingPass
from graaal.RenderContext import RenderContext, Shader, Texture, TextureUsage


class PeripheralMotionBlurEffect(PostprocessingEffect):
	label = "peripheral_motion_blur"
	def shader_function(self, index): return "peripheral_motion_blur"
	def shader_paths(self): return (Path(__file__).with_name("shaders"),)
	def shader_fragment(self, index):
		return '#include "peripheral_motion_blur.shaderlib"\n', {}

	def uniform_values(self, index): return {"motion": self.values}

	def __init__(self, *, linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0), fovy_deg=60, exposure=1/60, reference_distance=10.0, strength=1.0, inner_radius=0.25, outer_radius=0.85, max_blur=0.08, debug=False, minimum_velocity=0.0, blur_exponent=1.0):
		self.set_motion(linear_velocity=linear_velocity, angular_velocity=angular_velocity, fovy_deg=fovy_deg, exposure=exposure, reference_distance=reference_distance, strength=strength, inner_radius=inner_radius, outer_radius=outer_radius, max_blur=max_blur, debug=debug, minimum_velocity=minimum_velocity, blur_exponent=blur_exponent)

	def set_motion(self, *, linear_velocity, angular_velocity, fovy_deg, exposure=1/60, reference_distance=10.0, strength=1.0, inner_radius=0.25, outer_radius=0.85, max_blur=0.08, debug=False, minimum_velocity=0.0, blur_exponent=1.0):
		linear, angular = (np.asarray(v, dtype=np.float32) for v in (linear_velocity, angular_velocity))
		if linear.shape != (3,) or angular.shape != (3,) or not np.all(np.isfinite(linear)) or not np.all(np.isfinite(angular)): raise ValueError("Expected finite three-component velocities")
		values = (fovy_deg, exposure, reference_distance, strength, inner_radius, outer_radius, max_blur, minimum_velocity, blur_exponent)
		if not all(math.isfinite(v) for v in values) or not (0 < fovy_deg < 180 and exposure >= 0 and reference_distance > 0 and strength >= 0 and 0 <= inner_radius < outer_radius and max_blur >= 0 and minimum_velocity >= 0 and blur_exponent > 0): raise ValueError("Invalid motion blur parameters")
		speed = np.linalg.norm(linear.astype(np.float64))
		if speed > 0: linear = linear * (max(0.0, speed-minimum_velocity)/speed)
		self.values = {"linear": (*linear, 0), "angular": (*angular, 0), "params": (math.tan(math.radians(fovy_deg)/2), exposure*strength, reference_distance, max_blur), "radii": (inner_radius, outer_radius, float(debug), blur_exponent)}



DEFAULT_BLUR_ENABLED = True
DEFAULT_BLUR_DEBUG = False
DEFAULT_MAX_STRENGTH = 2.5
DEFAULT_INNER_RADIUS = 0.1
DEFAULT_OUTER_RADIUS = 0.8
DEFAULT_BLUR_EXPONENT = 0.5
DEFAULT_ACCELERATION_GAIN = 0.1
DEFAULT_RECOVERY_TIME = 0.4
DEFAULT_MINIMUM_VELOCITY = 5.0
BLUR_EXPOSURE = 1/30
BLUR_REFERENCE_DISTANCE = 8
BLUR_MAX_LENGTH = 0.3


def update_acceleration_boost(filtered_acceleration, previous_speed, speed, dt, response_time):
	"""Smooth increases in speed while letting the boost decay during braking."""
	if dt <= 0: return filtered_acceleration
	increase = abs(speed) if speed*previous_speed < 0 else max(0.0, abs(speed)-abs(previous_speed))
	acceleration = increase/dt
	decay = exp(-dt/response_time)
	return filtered_acceleration*decay + acceleration*(1.0-decay)


def run_scene():
	RenderContext.init_window(1200, 720, "Drive: W/S or Up/Down | Look: A/D or Left/Right | Space: blur", required_gpu_features=["indirect-first-instance"])
	depth = gui = None
	try:
		shader = Shader(filepath=str(Path(__file__).parent / "shaders" / "complex.shader"))
		uniforms = shader.uniform_buffer()
		draws = DrawBatches(MESH_INSTANCE_DTYPE)
		draws.register_shader(0, standard_render_shader(shader, uniforms))
		draws.register_mesh(0, make_cube_mesh())
		draws.register_lod_group(0, (0,))
		rng = np.random.default_rng(4)
		positions = []
		scales = []
		colors = []
		for z in range(-490, 10, 10):
			for side in (-1, 1):
				height = float(rng.uniform(8, 32))
				positions.append((side * float(rng.uniform(11, 16)), height/2, z))
				scales.append((float(rng.uniform(6, 10)), height, float(rng.uniform(6, 9))))
				colors.append((*rng.uniform(0.25, 0.65, 3), 1))
			positions.append((0, 0.02, z))
			scales.append((0.18, 0.04, 4))
			colors.append((0.9, 0.8, 0.45, 1))
		# Three copies keep the visible street identical at either wrap point.
		base_positions = np.asarray(positions, dtype=np.float32)
		base_scales, base_colors = scales[:], colors[:]
		positions, scales, colors = [], [], []
		for offset in (-500, 0, 500):
			positions.append((0, -0.5, -250+offset))
			scales.append((12, 1, 500))
			colors.append((0.13, 0.15, 0.18, 1))
			positions.extend(base_positions + (0, 0, offset))
			scales.extend(base_scales)
			colors.extend(base_colors)
		n = len(positions)
		instances = np.zeros(n, dtype=MESH_INSTANCE_DTYPE)
		instances["iPosition"] = positions
		instances["iRotation"] = pack_quaternion(np.tile((0, 0, 0, 1), (n, 1)))
		instances["iScale"] = pack_scale(scales)
		instances["iTint"][:, 0] = [pack_rgba8_srgb(color) for color in colors]
		draws.sync(np.zeros(n, dtype=np.uint32), np.zeros(n, dtype=np.uint32))
		draws.write_instances(instances[draws.instance_order])
		visibility = draws.create_visibility().all_visible()
		motion = PeripheralMotionBlurEffect()
		blur = PostprocessingPass([motion])
		camera = Camera((0, 2, 0), (0, 2, -1), (0, 1, 0), 65, near=0.1, far=250)
		enabled, debug = DEFAULT_BLUR_ENABLED, DEFAULT_BLUR_DEBUG
		max_strength, inner_radius, outer_radius, blur_exponent = DEFAULT_MAX_STRENGTH, DEFAULT_INNER_RADIUS, DEFAULT_OUTER_RADIUS, DEFAULT_BLUR_EXPONENT
		acceleration_gain, recovery_time, minimum_velocity = DEFAULT_ACCELERATION_GAIN, DEFAULT_RECOVERY_TIME, DEFAULT_MINIMUM_VELOCITY
		velocity = 0.0
		gui = ImGui()
		def draw_controls():
			nonlocal enabled, debug, max_strength, inner_radius, outer_radius, blur_exponent, acceleration_gain, recovery_time, minimum_velocity
			imgui = gui.imgui
			imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
			imgui.set_next_window_size((370, 430), imgui.Cond_.first_use_ever)
			imgui.begin("Motion blur")
			try:
				_, enabled = imgui.checkbox("Enabled (Space)", enabled)
				_, debug = imgui.checkbox("Show blur direction and strength", debug)
				if debug: imgui.text_wrapped("Hue: right red, down yellow-green, left cyan, up violet. Color mix shows blur strength.")
				imgui.separator_text("Motion")
				imgui.progress_bar(min(abs(velocity)/55.0, 1.0), (-1, 22), f"Speed: {abs(velocity):.1f} units/s ({'reverse' if velocity < 0 else 'forward'})")
				_, minimum_velocity = imgui.slider_float("Minimum speed (units/s)", minimum_velocity, 0.0, 55.0, "%.1f")
				_, max_strength = imgui.slider_float("Max strength", max_strength, 0.0, 6.0, "%.2f")
				imgui.text_wrapped("Baseline is 25% of maximum; acceleration raises it toward maximum.")
				_, acceleration_gain = imgui.slider_float("Acceleration boost", acceleration_gain, 0.0, 1.0, "%.3f")
				_, recovery_time = imgui.slider_float("Boost decay (seconds)", recovery_time, 0.1, 4.0, "%.1f")
				imgui.separator_text("Vignette")
				_, inner_radius = imgui.slider_float("Sharp center radius", inner_radius, 0.0, outer_radius-0.01, "%.2f")
				_, outer_radius = imgui.slider_float("Max blur radius", outer_radius, inner_radius+0.01, 1.0, "%.2f")
				_, blur_exponent = imgui.slider_float("Blur exponent", blur_exponent, 0.01, 4.0, "%.2f")
				imgui.text_wrapped("W/Up: forward  S/Down: reverse  A/D or Left/Right: look")
			finally: imgui.end()
		def on_char(code):
			nonlocal enabled
			if chr(code) != " ": return False
			enabled = not enabled
			return True
		RenderContext.subscribe_event("char", on_char)
		speed = distance = yaw = filtered_acceleration = 0.0
		def pressed(*keys):
			return not gui.io.want_capture_keyboard and any(glfw.get_key(RenderContext.window, key) == glfw.PRESS for key in keys)
		while RenderContext.window_loop():
			size = RenderContext.window_dimensions
			if min(size) <= 0: continue
			if blur.resize(size):
				if depth is not None: depth.close()
				depth = Texture(size, format="depth32float", usage=TextureUsage.RENDER_ATTACHMENT)
			dt = min(RenderContext.frame_time, 0.1)
			accelerate = pressed(glfw.KEY_W, glfw.KEY_UP)
			brake = pressed(glfw.KEY_S, glfw.KEY_DOWN)
			turn = int(pressed(glfw.KEY_D, glfw.KEY_RIGHT)) - int(pressed(glfw.KEY_A, glfw.KEY_LEFT))
			previous_speed, previous_yaw = speed, yaw
			throttle = int(accelerate) - int(brake)
			if throttle:
				speed = max(-30.0, min(55.0, speed + 20.0*throttle*dt))
			elif speed > 0:
				speed = max(0.0, speed-3.0*dt)
			else:
				speed = min(0.0, speed+3.0*dt)
			distance_step = (previous_speed+speed)*0.5*dt
			distance = (distance + distance_step) % 500.0
			z = -distance
			yaw = max(-pi/2, min(pi/2, yaw + turn*1.5*dt))
			camera.position = Vec3(0, 2, z)
			camera.target = Vec3(sin(yaw), 2, z-cos(yaw))
			uniforms.content["view"] = camera.view()
			uniforms.content["proj"] = camera.projection(size[0]/size[1])
			uniforms.content["light_dir"] = (0.4, 0.8, 0.3, 0)
			uniforms.upload()
			commands = RenderContext.commands("postprocessing")
			with commands.render_pass(color=blur.color_attachment(clear=(0.3, 0.48, 0.65, 1)), depth=depth.depth_attachment(clear=1.0), label="city") as render:
				draws.draw(render, visibility)
			elapsed = RenderContext.frame_time
			filtered_acceleration = update_acceleration_boost(filtered_acceleration, previous_speed, speed, elapsed, recovery_time)
			velocity = distance_step/elapsed if elapsed > 0 else 0.0
			turn_rate = (yaw-previous_yaw)/elapsed if elapsed > 0 else 0.0
			linear = (-velocity*sin(yaw), 0, -velocity*cos(yaw)) if enabled else (0, 0, 0)
			angular = (0, -turn_rate, 0) if enabled else (0, 0, 0)
			motion.set_motion(linear_velocity=linear, angular_velocity=angular, fovy_deg=camera.fovy_deg, exposure=BLUR_EXPOSURE, reference_distance=BLUR_REFERENCE_DISTANCE, strength=max_strength*(1.0-0.75*exp(-acceleration_gain*filtered_acceleration)), inner_radius=inner_radius, outer_radius=outer_radius, blur_exponent=blur_exponent, max_blur=BLUR_MAX_LENGTH, debug=debug, minimum_velocity=minimum_velocity)
			blur.record(commands, RenderContext.screen())
			RenderContext.submit(commands.finish())
			gui.render(draw_controls)
	finally:
		if gui is not None: gui.close()
		if depth is not None: depth.close()
		RenderContext.cleanup()


if __name__ == "__main__":
	run_scene()

