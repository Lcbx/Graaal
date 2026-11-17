"""Stochastic OIT with frustum and Hi-Z culling via DrawBatches.

Run: python main.py scenes/transparency_example.py [--frames 120]
Space toggles orbit, scrolling zooms, R reverses transparent submission order.
1/3/5/9 select 0/1/2/3 Kawase pairs.
D cycles diagnostic views; [ / ] adjusts screen noise scale.
G toggles grazing-angle opacity fade (angles are adjustable in settings).
M toggles 4x MSAA; A toggles SSAO; O selects SSAO in the compose pass or a separate pass.
F shows/hides ImGui settings.
P shows/hides the performance profiler.
FPS and average frame time remain visible in the top-right corner.
Drag settings or Ctrl+click to type; Left/Right changes Kawase blur pairs.
Imports only prepare CPU helpers; graphics are initialized by run_scene().
"""

from dataclasses import replace
from math import cos, sin
from pathlib import Path

import numpy as np

from graaal.Assets import DEFAULT_BLUE_NOISE_PATH, load_blue_noise
from graaal.AmbientOcclusionPass import AmbientOcclusionPass
from graaal.AntialiasingPass import AntialiasingPass
from graaal.DrawBatches import DrawBatches, FrustumCuller, HiZCuller, ShaderPass
from graaal.HZB import HZB
from graaal.InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_scale
from graaal.Math import Camera, Vec3
from graaal.RenderContext import (
    BufferUsage,
    GpuBuffer,
    Mesh,
    RenderContext,
    RenderPipeline,
    Texture,
    TextureUsage,
)
from graaal.TextureProcessing import build_material_texture_array
from graaal.TransparencyData import load_transparency_scene as load_transparency_data
from graaal.transparency_utils import record_transparency
from graaal.TransparencyPass import GrazingFade, TransparencyPass
from scenes.example_overlay import ExampleOverlay
from graaal.render_resolution import ResolutionPresenter
from graaal.rendering_settings import FilterSettings, RenderingSettings


PATH = Path(__file__).parent / "resources/small_scene.glb"
BLUE_NOISE_PATH = DEFAULT_BLUE_NOISE_PATH
TEXTURE_SIZE = 512


def load_transparency_scene(path):
	# This asset authors binary tree leaves as BLEND. Use MASK cutoff/mips for
	# this named foliage material while keeping genuinely soft cards as BLEND.
	return load_transparency_data(path, mask_materials=("Simple_foliage.001",))


class GeometryBatches:
	"""Static glTF primitives submitted directly through DrawBatches."""
	def __init__(self, primitives, materials, transparency:TransparencyPass, transparent:bool):
		primitives = [p for p in primitives if (materials[p.material_id]["surface"][2] == 2) == transparent]
		self.data = DrawBatches(MESH_INSTANCE_DTYPE)
		self.metadata = GpuBuffer(np.zeros((1, 4), dtype=np.uint32), BufferUsage.STORAGE | BufferUsage.COPY_DST)
		self.passes = transparency.create_geometry_passes(self.metadata, transparent=transparent, instances=self.data.buffers["instances"])
		if not transparent:
			self.depth_pass = ShaderPass(RenderPipeline(transparency.depth_shader, vertex_entry="vertex",
				fragment_entry="opaque_prepass", cull_mode="none", label="opaque_depth"),
				self.passes.render_shader["prepass"].bindings)
		self.data.register_shader(0, self.passes.render_shader)
		for primitive in primitives:
			mesh_id = primitive.stable_id
			mesh = Mesh(primitive.vertices, np.arange(len(primitive.vertices), dtype=np.uint32))
			self.data.register_mesh(mesh_id, mesh)
			self.data.register_lod_group(mesh_id, (mesh_id,))
		mesh_ids = np.array([p.stable_id for p in primitives], dtype=np.uint32)
		self.data.sync(mesh_ids, np.zeros(len(mesh_ids), dtype=np.uint32))
		self.visibility = self.data.create_visibility()
		self.data.write_instance_fields(iPosition=(0, 0, 0), iTint=0,
			iRotation=pack_quaternion([0, 0, 0, 1]), iScale=pack_scale([1, 1, 1]))
		# Metadata follows packed instance order, which may differ from input order.
		metadata = np.zeros((max(1, len(primitives)), 4), dtype=np.uint32)
		for row, source in enumerate(self.data.instance_order):
			p = primitives[source]
			metadata[row] = (p.triangle_offset, p.material_id, p.stable_id, 0)
		self.metadata.write(metadata)

	def prepare(self, commands, camera, view_projection):
		FrustumCuller().record(commands, self.visibility, camera_position=camera.position,
			view_projection=view_projection, fovy_deg=camera.fovy_deg)

	def draw(self, render, stage, **options):
		self.data.draw(render, self.visibility, stage, **options)


transparency: TransparencyPass
ao_pass: AmbientOcclusionPass
aa_pass: AntialiasingPass
opaque_batches: GeometryBatches
transparent_batches: GeometryBatches
camera: Camera
hzb: HZB
center: np.ndarray
radius: float
angle = 0.65
orbit = False
distance = 0.0
msaa_enabled = False
ao_enabled = False
ao_method = "compose"
overlay = None


def init_scene(path=PATH, *, blue_noise_path=BLUE_NOISE_PATH, **transparency_options):
	"""Initialize the example's global render data after device setup."""
	global transparency, ao_pass, aa_pass, opaque_batches, transparent_batches, camera, hzb, center, radius, angle, orbit, distance, msaa_enabled, ao_enabled, ao_method
	primitives, materials, images, vertices = load_transparency_scene(path)
	# Ignore a large flat ground plane when framing the example.
	objects = [p.vertices["position"] for p in primitives if np.ptp(p.vertices["position"][:, 1]) > 0.01]
	positions = np.concatenate(objects or [p.vertices["position"] for p in primitives])
	center = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
	radius = max(float(np.linalg.norm(np.ptp(positions, axis=0))) * 0.8, 1.0)
	textures = build_material_texture_array(images, materials, (TEXTURE_SIZE, TEXTURE_SIZE))
	blue_noise = None
	if blue_noise_path is not None and Path(blue_noise_path) != DEFAULT_BLUE_NOISE_PATH:
		noise = load_blue_noise(blue_noise_path)
		noise_size = (noise.shape[1], noise.shape[0])
		blue_noise = Texture(noise_size, format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
		RenderContext.device.queue.write_texture({"texture": blue_noise.handle}, noise,
			{"bytes_per_row": noise.shape[1]*4}, (*noise_size, 1))
	transparency = TransparencyPass(
		materials=GpuBuffer(materials, BufferUsage.STORAGE | BufferUsage.COPY_DST),
		base_textures=textures.view(dimension="2d-array"),
		vertices=GpuBuffer(vertices, BufferUsage.STORAGE | BufferUsage.COPY_DST), blue_noise=blue_noise,
		**transparency_options)
	ao_pass = AmbientOcclusionPass(target_format=transparency.target_format)
	aa_pass = AntialiasingPass(target_format=transparency.target_format)
	msaa_enabled = transparency.sample_count == 4
	ao_enabled, ao_method = False, "compose"
	hzb = HZB()
	opaque_batches = GeometryBatches(primitives, materials, transparency, False)
	transparent_batches = GeometryBatches(primitives, materials, transparency, True)
	camera = Camera(tuple(center + (radius, radius*0.65, radius)), tuple(center), (0, 1, 0), 55,
		near=max(0.01, radius/1000), far=radius*100)
	angle, orbit, distance = 0.65, False, radius*1.4
	transparency.set_ambient_occlusion(False)


def set_msaa_enabled(enabled):
	global msaa_enabled
	if type(enabled) is not bool: raise ValueError("MSAA enabled must be boolean")
	transparency.set_sample_count(4 if enabled else 1)
	msaa_enabled = enabled


def set_ao_enabled(enabled):
	global ao_enabled
	if type(enabled) is not bool: raise ValueError("AO enabled must be boolean")
	ao_enabled = enabled
	transparency.set_ambient_occlusion(enabled and ao_method == "compose")


def set_ao_method(method):
	global ao_method
	if method not in ("compose", "pass"):
		raise ValueError("AO method must be compose or pass")
	ao_method = method
	transparency.set_ambient_occlusion(ao_enabled and method == "compose")


def scroll_callback(x, y):
	global distance
	distance = np.clip(distance * np.exp(-0.1*y), radius*0.15, radius*8)
	return True


def on_char(code):
	global orbit
	if overlay is not None and overlay.handle_char(code): return True
	if chr(code) == " ":
		orbit = not orbit
	elif chr(code).lower() == "r":
		transparent_batches.data.draw_batches.reverse()
	elif chr(code) in ("1", "3", "5", "9"):
		transparency.set_kawase_passes(("1", "3", "5", "9").index(chr(code)))
	elif chr(code).lower() == "d":
		views = RenderingSettings.sync_debug_view(transparency)
		index = (views.index(transparency.debug_view)+1) % len(views)
		transparency.set_debug_view(views[index])
		print(f"View: {RenderingSettings.debug_view_label(transparency, transparency.debug_view)}")
	elif chr(code) in ("[", "]"):
		factor = 2**(-0.25 if chr(code) == "[" else 0.25)
		transparency.set_noise_scale(max(0.25, min(4.0, transparency.noise_scale*factor)))
		print(f"Noise cell size: {transparency.noise_scale:.3g} px")
	elif chr(code).lower() == "g":
		options = transparency.grazing_fade
		transparency.set_grazing_fade(replace(options, enabled=not options.enabled))
		print(f"Grazing fade: {not options.enabled}, {options.start_angle:g} to {options.end_angle:g} degrees")
	return True


def render_frame(*, profiler=None, overlay=None, render_size=None, presenter=None):
	"""Record frustum culling, opaque depth, Hi-Z culling, and transparency."""
	window_size = RenderContext.window_dimensions
	if min(window_size) <= 0:
		return False
	width, height = render_size if render_size is not None else window_size
	scaled = (width, height) != window_size
	if scaled and presenter is None: raise ValueError("Scaled rendering requires a presenter")
	transparency.resize((width, height))
	view_projection = transparency.update_camera(camera)
	commands = overlay.commands("stochastic_renderer") if overlay is not None else RenderContext.commands("stochastic_renderer")
	if profiler is not None: commands = profiler.wrap(commands)
	for geometry in (opaque_batches, transparent_batches):
		geometry.prepare(commands, camera, view_projection)
	if transparency.sample_count == 1:
		hzb.resize((width, height))
		# Stochastic cutout surfaces cannot contribute conservative occluder depth.
		with commands.render_pass(color=(), depth=hzb.depth_texture.depth_attachment(clear=1.0),
			label="depth_prepass") as render:
			opaque_batches.draw(render, "prepass", pass_override=opaque_batches.depth_pass)
		hzb.build(commands)
		for geometry in (opaque_batches, transparent_batches):
			HiZCuller().record(commands, geometry.visibility, hzb_view=hzb.view)
	# Pixel-center depth cannot conservatively occlude all MSAA samples. At 4x,
	# draw both lists directly from the frustum visibility prepared above.
	with transparency.opaque_pass(commands) as render:
		opaque_batches.draw(render, "main")
	def draw_transparent(render, name):
		if name in ("nearest", "guide"):
			transparent_batches.draw(render, "main", pass_override=getattr(transparent_batches.passes, name))
		else:
			transparent_batches.draw(render, name)
	record_transparency(commands, transparency, transparent_batches.passes, draw_transparent)
	screen = RenderContext.screen()
	if scaled: presenter.resize((width, height))
	output = presenter.color_attachment() if scaled else screen
	if aa_pass.mode in ("lxaa", "faaa", "ddaa2"):
		aa_pass.resize((width, height))
		output = aa_pass.color_attachment()
	aa_output = output
	if ao_enabled and ao_method == "pass":
		ao_pass.resize((width, height))
		output = ao_pass.color_attachment()
	transparency.compose(commands, output)
	if ao_enabled and ao_method == "pass":
		ao_pass.update_camera(camera, sample_count=transparency.sample_count)
		ao_pass.record(commands, transparency.depth, aa_output)
	if aa_pass.mode in ("lxaa", "faaa", "ddaa2"):
		aa_pass.record(commands, presenter.color_attachment() if scaled else screen, aa_pass.mode)
	if scaled: presenter.record(commands, screen)
	if overlay is not None: overlay.submit(commands)
	else: RenderContext.submit(commands.finish())
	return True


def run_scene():
	global settings, overlay, angle
	from graaal.ImGui import ImGui

	RenderContext.init_window(1200, 900, "Stochastic transparency | M: MSAA | A: SSAO | O: SSAO path | F: settings | P: profiler | D: debug | Space: orbit",
		required_gpu_features=["indirect-first-instance"], optional_gpu_features=["timestamp-query"])
	init_scene()
	presenter = ResolutionPresenter(target_format=transparency.target_format)
	settings = RenderingSettings(transparency, camera=camera, on_msaa_changed=set_msaa_enabled,
		on_ssao_changed=set_ao_enabled, ssao_enabled=ao_enabled, ao_pass=ao_pass, aa_pass=aa_pass)
	overlay = ExampleOverlay(settings, get_ao_method=lambda: ao_method, set_ao_method=set_ao_method)
	RenderContext.event_handlers["mouse_scroll"].append(scroll_callback)
	RenderContext.event_handlers["char"].append(on_char)
	RenderContext.event_handlers["key"].append(overlay.handle_key)
	try:
		with ImGui() as gui:
			while RenderContext.window_loop():
				if orbit:
					angle += RenderContext.frame_time * 0.2
				camera.position = Vec3(center + distance * np.array([cos(angle), 0.5, sin(angle)]))
				if render_frame(overlay=overlay, render_size=settings.render_size(RenderContext.window_dimensions), presenter=presenter):
					gui.render(overlay.draw)
	finally:
		RenderContext.event_handlers["mouse_scroll"].remove(scroll_callback)
		RenderContext.event_handlers["char"].remove(on_char)
		RenderContext.event_handlers["key"].remove(overlay.handle_key)

if __name__ == "__main__":
	run_scene()
