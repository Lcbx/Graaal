"""Dictionary materials, shared values, instance overrides and inherited shaders.

Run: python main.py scenes/material_example.py [--frames 120]
M toggles 4x MSAA; A toggles SSAO; O selects SSAO in the compose pass or a separate pass. Space toggles orbit; D replaces the opaque shader's default.
F shows/hides rendering settings; P shows/hides the performance profiler.
The bottom-left pair shares a dictionary; the transparent spheres share a checker
texture. The bottom-right sphere adds rim lighting through a Jinja child template.
No transparency code is copied.
The top-middle sphere uses time-driven displacement in the Jinja material below.
"""
from pathlib import Path
from math import cos, sin

import numpy as np

from graaal.Materials.DefaultMaterial import DefaultMaterialShader
from graaal.DrawBatches import FrustumCuller
from graaal.RenderPasses.PostprocessingPass import AmbientOcclusionEffect, AntialiasingEffect, PostprocessingPass
from graaal.ECS import ECS
from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.Materials.MaterialRenderer import MaterialMesh, DrawInstance, MaterialRenderer
from graaal.Math import Camera, Vec3
from graaal.RenderContext import RenderContext, Texture, TextureUsage
from graaal.RenderPasses.TransparencyData import VERTEX_DTYPE
from scenes.example_overlay import ExampleOverlay
from graaal.RenderPasses.render_resolution import ResolutionPresenter
from graaal.UI.rendering_settings import RenderingSettings
from graaal.RenderPasses.TransparencyPass import record_transparency


STRIPED_SHADER = '''
#extends "default_material.shader"

#block parameters
{{ super() }}
<MaterialParameters> stripe_color: vec3f = (1, 0.15, 0.03);
<MaterialParameters> stripe_frequency: f32 = 8.0;
<MaterialParameters> displacement: f32 = 0.18;
<MaterialParameters> speed: f32 = 2.0;
#endblock

#block vertex
	let radius = max(length(position), 1e-6);
	let direction = position/radius;
	let phase = direction.y*8.0-frame.time*p.speed;
	let displaced_radius = radius+p.displacement*sin(phase);
	let gradient = p.displacement*8.0*cos(phase)*(vec3f(0.0, 1.0, 0.0)-direction.y*direction);
	let displaced_normal = normalize(direction-gradient/max(displaced_radius, 1e-6));
	return MaterialVertex(direction*displaced_radius, displaced_normal, uv);
#endblock

#block lighting
	let band = smoothstep(-0.2, 0.2, sin(uv.y*p.stripe_frequency*6.2831853));
	let color = mix(material.base_color.rgb, p.stripe_color, band);
	var striped = material;
	striped.base_color = vec4f(color, material.base_color.a);
	return default_lighting_{{ PREFIX }}(striped, p, uv, position, normal, lod);
#endblock
'''



def sphere_vertices():
	u, v = np.meshgrid(np.linspace(0, 2*np.pi, 33), np.linspace(0, np.pi, 17))
	normals = np.stack((np.sin(v)*np.cos(u), np.cos(v), np.sin(v)*np.sin(u)), axis=-1)
	grid = np.zeros((17, 33), dtype=VERTEX_DTYPE)
	grid["position"], grid["normal"] = normals, normals
	grid["uv"] = np.stack((u/(2*np.pi), v/np.pi), axis=-1)
	return np.stack((grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:],
		grid[:-1, :-1], grid[1:, 1:], grid[:-1, 1:]), axis=2).reshape(-1, 3)[:, (0, 2, 1)].reshape(-1)


def make_instances():
	"""CPU-only setup; materials and overrides are ordinary mutable dictionaries."""
	vertices = sphere_vertices()
	opaque = DefaultMaterialShader(default_material={"base_color": (0.8, 0.2, 0.08, 1)})
	glass = DefaultMaterialShader(transparent=True, template=Path(__file__).parent / "shaders/glass_material.shader")
	rim = DefaultMaterialShader(transparent=True, template=Path(__file__).parent / "shaders/rim_material.shader",
		default_material={"base_color": (0.25, 0.6, 0.8, 0.45)})
	striped = DefaultMaterialShader(source=STRIPED_SHADER,
		default_material={"base_color": (0.08, 0.2, 0.75, 1), "roughness": 0.3})
	shared_glass = {"base_color": (0.08, 0.35, 0.55, 0.65), "roughness": 0.2, "specular": 3.0,
		"metallic": 0.65, "ambient": 0.08, "diffuse": 0.35,
		"grazing_start": 55.0, "grazing_end": 85.0}
	glass_mesh = MaterialMesh(vertices, glass, material=shared_glass)
	opaque_mesh = MaterialMesh(vertices, opaque)
	instances = [
		DrawInstance(opaque_mesh, position=(-2.6, 2.6, 0), scale=(0.8, 0.8, 0.8)),
		DrawInstance(opaque_mesh, material={"base_color": (0.15, 0.65, 0.2, 1)}, position=(2.6, 2.6, 0), scale=(0.8, 0.8, 0.8)),
		DrawInstance(glass_mesh, overrides={"grazing_fade": True}, position=(-2.6, 0, 0)),
		DrawInstance(glass_mesh, overrides={"tint": (1, 0.7, 0.45, 1)}, position=(0, 0, 0)),
		DrawInstance(MaterialMesh(vertices, rim), position=(2.6, 0, 0)),
		DrawInstance(MaterialMesh(vertices, striped, bounds=((0, 0, 0), (1.3, 1.3, 1.3))),
			position=(0, 2.6, 0), scale=(0.8, 0.8, 0.8)),
	]
	return instances, shared_glass, opaque


def make_world():
	"""Entity-owned materials and separate numeric transform components."""
	items, shared, opaque = make_instances()
	world = ECS()
	world.register(MaterialInstance, MaterialTransform)
	entities = world.create(len(items))
	for entity, item in zip(entities, items):
		world.add(entity, MaterialInstance(item.mesh, item.material, item.overrides, item.shader),
			MaterialTransform(Vec3(item.position), scale=Vec3(item.scale)))
	return world, entities, list(dict.fromkeys(item.mesh for item in items)), shared, opaque


def run_scene():
	from graaal.UI.ImGui import ImGui
	from imgui_bundle import imgui
	frustum_culler = FrustumCuller()

	RenderContext.init_window(1100, 760, "Materials | M: MSAA | A: SSAO | O: SSAO path | D: default | F: rendering | P: profiler | Space: orbit",
		required_gpu_features=["indirect-first-instance"], optional_gpu_features=["timestamp-query"])
	world, entities, meshes, shared, opaque = make_world()
	shared["texture_layer"] = 1
	meshes[2].shader.default_material["texture_layer"] = 1
	size = 64
	checks = (np.indices((size, size))//4).sum(axis=0)%2
	pixels = np.full((2, size, size, 4), 255, dtype=np.uint8)
	pixels[1, :, :, :3] = np.where(checks[..., None], 48, 255)
	checker_texture = Texture((size, size, 2), format="rgba8unorm-srgb", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
	RenderContext.device.queue.write_texture({"texture": checker_texture.handle}, pixels,
		{"bytes_per_row": size*4, "rows_per_image": size}, (size, size, 2))
	renderer = MaterialRenderer(meshes=meshes, base_textures=checker_texture.view(dimension="2d-array"), frame_fields={"time": ("f32", 0.0)})
	ao_effect = AmbientOcclusionEffect()
	aa_effect = AntialiasingEffect()
	aa_effect.mode = "off"
	postprocessing_passes = {}
	presenter = ResolutionPresenter(target_format=renderer.transparency.target_format)
	adapter = EcsMaterialAdapter(renderer, world)
	overrides = world.get(MaterialInstance)[entities].overrides
	striped = meshes[-1].shader.default_material
	camera = Camera((0, 1.3, 9), (0, 1.3, 0), (0, 1, 0), 50, near=0.05, far=100)
	orbit, angle, alternate = False, 0.0, False
	msaa_enabled, ao_enabled, ao_method = False, False, "compose"
	animation_time, animate = 0.0, True
	def set_msaa_enabled(enabled):
		nonlocal msaa_enabled
		msaa_enabled = enabled
		renderer.set_sample_count(4 if enabled else 1)
	def set_ssao_enabled(enabled):
		nonlocal ao_enabled
		ao_enabled = enabled
		renderer.transparency.set_ambient_occlusion(enabled and ao_method == "compose")
	settings = RenderingSettings(renderer, visible=False, position=(370, 12), camera=camera,
		on_msaa_changed=set_msaa_enabled, on_ssao_changed=set_ssao_enabled, ssao_enabled=ao_enabled, ao_pass=ao_effect, aa_pass=aa_effect)
	def set_ao_method(method):
		nonlocal ao_method
		ao_method = method
		renderer.transparency.set_ambient_occlusion(ao_enabled and method == "compose")
	overlay = ExampleOverlay(settings, renderer=renderer, get_ao_method=lambda: ao_method,
		set_ao_method=set_ao_method, profiler_position=(370, 12))
	def set_alternate_color(enabled):
		nonlocal alternate
		alternate = enabled
		opaque.default_material = {"base_color": (0.6, 0.1, 0.8, 1) if alternate else (0.8, 0.2, 0.08, 1)}
		adapter.touch_materials(opaque.default_material)
	def on_char(code):
		nonlocal orbit
		if overlay.handle_char(code): return True
		if chr(code) == " ": orbit = not orbit
		elif chr(code).lower() == "d":
			set_alternate_color(not alternate)
		return True
	def draw_controls():
		nonlocal animate
		# Labels follow the objects during orbit and use ImGui's logical pixels.
		width, height = imgui.get_io().display_size
		vp = camera.view() @ camera.projection(width/height)
		labels = ("Default material (D)", "Explicit green override", "Shared glass A: instance fade",
			"Shared glass B: instance tint", "Inherited rim shader", "Animated WGSL shader")
		transforms = world.get(MaterialTransform)[entities]
		for position, scale, label in zip(transforms.position, transforms.scale, labels):
			anchor = np.array((*position, 1.0))
			anchor[1] -= scale[1]+0.15
			clip = anchor @ vp
			if clip[3] <= 0: continue
			xy = clip[:2]/clip[3]
			x, y = (xy[0]*0.5+0.5)*width, (0.5-xy[1]*0.5)*height
			imgui.get_background_draw_list().add_text((x-imgui.calc_text_size(label).x/2, y), 0xffffffff, label)
		imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((340, 510), imgui.Cond_.first_use_ever)
		imgui.begin("Dictionary materials")
		try:
			imgui.text_wrapped("Six spheres: three opaque above, three transparent below. The bottom-left and bottom-middle spheres share the sliders below.")
			changed, value = imgui.checkbox("Alternate opaque base color (D)", alternate)
			if changed: set_alternate_color(value)
			changed, value = imgui.slider_float("Shared roughness", shared["roughness"], 0.02, 1)
			if changed:
				shared["roughness"] = value
				adapter.touch_materials(shared)
			changed, value = imgui.slider_float("Shared specular", shared["specular"], 0, 8)
			if changed:
				shared["specular"] = value
				adapter.touch_materials(shared)
			changed, value = imgui.checkbox("Left glass grazing fade", overrides[2]["grazing_fade"])
			if changed:
				overrides[2]["grazing_fade"] = value
				adapter.touch_materials(overrides[2])
			changed, value = imgui.slider_float("Middle glass opacity multiplier", overrides[3].get("opacity_multiplier", 1), 0, 1)
			if changed:
				overrides[3]["opacity_multiplier"] = value
				adapter.touch_materials(overrides[3])
			imgui.separator()
			imgui.text("Top-middle: shader written in this script")
			changed, value = imgui.slider_float("Stripe frequency", striped.get("stripe_frequency", 8.0), 1, 20)
			if changed:
				striped["stripe_frequency"] = value
				adapter.touch_materials(striped)
			changed, value = imgui.color_edit3("Stripe color", striped.get("stripe_color", (1.0, 0.15, 0.03)))
			if changed:
				striped["stripe_color"] = tuple(value)
				adapter.touch_materials(striped)
			_, animate = imgui.checkbox("Animate displacement", animate)
			for name, label, default, low, high in (("displacement", "Displacement", 0.18, 0.0, 0.3),
				("speed", "Animation speed", 2.0, 0.0, 5.0)):
				changed, value = imgui.slider_float(label, striped.get(name, default), low, high)
				if changed:
					striped[name] = value
					adapter.touch_materials(striped)
			imgui.text_wrapped("D changes only the top-left default. Top-right has an explicit green material. Bottom-right adds inherited rim lighting. M: MSAA; A: SSAO; O: SSAO path; F: rendering settings; P: profiler; Space: orbit.")
		finally: imgui.end()
		overlay.draw()
	RenderContext.subscribe_event("char", on_char)
	RenderContext.subscribe_event("key", overlay.handle_key)
	try:
		with ImGui() as gui:
			while RenderContext.window_loop():
				if min(RenderContext.window_dimensions) <= 0: continue
				if orbit: angle += RenderContext.frame_time*0.3
				if animate: animation_time += RenderContext.frame_time
				camera.position = Vec3(9*sin(angle), 1.3, 9*cos(angle))
				adapter.sync()
				commands = overlay.commands("material_example")
				window_size = RenderContext.window_dimensions
				render_size = settings.render_size(window_size)
				scaled = render_size != window_size
				screen = RenderContext.screen()
				if scaled: presenter.resize(render_size)
				output = presenter.color_attachment() if scaled else screen
				use_ao = ao_enabled and ao_method == "pass"
				use_aa = aa_effect.mode in ("lxaa", "faaa", "ddaa2")
				post = None
				if use_ao or use_aa:
					key = (use_ao, use_aa)
					if key not in postprocessing_passes:
						postprocessing_passes[key] = PostprocessingPass(([ao_effect] if use_ao else []) + ([aa_effect] if use_aa else []), target_format=renderer.transparency.target_format)
					post = postprocessing_passes[key]
					post.resize(render_size)
					output = post.color_attachment()
				oit = renderer.transparency
				oit.resize(render_size)
				vp = oit.update_camera(camera, frame={"time": animation_time}, lighting={"light_dir": (0.4, 0.8, 0.3)})
				frustum_culler.record(commands, renderer.visibility, camera_position=camera.position, view_projection=vp, fovy_deg=camera.fovy_deg)
				with oit.opaque_pass(commands) as render:
					renderer.draws.draw(render, renderer.visibility, "opaque")
				def draw_transparent(render, name):
					renderer.draws.draw(render, renderer.visibility, name)
				record_transparency(commands, oit, renderer.geometry[1], draw_transparent)
				oit.compose(commands, output)
				if use_ao: ao_effect.update_camera(camera, render_size[0]/render_size[1])
				if post is not None: post.record(commands, presenter.color_attachment() if scaled else screen, depth=oit.depth if use_ao else None)
				if scaled: presenter.record(commands, screen)
				overlay.submit(commands)
				gui.render(draw_controls)
	finally:
		RenderContext.event_handlers["char"].remove(on_char)
		RenderContext.event_handlers["key"].remove(overlay.handle_key)


if __name__ == "__main__":
	run_scene()
