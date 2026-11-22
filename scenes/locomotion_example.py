"""GPU-skinned locomotion with forward root motion through a looping city.

Run: python main.py scenes/locomotion_example.py [--frames 120]
"""
from dataclasses import replace
import numpy as np
from pathlib import Path

from graaal.Animation.SkeletalAnimation import Clip, Skeleton
from scenes.animation_dummy import dummy_material_mesh, load_dummy
from scenes.root_motion_example import Movement, crossfade


CITY_PERIOD = 500.0
CHARACTER_X = -2.0


def make_locomotion_clips(rig, primitive, named_clips):
	"""Add a trajectory root above the animated hips of the in-place clips."""
	rig = Skeleton(np.r_[-1, rig.parents + 1], np.vstack(([0, 0, 0], rig.translation)),
		np.vstack(([0, 0, 0, 1], rig.rotation)), np.vstack(([1, 1, 1], rig.scale)),
		np.concatenate((np.eye(4)[None], rig.inverse_bind)))
	vertices = primitive.vertices.copy()
	vertices["joints"] += 1
	clips = {}
	for clip_id, ((_, clip), speed) in enumerate(zip(named_clips, (0.0, 1.5, 4.0), strict=True)):
		frames = len(clip.translation)
		trajectory = np.zeros((frames, 1, 3), dtype=np.float32)
		trajectory[:, 0, 2] = np.arange(frames, dtype=np.float32) * (speed / clip.fps)
		rotation = np.broadcast_to(rig.rotation[0], (frames, 1, 4))
		scale = np.broadcast_to(rig.scale[0], (frames, 1, 3))
		clips[clip_id] = Clip(np.concatenate((trajectory, clip.translation), axis=1),
			np.concatenate((rotation, clip.rotation), axis=1), np.concatenate((scale, clip.scale), axis=1), clip.fps)
	return rig, replace(primitive, vertices=vertices), clips


def run_scene():
	from imgui_bundle import imgui
	from graaal.DrawBatches import DrawBatches, RenderShader, ShaderPass
	from graaal.ECS import ECS, Scheduler
	from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.EcsAdapters.MaterialTransform import MaterialTransform
	from graaal.UI.ImGui import ImGui
	from graaal.RenderContext.Geometry import make_cube_mesh
	from graaal.DrawBatches.InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_rgba8_srgb, pack_scale
	from graaal.Materials.MaterialRenderer import MaterialRenderer
	from graaal.Math import Camera, Quaternion, Vec3
	from graaal.Animation.PosePalette import PosePalette
	from graaal.RenderContext import RenderContext, RenderPipeline, Shader
	from graaal.Animation.RootMotion import RootMotion
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
	from graaal.RenderPasses.TransparencyPass import record_transparency
	from graaal.RenderPasses.render_resolution import ResolutionPresenter
	from graaal.UI.rendering_settings import RenderingSettings
	from scenes.example_overlay import ExampleOverlay

	rig, primitive, clips = make_locomotion_clips(*load_dummy())
	RenderContext.init_window(1100, 760, "Root-motion locomotion | F: rendering | P: profiler | Space: pause", required_gpu_features=["indirect-first-instance"], optional_gpu_features=["timestamp-query"])
	mesh, textures = dummy_material_mesh(primitive)
	palette = SkinMatrixBuffer(len(rig.parents))
	poses = PosePalette(palette)
	renderer = MaterialRenderer(meshes=[mesh], skin_palette=palette, base_textures=textures.view(dimension="2d-array"))
	city_shader = Shader(filepath=str(Path(__file__).parent / "shaders" / "complex.shader"))
	city_uniforms = city_shader.uniform_buffer()
	city_bindings = (city_shader.bind_group(0, uniforms=city_uniforms),)
	city = DrawBatches(MESH_INSTANCE_DTYPE)
	city.register_shader(0, RenderShader({"main": ShaderPass(RenderPipeline(city_shader, vertex_entry="vertex", fragment_entry="fragment", depth_test="less-equal"), city_bindings)}))
	city.register_mesh(0, make_cube_mesh())
	city.register_lod_group(0, (0,))
	rng = np.random.default_rng(4)
	positions, scales, colors = [], [], []
	for z in range(-490, 10, 10):
		for side in (-1, 1):
			height = float(rng.uniform(8, 32))
			positions.append((side * float(rng.uniform(11, 16)), height/2, z))
			scales.append((float(rng.uniform(6, 10)), height, float(rng.uniform(6, 9))))
			colors.append((*rng.uniform(0.25, 0.65, 3), 1))
		positions.append((0, 0.02, z))
		scales.append((0.18, 0.04, 4))
		colors.append((0.9, 0.8, 0.45, 1))
	base_positions = np.asarray(positions, dtype=np.float32)
	base_scales, base_colors = scales[:], colors[:]
	positions, scales, colors = [], [], []
	for offset in (-CITY_PERIOD, 0, CITY_PERIOD):
		positions.append((0, -0.5, -CITY_PERIOD/2 + offset))
		scales.append((12, 1, CITY_PERIOD))
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
	city.sync(np.zeros(n, dtype=np.uint32), np.zeros(n, dtype=np.uint32))
	city.write_instances(instances[city.instance_order])
	city_visibility = city.create_visibility().all_visible()
	city_samples = 1
	camera = Camera((CHARACTER_X + 1.5, 1.9, 5.5), (CHARACTER_X, 0.9, 0), (0, 1, 0), 45, near=0.05, far=250)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, Movement, MaterialInstance, MaterialTransform)
	entity = world.create()[0]
	transform = MaterialTransform(position=Vec3(CHARACTER_X, 0, 0), rotation=Quaternion.from_y_rotation(np.pi))
	world.add(entity, SkeletonRef(0), ClipPlayer(1, extract_root_motion=True), Movement(), MaterialInstance(mesh), transform)
	root_motions = {(0, clip_id): RootMotion(rig, clip) for clip_id, clip in clips.items()}
	animation = EcsAnimationAdapter(world, {0: rig}, clips, root_motions=root_motions)
	materials = EcsMaterialAdapter(renderer, world)
	settings = RenderingSettings(renderer, visible=False, position=(370, 12), camera=camera)
	overlay = ExampleOverlay(settings, renderer=renderer, profiler_position=(370, 12))
	presenter = ResolutionPresenter(target_format=renderer.transparency.target_format)
	playing, selected = True, 1
	labels = ("Rest", "Walk", "Run")
	fixed_dt, accumulator = 1 / 60, 0.0

	def accept_movement(world, commands, dt):
		for batch in animation.root_motion_batches:
			movement = world.get(Movement)[batch.entities]
			transforms = world.get(MaterialTransform)[batch.entities]
			# Character-local +Z points down the street along world -Z.
			requested = batch.matrices[:, 2, 3].astype(np.float32)
			distance = (movement.x + requested) % CITY_PERIOD
			position = transforms.position.copy()
			position[:, 2] = -distance
			movement.assign(requested=requested, accepted=requested, x=distance)
			transforms.assign(position=position)

	schedule = Scheduler(world)
	schedule.add("animation_control", animation.advance, phase="fixed_update")
	schedule.add("movement", accept_movement, phase="fixed_update", after="animation_control")
	schedule.add("animation_pose", animation.finalize, phase="fixed_update", after="movement")
	schedule.run(0, fixed_steps=1, fixed_dt=fixed_dt)

	def on_char(code):
		nonlocal playing
		if overlay.handle_char(code): return True
		if chr(code) == " ":
			playing = not playing
			return True
		return False

	def draw_controls():
		nonlocal playing, selected
		imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((330, 360), imgui.Cond_.first_use_ever)
		imgui.begin("Root-motion locomotion")
		try:
			imgui.text_wrapped("The in-place walk and run use a separate forward root-motion track.")
			player = world.get(ClipPlayer)[[entity]]
			for clip_id, label in enumerate(labels):
				if clip_id: imgui.same_line()
				if imgui.button(label) and not player.transitioning[0]:
					crossfade(world, entity, clip_id)
					selected = clip_id
			_, playing = imgui.checkbox("Playing (Space)", playing)
			movement = world.get(Movement)[[entity]]
			imgui.text(f"Selected: {labels[selected]}")
			imgui.text(f"Distance within loop: {movement.x[0]:.2f} / {CITY_PERIOD:.0f}")
			imgui.text(f"Last requested: {movement.requested[0]:.4f}")
			imgui.text(f"Last accepted: {movement.accepted[0]:.4f}")
			imgui.text_wrapped("Movement continues indefinitely through repeating city blocks. F opens rendering settings; P opens the profiler.")
		finally: imgui.end()
		overlay.draw()

	RenderContext.subscribe_event("char", on_char)
	RenderContext.subscribe_event("key", overlay.handle_key)
	try:
		with ImGui() as gui:
			while RenderContext.window_loop():
				if min(RenderContext.window_dimensions) <= 0: continue
				world.get(ClipPlayer)[[entity]].assign(playing=playing)
				accumulator += min(RenderContext.frame_time, 0.25)
				steps = int(accumulator / fixed_dt)
				if steps:
					schedule.run(RenderContext.frame_time, fixed_steps=steps, fixed_dt=fixed_dt)
					accumulator -= steps * fixed_dt
				poses.update(animation.batches)
				materials.sync()
				character_z = float(world.get(MaterialTransform)[[entity]].position[0, 2])
				camera.position = Vec3(CHARACTER_X + 1.5, 1.9, character_z + 5.5)
				camera.target = Vec3(CHARACTER_X, 0.9, character_z)
				commands = overlay.commands("locomotion_example")
				window_size = RenderContext.window_dimensions
				render_size = settings.render_size(window_size)
				screen = RenderContext.screen()
				if render_size != window_size:
					presenter.resize(render_size)
					output = presenter.color_attachment()
				else: output = screen
				if city_samples != renderer.transparency.sample_count:
					city_samples = renderer.transparency.sample_count
					city.shaders[0] = RenderShader({"main": ShaderPass(RenderPipeline(city_shader, vertex_entry="vertex", fragment_entry="fragment", depth_test="less-equal", sample_count=city_samples), city_bindings)})
				city_uniforms.content["view"] = camera.view()
				city_uniforms.content["proj"] = camera.projection(render_size[0]/render_size[1])
				city_uniforms.content["light_dir"] = (0.4, 0.8, 0.3, 0)
				city_uniforms.upload()
				renderer.sync_skinning()
				oit = renderer.transparency
				oit.resize(render_size)
				vp = oit.update_camera(camera)
				renderer.culler.record(commands, renderer.visibility, camera_position=camera.position, view_projection=vp, fovy_deg=camera.fovy_deg)
				with oit.opaque_pass(commands, clear=(0.3, 0.48, 0.65, 1)) as render:
					city.draw(render, city_visibility)
					renderer.draws.draw(render, renderer.visibility, "opaque")
				def draw_transparent(render, name):
					renderer.draws.draw(render, renderer.visibility, name)
				record_transparency(commands, oit, renderer.geometry[1], draw_transparent)
				oit.compose(commands, output)
				if render_size != window_size: presenter.record(commands, screen)
				overlay.submit(commands)
				gui.render(draw_controls)
	finally:
		RenderContext.event_handlers["char"].remove(on_char)
		RenderContext.event_handlers["key"].remove(overlay.handle_key)
		palette.close()


if __name__ == "__main__":
	run_scene()
