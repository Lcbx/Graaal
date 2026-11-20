"""Compare manual clip mixing and ECS crossfades on the same skinned mesh.

Run: python main.py scenes/mixing_animation_example.py [--frames 120]
F opens rendering settings, P opens the profiler, and Space pauses playback.
"""

from scenes.simple_animation_example import load_greenman


def advance_time(time, dt, speed, duration, loop):
	if duration <= 0: return 0.0
	time += dt * speed
	return time % duration if loop else min(max(time, 0.0), duration)


def run_scene():
	from imgui_bundle import imgui
	from graaal.DefaultMaterial import DefaultMaterialShader
	from graaal.ECS import ECS
	from graaal.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
	from graaal.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.MaterialTransform import MaterialTransform
	from graaal.ImGui import ImGui
	from graaal.MaterialRenderer import MaterialMesh, MaterialRenderer
	from graaal.Math import Camera, Vec3
	from graaal.RenderContext import RenderContext
	from graaal.SkeletalAnimation import model_matrices, sample_crossfade, skin_matrices
	from graaal.SkinMatrixBuffer import SkinMatrixBuffer
	from graaal.render_resolution import ResolutionPresenter
	from graaal.rendering_settings import RenderingSettings
	from scenes.example_overlay import ExampleOverlay

	rig, vertices, indices, clips = load_greenman()
	if not clips: raise ValueError("The mixing example requires animation clips")
	labels = tuple(name for name, _ in clips)
	RenderContext.init_window(1100, 760, "Animation mixing | F: rendering | P: profiler | Space: pause", required_gpu_features=["indirect-first-instance"], optional_gpu_features=["timestamp-query"])
	shader = DefaultMaterialShader(default_material={"base_color": (0.22, 0.65, 0.25, 1), "roughness": 0.65})
	mesh = MaterialMesh(vertices, shader, indices=indices, bounds=((0, 1, 0), (4, 4, 4)))
	palette = SkinMatrixBuffer(2 * len(rig.parents))
	renderer = MaterialRenderer(meshes=[mesh], skin_palette=palette)
	camera = Camera((0, 2.6, 10), (0, 1, 0), (0, 1, 0), 45, near=0.05, far=100)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, MaterialInstance, MaterialTransform)
	entities = world.create(2)
	for i, entity in enumerate(entities):
		world.add(entity, MaterialInstance(mesh), MaterialTransform(position=Vec3((i * 2 - 1) * 1.7, 0, 0)))
		palette.acquire(entity, len(rig.parents))
	source, target = min(1, len(clips) - 1), min(2, len(clips) - 1)
	world.add(entities[1], SkeletonRef(0), ClipPlayer(source))
	players = world.get(ClipPlayer)
	animation = EcsAnimationAdapter(world, {0: rig}, {i: clip for i, (_, clip) in enumerate(clips)})
	materials = EcsMaterialAdapter(renderer, world)
	settings = RenderingSettings(renderer, visible=False, position=(370, 12), camera=camera)
	overlay = ExampleOverlay(settings, renderer=renderer, profiler_position=(370, 12))
	presenter = ResolutionPresenter(target_format=renderer.transparency.target_format)
	mix, source_time, target_time = 0.5, 0.0, 0.0
	speed, fade_duration, playing, loop = 1.0, 0.6, True, True

	def on_char(code):
		nonlocal playing
		if overlay.handle_char(code): return True
		if chr(code) == " ":
			playing = not playing
			return True
		return False

	def draw_controls():
		nonlocal source, target, source_time, target_time, mix, speed, fade_duration, playing, loop
		imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((320, 480), imgui.Cond_.first_use_ever)
		imgui.begin("Animation mixing")
		try:
			imgui.text_wrapped("Left: manual blend of clips A and B. Right: ECS player crossfading to B.")
			changed, value = imgui.combo("Clip A", source, labels)
			if changed: source, source_time = value, 0.0
			changed, value = imgui.combo("Clip B", target, labels)
			if changed: target, target_time = value, 0.0
			_, mix = imgui.slider_float("Manual mix", mix, 0.0, 1.0, "%.2f")
			_, speed = imgui.slider_float("Playback speed", speed, -2.0, 2.0, "%.2f")
			_, playing = imgui.checkbox("Playing (Space)", playing)
			_, loop = imgui.checkbox("Loop clips", loop)
			imgui.separator()
			_, fade_duration = imgui.slider_float("Fade duration (seconds)", fade_duration, 0.05, 3.0, "%.2f")
			player = players[entities[1:]]
			if imgui.button("Set right to A"):
				player.assign(clip_id=source, time=0.0, speed=speed, loop=loop, playing=playing, transitioning=False)
			imgui.same_line()
			if imgui.button("Crossfade right to B") and not player.transitioning[0]:
				player.assign(transitioning=True, target_clip_id=target, target_time=0.0, target_speed=speed, target_loop=loop, transition_time=0.0, transition_duration=fade_duration)
			player = players[entities[1:]]
			if player.transitioning[0]:
				imgui.text(f"Fade: {player.transition_time[0] / player.transition_duration[0]:.0%} to {labels[int(player.target_clip_id[0])]}")
			else:
				imgui.text(f"Right clip: {labels[int(player.clip_id[0])]}")
			imgui.text_wrapped("A and B control the left mix. Use the buttons to choose or crossfade the right player. F: rendering settings; P: profiler.")
		finally: imgui.end()
		overlay.draw()

	RenderContext.subscribe_event("char", on_char)
	RenderContext.subscribe_event("key", overlay.handle_key)
	try:
		with ImGui() as gui:
			while RenderContext.window_loop():
				if min(RenderContext.window_dimensions) <= 0: continue
				dt = RenderContext.frame_time if playing else 0.0
				source_time = advance_time(source_time, dt, speed, clips[source][1].duration, loop)
				target_time = advance_time(target_time, dt, speed, clips[target][1].duration, loop)
				players[entities[1:]].assign(speed=speed, target_speed=speed, loop=loop, target_loop=loop, playing=playing)
				for batch in animation.update(RenderContext.frame_time):
					palette.write(batch.entities, batch.matrices)
				pose = sample_crossfade(clips[source][1], clips[target][1], [source_time], [target_time], mix, source_loop=loop, target_loop=loop)
				palette.write(entities[:1], skin_matrices(rig, model_matrices(rig, *pose)))
				materials.sync()
				commands = overlay.commands("mixing_animation_example")
				window_size = RenderContext.window_dimensions
				render_size = settings.render_size(window_size)
				screen = RenderContext.screen()
				if render_size != window_size:
					presenter.resize(render_size)
					output = presenter.color_attachment()
				else: output = screen
				renderer.record(commands, render_size, camera, output)
				if render_size != window_size: presenter.record(commands, screen)
				overlay.submit(commands)
				gui.render(draw_controls)
	finally:
		RenderContext.event_handlers["char"].remove(on_char)
		RenderContext.event_handlers["key"].remove(overlay.handle_key)
		palette.close()


if __name__ == "__main__":
	run_scene()
