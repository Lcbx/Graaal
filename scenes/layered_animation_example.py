"""Compare a base animation with a masked override or additive layer.

Run: python main.py scenes/layered_animation_example.py [--frames 120]
"""
import numpy as np

from graaal.Animation.SkeletalAnimation import additive_pose, blend_poses, model_matrices, sample_clip, skin_matrices, subtree_mask
from scenes.animation_dummy import dummy_material_mesh, load_dummy
from scenes.mixing_animation_example import advance_time


def dummy_masks(rig):
	"""Masks rooted at the dummy's first spine joint, thighs and hands."""
	upper = subtree_mask(rig, 1)
	lower = np.maximum(subtree_mask(rig, 44), subtree_mask(rig, 48))
	return (np.ones(len(rig.parents), dtype=np.float32), upper, lower, subtree_mask(rig, 9), subtree_mask(rig, 28))


def run_scene():
	from imgui_bundle import imgui
	from graaal.ECS import ECS
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.EcsAdapters.MaterialTransform import MaterialTransform
	from graaal.UI.ImGui import ImGui
	from graaal.Materials.MaterialRenderer import MaterialRenderer
	from graaal.Math import Camera, Vec3
	from graaal.RenderContext import RenderContext
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
	from graaal.RenderPasses.render_resolution import ResolutionPresenter
	from graaal.UI.rendering_settings import RenderingSettings
	from scenes.example_overlay import ExampleOverlay

	rig, primitive, clips = load_dummy()
	labels = tuple(name for name, _ in clips)
	masks = dummy_masks(rig)
	mask_labels = ("Whole body", "Upper body and hands", "Legs", "Left hand", "Right hand")
	RenderContext.init_window(1100, 760, "Layered animation | F: rendering | P: profiler | Space: pause", required_gpu_features=["indirect-first-instance"], optional_gpu_features=["timestamp-query"])
	mesh, textures = dummy_material_mesh(primitive)
	palette = SkinMatrixBuffer(2 * len(rig.parents))
	renderer = MaterialRenderer(meshes=[mesh], skin_palette=palette, base_textures=textures.view(dimension="2d-array"))
	camera = Camera((-0.8, 1.5, 5.5), (-0.8, 0.9, 0), (0, 1, 0), 45, near=0.05, far=100)
	world = ECS()
	world.register(MaterialInstance, MaterialTransform)
	entities = world.create(2)
	for i, entity in enumerate(entities):
		world.add(entity, MaterialInstance(mesh), MaterialTransform(position=Vec3((i * 2 - 1) * 1.7, 0, 0)))
		palette.acquire(entity, len(rig.parents))
	materials = EcsMaterialAdapter(renderer, world)
	settings = RenderingSettings(renderer, visible=False, position=(370, 12), camera=camera)
	overlay = ExampleOverlay(settings, renderer=renderer, profiler_position=(370, 12))
	presenter = ResolutionPresenter(target_format=renderer.transparency.target_format)
	base_id, layer_id = 1, 2
	base_time, layer_time, base_speed, layer_speed = 0.0, 0.0, 1.0, 1.0
	weight, mask_id, mode, playing = 1.0, 1, 0, True

	def on_char(code):
		nonlocal playing
		if overlay.handle_char(code): return True
		if chr(code) == " ":
			playing = not playing
			return True
		return False

	def draw_controls():
		nonlocal base_id, layer_id, base_time, layer_time, base_speed, layer_speed, weight, mask_id, mode, playing
		imgui.set_next_window_pos((12, 12), imgui.Cond_.first_use_ever)
		imgui.set_next_window_size((325, 450), imgui.Cond_.first_use_ever)
		imgui.begin("Layered animation")
		try:
			imgui.text_wrapped("Left: base clip. Right: base with a masked animation layer.")
			changed, value = imgui.combo("Base clip", base_id, labels)
			if changed: base_id, base_time = value, 0.0
			changed, value = imgui.combo("Layer clip", layer_id, labels)
			if changed: layer_id, layer_time = value, 0.0
			_, mode = imgui.combo("Layer mode", mode, ("Override", "Additive"))
			_, mask_id = imgui.combo("Bone mask", mask_id, mask_labels)
			_, weight = imgui.slider_float("Layer weight", weight, 0.0, 1.0, "%.2f")
			_, base_speed = imgui.slider_float("Base speed", base_speed, -2.0, 2.0, "%.2f")
			_, layer_speed = imgui.slider_float("Layer speed", layer_speed, -2.0, 2.0, "%.2f")
			_, playing = imgui.checkbox("Playing (Space)", playing)
			imgui.text_wrapped("Additive mode uses the skeleton reference pose; these clips were not authored as additive layers. F: rendering settings; P: profiler.")
		finally: imgui.end()
		overlay.draw()

	RenderContext.subscribe_event("char", on_char)
	RenderContext.subscribe_event("key", overlay.handle_key)
	try:
		with ImGui() as gui:
			while RenderContext.window_loop():
				if min(RenderContext.window_dimensions) <= 0: continue
				dt = RenderContext.frame_time if playing else 0.0
				base_time = advance_time(base_time, dt, base_speed, clips[base_id][1].duration, True)
				layer_time = advance_time(layer_time, dt, layer_speed, clips[layer_id][1].duration, True)
				base = sample_clip(clips[base_id][1], [base_time], loop=False)
				layer = sample_clip(clips[layer_id][1], [layer_time], loop=False)
				layered = (blend_poses if mode == 0 else additive_pose)(rig, base, layer, weight, mask=masks[mask_id])
				combined = tuple(np.concatenate((left, right), axis=0) for left, right in zip(base, layered))
				palette.write(entities, skin_matrices(rig, model_matrices(rig, *combined)))
				materials.sync()
				commands = overlay.commands("layered_animation_example")
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
