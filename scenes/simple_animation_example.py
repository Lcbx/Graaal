"""Dummy skeletal animation: 1 rest pose, 2 walk, 3 run; Space pauses.

Run from the repository root: python main.py scenes/simple_animation_example.py
"""
from scenes.animation_dummy import dummy_material_mesh, load_dummy

def run_scene():
	from graaal.ECS import ECS
	from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.EcsAdapters.MaterialTransform import MaterialTransform
	from graaal.Materials.MaterialRenderer import MaterialRenderer
	from graaal.Math import Camera, Vec3
	from graaal.Animation.PosePalette import PosePalette
	from graaal.RenderContext import RenderContext
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer

	rig, primitive, clips = load_dummy()
	RenderContext.init_window(960, 720, "Dummy | 1-3: animation | Space: pause", required_gpu_features=["indirect-first-instance"])
	mesh, textures = dummy_material_mesh(primitive)
	palette = SkinMatrixBuffer(3 * len(rig.parents))
	poses = PosePalette(palette)
	renderer = MaterialRenderer(meshes=[mesh], skin_palette=palette, base_textures=textures.view(dimension="2d-array"))
	camera = Camera((0, 1.6, 5.5), (0, 0.9, 0), (0, 1, 0), 45, near=0.05, far=100)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, MaterialInstance, MaterialTransform)
	entities = world.create(3)
	for i, entity in enumerate(entities):
		world.add(entity, SkeletonRef(0), ClipPlayer(i), MaterialInstance(mesh), MaterialTransform(position=Vec3((i - 1) * 1.7, 0, 0)))
	players = world.get(ClipPlayer)
	anim = EcsAnimationAdapter(world, {0: rig}, {i: clip for i, (_, clip) in enumerate(clips)})
	materials = EcsMaterialAdapter(renderer, world)
	print("Animations:", ", ".join(f"{i + 1}: {name}" for i, (name, _) in enumerate(clips)))

	def on_char(code):
		key = chr(code)
		player = players[entities]
		if key == " ": player.assign(playing=not player.playing[0])
		elif key in "123456789" and int(key) <= len(clips):
			clip_index = int(key) - 1
			player.assign(transitioning=True, target_clip_id=clip_index, target_time=0.0, target_speed=1.0, target_loop=True, transition_time=0.0, transition_duration=0.3)
			print(f"Animation: {clips[clip_index][0]}")
		else: return False
		return True

	RenderContext.subscribe_event("char", on_char)
	while RenderContext.window_loop():
		poses.update(anim.update(RenderContext.frame_time))
		materials.sync()
		commands = RenderContext.commands("simple_animation_example")
		renderer.record(commands, RenderContext.window_dimensions, camera, RenderContext.screen())
		RenderContext.submit(commands.finish())


if __name__ == "__main__":
	run_scene()
