"""Greenman skeletal animation: 1 pose, 2 idle, 3 move, 4 attack; Space pauses.

Run from the repository root: python main.py scenes/simple_animation_example.py
"""
from pathlib import Path

from graaal.SkeletalGltf import load_skinned_gltf


ASSET = Path(__file__).parent / "resources" / "greenman.glb"
def load_greenman(path=ASSET):
	"""Load the example asset with the reusable skinned glTF importer."""
	asset = load_skinned_gltf(path)
	if len(asset.primitives) != 1: raise ValueError("The Greenman example expects one primitive")
	primitive = asset.primitives[0]
	return asset.skeleton, primitive.vertices, primitive.indices, asset.clips



def run_scene():
	from graaal.DefaultMaterial import DefaultMaterialShader
	from graaal.ECS import ECS
	from graaal.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
	from graaal.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.MaterialTransform import MaterialTransform
	from graaal.MaterialRenderer import MaterialMesh, MaterialRenderer
	from graaal.Math import Camera, Vec3
	from graaal.PosePalette import PosePalette
	from graaal.RenderContext import RenderContext
	from graaal.SkinMatrixBuffer import SkinMatrixBuffer

	rig, vertices, indices, clips = load_greenman()
	RenderContext.init_window(960, 720, "Greenman | 1-4: animation | Space: pause", required_gpu_features=["indirect-first-instance"])
	shader = DefaultMaterialShader(default_material={"base_color": (0.22, 0.65, 0.25, 1), "roughness": 0.65})
	mesh = MaterialMesh(vertices, shader, indices=indices, bounds=((0, 1, 0), (4, 4, 4)))
	palette = SkinMatrixBuffer(3 * len(rig.parents))
	poses = PosePalette(palette)
	renderer = MaterialRenderer(meshes=[mesh], skin_palette=palette)
	camera = Camera((0, 2.6, 11), (0, 1, 0), (0, 1, 0), 45, near=0.05, far=100)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, MaterialInstance, MaterialTransform)
	entities = world.create(3)
	for i, entity in enumerate(entities):
		world.add(entity, SkeletonRef(0), ClipPlayer(min(i + 1, len(clips) - 1)),
			MaterialInstance(mesh), MaterialTransform(position=Vec3((i - 1) * 3, 0, 0)))
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
		commands = RenderContext.commands("greenman")
		renderer.record(commands, RenderContext.window_dimensions, camera, RenderContext.screen())
		RenderContext.submit(commands.finish())


if __name__ == "__main__":
	run_scene()
