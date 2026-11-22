"""Native humanoid: animation -> powered push -> limp -> blended recovery.

python -m scenes.ragdoll_example --headless --frames 960
python -m scenes.ragdoll_example --offscreen --frames 6
python main.py scenes/ragdoll_example.py
"""

import argparse

import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, PoseBatch, SkeletonRef
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, PhysicsRagdoll, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import RagdollAsset
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip_pairs
from graaal.Animation.SkeletalGltf import SKINNED_VERTEX_DTYPE


def create_demo():
	import pyjoltphysics as pj
	settings = pj.RagdollSettings.humanoid(root_force=1400, root_torque=600)
	count = len(settings)
	rig = Skeleton(settings.parents, settings.rest_translations, settings.rest_rotations, np.ones((count, 3)), np.tile(np.eye(4), (count, 1, 1)))
	rest = (rig.translation[None], rig.rotation[None], rig.scale[None])
	rig = Skeleton(rig.parents, rig.translation, rig.rotation, rig.scale, np.linalg.inv(model_matrices(rig, *rest)[0]))
	asset = RagdollAsset(rig, settings)
	translation, rotation, scale = (np.tile(value, (5, 1, 1)) for value in (rig.translation, rig.rotation, rig.scale))
	rotation[:, 3] = np.array([Quaternion.from_z_rotation(angle) for angle in (0, 0.3, 0, -0.3, 0)])
	rotation[:, 5] = np.array([Quaternion.from_z_rotation(-angle) for angle in (0, 0.3, 0, -0.3, 0)])
	clip = Clip(translation, rotation, scale, fps=2)
	ecs = ECS()
	ecs.register(Collider, RigidBody, PhysicsRagdoll, MaterialTransform, SkeletonRef, ClipPlayer)
	owner, floor = ecs.create(2)
	ecs.add(owner, PhysicsRagdoll(0), MaterialTransform(rotation=Quaternion.from_y_rotation(0.35)), SkeletonRef(0), ClipPlayer(0))
	ecs.add(floor, Collider(0), MaterialTransform(position=(0, -0.5, 0)))
	physics = EcsPhysicsAdapter(ecs, {0: pj.Shape.box((20, 1, 20))}, settings=pj.WorldSettings(threads=1), ragdolls={0: asset}, interpolate=True)
	animation = EcsAnimationAdapter(ecs, {0: rig}, {0: clip})
	frames = {"batches": [], "stage": "animated"}

	def control(world, commands, dt):
		tick = physics.world.tick % 960
		if tick == 0:
			world.get(PhysicsRagdoll)[[owner]].assign(mode=0, blend=0)
			frames["stage"] = "animated"
		elif tick == 120:
			world.get(PhysicsRagdoll)[[owner]].assign(mode=1, blend=0.2)
			frames["stage"] = "powered"
		elif tick == 180:
			physics.ragdolls[int(owner)].hit(1, (45, 5, 0))
		elif tick == 360:
			world.get(PhysicsRagdoll)[[owner]].assign(mode=2, blend=0.2)
			frames["stage"] = "limp"
		elif tick == 600:
			world.get(PhysicsRagdoll)[[owner]].assign(mode=0, blend=1)
			frames["stage"] = "recovery"

	def targets(world, commands, dt):
		for batch in animation.sample_batches:
			pose = sample_clip_pairs(rig, {0: clip}, batch.clip_ids, batch.times, batch.weights)
			physics.set_ragdoll_pose(batch.entities, pose)

	def finalize(world, commands, dt):
		frames["batches"] = [PoseBatch(0, 0, np.array([owner]), physics.ragdoll_skin([owner]))]

	schedule = Scheduler(ecs)
	schedule.add("animation", animation.advance, phase="fixed_update")
	schedule.add("control", control, phase="fixed_update", after="animation")
	schedule.add("physics_sync", physics.reconcile, phase="fixed_update", after="control")
	schedule.add("pose_targets", targets, phase="fixed_update", after="physics_sync")
	schedule.add("physics_input", physics.prepare_inputs, phase="fixed_update", after="pose_targets")
	schedule.add("physics_step", physics.step, phase="fixed_update", after="physics_input")
	schedule.add("physics_output", physics.pull, phase="fixed_update", after="physics_step")
	schedule.add("physics_events", physics.dispatch_events, phase="fixed_update", after="physics_output")
	schedule.add("final_pose", finalize, phase="fixed_update", after="physics_events")
	return ecs, owner, floor, physics, schedule, frames, asset


def skinned_boxes(asset, cube):
	"""One skinned mesh of boxes showing the prefab's offset capsule volumes."""
	rig = asset.skeleton
	model = model_matrices(rig, rig.translation[None], rig.rotation[None], rig.scale[None])[0]
	vertices = np.zeros((len(asset.bones), len(cube)), dtype=SKINNED_VERTEX_DTYPE)
	for i, bone in enumerate(asset.settings.bones):
		radius, cylinder_length = bone.shape.params[:2]
		size = np.array([radius * 2, radius * 2 + cylinder_length, radius * 2])
		shape_rotation = Quaternion(bone.shape_rotation).matrix33
		position = (cube["position"] * size) @ shape_rotation.T + bone.shape_offset
		vertices[i]["position"] = position @ model[asset.bones[i], :3, :3].T + model[asset.bones[i], :3, 3]
		vertices[i]["normal"] = cube["normal"] @ shape_rotation.T @ model[asset.bones[i], :3, :3].T
		vertices[i]["uv"] = cube["uv"]
		vertices[i]["joints"][:, 0] = asset.bones[i]
		vertices[i]["weights"][:, 0] = 1
	return vertices.reshape(-1)


def run_headless(frames=960):
	ecs, owner, floor, physics, schedule, state, asset = create_demo()
	with physics:
		for _ in range(frames):
			schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
			if not np.isfinite(state["batches"][0].matrices).all(): raise RuntimeError("nonfinite ragdoll palette")
			if physics.world.tick % 120 == 0:
				positions, _ = physics.ragdolls[int(owner)].pose()
				print(f"tick={physics.world.tick} stage={state['stage']} pelvis_y={positions[0, 1]:.3f}")


def run_scene(*, offscreen=False, frames=None):
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.Materials.MaterialRenderer import MaterialMesh, MaterialRenderer
	from graaal.Math import Camera
	from graaal.Animation.PosePalette import PosePalette
	from graaal.RenderContext import RenderContext, Texture, TextureUsage
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
	from scenes.physics_geometry import cube_vertices

	ecs, owner, floor, physics, schedule, state, asset = create_demo()
	try:
		if offscreen: RenderContext.setup_graphics_backend(True, required_features=["indirect-first-instance"])
		else: RenderContext.init_window(960, 720, "Native ragdoll | animated, powered push, limp, recovery", required_gpu_features=["indirect-first-instance"])
		cube = cube_vertices()
		shader = DefaultMaterialShader()
		mesh = MaterialMesh(skinned_boxes(asset, cube), shader, bounds=((0, 0, 0), (20, 20, 20)))
		floor_vertices = cube.copy()
		floor_vertices["position"] *= [20, 1, 20]
		ground = MaterialMesh(floor_vertices, shader)
		palette = SkinMatrixBuffer(len(asset.skeleton.parents))
		poses = PosePalette(palette)
		renderer = MaterialRenderer(meshes=[mesh, ground], skin_palette=palette)
		ecs.register(MaterialInstance)
		ecs.add(owner, MaterialInstance(mesh, {"base_color": (0.2, 0.65, 0.9, 1)}))
		ecs.add(floor, MaterialInstance(ground, {"base_color": (0.3, 0.3, 0.35, 1)}))
		drawing = EcsMaterialAdapter(renderer, ecs)
		camera = Camera((4, 3, 7), (0, 0.9, 0), (0, 1, 0), 45, near=0.05, far=100)
		size = (640, 480)
		output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC) if offscreen else None
		accumulator, displayed = 0.0, 0
		schedule.run(0, fixed_steps=1, fixed_dt=1 / 120)
		if offscreen and frames is None: frames = 6
		while frames is None or displayed < frames:
			if not offscreen and not RenderContext.window_loop(): break
			elapsed = 1 / 60 if offscreen else min(RenderContext.frame_time, 0.1)
			accumulator += elapsed
			steps = int(accumulator * 120)
			accumulator = max(0.0, accumulator - steps / 120)
			schedule.run(elapsed, fixed_steps=steps, fixed_dt=1 / 120)
			poses.update([PoseBatch(0, 0, np.array([owner]), physics.ragdoll_skin([owner], alpha=accumulator * 120))])
			drawing.sync()
			commands = RenderContext.commands("ragdoll")
			size = size if offscreen else RenderContext.window_dimensions
			renderer.record(commands, size, camera, output.color_attachment() if offscreen else RenderContext.screen())
			RenderContext.submit(commands.finish())
			displayed += 1
		if offscreen:
			pixels = RenderContext.device.queue.read_texture({"texture": output.handle}, {"bytes_per_row": size[0] * 4}, (*size, 1))
			if not np.any(np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 4)[:, 2] > 100): raise RuntimeError("ragdoll scene rendered no colored mesh")
			print(f"Ragdoll scene rendered {displayed} offscreen frames")
	finally:
		physics.close()
		RenderContext.cleanup()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("--headless", action="store_true")
	mode.add_argument("--offscreen", action="store_true")
	parser.add_argument("--frames", type=int)
	args = parser.parse_args()
	if args.frames is not None and args.frames <= 0: parser.error("--frames must be positive")
	if args.headless: run_headless(960 if args.frames is None else args.frames)
	else: run_scene(offscreen=args.offscreen, frames=args.frames)
