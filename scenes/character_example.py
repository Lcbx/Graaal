"""Graph-driven native character, collision foot IK and explicit ragdoll hand-off.

python main.py scenes/character_example.py
python -m scenes.character_example --headless --frames 960
python -m scenes.character_example --offscreen --frames 6
WASD moves, Space jumps, R falls, G recovers, T resumes the scripted cycle.
"""

import argparse

import numpy as np

from graaal.Animation.AnimationEvents import EventTrack
from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsAnimation import PoseBatch
from graaal.EcsAdapters.EcsAnimationGraph import EcsAnimationGraphAdapter, GraphPlayer
from graaal.EcsAdapters.EcsPhysics import CharacterState, Collider, EcsPhysicsAdapter, PhysicsCharacter, PhysicsRagdoll, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import RagdollAsset
from graaal.Animation.PoseTasks import PosePlan
from graaal.Animation.RootMotion import RootMotion
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, skin_matrices
from scenes.physics_geometry import cube_vertices, skinned_boxes
from scenes.physics_utils import advance_fixed_steps
from scenes.physics_viewing_utils import draw_frame, read_pixels, viewing_target


def create_demo():
	import pyjoltphysics as pj
	settings = pj.RagdollSettings.humanoid()
	physical_bones = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
	parents = [-1, 0, 1, 1, 3, 1, 5, 0, 7, 8, 0, 10, 11]
	translation = np.zeros((13, 3), dtype=np.float32)
	translation[physical_bones] = settings.rest_translations
	translation[[9, 12]] = [0, -0.43, 0]
	rotation, scale = np.tile([0, 0, 0, 1], (13, 1)), np.ones((13, 3))
	rig = Skeleton(parents, translation, rotation, scale, np.tile(np.eye(4), (13, 1, 1)))
	rig = Skeleton(parents, translation, rotation, scale, np.linalg.inv(model_matrices(rig, translation[None], rotation[None], scale[None])[0]))
	asset = RagdollAsset(rig, settings, bones=physical_bones)
	translations, rotations, scales = (np.tile(value, (5, 1, 1)) for value in (rig.translation, rig.rotation, rig.scale))
	translations[:, 0, 2] = np.linspace(0, 2, 5)
	rotations[:, 7] = np.array([Quaternion.from_x_rotation(angle) for angle in (0, 0.2, 0, -0.2, 0)])
	rotations[:, 10] = np.array([Quaternion.from_x_rotation(-angle) for angle in (0, 0.2, 0, -0.2, 0)])
	walk = Clip(translations, rotations, scales, fps=2)
	translations[:, 0, 2] *= 3
	brisk = Clip(translations, rotations, scales, fps=4)
	idle = Clip(np.tile(rig.translation, (5, 1, 1)), np.tile(rig.rotation, (5, 1, 1)), scales, fps=2)
	tracks = {name: EventTrack(clip.duration, [0, clip.duration / 2], ["left_step", "right_step"]) for name, clip in (("walk", walk), ("brisk", brisk))}
	graph = AnimationGraphDefinition({
		"version": 1, "parameters": {"move": 1, "pace": 0.5},
		"states": {"idle": {"clip": "idle"}, "locomotion": {"blend_space": "pace"}}, "initial_state": "locomotion",
		"blend_spaces": {"pace": {"parameter": "pace", "sync_reference": "walk", "points": [{"position": 0, "clip": "walk"}, {"position": 1, "clip": "brisk"}]}},
		"transitions": [
			{"source": "idle", "target": "locomotion", "conditions": [{"parameter": "move", "comparison": ">", "threshold": 0.5}], "duration": 0.15},
			{"source": "locomotion", "target": "idle", "conditions": [{"parameter": "move", "comparison": "<", "threshold": 0.5}], "duration": 0.15},
		],
		"sample_slots": ["locomotion"], "root_motion_slots": ["locomotion"],
		"tasks": [{"name": "sample", "op": "sample", "slot": "locomotion"}, {"name": "in_place", "op": "in_place", "base": "sample", "slot": "locomotion"}],
	}).compile(rig, {"idle": idle, "walk": walk, "brisk": brisk}, sync_tracks=tracks)
	post = PosePlan(rig, [("input", 0), ("two_bone_ik", 0, (7, 8, 9), 0, 0), ("two_bone_ik", 1, (10, 11, 12), 1, 1)])
	ecs = ECS()
	ecs.register(Collider, RigidBody, PhysicsCharacter, CharacterState, PhysicsRagdoll, MaterialTransform, GraphPlayer)
	owner, floor, wall, step, sensor = ecs.create(5)
	ecs.add(owner, PhysicsCharacter(), CharacterState(), MaterialTransform(position=(-4, 0, 0)), GraphPlayer(0))
	ecs.add(floor, Collider(0), MaterialTransform(position=(0, -0.5, 0), scale=(30, 1, 20)))
	ecs.add(wall, Collider(0), MaterialTransform(position=(3, 0.375, 0), scale=(0.5, 0.75, 8)))
	ecs.add(step, Collider(0), MaterialTransform(position=(-2, 0.1, 0), scale=(1, 0.2, 3)))
	ecs.add(sensor, Collider(0, sensor=True), MaterialTransform(position=(-1, 1, 0), scale=(2, 2, 3)))
	capabilities = pj.CharacterCapabilities(airControl=1)
	physics = EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, characters={0: capabilities}, ragdolls={0: asset}, settings=pj.WorldSettings(threads=1), interpolate=True)
	motions = {key: RootMotion(rig, clip) for key, clip in graph.clips.items()}
	animation = EcsAnimationGraphAdapter(ecs, 0, graph, event_tracks={graph.state_ids["locomotion"]: tracks["walk"]}, root_motions=motions)
	state = {"auto": True, "direction": np.array([1, 0, 0]), "jump": False, "fall": False, "recover": False, "stage": "controller", "pose": None, "final_pose": None, "batches": [], "launch": None, "return_velocity": None, "fall_tick": -1, "foot_hits": np.zeros(2, dtype=bool), "requested_speed": 0.0, "accepted_speed": 0.0, "trigger_begins": 0, "trigger_ends": 0}
	state.update(animation=animation, pace=0.5)

	def prepare_pose(world, commands, dt):
		moving = state["stage"] == "controller" and (state["auto"] or np.any(state["direction"]))
		base = animation.update(dt, [[float(moving), state["pace"]]])
		pose = graph.input_plan.evaluate({}, [], np.empty((1, 0)), poses=[base], root_motions=[(motions[graph.clip_ids["walk"]], np.array([True]))])
		state["pose"] = tuple(value.copy() for value in pose)

	def control(world, commands, dt):
		tick = physics.world.tick
		if state["auto"]:
			state["direction"] = np.array([1, 0, 0])
			if state["stage"] == "controller":
				obstacle = world.get(MaterialTransform)[[wall]]
				distance = obstacle.position[0, 0] - obstacle.scale[0, 0] * 0.5 - world.get(MaterialTransform)[[owner]].position[0, 0]
				state["jump"] |= 0.7 < distance < 0.9 and world.get(CharacterState)[[owner]].grounded[0]
			state["fall"] |= tick % 960 == 240
			state["recover"] |= tick % 960 == 600
		if state["fall"] and state["stage"] == "controller" and int(owner) in physics.characters:
			state["launch"] = physics.characters[int(owner)].velocity.copy()
			world.get(PhysicsCharacter)[[owner]].enabled = False
			world.add(owner, PhysicsRagdoll(0, mode=0))
			base = state["final_pose"] if state["final_pose"] is not None else state["pose"]
			state["frozen"] = tuple(value.copy() for value in base)
			state["fall_tick"], state["stage"] = tick, "settling"
		elif state["stage"] == "settling" and tick >= state["fall_tick"] + 2:
			world.get(PhysicsRagdoll)[[owner]].mode = 2
			state["stage"] = "ragdoll"
		if state["recover"] and state["stage"] in ("ragdoll", "settling"):
			handle = physics.ragdolls[int(owner)]
			position, _ = handle.pose()
			feet = position[0].copy()
			hit = physics.raycast(feet + [0, 0.05, 0], [0, -1, 0], 6, exclude=[owner])
			feet[1] = hit.point[1] if hit is not None else feet[1] - rig.translation[0, 1]
			capsule = pj.Shape.capsule(capabilities.radius, capabilities.standHeight - 2 * capabilities.radius)
			blocked = physics.overlap(capsule, feet + [0, capabilities.standHeight * 0.5 + 0.02, 0], exclude=[owner])
			if hit is not None and not len(blocked):
				state["return_velocity"] = physics.world.bodies_state(handle.bones[:1]).linear_velocity[0].copy()
				world.get(MaterialTransform)[[owner]].position = feet
				world.remove(owner, PhysicsRagdoll)
				world.get(PhysicsCharacter)[[owner]].enabled = True
				state["stage"] = "controller"
		if state["stage"] == "controller" and np.linalg.norm(state["direction"][[0, 2]]) > 0:
			direction = state["direction"]
			world.get(MaterialTransform)[[owner]].rotation = Quaternion.from_y_rotation(np.arctan2(direction[0], direction[2]))
		state["fall"], state["recover"] = False, False

	def inputs(world, commands, dt):
		if state["stage"] == "controller":
			request = animation.playback.root_motion[0, :3, 3]
			speed = np.linalg.norm(request[[0, 2]]) / dt
			state["requested_speed"] = float(speed) if np.linalg.norm(state["direction"]) else 0
			physics.move_characters([owner], state["direction"] if state["return_velocity"] is None else [0, 0, 0], speed=speed)
			if state["jump"]: physics.jump_characters([owner])
			if state["return_velocity"] is not None:
				physics.character_commands([owner], pj.CharacterOp.IMPULSE, values=(*state["return_velocity"], 0))
				state["return_velocity"] = None
		else:
			physics.set_ragdoll_pose([owner], state["frozen"])
			if state["stage"] == "ragdoll" and state["launch"] is not None:
				handle = physics.ragdolls[int(owner)]
				physics.world.set_velocities(handle.bones, linear=np.broadcast_to(state["launch"], (len(handle.bones), 3)))
				handle.hit(1, (0, 3, 8))
				state["launch"] = None
		state["jump"] = False

	def finalize(world, commands, dt):
		state["foot_hits"][:] = False
		if state["stage"] == "controller":
			base = state["pose"]
			model = model_matrices(rig, *base)
			transform = world.get(MaterialTransform)[[owner]]
			frame = transform.position[0]
			turn = Quaternion.matrix_batch(transform.rotation)[0]
			feet = model[0, [9, 12], :3, 3] @ turn.T + frame
			hits = physics.raycast_batch(feet + [0, 0.5, 0], np.tile([0, -1, 0], (2, 1)), 1.5, exclude=[owner])
			valid = np.isfinite(hits.point).all(axis=1) & (hits.normal[:, 1] > 0.5) & world.get(CharacterState)[[owner]].grounded[0]
			targets = feet.copy()
			targets[valid] = hits.point[valid] + [0, 0.06, 0]
			targets = (targets - frame) @ turn
			poles = model[0, [7, 10], :3, 3] + [0, 0, 1]
			pose = post.evaluate({}, [], valid[None].astype(np.float32), poses=[base], goals=[(targets[:1], poles[:1]), (targets[1:], poles[1:])])
			state["final_pose"] = tuple(value.copy() for value in pose)
			matrices = skin_matrices(rig, model_matrices(rig, *pose))
			state["foot_hits"] = valid
			state["accepted_speed"] = float(np.linalg.norm(world.get(CharacterState)[[owner]].velocity[0, [0, 2]]))
		else:
			matrices = physics.ragdoll_skin([owner])
		state["batches"] = [PoseBatch(0, 0, np.array([owner]), matrices)]
		events = physics.events[physics.events["kind"] == pj.EventKind.TRIGGER]
		state["trigger_begins"] += int(np.count_nonzero(events["phase"] == pj.EventPhase.BEGIN))
		state["trigger_ends"] += int(np.count_nonzero(events["phase"] == pj.EventPhase.END))

	schedule = Scheduler(ecs)
	previous = ()
	for name, callback in (("pre_pose", prepare_pose), ("control", control), ("sync", physics.reconcile), ("targets", inputs), ("input", physics.prepare_inputs), ("step", physics.step), ("pull", physics.pull), ("events", physics.dispatch_events), ("final_pose", finalize)):
		schedule.add(name, callback, phase="fixed_update", after=previous)
		previous = name
	return ecs, owner, (floor, wall, step, sensor), physics, schedule, state, asset


def run_headless(frames=960):
	ecs, owner, environment, physics, schedule, state, asset = create_demo()
	with physics:
		fixed_dt = 1 / physics.world.settings.tick_hz
		for _ in range(frames):
			schedule.run(fixed_dt, fixed_steps=1, fixed_dt=fixed_dt)
			if not np.isfinite(state["batches"][0].matrices).all(): raise RuntimeError("nonfinite character palette")
			if physics.world.tick % 120 == 0:
				position = ecs.get(MaterialTransform)[[owner]].position[0]
				print(f"tick={physics.world.tick} stage={state['stage']} feet={position.round(2)} requested={state['requested_speed']:.2f} accepted={state['accepted_speed']:.2f}")
		print(f"trigger begins={state['trigger_begins']} ends={state['trigger_ends']}")


def run_scene(*, offscreen=False, frames=None):
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.Materials.MaterialRenderer import MaterialMesh, MaterialRenderer
	from graaal.Math import Camera
	from graaal.Animation.PosePalette import PosePalette
	from graaal.RenderContext import RenderContext
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer

	ecs, owner, environment, physics, schedule, state, asset = create_demo()
	title = "Character | WASD move, Space jump, R fall, G recover, T scripted cycle"
	with physics, viewing_target(title, (1000, 720), offscreen=offscreen) as output:
		cube = cube_vertices()
		shader = DefaultMaterialShader()
		mesh = MaterialMesh(skinned_boxes(asset, cube), shader, bounds=((0, 0, 0), (20, 20, 20)))
		ground = MaterialMesh(cube, shader)
		palette = SkinMatrixBuffer(len(asset.skeleton.parents))
		poses = PosePalette(palette)
		renderer = MaterialRenderer(meshes=[mesh, ground], skin_palette=palette)
		ecs.register(MaterialInstance)
		ecs.add(owner, MaterialInstance(mesh, {"base_color": (0.1, 0.65, 0.95, 1)}))
		for entity, color in zip(environment[:3], ((0.3, 0.3, 0.35, 1), (0.65, 0.3, 0.15, 1), (0.4, 0.55, 0.3, 1))):
			ecs.add(entity, MaterialInstance(ground, {"base_color": color}))
		tick_hz = physics.world.settings.tick_hz
		fixed_dt = 1 / tick_hz
		accumulator, displayed = 0.0, 0
		drawing = EcsMaterialAdapter(renderer, ecs, transform_source=lambda owners: physics.render_transforms(owners, alpha=accumulator * tick_hz))
		camera = Camera((4, 4, 10), (-0.5, 1, 0), (0, 1, 0), 45, near=0.05, far=100)
		def on_char(code):
			key = chr(code).lower()
			if key == "t": state["auto"] = True
			elif key in (" ", "r", "g"):
				state["auto"] = False
				state[{" ": "jump", "r": "fall", "g": "recover"}[key]] = True
			else: return False
			return True
		if not offscreen:
			import glfw

			RenderContext.subscribe_event("char", on_char)
		schedule.run(0, fixed_steps=1, fixed_dt=fixed_dt)
		if offscreen and frames is None: frames = 6
		while frames is None or displayed < frames:
			if not offscreen:
				if not RenderContext.window_loop(): break
				pressed = lambda key: glfw.get_key(RenderContext.window, key) == glfw.PRESS
				direction = np.array([int(pressed(glfw.KEY_D)) - int(pressed(glfw.KEY_A)), 0, int(pressed(glfw.KEY_S)) - int(pressed(glfw.KEY_W))])
				if np.any(direction): state["auto"] = False
				if not state["auto"]: state["direction"] = direction
			elapsed = 1 / 60 if offscreen else min(RenderContext.frame_time, 0.1)
			steps, accumulator = advance_fixed_steps(accumulator, elapsed, tick_hz)
			schedule.run(elapsed, fixed_steps=steps, fixed_dt=fixed_dt)
			batches = state["batches"]
			if state["stage"] != "controller": batches = [PoseBatch(0, 0, np.array([owner]), physics.ragdoll_skin([owner], alpha=accumulator * tick_hz))]
			poses.update(batches)
			drawing.sync()
			draw_frame(renderer, camera, output, "character")
			displayed += 1
		if offscreen:
			rgb = read_pixels(output).astype(np.int16)
			if not np.any(rgb[..., 2] > rgb[..., 0] + 30): raise RuntimeError("character scene rendered no blue character pixels")
			print(f"Character scene rendered {displayed} offscreen frames")


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
