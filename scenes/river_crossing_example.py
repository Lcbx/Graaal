"""A crowd walks, runs and swims across a river with a Jolt water sensor.

python main.py scenes/river_crossing_example.py
python -m scenes.river_crossing_example --headless
python -m scenes.river_crossing_example --offscreen --frames 2
"""

import argparse

import numpy as np

from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.AnimationGraphPlayback import AnimationGraphPlayback
from graaal.Animation.SkeletalAnimation import Clip, model_matrices, sample_clip, skin_matrices, subtree_mask
from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from scenes.animation_dummy import load_dummy
from scenes.physics_utils import advance_fixed_steps


TICK_HZ = 60
FIXED_DT = 1 / TICK_HZ
LANES, ROWS = 12, 4
RIVER_WIDTH, RIVER_LENGTH = 12.0, 100.0
WATER_HEIGHT, SWIM_DEPTH = -0.14, 1.55
FINISH = 13.0
FADE_TIME = 0.35
LAND_SPEED_RANGE = (1.3, 3.6)
RUN_THRESHOLD = 2.25
SWIM_SPEED_FACTOR = 0.6


def make_animation_graph(rig, named_clips):
	clips = dict(named_clips)
	# The graph inherits one playback rate per dummy. Retiming preserves each gait's cadence.
	for name, nominal_speed in (("Walking", 1.5), ("Running", 4.0), ("Swimming_in_place", 1.45 / SWIM_SPEED_FACTOR)):
		clip = clips[name]
		clips[name] = Clip(clip.translation, clip.rotation, clip.scale, clip.fps * RUN_THRESHOLD / nominal_speed)
	# Keep the bind-pose stance and lower both arms using a mid-stride walk sample.
	arm_mask = np.logical_or(subtree_mask(rig, 6), subtree_mask(rig, 25))
	walk = clips["Walking"]
	_, walk_rotation, _ = sample_clip(walk, [walk.duration / 2])
	rest = clips["Rest pose"]
	clips["Rest pose"] = Clip(rest.translation, np.where(arm_mask[None, :, None], walk_rotation, rest.rotation), rest.scale, rest.fps)

	tread, forward = clips["Swimming_in_place"], clips["Swimming"]
	translation = forward.translation.copy()
	# The kinematic body supplies forward travel. Keep the cyclic hip motion in the pose.
	travel = translation[-1, 0, 2] - translation[0, 0, 2]
	translation[:, 0, 2] -= np.linspace(0, travel, len(translation), dtype=np.float32)
	# Keep average head height consistent at the waterline as the torso pitches forward.
	head_height = [model_matrices(rig, clip.translation, clip.rotation, clip.scale)[:, 5, 1, 3].mean() for clip in (tread, forward)]
	translation[:, 0, 1] += head_height[0] - head_height[1]
	clips["Swimming"] = Clip(translation, forward.rotation, forward.scale, (len(translation) - 1) / tread.duration)

	conditions = {
		"stand": [("in_water", "==", 0), ("moving", "==", 0)],
		"walk": [("in_water", "==", 0), ("moving", "==", 1), ("land_speed", "<", RUN_THRESHOLD)],
		"run": [("in_water", "==", 0), ("moving", "==", 1), ("land_speed", ">=", RUN_THRESHOLD)],
		"swim": [("in_water", "==", 1)],
	}
	definition = AnimationGraphDefinition({
		"version": 1,
		"parameters": {"in_water": 0, "moving": 1, "land_speed": 0, "swim_blend": 0},
		"initial_state": "stand",
		"states": {"stand": {"clip": "Rest pose"}, "walk": {"clip": "Walking"}, "run": {"clip": "Running"}, "swim": {"blend_space": "swim"}},
		"blend_spaces": {"swim": {"parameter": "swim_blend", "points": [{"position": 0, "clip": "Swimming_in_place"}, {"position": 1, "clip": "Swimming"}]}},
		"transitions": [
			{"source": source, "target": target, "duration": FADE_TIME,
				"conditions": [{"parameter": name, "comparison": op, "threshold": value} for name, op, value in rules]}
			for source in conditions for target, rules in conditions.items() if source != target
		],
		"sample_slots": ["locomotion"],
		"tasks": [{"name": "locomotion", "op": "sample", "slot": "locomotion"}],
	})
	return definition.compile(rig, clips)


def create_demo(rig, named_clips):
	import pyjoltphysics as pj

	graph = make_animation_graph(rig, named_clips)
	world = ECS()
	world.register(MaterialTransform, Collider, RigidBody)
	dummies = world.create(LANES * ROWS)
	river = world.create()[0]
	world.add(dummies, MaterialTransform(), Collider(0), RigidBody(motion=1, group=1, mask=2))
	world.add(river, MaterialTransform(position=(0, -0.7, 0)), Collider(1, sensor=True), RigidBody(motion=0, group=2, mask=1))
	rng = np.random.default_rng(27)
	lanes, rows = np.tile(np.arange(LANES), ROWS), np.repeat(np.arange(ROWS), LANES)
	start = np.zeros((len(dummies), 3), dtype=np.float32)
	start[:, 0] = (lanes - (LANES - 1) / 2) * 1.9 + rng.uniform(-0.22, 0.22, len(dummies))
	start[:, 2] = -10.0 - rows * 2.5 - rng.uniform(0, 1.8, len(dummies))
	land_speed = rng.uniform(*LAND_SPEED_RANGE, len(dummies))
	swim_speed = SWIM_SPEED_FACTOR * land_speed
	forward_weight = np.interp(land_speed, LAND_SPEED_RANGE, (0.65, 1.0))
	until_pause = rng.uniform(0.8, 6.0, len(dummies))
	pause_left = np.zeros(len(dummies))
	swimming = np.zeros(len(dummies), dtype=bool)
	transforms = world.get(MaterialTransform)
	transforms[dummies].position = start
	parameters = np.tile(graph.parameter_defaults, (len(dummies), 1))
	parameters[:, graph.parameter_columns["land_speed"]] = land_speed
	in_water = parameters[:, graph.parameter_columns["in_water"]]
	moving = parameters[:, graph.parameter_columns["moving"]]
	swim_blend = parameters[:, graph.parameter_columns["swim_blend"]]
	animation = AnimationGraphPlayback(graph, len(dummies))
	initial_states = np.where(land_speed < RUN_THRESHOLD, graph.state_ids["walk"], graph.state_ids["run"])
	durations = np.array([graph.clips[graph.state_clips[state]].duration for state in (graph.state_ids["walk"], graph.state_ids["run"])])
	initial_times = rng.random(len(dummies)) * np.where(land_speed < RUN_THRESHOLD, durations[0], durations[1])
	animation.reset(initial_states, initial_times, speeds=land_speed / RUN_THRESHOLD)
	physics = EcsPhysicsAdapter(world, {0: pj.Shape.sphere(0.17), 1: pj.Shape.box((RIVER_LENGTH, 4, RIVER_WIDTH))},
		settings=pj.WorldSettings(tick_hz=TICK_HZ, threads=1), interpolate=True)

	def move_crowd(world, commands, dt):
		pause_left[:] = np.maximum(pause_left - dt, 0)
		until_pause[:] -= dt * (pause_left == 0)
		start_pause = until_pause <= 0
		pause_left[start_pause] = rng.uniform(0.5, 1.6, start_pause.sum())
		until_pause[start_pause] = rng.uniform(2.0, 7.0, start_pause.sum())
		pose = transforms[dummies]
		position = pose.position
		next_z = position[:, 2] + np.where(swimming, swim_speed, land_speed) * (pause_left == 0) * dt
		finished = next_z >= FINISH
		position[:, 2] = next_z
		depth = np.where(swimming, -SWIM_DEPTH, 0.0)
		step = SWIM_DEPTH * dt / FADE_TIME
		position[:, 1] += np.clip(depth - position[:, 1], -step, step)
		if np.any(finished):
			# Both the kinematic target and the teleport command apply during this tick.
			position[finished] = start[finished]
			physics.teleport(dummies[finished], start[finished])
			until_pause[finished] = rng.uniform(0.8, 6.0, finished.sum())
		pose.position = position

	def update_water_overlap(world, commands, events):
		other = np.where(events["entity_a"] == river, events["entity_b"], events["entity_a"])
		owners, last = np.unique(other[::-1], return_index=True)
		active = events["phase"][::-1][last] != pj.EventPhase.END
		swimming[np.isin(dummies, owners[active])] = True
		swimming[np.isin(dummies, owners[~active])] = False

	def advance_animation(world, commands, dt):
		in_water[:] = swimming
		moving[:] = pause_left == 0
		step = dt / FADE_TIME
		swim_blend[:] += np.clip(moving * forward_weight - swim_blend, -step, step)
		animation.advance(dt, parameters)

	physics.subscribe(update_water_overlap, kind=pj.EventKind.TRIGGER, entity=river)
	schedule = Scheduler(world)
	schedule.add("move", move_crowd, phase="fixed_update")
	schedule.add("physics_sync", physics.reconcile, phase="fixed_update", after="move")
	schedule.add("physics_input", physics.prepare_inputs, phase="fixed_update", after="physics_sync")
	schedule.add("physics_step", physics.step, phase="fixed_update", after="physics_input")
	schedule.add("physics_output", physics.pull, phase="fixed_update", after="physics_step")
	schedule.add("physics_events", physics.dispatch_events, phase="fixed_update", after="physics_output")
	schedule.add("animation", advance_animation, phase="fixed_update", after="physics_events")
	return world, dummies, physics, schedule, animation, parameters


def add_landscape(world, ground, water):
	from graaal.EcsAdapters.EcsMaterials import MaterialInstance

	grass = {"base_color": (0.18, 0.31, 0.11, 1)}
	sand = {"base_color": (0.56, 0.44, 0.26, 1)}
	bed = {"base_color": (0.09, 0.27, 0.27, 1)}
	river = {"base_color": (0.07, 0.42, 0.54, 0.62), "roughness": 0.25, "alpha_mode": 2}
	ripple = {"base_color": (0.20, 0.62, 0.68, 0.34), "alpha_mode": 2}

	def box(mesh, material, position, scale):
		world.add(world.create()[0], MaterialInstance(mesh, material), MaterialTransform(position=position, scale=scale))

	for bank in (-1, 1):
		box(ground, grass, (0, -1, bank * (RIVER_WIDTH / 2 + 22)), (RIVER_LENGTH, 2, 40))
		box(ground, sand, (0, -1, bank * (RIVER_WIDTH / 2 + 1)), (RIVER_LENGTH, 2, 2))
	box(ground, bed, (0, -2.1, 0), (RIVER_LENGTH, 1, RIVER_WIDTH))
	box(water, river, (0, WATER_HEIGHT - 0.03, 0), (RIVER_LENGTH, 0.06, RIVER_WIDTH))
	rng = np.random.default_rng(4)
	for x, z in rng.uniform((-RIVER_LENGTH / 2 + 3, -RIVER_WIDTH / 2 + 0.2), (RIVER_LENGTH / 2 - 3, RIVER_WIDTH / 2 - 0.2), (100, 2)):
		box(water, ripple, (x, WATER_HEIGHT + 0.015, z), (rng.uniform(0.5, 3), 0.01, rng.uniform(0.04, 0.12)))


def run_scene(*, offscreen=False, frames=None):
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import MaterialMesh, MaterialRenderer
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
	from scenes.animation_dummy import dummy_material_mesh
	from scenes.physics_geometry import cube_vertices
	from scenes.physics_viewing_utils import draw_frame, has_blue_pixels, read_pixels, viewing_target

	rig, primitive, clips = load_dummy(swimming=True)
	world, dummies, physics, schedule, animation, parameters = create_demo(rig, clips)
	with physics, viewing_target("River crossing", window_size=(1100, 760), offscreen=offscreen) as output:
		mesh, textures = dummy_material_mesh(primitive, include_white=True)
		cube = cube_vertices()
		ground = MaterialMesh(cube, DefaultMaterialShader(default_material={"texture_layer": 1}))
		water = MaterialMesh(cube, DefaultMaterialShader(transparent=True, default_material={"texture_layer": 1}))
		world.register(MaterialInstance)
		world.add(dummies, MaterialInstance(mesh))
		add_landscape(world, ground, water)
		bones = len(rig.parents)
		palette = SkinMatrixBuffer(len(dummies) * bones)
		try:
			for dummy in dummies:
				palette.acquire(dummy, bones)
			renderer = MaterialRenderer(meshes=[mesh, ground, water], skin_palette=palette, base_textures=textures.view(dimension="2d-array"))
			camera = Camera((15, 10, -5), (0, 0, -2), (0, 1, 0), 50, near=0.1, far=120)
			accumulator, displayed = 0.0, 0
			drawing = EcsMaterialAdapter(renderer, world, transform_source=lambda owners: physics.render_transforms(owners, alpha=accumulator * TICK_HZ))
			if offscreen and frames is None: frames = 1
			while frames is None or displayed < frames:
				if not offscreen and not RenderContext.window_loop(): break
				elapsed = FIXED_DT if offscreen else min(RenderContext.frame_time, 0.1)
				steps, accumulator = advance_fixed_steps(accumulator, elapsed, TICK_HZ)
				if steps:
					schedule.run(elapsed, fixed_steps=steps, fixed_dt=FIXED_DT)
				if steps or displayed == 0:
					model = model_matrices(rig, *animation.pose(parameters))
					palette.write(dummies, skin_matrices(rig, model))
				drawing.sync()
				draw_frame(renderer, camera, output, "river_crossing")
				displayed += 1
			if offscreen:
				if not has_blue_pixels(read_pixels(output)): raise RuntimeError("River crossing rendered no water pixels")
				print(f"River crossing rendered {displayed} offscreen frames")
		finally:
			palette.close()


def run_headless(ticks=2400):
	rig, _, clips = load_dummy(swimming=True)
	world, dummies, physics, schedule, animation, _ = create_demo(rig, clips)
	transforms = world.get(MaterialTransform)
	swim = animation.graph.state_ids["swim"]
	previous_z = transforms[dummies].position[:, 2]
	swam = np.zeros(len(dummies), dtype=bool)
	paused = swam.copy()
	returned = swam.copy()
	with physics:
		for _ in range(ticks):
			schedule.run(FIXED_DT, fixed_steps=1, fixed_dt=FIXED_DT)
			z = transforms[dummies].position[:, 2]
			swam |= animation.states == swim
			paused |= z == previous_z
			returned |= z < previous_z
			previous_z = z
	print(f"ticks={ticks} dummies={len(dummies)} swam={swam.sum()} paused={paused.sum()} returned={returned.sum()}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("--headless", action="store_true")
	mode.add_argument("--offscreen", action="store_true")
	parser.add_argument("--frames", type=int)
	args = parser.parse_args()
	if args.frames is not None and args.frames <= 0: parser.error("--frames must be positive")
	if args.headless: run_headless(2400 if args.frames is None else args.frames)
	else: run_scene(offscreen=args.offscreen, frames=args.frames)
