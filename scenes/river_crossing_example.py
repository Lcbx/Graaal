"""A crowd walks, runs and swims across a river with a Jolt water sensor.

python main.py scenes/river_crossing_example.py
python -m scenes.river_crossing_example --headless
python -m scenes.river_crossing_example --offscreen --frames 2
"""

import argparse
from contextlib import contextmanager

import numpy as np

from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.SkeletalAnimation import Clip, model_matrices, sample_clip, subtree_mask
from graaal.ECS import ECS, Scheduler
from graaal.EcsAdapters.EcsAnimationGraph import EcsAnimationGraphAdapter, GraphPlayer
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from scenes.animation_dummy import load_dummy
from scenes.physics_utils import advance_fixed_steps


TICK_HZ = 60
FIXED_DT = 1 / TICK_HZ
HEADLESS_TICKS = 3600
LANES, ROWS = 12, 4
DUMMY_RADIUS, DUMMY_HEIGHT = 0.25, 1.8
DUMMY_CENTER = DUMMY_HEIGHT / 2
SWIM_RADIUS, SWIM_LENGTH = 0.30, 1.6
LAND_COLLIDER, WATER_SENSOR, SWIM_COLLIDER = range(3)
WATER_SENSOR_POSITION = (0, -0.7, 0)
UPRIGHT, HORIZONTAL = (0, 0, 0, 1), (np.sqrt(0.5), 0, 0, np.sqrt(0.5))
RIVER_WIDTH, RIVER_LENGTH = 12.0, 100.0
WATER_HEIGHT, SWIM_DEPTH = -0.14, 1.55
FINISH = 13.0
FADE_TIME = 0.35
YIELD_TIME, CLEARANCE = 0.8, 0.08
LAND_SPEED_RANGE = (1.3, 3.6)
RUN_THRESHOLD = 2.25
POSE_LOD_NEAR, POSE_LOD_FAR = 18.0, 22.0


def make_animation_graph(rig, named_clips):
	clips = dict(named_clips)
	# The graph inherits one playback rate per dummy. Retiming preserves each land gait's cadence.
	for name, nominal_speed in (("Walking", 1.5), ("Running", 4.0)):
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
	clips["Swimming"] = Clip(translation, forward.rotation, forward.scale, forward.fps)
	# Both poses share the forward stroke's cycle so their phases stay aligned during a pause.
	clips["Swimming_in_place"] = Clip(tread.translation, tread.rotation, tread.scale, (len(tread.translation) - 1) / forward.duration)

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
	return definition.compile(rig, clips), travel / forward.duration


def create_demo(rig, named_clips, *, row_count=ROWS):
	import pyjoltphysics as pj

	graph, authored_swim_speed = make_animation_graph(rig, named_clips)
	world = ECS()
	world.register(MaterialTransform, Collider, RigidBody, GraphPlayer)
	dummies = world.create(LANES * row_count)
	river = world.create()[0]
	world.add(dummies, MaterialTransform(), Collider(LAND_COLLIDER), RigidBody(motion=1, group=1, mask=3), GraphPlayer(0))
	world.add(river, MaterialTransform(position=WATER_SENSOR_POSITION), Collider(WATER_SENSOR, sensor=True), RigidBody(motion=0, group=2, mask=1))
	rng = np.random.default_rng(27)
	lanes, rows = np.tile(np.arange(LANES), row_count), np.repeat(np.arange(row_count), LANES)
	start = np.zeros((len(dummies), 3), dtype=np.float32)
	start[:, 0] = (lanes - (LANES - 1) / 2) * 1.9 + rng.uniform(-0.22, 0.22, len(dummies))
	start[:, 1] = DUMMY_CENTER
	start[:, 2] = -10.0 - rows * 2.5 - rng.uniform(0, 1.8, len(dummies))
	nearby = np.abs(start[:, None, 0] - start[None, :, 0]) < 2 * max(DUMMY_RADIUS, SWIM_RADIUS)
	np.fill_diagonal(nearby, False)
	land_speed = rng.uniform(*LAND_SPEED_RANGE, len(dummies))
	playback_rate = land_speed / RUN_THRESHOLD
	swim_speed = authored_swim_speed * playback_rate
	# Share the fraction of cruising speed between movement and the swim blend.
	pace = np.ones(len(dummies))
	accepted_pace = pace.copy()
	pause_left = np.zeros(len(dummies))
	waiting_at_finish = np.zeros(len(dummies), dtype=bool)
	swimming = np.zeros(len(dummies), dtype=bool)
	transforms = world.get(MaterialTransform)
	colliders = world.get(Collider)
	transforms[dummies].position = start
	parameters = np.tile(graph.parameter_defaults, (len(dummies), 1))
	parameters[:, graph.parameter_columns["land_speed"]] = land_speed
	in_water = parameters[:, graph.parameter_columns["in_water"]]
	moving = parameters[:, graph.parameter_columns["moving"]]
	swim_blend = parameters[:, graph.parameter_columns["swim_blend"]]
	animation = EcsAnimationGraphAdapter(world, 0, graph)
	animation.advance(0)
	rows = np.searchsorted(dummies, animation.owners)
	initial_states = np.where(land_speed < RUN_THRESHOLD, graph.state_ids["walk"], graph.state_ids["run"])
	durations = np.array([graph.clips[graph.state_clips[state]].duration for state in (graph.state_ids["walk"], graph.state_ids["run"])])
	initial_times = rng.random(len(dummies)) * np.where(land_speed < RUN_THRESHOLD, durations[0], durations[1])
	animation.playback.reset(initial_states[rows], initial_times[rows], speeds=playback_rate[rows])
	shapes = {
		LAND_COLLIDER: pj.Shape.capsule(DUMMY_RADIUS, DUMMY_HEIGHT - 2 * DUMMY_RADIUS),
		WATER_SENSOR: pj.Shape.box((RIVER_LENGTH, 4, RIVER_WIDTH)),
		SWIM_COLLIDER: pj.Shape.capsule(SWIM_RADIUS, SWIM_LENGTH - 2 * SWIM_RADIUS),
	}
	physics = EcsPhysicsAdapter(world, shapes, settings=pj.WorldSettings(tick_hz=TICK_HZ, threads=1), interpolate=True)
	water_filter = physics.query_filter(group=2, mask=1)

	def move_crowd(world, commands, dt):
		pose = transforms[dummies]
		position = pose.position
		z = position[:, 2]
		# Reserve the longer footprint on approach so the collider can switch with room to spare.
		near_water = (np.abs(z) <= RIVER_WIDTH / 2 + SWIM_LENGTH) | swimming
		half_length = np.where(near_water, SWIM_LENGTH / 2, DUMMY_RADIUS)
		# The kinematic bodies follow targets, so keep their projected capsules apart along each lane.
		separation = z[None, :] - z[:, None]
		gap = separation - half_length[:, None] - half_length[None, :] - CLEARANCE
		gap_ahead = np.min(np.where(nearby & (separation > 0), gap, np.inf), axis=1)
		cruise_speed = np.where(swimming, swim_speed, land_speed)
		pause_left[:] = np.maximum(pause_left - dt, 0)
		braking_distance = cruise_speed * (0.5 * FADE_TIME * pace**2 + dt)
		pause_left[(pause_left == 0) & (gap_ahead <= braking_distance)] = YIELD_TIME
		step = dt / FADE_TIME
		pace[:] += np.clip((pause_left == 0) - pace, -step, step)
		travel = np.minimum(np.minimum(cruise_speed * pace * dt, np.maximum(gap_ahead, 0)), FINISH - z)
		accepted_pace[:] = travel / (cruise_speed * dt)
		next_z = z + travel
		spawn_gap = np.abs(start[:, None, 2] - next_z[None, :]) - DUMMY_RADIUS - half_length[None, :]
		spawn_blocked = np.any(nearby & (spawn_gap < CLEARANCE), axis=1)
		waiting_at_finish[:] = (next_z >= FINISH) & spawn_blocked
		recycle = (next_z >= FINISH) & ~spawn_blocked
		position[:, 2] = next_z
		center = DUMMY_CENTER - swimming * SWIM_DEPTH
		position[:, 1] += np.clip(center - position[:, 1], -SWIM_DEPTH * step, SWIM_DEPTH * step)
		if np.any(recycle):
			# Both the kinematic target and the teleport command apply during this tick.
			position[recycle] = start[recycle]
			physics.teleport(dummies[recycle], start[recycle])
			pause_left[recycle] = 0
			pace[recycle] = 1
		pose.position = position
		wanted = np.where(swimming, SWIM_COLLIDER, LAND_COLLIDER)
		changed = colliders[dummies].shape != wanted
		if np.any(changed):
			orientations = np.where(swimming[changed, None], HORIZONTAL, UPRIGHT)
			colliders[dummies[changed]].assign(shape=wanted[changed], rotation=orientations)

	def update_water_overlap(world, commands, dt):
		# Collider swaps replace native bodies. Query the sensor volume against the live bodies.
		overlaps = physics.overlap(WATER_SENSOR, WATER_SENSOR_POSITION, filter=water_filter)
		swimming[:] = np.isin(dummies, physics.entities_of(overlaps))

	def advance_animation(world, commands, dt):
		in_water[:] = swimming
		moving[:] = (pause_left == 0) & ~waiting_at_finish
		swim_blend[:] = accepted_pace
		rows = np.searchsorted(dummies, animation.entities())
		animation.advance(dt, parameters[rows])

	def current_pose():
		return animation.playback.pose(parameters[np.searchsorted(dummies, animation.owners)])

	schedule = Scheduler(world)
	schedule.add("move", move_crowd, phase="fixed_update")
	schedule.add("physics_sync", physics.reconcile, phase="fixed_update", after="move")
	schedule.add("physics_input", physics.prepare_inputs, phase="fixed_update", after="physics_sync")
	schedule.add("physics_step", physics.step, phase="fixed_update", after="physics_input")
	schedule.add("physics_output", physics.pull, phase="fixed_update", after="physics_step")
	schedule.add("physics_events", physics.dispatch_events, phase="fixed_update", after="physics_output")
	schedule.add("water_overlap", update_water_overlap, phase="fixed_update", after="physics_events")
	schedule.add("animation", advance_animation, phase="fixed_update", after="water_overlap")
	return world, dummies, physics, schedule, animation, parameters, current_pose


def add_landscape(world, ground, water, *, bank_length=40):
	from graaal.EcsAdapters.EcsMaterials import MaterialInstance

	grass = {"base_color": (0.18, 0.31, 0.11, 1)}
	sand = {"base_color": (0.56, 0.44, 0.26, 1)}
	bed = {"base_color": (0.09, 0.27, 0.27, 1)}
	river = {"base_color": (0.07, 0.42, 0.54, 0.62), "roughness": 0.25, "alpha_mode": 2}
	ripple = {"base_color": (0.20, 0.62, 0.68, 0.34), "alpha_mode": 2}

	def box(mesh, material, position, scale):
		world.add(world.create()[0], MaterialInstance(mesh, material), MaterialTransform(position=position, scale=scale))

	for bank in (-1, 1):
		box(ground, grass, (0, -1, bank * (RIVER_WIDTH / 2 + 2 + bank_length / 2)), (RIVER_LENGTH, 2, bank_length))
		box(ground, sand, (0, -1, bank * (RIVER_WIDTH / 2 + 1)), (RIVER_LENGTH, 2, 2))
	box(ground, bed, (0, -2.1, 0), (RIVER_LENGTH, 1, RIVER_WIDTH))
	box(water, river, (0, WATER_HEIGHT - 0.03, 0), (RIVER_LENGTH, 0.06, RIVER_WIDTH))
	rng = np.random.default_rng(4)
	for x, z in rng.uniform((-RIVER_LENGTH / 2 + 3, -RIVER_WIDTH / 2 + 0.2), (RIVER_LENGTH / 2 - 3, RIVER_WIDTH / 2 - 0.2), (100, 2)):
		box(water, ripple, (x, WATER_HEIGHT + 0.015, z), (rng.uniform(0.5, 3), 0.01, rng.uniform(0.04, 0.12)))


def crowd_eligibility(graph, primitive):
	from graaal.Animation.BakedClipPool import BakedClipPool
	from graaal.Animation.CrowdGraphValidation import validate_crowd_graph

	pool = BakedClipPool.from_graph(graph)
	rates = np.array([LAND_SPEED_RANGE[0], RUN_THRESHOLD, LAND_SPEED_RANGE[1]]) / RUN_THRESHOLD
	error_budget = 0.02 * float(np.ptp(primitive.vertices["position"][:, 1]))
	return validate_crowd_graph(graph, pool, error_budget=error_budget, phase_samples=32, playback_rates=rates)


@contextmanager
def river_rendering(world, dummies, physics, rig, primitive, render_alpha):
	from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter, MaterialInstance
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import MaterialMesh, MaterialRenderer
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
	from scenes.animation_dummy import dummy_material_mesh
	from scenes.physics_geometry import cube_vertices

	mesh, textures = dummy_material_mesh(primitive, include_white=True)
	cube = cube_vertices()
	ground = MaterialMesh(cube, DefaultMaterialShader(default_material={"texture_layer": 1}))
	water = MaterialMesh(cube, DefaultMaterialShader(transparent=True, default_material={"texture_layer": 1}))
	world.register(MaterialInstance)
	world.add(dummies, MaterialInstance(mesh))
	spawn_z = world.get(MaterialTransform)[dummies].position[:, 2].min()
	add_landscape(world, ground, water, bank_length=max(40, -spawn_z - RIVER_WIDTH / 2))
	bones = len(rig.parents)
	palette = SkinMatrixBuffer(len(dummies) * bones)
	try:
		for dummy in dummies:
			palette.acquire(dummy, bones)
		renderer = MaterialRenderer(meshes=[mesh, ground, water], skin_palette=palette, base_textures=textures.view(dimension="2d-array"))

		def render_transforms(owners):
			position, rotation, scale = physics.render_transforms(owners, alpha=render_alpha())
			position[np.isin(owners, dummies), 1] -= DUMMY_CENTER
			return position, rotation, scale

		drawing = EcsMaterialAdapter(renderer, world, transform_source=render_transforms)
		yield drawing, renderer, palette
	finally:
		palette.close()


def run_scene(*, offscreen=False, frames=None, row_count=ROWS, crowd_poses=True):
	from graaal.Animation.AnimationLod import DistancePoseLod
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext
	from graaal.RenderPasses.CrowdPose import GraphPosePass
	from scenes.physics_viewing_utils import has_blue_pixels, read_pixels, viewing_target

	rig, primitive, clips = load_dummy(swimming=True)
	world, dummies, physics, schedule, animation, parameters, _ = create_demo(rig, clips, row_count=row_count)
	accumulator, displayed = 0.0, 0
	with (
		physics,
		viewing_target("River crossing", window_size=(1100, 760), offscreen=offscreen) as output,
		river_rendering(world, dummies, physics, rig, primitive, lambda: accumulator * TICK_HZ) as (drawing, renderer, palette),
	):
		transforms = world.get(MaterialTransform)
		eligibility = crowd_eligibility(animation.playback.graph, primitive) if crowd_poses else None
		poses = GraphPosePass(palette, len(dummies), eligibility=eligibility)
		try:
			lod = DistancePoseLod(POSE_LOD_NEAR, POSE_LOD_FAR) if crowd_poses else None
			gpu_mask = None
			camera = Camera((15, 10, -5), (0, 0, -2), (0, 1, 0), 50, near=0.1, far=120)
			poses_dirty = True
			if offscreen and frames is None: frames = 1
			while frames is None or displayed < frames:
				if not offscreen and not RenderContext.window_loop(): break
				elapsed = FIXED_DT if offscreen else min(RenderContext.frame_time, 0.1)
				steps, accumulator = advance_fixed_steps(accumulator, elapsed, TICK_HZ)
				if steps:
					schedule.run(elapsed, fixed_steps=steps, fixed_dt=FIXED_DT)
					poses_dirty = True
				size = output.size[:2] if output is not None else RenderContext.window_dimensions
				if min(size) > 0:
					commands = RenderContext.commands("river_crossing")
					owners, playback = animation.owners, animation.playback
					rows = np.searchsorted(dummies, owners)
					values = parameters[rows]
					if lod is not None:
						mask = lod.update(owners, transforms[owners].position, camera.position, hold=playback.blending(values))
						poses_dirty |= not np.array_equal(gpu_mask, mask)
						gpu_mask = mask
					if poses_dirty:
						poses.record(commands, owners, playback, values, gpu_mask=gpu_mask)
						poses_dirty = False
					drawing.sync()
					renderer.record(commands, size, camera, output.color_attachment() if output is not None else RenderContext.screen())
					RenderContext.submit(commands.finish())
				displayed += 1
			if offscreen:
				if not has_blue_pixels(read_pixels(output)): raise RuntimeError("River crossing rendered no water pixels")
				print(f"River crossing rendered {displayed} offscreen frames")
		finally:
			poses.close()


def run_headless(ticks=HEADLESS_TICKS, *, row_count=ROWS):
	rig, _, clips = load_dummy(swimming=True)
	world, dummies, physics, schedule, animation, _, _ = create_demo(rig, clips, row_count=row_count)
	transforms = world.get(MaterialTransform)
	swim = animation.playback.graph.state_ids["swim"]
	previous_z = transforms[dummies].position[:, 2]
	swam = np.zeros(len(dummies), dtype=bool)
	paused = swam.copy()
	returned = swam.copy()
	with physics:
		for _ in range(ticks):
			schedule.run(FIXED_DT, fixed_steps=1, fixed_dt=FIXED_DT)
			z = transforms[dummies].position[:, 2]
			swam[np.searchsorted(dummies, animation.owners)] |= animation.playback.states == swim
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
	parser.add_argument("--rows", type=int, default=ROWS, help="rows of dummies per lane")
	parser.add_argument("--cpu-poses", action="store_true", help="evaluate all rendered poses on the CPU")
	args = parser.parse_args()
	if args.frames is not None and args.frames <= 0: parser.error("--frames must be positive")
	if args.rows < 1: parser.error("--rows must be positive")
	if args.headless: run_headless(HEADLESS_TICKS if args.frames is None else args.frames, row_count=args.rows)
	else: run_scene(offscreen=args.offscreen, frames=args.frames, row_count=args.rows, crowd_poses=not args.cpu_poses)
