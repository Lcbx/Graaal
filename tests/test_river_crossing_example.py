import numpy as np
import pytest

from graaal.Animation.AnimationGraphPlayback import state_sample
from graaal.EcsAdapters.EcsPhysics import Collider
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from scenes.animation_dummy import load_dummy
from scenes.river_crossing_example import CLEARANCE, DUMMY_RADIUS, FADE_TIME, FINISH, FIXED_DT, HEADLESS_TICKS, LAND_COLLIDER, RIVER_WIDTH, SWIM_COLLIDER, SWIM_LENGTH, SWIM_RADIUS, create_demo


def test_crowd_swims_between_sensor_shores_and_teleports_to_spawn():
	pj = pytest.importorskip("pyjoltphysics")
	pytest.importorskip("pygltflib")
	rig, _, clips = load_dummy(swimming=True)
	world, dummies, physics, schedule, animation, parameters, _ = create_demo(rig, clips)
	np.testing.assert_array_equal(animation.owners, dummies)
	graph = animation.playback.graph
	swim = graph.state_ids["swim"]
	pace = graph.parameter_columns["swim_blend"]
	water = graph.parameter_columns["in_water"]
	moving = graph.parameter_columns["moving"]
	authored_swim = dict(clips)["Swimming"]
	travel = authored_swim.translation[-1, 0, 2] - authored_swim.translation[0, 0, 2]
	cruise_speed = travel / authored_swim.duration * animation.playback.speeds
	transforms = world.get(MaterialTransform)
	colliders = world.get(Collider)
	start = transforms[dummies].position
	previous_z = start[:, 2]
	nearby = np.abs(start[:, None, 0] - start[None, :, 0]) < 2 * max(DUMMY_RADIUS, SWIM_RADIUS)
	np.fill_diagonal(nearby, False)
	yield_distance = SWIM_LENGTH + CLEARANCE + parameters[:, graph.parameter_columns["land_speed"]] * (FADE_TIME / 2 + FIXED_DT)
	assert set(animation.playback.states) == {graph.state_ids["walk"], graph.state_ids["run"]}
	forward = graph.clips[graph.clip_ids["Swimming"]]
	tread = graph.clips[graph.clip_ids["Swimming_in_place"]]
	assert forward.duration == pytest.approx(authored_swim.duration)
	assert forward.duration == pytest.approx(tread.duration)
	np.testing.assert_allclose(forward.translation[0, 0, 2], forward.translation[-1, 0, 2], atol=1e-6)
	blend_parameters = parameters[:2].copy()
	blend_parameters[:, pace] = (0, 0.8)
	ids, _, weights = state_sample(graph, swim, np.zeros(2), blend_parameters)
	assert ids[1].tolist() == [graph.clip_ids[name] for name in ("Swimming_in_place", "Swimming")]
	np.testing.assert_allclose(weights, [[1, 0], [0.2, 0.8]])
	entered = np.zeros(len(dummies), dtype=bool)
	exited, swam, paused, treaded, returned = (entered.copy() for _ in range(5))
	was_in_water = entered.copy()
	previous_blend = parameters[:, pace].copy()
	was_moving = np.ones(len(dummies), dtype=bool)
	checked_swim_capsule = False
	with physics:
		for tick in range(HEADLESS_TICKS):
			schedule.run(FIXED_DT, fixed_steps=1, fixed_dt=FIXED_DT)
			position = transforms[dummies].position
			z = position[:, 2]
			if tick == 0:
				uids = dict(physics.body_uids)
				upper_body = physics.overlap(pj.Shape.sphere(0.05), position[0] + (0, 0.65, 0))
				assert dummies[0] in physics.entities_of(upper_body)
			shape_ids = colliders[dummies].shape
			np.testing.assert_array_equal(shape_ids, np.where(was_in_water, SWIM_COLLIDER, LAND_COLLIDER))
			half_length = np.where(was_in_water, SWIM_LENGTH / 2, DUMMY_RADIUS)
			gap = np.abs(z[None, :] - z[:, None]) - half_length[:, None] - half_length[None, :]
			assert np.all(gap[nearby] >= CLEARANCE - 1e-5)
			in_water = parameters[:, water] != 0
			begin = ~was_in_water & in_water
			end = was_in_water & ~in_water
			for seen, at_shore, shore, extent in ((entered, begin, -RIVER_WIDTH / 2, DUMMY_RADIUS), (exited, end, RIVER_WIDTH / 2, SWIM_LENGTH / 2)):
				assert np.all(np.abs(z[at_shore] - shore) < extent + 0.1)
				seen |= at_shore
			if not checked_swim_capsule and np.any(was_in_water):
				row = np.flatnonzero(was_in_water)[0]
				swim_body = physics.overlap(pj.Shape.sphere(0.05), position[row] + (0, 0, 0.65))
				assert dummies[row] in physics.entities_of(swim_body)
				assert physics.body_uids[int(dummies[row])] != uids[int(dummies[row])]
				checked_swim_capsule = True
			is_moving = parameters[:, moving] != 0
			yielding = was_moving & ~is_moving & (z < FINISH)
			ahead = previous_z[None, :] - previous_z[:, None]
			assert np.all(np.any(nearby & (ahead > 0) & (ahead <= yield_distance[:, None]), axis=1)[yielding])
			stayed_in_water = was_in_water & in_water
			blend = parameters[:, pace]
			water_speed = (z - previous_z)[stayed_in_water] / FIXED_DT
			np.testing.assert_allclose(water_speed, (cruise_speed * blend)[stayed_in_water], atol=4e-5, rtol=0)
			assert np.all(np.abs(blend - previous_blend)[stayed_in_water] <= FIXED_DT / FADE_TIME + 1e-6)
			in_swim = animation.playback.states == swim
			stopped = z == previous_z
			swam |= in_swim & stayed_in_water & (blend == 1)
			paused |= stopped & (z < FINISH)
			treaded |= in_swim & stopped & (blend == 0)
			recycled = z < previous_z
			if np.any(recycled):
				assert np.all(entered[recycled] & exited[recycled] & swam[recycled])
				assert np.all(previous_z[recycled] > FINISH - 0.1)
				np.testing.assert_array_equal(position[recycled], start[recycled])
				render_position, _, _ = physics.render_transforms(dummies[recycled], alpha=0)
				np.testing.assert_allclose(render_position, start[recycled], atol=1e-5, rtol=0)
			returned |= recycled
			previous_z = z
			previous_blend = blend.copy()
			was_in_water = in_water
			was_moving = is_moving
			if np.all(returned): break
	assert checked_swim_capsule and np.all(returned) and np.any(treaded)
	assert 0 < np.count_nonzero(paused) < len(dummies)


def test_river_routes_gpu_and_cpu_poses_through_the_same_palette():
	pytest.importorskip("pyjoltphysics")
	pytest.importorskip("pygltflib")
	pytest.importorskip("wgpu")
	from graaal.Animation.AnimationLod import DistancePoseLod
	from graaal.Animation.BakedClipPool import BakedClipPool
	from graaal.Animation.CrowdGraphSamples import CrowdGraphRouter
	from graaal.Animation.CrowdGraphValidation import validate_crowd_graph
	from graaal.Animation.SkeletalAnimation import model_matrices, skin_matrices
	from graaal.RenderContext import RenderContext
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
	from graaal.RenderPasses.CrowdPose import GraphPosePass
	from scenes.physics_viewing_utils import viewing_target

	rig, _, clips = load_dummy(swimming=True)
	_, owners, physics, _, animation, parameters, _ = create_demo(rig, clips, row_count=1)
	graph = animation.playback.graph
	pool = BakedClipPool.from_graph(graph)
	budget = 0.035
	eligibility = validate_crowd_graph(graph, pool, error_budget=budget)
	router = CrowdGraphRouter(eligibility)
	lod = DistancePoseLod(8, 12)
	positions = np.zeros((len(owners), 3))
	positions[1:, 2] = 15
	water = graph.parameter_columns["in_water"]
	pace = graph.parameter_columns["swim_blend"]
	with physics, viewing_target("River pose routing", offscreen=True):
		palette = SkinMatrixBuffer(len(owners) * pool.bones)
		poses = GraphPosePass(palette, len(owners), eligibility=eligibility)
		try:
			for owner in owners: palette.acquire(owner, pool.bones)

			def draw_pose(expected_cpu, distance=12):
				playback = animation.playback
				gpu_mask = lod.update(owners, positions, [distance, 0, 0], hold=playback.blending(parameters))
				cpu, batch = router.split(playback, parameters, gpu_mask=gpu_mask)
				assert cpu.tolist() == expected_cpu
				assert sorted(np.r_[cpu, batch.rows]) == list(range(len(owners)))
				models = model_matrices(rig, *animation.playback.pose(parameters))
				gpu = pool.model(batch.clip_ids, batch.times, batch.weights)
				assert np.max(np.linalg.norm(gpu[..., :3, 3] - models[batch.rows, :, :3, 3], axis=-1)) <= budget
				models[batch.rows] = gpu
				commands = RenderContext.commands("river pose routing")
				poses.record(commands, animation.owners, playback, parameters, gpu_mask=gpu_mask)
				RenderContext.submit(commands.finish())
				actual = np.frombuffer(RenderContext.device.queue.read_buffer(palette.buffer.handle), dtype=np.float32)
				expected = skin_matrices(rig, models)
				np.testing.assert_allclose(actual.reshape(expected.shape), expected, atol=3e-6)
				return models[0, :, :3, 3]

			start_time = animation.playback.times[0]
			near = draw_pose([0], distance=8)
			np.testing.assert_array_equal(draw_pose([0], distance=10), near)
			before = draw_pose([])
			assert np.max(np.linalg.norm(before - near, axis=-1)) <= budget
			np.testing.assert_array_equal(draw_pose([], distance=10), before)
			np.testing.assert_array_equal(draw_pose([0], distance=8), near)
			before = draw_pose([])
			assert animation.playback.times[0] == start_time
			parameters[[0, 2], water] = 1
			parameters[[0, 2], pace] = (0.5, 0.8)
			animation.advance(0, parameters)
			assert animation.playback.times[0] == start_time
			assert np.max(np.linalg.norm(draw_pose([0, 2]) - before, axis=-1)) <= budget
			animation.advance(FADE_TIME, parameters)
			assert animation.playback.states[0] == graph.state_ids["swim"]
			draw_pose([0, 2])
			parameters[[0, 2], water] = 0
			animation.advance(FADE_TIME, parameters)
			draw_pose([])
		finally:
			poses.close()
			palette.close()


def test_river_pose_culling_preserves_first_draw_after_camera_turn():
	pytest.importorskip("pyjoltphysics")
	pytest.importorskip("pygltflib")
	pytest.importorskip("wgpu")
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext
	from graaal.RenderPasses.CrowdPose import GraphPosePass
	from scenes.physics_viewing_utils import read_pixels, viewing_target
	from scenes.river_crossing_example import river_rendering

	rig, primitive, clips = load_dummy(swimming=True)
	world, dummies, physics, schedule, animation, parameters, _ = create_demo(rig, clips, row_count=6)
	camera = Camera((0, 6, 6), (0, 6, 20), (0, 1, 0), 28, far=120)
	with (
		physics,
		viewing_target("River pose visibility", offscreen=True, offscreen_size=(320, 240)) as output,
		river_rendering(world, dummies, physics, rig, primitive, lambda: 0) as (drawing, renderer, palette, pose_visibility),
	):
		poses = GraphPosePass(palette, len(dummies))
		try:
			size = output.size[:2]
			drawing.sync()
			rows = np.searchsorted(dummies, animation.owners)
			hidden = pose_visibility(camera, size, rows)
			assert not hidden.any()
			commands = RenderContext.commands("offscreen graph poses")
			poses.record(commands, animation.owners, animation.playback, parameters[rows], pose_mask=hidden)
			RenderContext.submit(commands.finish())
			actual = np.frombuffer(RenderContext.device.queue.read_buffer(palette.buffer.handle), dtype=np.float32)
			assert not actual.any()

			schedule.run(FIXED_DT, fixed_steps=1, fixed_dt=FIXED_DT)
			camera.position[:] = (15, 7, -10)
			camera.target[:] = (0, 0, -10)
			drawing.sync()
			visible = pose_visibility(camera, size, rows)
			assert 0 < visible.sum() < len(dummies)
			clock = animation.playback.times.copy()

			def draw(pose_mask):
				commands = RenderContext.commands("river re-entry")
				poses.record(commands, animation.owners, animation.playback, parameters[rows], pose_mask=pose_mask)
				renderer.record(commands, size, camera, output.color_attachment())
				RenderContext.submit(commands.finish())
				matrices = np.frombuffer(RenderContext.device.queue.read_buffer(palette.buffer.handle), dtype=np.float32).reshape(len(dummies), -1)
				return read_pixels(output), matrices

			culled, selected = draw(visible)
			full, reference = draw(None)
			assert not selected[~visible].any()
			np.testing.assert_allclose(selected[visible], reference[visible], rtol=0, atol=1e-6)
			# Transparent contributions can differ by a few 8-bit output levels.
			np.testing.assert_allclose(culled, full, rtol=0, atol=3)
			np.testing.assert_array_equal(animation.playback.times, clock)
		finally:
			poses.close()


def test_river_reduced_poses_write_the_full_palette_and_fall_back():
	pytest.importorskip("pyjoltphysics")
	pytest.importorskip("pygltflib")
	pytest.importorskip("wgpu")
	from graaal.Animation.CrowdGraphSamples import CrowdGraphRouter
	from graaal.Animation.SkeletalAnimation import model_matrices, skin_matrices, skin_vertices
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext
	from graaal.RenderPasses.CrowdPose import GraphPosePass
	from scenes.physics_viewing_utils import has_blue_pixels, read_pixels, viewing_target
	from scenes.river_crossing_example import POSE_LOD_NEAR, crowd_skeleton_lod, river_rendering

	rig, primitive, clips = load_dummy(swimming=True)
	world, dummies, physics, _, animation, parameters, _ = create_demo(rig, clips, row_count=1)
	playback = animation.playback
	graph = playback.graph
	variant, report = crowd_skeleton_lod(graph, primitive)
	assert variant is not None and report.pool.bones < len(rig.parents)
	assert report.states["walk"].eligible and report.states["run"].eligible
	assert not report.states["swim"].eligible
	assert all(not edge.result.eligible for edge in report.transitions if "swim" in (edge.source, edge.target))
	with (
		physics,
		viewing_target("River skeleton LOD", offscreen=True, offscreen_size=(640, 480)) as output,
		river_rendering(world, dummies, physics, rig, primitive, lambda: 0) as (drawing, renderer, palette, _),
	):
		poses = GraphPosePass(palette, len(dummies), eligibility=report)
		router = CrowdGraphRouter(report)
		try:
			owners = animation.owners
			drawing.sync()
			gpu = np.ones(len(owners), dtype=bool)
			gpu[0] = False
			visible = np.ones_like(gpu)
			visible[1] = False
			source = playback.states[2]
			parameters[2, graph.parameter_columns["in_water"]] = 1
			parameters[2, graph.parameter_columns["swim_blend"]] = 0.5
			animation.advance(FADE_TIME / 2, parameters)
			playback = animation.playback
			assert playback.states[2] == source and playback.target_states[2] == graph.state_ids["swim"]
			clock = playback.times.copy()
			exact = skin_matrices(rig, model_matrices(rig, *playback.pose(parameters)))

			def record(render=False):
				commands = RenderContext.commands("river skeleton LOD")
				poses.record(commands, owners, playback, parameters, gpu_mask=gpu, pose_mask=visible)
				if render:
					camera = Camera((15, 10, -5), (0, 0, -2), (0, 1, 0), 50, near=0.1, far=120)
					renderer.record(commands, output.size[:2], camera, output.color_attachment())
				RenderContext.submit(commands.finish())
				data = RenderContext.device.queue.read_buffer(palette.buffer.handle)
				return np.frombuffer(data, dtype=np.float32).reshape(exact.shape)

			first = record()
			_, batch = router.split(playback, parameters, gpu_mask=visible & gpu)
			rows = batch.rows
			np.testing.assert_array_equal(rows, np.arange(3, len(owners)))
			model = report.pool.model(batch.clip_ids, batch.times, batch.weights)
			approx = skin_matrices(variant.skeleton, model)
			np.testing.assert_allclose(first[rows], approx[:, variant.joint_map], rtol=0, atol=3e-6)
			np.testing.assert_allclose(first[[0, 2]], exact[[0, 2]], rtol=0, atol=1e-6)
			assert not first[1].any()
			full_vertices, reduced_vertices = primitive.vertices, variant.primitives[0].vertices
			skinned = [skin_vertices(v["position"], None, v["joints"], v["weights"], matrices)[0]
				for v, matrices in ((full_vertices, first[rows[0]]), (reduced_vertices, approx[0]))]
			np.testing.assert_allclose(*skinned, rtol=0, atol=4e-6)
			reference, _ = skin_vertices(full_vertices["position"], None, full_vertices["joints"], full_vertices["weights"], exact[rows[0]])
			camera = Camera((0, 0.9, POSE_LOD_NEAR), (0, 0.9, 0), (0, 1, 0), 50, near=0.1, far=120)
			width, height = output.size[:2]
			view_projection = camera.view() @ camera.projection(width / height)
			projected = np.c_[np.concatenate((skinned[0], reference)), np.ones(2 * len(reference))] @ view_projection
			pixels = (projected[:, :2] / projected[:, 3, None]) * (width / 2, height / 2)
			assert np.linalg.norm(pixels[:len(reference)] - pixels[len(reference):], axis=1).max() < 1.25

			gpu[3] = False
			visible[1] = True
			second = record(render=True)
			assert has_blue_pixels(read_pixels(output))
			assert second[1].any()
			np.testing.assert_allclose(second[3], exact[3], rtol=0, atol=1e-6)
			np.testing.assert_array_equal(playback.times, clock)
		finally:
			poses.close()
