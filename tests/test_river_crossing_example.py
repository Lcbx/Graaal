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
