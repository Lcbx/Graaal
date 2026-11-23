import numpy as np
import pytest

from graaal.Animation.AnimationGraphPlayback import state_sample
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from scenes.animation_dummy import load_dummy
from scenes.river_crossing_example import FINISH, FIXED_DT, RIVER_WIDTH, create_demo


def test_crowd_swims_between_sensor_shores_and_teleports_to_spawn():
	pj = pytest.importorskip("pyjoltphysics")
	pytest.importorskip("pygltflib")
	rig, _, clips = load_dummy(swimming=True)
	world, dummies, physics, schedule, animation, parameters = create_demo(rig, clips)
	graph = animation.graph
	swim = graph.state_ids["swim"]
	pace = graph.parameter_columns["swim_blend"]
	transforms = world.get(MaterialTransform)
	start = transforms[dummies].position
	previous_z = start[:, 2]
	assert set(animation.states) == {graph.state_ids["walk"], graph.state_ids["run"]}
	forward = graph.clips[graph.clip_ids["Swimming"]]
	tread = graph.clips[graph.clip_ids["Swimming_in_place"]]
	assert forward.duration == pytest.approx(tread.duration)
	np.testing.assert_allclose(forward.translation[0, 0, 2], forward.translation[-1, 0, 2], atol=1e-6)
	blend_parameters = parameters[:2].copy()
	blend_parameters[:, pace] = (0, 0.8)
	ids, _, weights = state_sample(graph, swim, np.zeros(2), blend_parameters)
	assert ids[1].tolist() == [graph.clip_ids[name] for name in ("Swimming_in_place", "Swimming")]
	np.testing.assert_allclose(weights, [[1, 0], [0.2, 0.8]])
	entered = np.zeros(len(dummies), dtype=bool)
	exited, swam, paused, treaded, returned = (entered.copy() for _ in range(5))
	with physics:
		for _ in range(2400):
			schedule.run(FIXED_DT, fixed_steps=1, fixed_dt=FIXED_DT)
			position = transforms[dummies].position
			z = position[:, 2]
			events = physics.events
			triggers = events[events["kind"] == pj.EventKind.TRIGGER]
			for seen, phase, shore in ((entered, pj.EventPhase.BEGIN, -RIVER_WIDTH / 2), (exited, pj.EventPhase.END, RIVER_WIDTH / 2)):
				at_shore = np.isin(dummies, triggers["entity_b"][triggers["phase"] == phase])
				assert np.all(np.abs(z[at_shore] - shore) < 0.35)
				seen |= at_shore
			in_swim = animation.states == swim
			stopped = z == previous_z
			swam |= in_swim & (z > previous_z) & (parameters[:, pace] >= 0.6)
			paused |= stopped
			treaded |= in_swim & stopped & (parameters[:, pace] == 0)
			recycled = z < previous_z
			if np.any(recycled):
				assert np.all(entered[recycled] & exited[recycled] & swam[recycled] & paused[recycled])
				assert np.all(previous_z[recycled] > FINISH - 0.1)
				np.testing.assert_array_equal(position[recycled], start[recycled])
				render_position, _, _ = physics.render_transforms(dummies[recycled], alpha=0)
				np.testing.assert_allclose(render_position, start[recycled], atol=1e-5, rtol=0)
			returned |= recycled
			previous_z = z
			if np.all(returned): break
	assert np.all(returned) and np.any(treaded)
