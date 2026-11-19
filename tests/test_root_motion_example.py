import numpy as np
import pytest

from graaal.EcsAnimation import ClipPlayer
from scenes.root_motion_example import Movement, create_demo, crossfade


def test_root_motion_example_crossfades_and_limits_accepted_movement():
	world, entity, animation, schedule = create_demo(limit=0.3)
	crossfade(world, entity, 1)
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	movement = world.get(Movement)[[entity]]
	assert movement.requested[0] == pytest.approx(0.0625)
	assert movement.accepted[0] == pytest.approx(0.0625)
	assert animation.batches[0].matrices[0, 0, 0, 3] == pytest.approx(0)
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	assert movement.x[0] == pytest.approx(0.25)
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	assert movement.requested[0] == pytest.approx(0.25)
	assert movement.accepted[0] == pytest.approx(0.05)
	assert movement.x[0] == pytest.approx(0.3)
	world.get(ClipPlayer)[[entity]].assign(playing=False)
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	np.testing.assert_allclose([movement.requested[0], movement.accepted[0]], [0, 0])
