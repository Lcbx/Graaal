import numpy as np
import pytest

from graaal.EcsAnimation import ClipPlayer
from graaal.MaterialTransform import MaterialTransform
from scenes.physics_locomotion_example import create_demo
from scenes.root_motion_example import Movement


def test_native_sweep_accepts_only_unblocked_root_motion():
	pytest.importorskip("pyjoltphysics")
	world, character, obstacle, animation, physics, schedule = create_demo()
	with physics:
		positions, requests, accepted = [], [], []
		for _ in range(10):
			schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
			movement = world.get(Movement)[[character]]
			positions.append(float(movement.x[0]))
			requests.append(float(movement.requested[0]))
			accepted.append(float(movement.accepted[0]))
			assert animation.batches and np.isfinite(animation.batches[0].matrices).all()
		assert positions[0] > 0
		assert 1.48 < positions[-1] < 1.51
		assert all(request > 0 for request in requests)
		assert accepted[-1] < 0.01
		np.testing.assert_allclose(world.get(MaterialTransform)[[character]].position[0, 0], positions[-1])
		assert world.get(ClipPlayer)[[character]].extract_root_motion[0]
