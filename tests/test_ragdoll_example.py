"""One animation clock, native mode cycle and a single final palette per owner."""

import numpy as np
import pytest

def test_native_ragdoll_demo_animation_push_fall_and_recovery():
	pj = pytest.importorskip("pyjoltphysics")
	from scenes.passive_ragdoll_example import HUMANOID_NAMES
	from scenes.ragdoll_example import create_demo
	_, owner, _, physics, schedule, state, asset = create_demo()
	with physics:
		assert tuple(bone.name for bone in asset.settings.bones) == HUMANOID_NAMES
		assert asset.settings.root_force == asset.settings.root_torque == 0
		stages, heights, velocities = {}, {}, {}
		for tick in range(840):
			schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
			assert len(state["batches"]) == 1
			batch = state["batches"][0]
			assert batch.entities.tolist() == [owner]
			assert batch.matrices.shape == (1, len(asset.skeleton.parents), 3, 4)
			assert np.isfinite(batch.matrices).all()
			if tick in (179, 180): velocities[tick] = np.linalg.norm(physics.world.bodies_state(physics.ragdolls[int(owner)].bones).linear_velocity)
			if tick in (119, 239, 479, 599, 839):
				stages[tick] = state["stage"]
				positions, _ = physics.ragdolls[int(owner)].pose()
				heights[tick] = positions[0, 1]
		assert list(stages.values()) == ["animated", "powered", "limp", "limp", "recovery"]
		assert heights[599] < heights[119] - 0.5
		assert heights[839] == pytest.approx(heights[119], abs=0.01)
		assert velocities[180] > velocities[179] + 1
		assert physics.world.latest().ragdolls.mode[0] == pj.RagdollMode.KINEMATIC
	assert physics.world.latest().count == 0


def test_native_ragdoll_demo_standing_mode_uses_foot_support_and_holds_its_stance():
	pj = pytest.importorskip("pyjoltphysics")
	from scenes.ragdoll_example import TICK_HZ, create_demo
	_, _, _, physics, schedule, state, asset = create_demo(standing=True)
	with physics:
		feet = [asset.settings.bone_index(name) for name in ("foot_l", "foot_r")]
		assert all(asset.settings.bones[i].strength == 0 for i in feet)
		dt = 1 / TICK_HZ
		for tick in range(10 * TICK_HZ):
			schedule.run(dt, fixed_steps=1, fixed_dt=dt)
			observation = state["observation"]
			assert observation.position[0, 0, 1] > 0.9
			if tick >= 9 * TICK_HZ:
				assert observation.contacts[0, feet].any()
				assert np.linalg.norm(observation.center_of_mass_velocity[0, [0, 2]]) < 0.05
		assert physics.world.latest().ragdolls.mode[0] == pj.RagdollMode.POWERED
		assert state["stage"] == "powered"
