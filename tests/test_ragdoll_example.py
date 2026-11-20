"""One animation clock, native mode cycle and a single final palette per owner."""

import numpy as np
import pytest

from graaal.SkeletalAnimation import skin_vertices


def test_native_ragdoll_demo_animation_push_fall_and_recovery():
	pj = pytest.importorskip("pyjoltphysics")
	from scenes.ragdoll_example import create_demo
	ecs, owner, floor, physics, schedule, state, asset = create_demo()
	with physics:
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
		# Procedural demo geometry uses the same joint/weight layout as imported meshes.
		from scenes.ragdoll_example import skinned_boxes
		from scenes.physics_example import cube_vertices
		vertices = skinned_boxes(asset, cube_vertices())
		deformed, normals = skin_vertices(vertices["position"], vertices["normal"], vertices["joints"], vertices["weights"], state["batches"][0].matrices[0])
		assert np.isfinite(deformed).all() and np.isfinite(normals).all()
	assert physics.world.latest().count == 0
