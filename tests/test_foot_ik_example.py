import numpy as np

from graaal.EcsAnimation import ClipPlayer, SkeletonRef
from scenes.foot_ik_example import Character, create_demo, terrain_height, terrain_normal


def test_fixed_step_moves_then_places_foot_on_terrain():
	world, entities, animation, schedule, frames = create_demo()
	assert schedule.compile() == ("animation", "pre_pose", "terrain", "post_pose")
	for _ in range(5):
		schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
		characters = world.get(Character)[entities]
		assert characters.x[0] > 0
		owners, matrices = frames["batches"][0]
		np.testing.assert_array_equal(owners, entities)
		assert len(animation.root_motion_batches) == 1
		np.testing.assert_allclose(matrices[0, 2, 1, 3], terrain_height(characters.x[0]), atol=2e-6)
		np.testing.assert_allclose(matrices[1, 2, 1, 3], 0, atol=2e-6)
		normal = terrain_normal(characters.x[:1])[0]
		normal /= np.linalg.norm(normal)
		np.testing.assert_allclose(matrices[0, 2, :, 1], normal, atol=2e-6)
		np.testing.assert_allclose(matrices[1, 2, :, 1], [0, 1, 0], atol=2e-6)


def test_example_refetches_entities_after_recycling():
	world, entities, _, schedule, frames = create_demo()
	world.delete(entities[:1])
	new = world.create()[0]
	world.add(new, SkeletonRef(0), ClipPlayer(0, extract_root_motion=True), Character(ik_weight=1))
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	owners, matrices = frames["batches"][0]
	assert set(map(int, owners)) == {int(entities[1]), int(new)}
	row = np.flatnonzero(owners == new)[0]
	x = world.get(Character)[[new]].x[0]
	np.testing.assert_allclose(matrices[row, 2, 1, 3], terrain_height(x), atol=2e-6)
