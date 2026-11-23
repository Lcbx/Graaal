import numpy as np

from graaal.EcsAdapters.EcsAnimation import ClipPlayer, SkeletonRef
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


def test_native_ray_batch_can_replace_analytic_ground():
	import pytest
	pj = pytest.importorskip("pyjoltphysics")
	from graaal.ECS import ECS
	from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
	from graaal.EcsAdapters.MaterialTransform import MaterialTransform
	from scenes.physics_queries_example import query_ground

	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform)
	ground = ecs.create(1)[0]
	ecs.add(ground, Collider(0), MaterialTransform(position=(0, -0.3, 0)))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box((20, 1, 20))}, settings=pj.WorldSettings(threads=1)) as physics:
		physics.reconcile()
		physics.step()
		world, entities, _, schedule, frames = create_demo(lambda points: query_ground(physics, points))
		schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
		matrices = frames["batches"][0][1]
		np.testing.assert_allclose(matrices[0, 2, 1, 3], 0.2, atol=2e-6)
		np.testing.assert_allclose(matrices[1, 2, 1, 3], 0, atol=2e-6)


def test_ground_query_miss_disables_foot_correction():
	world, entities, _, schedule, frames = create_demo(lambda points: (np.full(len(points), np.nan), np.zeros_like(points)))
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	assert not np.any(frames["ground_hits"][0])
	assert np.all(np.isfinite(frames["batches"][0][1]))
	np.testing.assert_allclose(frames["batches"][0][1][:, 2, 1, 3], 0, atol=1e-6)


def test_supported_stance_holds_world_position_and_normal_until_release():
	world, entities, _, schedule, frames = create_demo(plant=True)
	feet, normals = [], []
	for _ in range(3):
		schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
		x = world.get(Character)[entities].x[0]
		matrix = frames["batches"][0][1][0, 2]
		feet.append(matrix[:, 3] + [x, 0, 0])
		normals.append(matrix[:, 1].copy())
	np.testing.assert_allclose(feet[0], feet[1], atol=2e-6)
	np.testing.assert_allclose(normals[0], normals[1], atol=2e-6)
	assert feet[2][0] > feet[1][0]
	assert not world.get(Character)[entities].locked.any()


def test_planted_foot_checks_support_at_its_anchor():
	def ground_query(points):
		height = np.where(points[:, 0] < 0.1, 0.2, np.nan)
		return height, np.tile([0, 1, 0], (len(points), 1))

	world, entities, _, schedule, frames = create_demo(ground_query, plant=True)
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	anchor = world.get(Character)[entities].anchor[0].copy()
	assert world.get(Character)[entities].locked[0]
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	characters = world.get(Character)[entities]
	assert characters.x[0] > 0.1
	assert characters.locked[0] and frames["ground_hits"][0][0]
	np.testing.assert_array_equal(characters.anchor[0], anchor)
	foot = frames["batches"][0][1][0, 2, :, 3] + [characters.x[0], 0, 0]
	np.testing.assert_allclose(foot, anchor, atol=2e-6)
	schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
	assert not world.get(Character)[entities].locked[0]
	assert not frames["ground_hits"][0][0]
