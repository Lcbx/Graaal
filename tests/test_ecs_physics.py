"""Native ECS lifecycle and transform authority contracts; no GPU required."""

import numpy as np
import pytest

from graaal.ECS import ECS, Scheduler, component, entity_index
from graaal.MaterialTransform import MaterialTransform
from graaal.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
from graaal.Math import Vec3, Vec4


@component
class Transform:
	position: Vec3 = (0, 0, 0)
	rotation: Vec4 = (0, 0, 0, 1)


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


@pytest.fixture
def setup(pj):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform)
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box(), 1: pj.Shape.box((20, 1, 20))}, settings=pj.WorldSettings(threads=1)) as physics:
		yield ecs, physics


def spawn(ecs, *, position=(0, 5, 0), body=True, shape=0, sensor=False):
	entity = ecs.create(1)[0]
	ecs.add(entity, MaterialTransform(position=position), Collider(shape, sensor=sensor))
	if body: ecs.add(entity, RigidBody())
	return entity


def tick(physics):
	physics.reconcile()
	physics.prepare_inputs()
	physics.step()
	physics.pull()


def state(physics, entities):
	return physics.world.bodies_state([physics.body_uids[int(entity)] for entity in np.atleast_1d(entities)])


def test_spawn_remove_recycled_handles_and_close(setup):
	ecs, physics = setup
	entities = [spawn(ecs, position=(x, 5, 0)) for x in (0, 2, 4)]
	tick(physics)
	old_entity = entities[0]
	old_uid = physics.body_uids[int(old_entity)]
	ecs.delete(old_entity)
	replacement = spawn(ecs, position=(10, 5, 0))
	assert entity_index(replacement) == entity_index(old_entity)
	assert replacement != old_entity
	physics.add_impulse([old_entity], [1000, 0, 0])
	tick(physics)
	assert int(old_entity) not in physics.body_uids
	assert old_uid not in physics.body_owners
	assert physics.body_uids[int(replacement)] != old_uid
	assert state(physics, replacement).linear_velocity[0, 0] == 0
	assert physics.world.latest().rows_of([old_uid])[0] == -1
	ecs.remove(entities[1], Collider)
	ecs.remove(entities[2], MaterialTransform)
	tick(physics)
	assert len(physics.body_uids) == 1
	physics.close()
	physics.close()
	assert not physics.body_uids
	assert physics.world.latest().count == 0
	with pytest.raises(RuntimeError, match="closed"): physics.reconcile()


def test_dynamic_mirror_never_teleports_and_static_moves_are_explicit(setup):
	ecs, physics = setup
	dynamic = spawn(ecs)
	static = spawn(ecs, position=(5, 0, 0), body=False)
	tick(physics)
	transforms = ecs.get(MaterialTransform)
	transforms[[dynamic]].position = (100, 100, 100)
	transforms[[static]].position = (20, 20, 20)
	assert physics.reconcile() is False
	tick(physics)
	assert state(physics, dynamic).position[0, 0] == pytest.approx(0)
	np.testing.assert_allclose(state(physics, static).position, [[5, 0, 0]])
	physics.teleport([static, dynamic], [[8, 0, 0], [0, 10, 0]])
	tick(physics)
	np.testing.assert_allclose(transforms[[static]].position, [[8, 0, 0]])
	assert state(physics, dynamic).position[0, 1] > 9.9


def test_kinematic_targets_and_teleport(setup):
	ecs, physics = setup
	entity = spawn(ecs)
	ecs.get(RigidBody)[[entity]].motion = 1
	tick(physics)
	ecs.get(MaterialTransform)[[entity]].position = (3, 5, 0)
	tick(physics)
	np.testing.assert_allclose(state(physics, entity).position, [[3, 5, 0]], atol=1e-5)
	tick(physics)
	np.testing.assert_allclose(state(physics, entity).linear_velocity, 0, atol=1e-5)
	physics.teleport([entity], [7, 5, 0])
	tick(physics)
	tick(physics)
	np.testing.assert_allclose(state(physics, entity).position, [[7, 5, 0]], atol=1e-5)


def test_batched_commands_copy_inputs_and_rebuild_retains_state(setup):
	ecs, physics = setup
	entities = [spawn(ecs, position=(x, 5, 0)) for x in (0, 4)]
	tick(physics)
	impulses = np.array([[1, 0, 0], [2, 0, 0]], dtype=float)
	physics.add_impulse(entities, impulses)
	impulses[:] = 1000
	physics.add_force(entities, [120, 0, 0])
	tick(physics)
	np.testing.assert_allclose(state(physics, entities).linear_velocity[:, 0], [2, 3], atol=1e-5)
	old_uids = dict(physics.body_uids)
	before = state(physics, entities)
	ecs.get(MaterialTransform)[entities].assign(scale=(2, 2, 2), position=(50, 50, 50))
	tick(physics)
	assert all(physics.body_uids[int(entity)] != old_uids[int(entity)] for entity in entities)
	after = state(physics, entities)
	np.testing.assert_allclose(after.linear_velocity[:, 0], before.linear_velocity[:, 0], atol=1e-5)
	assert np.all(after.position[:, 0] < 5)
	physics.set_velocity(entities, linear=(0, 0, 0), angular=(0, 1, 0))
	tick(physics)
	np.testing.assert_allclose(state(physics, entities).angular_velocity[:, 1], 1, atol=1e-5)


def test_writable_settings_keep_uids_and_structural_flags_rebuild(setup, pj):
	ecs, physics = setup
	entity = spawn(ecs)
	tick(physics)
	uid = physics.body_uids[int(entity)]
	ecs.get(RigidBody)[[entity]].assign(mass=2, friction=0.2, restitution=0.3, linear_damping=0.1, angular_damping=0.2, gravity_factor=0, flags=int(pj.BodyFlag.REPORT | pj.BodyFlag.NO_SLEEP), material=0, filter=1)
	tick(physics)
	assert physics.body_uids[int(entity)] == uid
	ecs.get(RigidBody)[[entity]].flags = 0
	tick(physics)
	assert physics.body_uids[int(entity)] == uid
	ecs.get(RigidBody)[[entity]].group = 2
	tick(physics)
	assert physics.body_uids[int(entity)] != uid


def test_compound_cache_sensor_and_colliders_removed(setup, pj):
	ecs, physics = setup
	entities = [spawn(ecs, position=(x, 5, 0), body=False) for x in (0, 5)]
	for entity in entities: ecs.add(entity, Collider(0, position=(2, 0, 0)))
	tick(physics)
	assert len(physics._shape_cache) == 1
	for entity in entities:
		hit = physics.world.raycast((state(physics, entity).position[0, 0] + 2, 10, 0), (0, -1, 0), 10)
		assert hit is not None
	old_uid = physics.body_uids[int(entities[0])]
	rows = ecs.get(Collider)[[entities[0]]].take([1])
	ecs.get(Collider).remove(rows)
	tick(physics)
	assert physics.body_uids[int(entities[0])] != old_uid
	sensor = spawn(ecs, position=(10, 1, 0), body=False, sensor=True)
	box = spawn(ecs, position=(10, 1, 0))
	tick(physics)
	events = physics.world.drain_events()
	assert np.any(events["kind"] == pj.EventKind.TRIGGER)
	assert state(physics, box).position[0, 1] < 1


def test_invalid_topology_leaves_existing_bodies(setup, pj):
	ecs, physics = setup
	entity = spawn(ecs)
	tick(physics)
	uid = physics.body_uids[int(entity)]
	ecs.add(entity, Collider(0, sensor=True))
	with pytest.raises(ValueError, match="mixed solid and sensor"): physics.reconcile()
	assert state(physics, entity).found[0]
	assert physics.body_uids[int(entity)] == uid
	ecs.remove(entity, Collider)
	ecs.add(entity, Collider(0, scale=(2, 1, 1)), Collider(0))
	with pytest.raises(ValueError, match="unit local scale"): physics.reconcile()


def test_mesh_dynamic_rejected_and_static_supported(pj):
	ecs = ECS()
	ecs.register(Collider, RigidBody, Transform)
	mesh = pj.Shape.mesh([[-2, 0, -2], [2, 0, -2], [0, 0, 2]], [[0, 2, 1]])
	with EcsPhysicsAdapter(ecs, {0: mesh}, transform_type=Transform, settings=pj.WorldSettings(threads=1)) as physics:
		entity = ecs.create(1)[0]
		ecs.add(entity, Transform(), Collider(0), RigidBody())
		with pytest.raises(ValueError, match="not dynamic"): physics.reconcile()
		ecs.remove(entity, RigidBody)
		tick(physics)
		assert state(physics, entity).found[0]


def test_custom_transform_orientation_and_fixed_schedule(pj):
	ecs = ECS()
	ecs.register(Collider, RigidBody, Transform)
	entity = ecs.create(1)[0]
	rotation = (0, np.sqrt(0.5), 0, np.sqrt(0.5))
	ecs.add(entity, Transform((0, 5, 0), rotation), Collider(0), RigidBody())
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, transform_type=Transform, settings=pj.WorldSettings(threads=1)) as physics:
		schedule = Scheduler(ecs)
		for name, callback in (("sync", physics.reconcile), ("input", physics.prepare_inputs), ("step", physics.step), ("pull", physics.pull)):
			schedule.add(name, callback, phase="fixed_update")
		schedule.run(1 / 60, fixed_steps=2, fixed_dt=1 / 120)
		assert physics.world.tick == 2
		np.testing.assert_allclose(ecs.get(Transform)[[entity]].rotation, [rotation], atol=1e-6)
		with pytest.raises(ValueError, match="fixed dt"): schedule.run(0, fixed_steps=1, fixed_dt=1 / 60)
		assert physics.world.tick == 2


def test_stale_native_uid_and_deleted_owner_scatter(setup):
	ecs, physics = setup
	entity = spawn(ecs)
	tick(physics)
	uid = physics.body_uids[int(entity)]
	physics.world.remove_bodies([uid])
	physics.step()
	with pytest.raises(RuntimeError, match="disappeared"): physics.pull()
	ecs.delete(entity)
	physics.reconcile()
	physics.step()
	physics.pull()
	assert not physics.body_uids


def test_continuous_fast_body_hits_thin_floor(setup, pj):
	ecs, physics = setup
	floor = spawn(ecs, position=(0, -0.05, 0), body=False)
	ecs.get(MaterialTransform)[[floor]].scale = (20, 0.1, 20)
	entity = spawn(ecs, position=(0, 2, 0))
	ecs.get(MaterialTransform)[[entity]].scale = (0.1, 0.1, 0.1)
	ecs.get(RigidBody)[[entity]].flags = int(pj.BodyFlag.CONTINUOUS)
	physics.set_velocity([entity], linear=(0, -1000, 0))
	tick(physics)
	assert state(physics, entity).position[0, 1] > -0.1


def test_rebuild_at_capacity_and_pending_creation(pj):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform)
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, settings=pj.WorldSettings(threads=1, max_bodies=2)) as physics:
		entities = [spawn(ecs, position=(x, 5, 0)) for x in (0, 5)]
		physics.reconcile()
		# A shape change can happen before the first native tick applies creation.
		ecs.get(MaterialTransform)[entities].scale = (2, 2, 2)
		tick(physics)
		assert physics.world.latest().count == 2
		ecs.get(MaterialTransform)[entities].scale = (3, 3, 3)
		tick(physics)
		assert physics.world.latest().count == 2
		assert np.all(state(physics, entities).found)


def test_creation_and_removal_are_native_batches(setup, monkeypatch):
	ecs, physics = setup
	entities = [spawn(ecs, position=(x * 2, 5, 0)) for x in range(16)]
	adds, removes = [], []
	add, remove = physics.world.add_bodies, physics.world.remove_bodies
	def add_batch(batch):
		adds.append(len(batch))
		return add(batch)
	def remove_batch(uids):
		removes.append(len(uids))
		return remove(uids)
	monkeypatch.setattr(physics.world, "add_bodies", add_batch)
	monkeypatch.setattr(physics.world, "remove_bodies", remove_batch)
	tick(physics)
	assert adds == [16]
	ecs.delete(entities)
	tick(physics)
	assert removes == [16]
	assert physics.world.latest().count == 0


def test_mass_gravity_and_impulse_at_point(setup):
	ecs, physics = setup
	entity = spawn(ecs)
	tick(physics)
	ecs.get(RigidBody)[[entity]].assign(mass=2, gravity_factor=0)
	physics.set_velocity([entity], linear=(0, 0, 0))
	physics.add_impulse([entity], [2, 0, 0], points=[[0, 6, 0]])
	tick(physics)
	current = state(physics, entity)
	np.testing.assert_allclose(current.linear_velocity, [[1, 0, 0]], atol=1e-5)
	assert abs(current.angular_velocity[0, 2]) > 0.1


def test_motion_transitions_and_rigidbody_removal(setup):
	ecs, physics = setup
	entity = spawn(ecs, body=False)
	tick(physics)
	uid = physics.body_uids[int(entity)]
	ecs.add(entity, RigidBody())
	tick(physics)
	assert physics.body_uids[int(entity)] != uid
	assert state(physics, entity).linear_velocity[0, 1] < 0
	ecs.get(RigidBody)[[entity]].motion = 1
	ecs.get(MaterialTransform)[[entity]].position = (3, 5, 0)
	tick(physics)
	np.testing.assert_allclose(state(physics, entity).position, [[3, 5, 0]], atol=1e-5)
	ecs.remove(entity, RigidBody)
	tick(physics)
	assert physics.world.latest().motion_type[0] == 0


def test_local_collider_scale_and_transform_removed_before_pull(setup):
	ecs, physics = setup
	entity = spawn(ecs, body=False)
	ecs.get(Collider)[[entity]].assign(position=(2, 0, 0), scale=(2, 1, 1))
	tick(physics)
	assert physics.world.raycast((2.9, 10, 0), (0, -1, 0), 10) is not None
	assert physics.world.raycast((0, 10, 0), (0, -1, 0), 10) is None
	ecs.remove(entity, MaterialTransform)
	physics.pull()
	physics.reconcile()
	physics.step()
	assert physics.world.latest().count == 0


def test_commands_reject_invalid_values_and_drop_deleted_entities(setup):
	ecs, physics = setup
	entity = spawn(ecs)
	for bad in ([np.nan, 0, 0], [1, 2], [[1, 2, 3], [4, 5, 6]]):
		with pytest.raises(ValueError): physics.add_impulse([entity], bad)
	with pytest.raises(ValueError, match="unit xyzw"): physics.teleport([entity], [0, 0, 0], [0, 0, 0, 0])
	physics.add_force([entity], [100, 0, 0])
	ecs.delete(entity)
	tick(physics)
	assert physics.world.latest().count == 0


def test_rebuild_does_not_turn_static_pose_edits_into_moves(setup):
	ecs, physics = setup
	static = spawn(ecs, position=(4, 5, 0), body=False)
	dynamic = spawn(ecs)
	tick(physics)
	before = state(physics, dynamic).position.copy()
	ecs.get(MaterialTransform)[[static, dynamic]].position = (100, 100, 100)
	ecs.get(MaterialTransform)[[static]].scale = (2, 2, 2)
	ecs.remove(dynamic, RigidBody)
	tick(physics)
	np.testing.assert_allclose(state(physics, static).position, [[4, 5, 0]])
	np.testing.assert_allclose(state(physics, dynamic).position, before)
	np.testing.assert_allclose(ecs.get(MaterialTransform)[[static]].position, [[4, 5, 0]])
	np.testing.assert_allclose(ecs.get(MaterialTransform)[[dynamic]].position, before)
