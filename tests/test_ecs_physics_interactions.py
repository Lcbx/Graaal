"""Native routed events, spatial queries, filters and constraint lifetimes."""

import threading

import numpy as np
import pytest

from graaal.ECS import ECS
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, PhysicsJoint, RigidBody, WORLD_ENTITY
from graaal.EcsAdapters.MaterialTransform import MaterialTransform


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


@pytest.fixture
def setup(pj):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsJoint)
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box(), 1: pj.Shape.box((20, 1, 20))}, settings=pj.WorldSettings(threads=1), joint_type=PhysicsJoint) as physics:
		yield ecs, physics


def add(ecs, position, *, motion=None, shape=0, sensor=False, flags=0):
	entity = ecs.create(1)[0]
	ecs.add(entity, MaterialTransform(position=position), Collider(shape, sensor=sensor))
	if motion is not None: ecs.add(entity, RigidBody(motion=motion, flags=flags))
	return entity


def tick(physics, commands=None):
	physics.reconcile()
	physics.prepare_inputs()
	physics.step()
	physics.pull()
	return physics.dispatch_events(commands=commands)


def contacts(physics, phase):
	events = physics.events
	return events[(events["kind"] == physics.pj.EventKind.CONTACT) & (events["phase"] == phase)]


def test_contact_begin_persist_end_and_once_per_tick_dispatch(setup, pj):
	ecs, physics = setup
	floor = add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 0.5, 0), motion=2, flags=int(pj.BodyFlag.REPORT))
	calls = []
	thread = threading.get_ident()
	def receive(world, commands, events):
		assert threading.get_ident() == thread
		calls.append(events.copy())
	token = physics.subscribe(receive, kind=pj.EventKind.CONTACT, entity=box)
	tick(physics)
	begin = contacts(physics, pj.EventPhase.BEGIN)
	assert len(begin) == 1
	assert {int(begin["entity_a"][0]), int(begin["entity_b"][0])} == {int(floor), int(box)}
	assert not physics.events.flags.writeable
	physics.dispatch_events()
	assert len(calls) == 1
	tick(physics)
	assert len(contacts(physics, pj.EventPhase.PERSIST)) == 1
	physics.teleport([box], (0, 5, 0))
	tick(physics)
	assert len(contacts(physics, pj.EventPhase.END)) == 1
	physics.unsubscribe(token)
	tick(physics)
	assert len(calls) == 3
	assert physics.events_dropped == 0


def test_default_contacts_do_not_emit_persist(setup, pj):
	ecs, physics = setup
	add(ecs, (0, -0.5, 0), shape=1)
	add(ecs, (0, 0.5, 0), motion=2)
	tick(physics)
	tick(physics)
	assert len(contacts(physics, pj.EventPhase.PERSIST)) == 0


def test_removal_routes_to_live_owner_without_retargeting_recycled_slot(setup, pj):
	ecs, physics = setup
	floor = add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 0.5, 0), motion=2)
	tick(physics)
	old_uid = physics.body_uids[int(box)]
	ecs.delete(box)
	new = add(ecs, (8, 5, 0), motion=2)
	calls = []
	physics.subscribe(lambda world, commands, events: calls.append(events), entity=new)
	tick(physics)
	end = contacts(physics, pj.EventPhase.END)
	assert len(end) == 1 and end["reason"][0] == pj.EndReason.REMOVED
	assert old_uid in (end["a"][0], end["b"][0])
	assert set(end[["entity_a", "entity_b"]][0]) == {int(floor), int(WORLD_ENTITY)}
	assert not calls
	assert physics.entities_of([old_uid])[0] == WORLD_ENTITY


def test_removed_collider_retains_owner_for_end_event(setup, pj):
	ecs, physics = setup
	add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 0.5, 0), motion=2)
	tick(physics)
	ecs.remove(box, Collider)
	calls = []
	physics.subscribe(lambda world, commands, events: calls.append(events.copy()), entity=box)
	tick(physics)
	assert len(calls) == 1
	assert calls[0]["reason"][0] == pj.EndReason.REMOVED
	assert box in (calls[0]["entity_a"][0], calls[0]["entity_b"][0])


def test_listeners_use_command_buffer_and_deleted_subscribers_are_suppressed(setup, pj):
	ecs, physics = setup
	add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 0.5, 0), motion=2)
	commands = ecs.commands()
	physics.subscribe(lambda world, buffer, events: buffer.delete(box), kind=pj.EventKind.CONTACT)
	tick(physics, commands)
	assert ecs.is_alive(box)[0]
	commands.flush()
	assert not ecs.is_alive(box)[0]


def test_trigger_sensor_overlap_sleep_and_wake(setup, pj):
	ecs, physics = setup
	physics.world.set_gravity((0, 0, 0))
	sensor = add(ecs, (0, 5, 0), sensor=True)
	box = add(ecs, (0, 5, 0), motion=2)
	events = tick(physics)
	trigger = events[events["kind"] == pj.EventKind.TRIGGER]
	assert len(trigger) == 1
	assert trigger["entity_a"][0] == sensor and trigger["entity_b"][0] == box
	physics.world.run_ticks(120)
	physics.pull()
	events = physics.dispatch_events()
	assert np.any((events["kind"] == pj.EventKind.STATE) & (events["phase"] == pj.EventPhase.SLEEP))
	physics.add_impulse([box], (1, 0, 0))
	events = tick(physics)
	assert np.any((events["kind"] == pj.EventKind.STATE) & (events["phase"] == pj.EventPhase.WAKE))
	physics.teleport([box], (5, 5, 0))
	events = tick(physics)
	assert np.any((events["kind"] == pj.EventKind.TRIGGER) & (events["phase"] == pj.EventPhase.END))


def test_query_filters_entity_exclusion_batch_sweep_and_overlap(setup, pj):
	ecs, physics = setup
	floor = add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 2, 0), motion=2)
	sensor = add(ecs, (0, 4, 0), sensor=True)
	tick(physics)
	hit = physics.raycast((0, 10, 0), (0, -1, 0), 20)
	assert physics.entities_of(hit.uid) == box
	hit = physics.raycast((0, 10, 0), (0, -1, 0), exclude=[box])
	assert physics.entities_of(hit.uid) == floor
	hit = physics.raycast((0, 10, 0), (0, -1, 0), filter=physics.query_filter(triggers=True))
	assert physics.entities_of(hit.uid) == sensor
	hits = physics.raycast_batch([[0, 10, 0], [30, 10, 0]], [[0, -1, 0]] * 2, 20)
	assert hits.hit.tolist() == [True, False]
	np.testing.assert_array_equal(physics.entities_of(hits.uid), [box, WORLD_ENTITY])
	assert physics.entities_of(physics.sweep(pj.Shape.sphere(0.1), (0, 10, 0), (0, -2, 0)).uid) == box
	assert set(physics.entities_of(physics.overlap(0, (0, 2, 0)))) == {box}
	assert len(physics.overlap(0, (0, 2, 0), exclude=[box])) == 0
	assert physics.materials is physics.world.materials


def test_filter_tables_change_contact_behavior(setup, pj):
	ecs, physics = setup
	add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 0.5, 0), motion=2)
	ecs.get(RigidBody)[[box]].filter = 1
	physics.set_filter_table([[True, False], [False, True]])
	tick(physics)
	assert len(contacts(physics, pj.EventPhase.BEGIN)) == 0
	physics.set_filter_table(np.ones((2, 2), dtype=bool))
	tick(physics)
	assert len(contacts(physics, pj.EventPhase.BEGIN)) == 1


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4, 5])
def test_joint_kinds_create_native_constraints(setup, kind):
	ecs, physics = setup
	box = add(ecs, (0, 5, 0), motion=2)
	ecs.add(box, PhysicsJoint(kind=kind, anchor=(0, 5, 0), to_anchor=(0, 6, 0)))
	tick(physics)
	assert len(physics.joint_ids) == 1
	assert physics.world.latest().joints.kind.tolist() == [kind]


def test_joint_lifecycle_keys_target_rebuild_and_owner_removal(setup):
	ecs, physics = setup
	target = add(ecs, (0, 5, 0))
	box = add(ecs, (2, 5, 0), motion=2)
	ecs.add(box, PhysicsJoint(key=1, target=target), PhysicsJoint(key=2, target=target, kind=1, anchor=(1, 5, 0)))
	tick(physics)
	old = dict(physics.joint_ids)
	tick(physics)
	assert dict(physics.joint_ids) == old
	ecs.get(MaterialTransform)[[target]].scale = (2, 2, 2)
	tick(physics)
	assert all(physics.joint_ids[key] != uid for key, uid in old.items())
	assert len(physics.world.latest().joints.id) == 2
	ecs.delete(target)
	tick(physics)
	assert not physics.joint_ids
	assert len(physics.world.latest().joints.id) == 0


def test_joint_deferred_creation_duplicate_keys_and_component_removal(setup):
	ecs, physics = setup
	target = ecs.create(1)[0]
	box = add(ecs, (2, 5, 0), motion=2)
	ecs.add(box, PhysicsJoint(target=target))
	tick(physics)
	assert not physics.joint_ids
	ecs.add(target, MaterialTransform(position=(0, 5, 0)), Collider(0))
	tick(physics)
	assert len(physics.joint_ids) == 1
	ecs.add(box, PhysicsJoint(target=target))
	with pytest.raises(ValueError, match="unique"): physics.reconcile()
	ecs.remove(box, PhysicsJoint)
	tick(physics)
	assert not physics.joint_ids
	assert len(physics.world.latest().joints.id) == 0


def test_hinge_motor_limits_and_runtime_motor_commands(setup, pj):
	ecs, physics = setup
	physics.world.set_gravity((0, 0, 0))
	box = add(ecs, (1, 5, 0), motion=2)
	ecs.add(box, PhysicsJoint(kind=2, anchor=(0, 5, 0), axis=(0, 0, 1), lower=(-0.3,) * 6, upper=(0.3,) * 6, motor_mode=(1, 0, 0, 0, 0, 0), motor_target=(2, 0, 0, 0, 0, 0), motor_max=(100,) * 6))
	tick(physics)
	uid = physics.joint_ids[(int(box), 0)]
	physics.world.run_ticks(120)
	assert abs(physics.world.latest().joints.value[0] - 0.3) < 0.05
	physics.set_joint_motor([box], 0, mode=pj.MotorMode.POSITION, target=-0.2, max_force=100)
	physics.world.run_ticks(180)
	assert abs(physics.world.latest().joints.value[0] + 0.2) < 0.05
	ecs.get(PhysicsJoint)[[box]].assign(motor_mode=(0,) * 6, enabled=False)
	tick(physics)
	assert physics.joint_ids[(int(box), 0)] == uid
	assert physics.world.latest().joints.state[0] == pj.JointState.DISABLED


def test_broken_joint_events_use_joint_namespace_and_preserve_component_key(setup, pj):
	ecs, physics = setup
	add(ecs, (8, 5, 0))
	box = add(ecs, (0, 5, 0), motion=2)
	ecs.add(box, PhysicsJoint(key=7, break_force=0.001))
	events = tick(physics)
	broken = events[(events["kind"] == pj.EventKind.JOINT) & (events["phase"] == pj.EventPhase.BREAK)]
	assert len(broken) == 1
	assert broken["entity_a"][0] == box and broken["entity_b"][0] == WORLD_ENTITY
	assert broken["joint_key"][0] == 7
	assert physics.world.latest().joints.state[0] == pj.JointState.BROKEN
	uid = physics.joint_ids[(int(box), 7)]
	tick(physics)
	assert physics.joint_ids[(int(box), 7)] == uid
	ecs.remove(box, PhysicsJoint)
	events = tick(physics)
	assert np.any((events["kind"] == pj.EventKind.JOINT) & (events["phase"] == pj.EventPhase.END))


def test_character_events_do_not_alias_body_namespace(setup, pj):
	ecs, physics = setup
	floor = add(ecs, (0, -0.5, 0), shape=1)
	tick(physics)
	character = physics.world.add_character((0, 1, 0))
	physics.world.run_ticks(120)
	events = physics.dispatch_events()
	character_events = events[events["kind"] == pj.EventKind.CHARACTER]
	assert len(character_events) > 0
	assert np.all(character_events["entity_a"] == WORLD_ENTITY)
	assert np.all(character_events["entity_b"] == floor)


def test_event_overflow_counter_and_sensor_sensor_overlap(pj):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform)
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, settings=pj.WorldSettings(threads=1, event_capacity=1)) as physics:
		for _ in range(4): add(ecs, (0, 5, 0), sensor=True)
		add(ecs, (0, 5, 0), sensor=True, motion=2, flags=int(pj.BodyFlag.REPORT | pj.BodyFlag.NO_SLEEP))
		physics.reconcile()
		physics.step()
		physics.world.run_ticks(5)
		events = physics.dispatch_events()
		assert len(events) > 0
		assert np.all(events["kind"] == pj.EventKind.TRIGGER)
		assert physics.events_dropped > 0


def test_direct_dispatch_flushes_commands_after_listeners(setup, pj):
	ecs, physics = setup
	add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 0.5, 0), motion=2)
	physics.subscribe(lambda world, commands, events: commands.delete(box), entity=box)
	tick(physics)
	assert not ecs.is_alive(box)[0]
	physics.reconcile()
	physics.close()
	assert physics.world.latest().count == 0


def test_immediate_owner_deletion_suppresses_later_entity_listener(setup, pj):
	ecs, physics = setup
	add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 0.5, 0), motion=2)
	called = []
	physics.subscribe(lambda world, commands, events: world.delete(box), entity=box)
	physics.subscribe(lambda world, commands, events: called.append(True), entity=box)
	tick(physics)
	assert not called
	assert physics.entities_of([]).shape == (0,)
	with pytest.raises(ValueError): physics.entities_of([-1])
	with pytest.raises(ValueError): physics.subscribe(lambda *args: None, entity=[0, 1])


def test_query_group_masks_and_all_sweep_hits(setup):
	ecs, physics = setup
	floor = add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 2, 0), motion=2)
	tick(physics)
	hits = physics.sweep(0, (0, 10, 0), (0, -2, 0), all_hits=True)
	np.testing.assert_array_equal(physics.entities_of([hit.uid for hit in hits]), [box, floor])
	ecs.get(RigidBody)[[box]].assign(group=2, mask=2)
	tick(physics)
	filter = physics.query_filter(group=2, mask=2)
	hit = physics.raycast((0, 10, 0), (0, -1, 0), filter=filter)
	assert physics.entities_of(hit.uid) == box
	assert physics.raycast((0, 10, 0), (0, -1, 0), filter=filter, exclude=[box]) is None


def test_native_material_table_indices_reach_contact_events(setup, pj):
	ecs, physics = setup
	add(ecs, (0, -0.5, 0), shape=1)
	box = add(ecs, (0, 0.5, 0), motion=2)
	material = physics.materials.define(friction=0.1, restitution=0.2)
	ecs.get(RigidBody)[[box]].material = material
	tick(physics)
	begin = contacts(physics, pj.EventPhase.BEGIN)
	assert len(begin) == 1
	assert material in (begin["material_a"][0], begin["material_b"][0])
