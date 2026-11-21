"""Same-world body checkpoints and selective resets preserve ownership boundaries."""

from dataclasses import replace

import numpy as np
import pytest

from graaal.ECS import ECS
from graaal.EcsAdapters.EcsPhysics import CharacterState, Collider, EcsPhysicsAdapter, PhysicsCharacter, PhysicsJoint, PhysicsRagdoll, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.PhysicsRagdoll import RagdollAsset
from graaal.Animation.SkeletalAnimation import Skeleton


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


@pytest.fixture(params=(1, 2))
def setup(pj, request):
	ecs = ECS()
	ecs.register(Collider, RigidBody, PhysicsJoint, MaterialTransform)
	box, sibling, platform, floor = ecs.create(4)
	ecs.add(box, Collider(0), RigidBody(), MaterialTransform(position=(0, 0.5, 0)))
	ecs.add(sibling, Collider(0), RigidBody(), MaterialTransform(position=(3, 4, 0)))
	ecs.add(platform, Collider(0), RigidBody(motion=1), MaterialTransform(position=(6, 2, 0)))
	ecs.add(floor, Collider(1), MaterialTransform(position=(0, -0.5, 0)))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box(), 1: pj.Shape.box((30, 1, 30))}, settings=pj.WorldSettings(threads=request.param, gravity=(0, 0, 0)), joint_type=PhysicsJoint, interpolate=True) as physics:
		yield ecs, physics, (box, sibling, platform, floor)


def tick(physics, count=1):
	for _ in range(count):
		physics.reconcile()
		physics.prepare_inputs()
		physics.step()
		physics.pull()
		physics.dispatch_events()


def body_state(physics, owners):
	return physics.world.bodies_state([physics.body_uids[int(owner)] for owner in owners])


def trajectory(physics, owners, count=12):
	frames = []
	for _ in range(count):
		tick(physics)
		state = body_state(physics, owners)
		frames.append(np.concatenate((state.position, state.orientation, state.linear_velocity, state.angular_velocity), axis=1))
	return np.array(frames)


def test_checkpoint_replays_queued_inputs_and_restores_kinematic_targets(setup):
	ecs, physics, owners = setup
	box, sibling, platform, _ = owners
	tick(physics, 3)
	physics.set_velocity([sibling], linear=(1, 0, 0), angular=(0, 0.2, 0))
	physics.add_force([sibling], (2, 0, 0))
	physics.teleport([box], (0, 3, 0))
	checkpoint = physics.save_state()
	expected = trajectory(physics, owners)
	advanced_tick = physics.world.tick
	physics.add_impulse([box, sibling], (1000, 0, 0))
	ecs.get(MaterialTransform)[[platform]].position = (8, 3, 0)
	assert physics.restore_state(checkpoint) == 3
	assert physics.world.tick == advanced_tick
	assert physics.events.size == 0 and not physics.events.flags.writeable
	pose = ecs.get(MaterialTransform)[[platform]].position
	np.testing.assert_allclose(pose, [[6, 2, 0]])
	for alpha in (0, 0.5, 1):
		np.testing.assert_allclose(physics.render_transforms(owners, alpha=alpha)[0], body_state(physics, owners).position, atol=1e-6)
	for _ in range(2):
		actual = trajectory(physics, owners)
		np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
		physics.restore_state(checkpoint)


def test_contact_and_joint_state_replay_without_recreating_resources(setup):
	ecs, physics, owners = setup
	box, sibling, _, _ = owners
	ecs.add(sibling, PhysicsJoint(key=1, kind=1, anchor=(3, 5, 0)))
	physics.world.set_gravity((0, -9.81, 0))
	physics.world.flush()
	tick(physics, 6)
	checkpoint = physics.save_state()
	ids = dict(physics.body_uids), dict(physics.joint_ids)
	expected = trajectory(physics, owners, 30)
	physics.restore_state(checkpoint)
	np.testing.assert_allclose(trajectory(physics, owners, 30), expected, atol=1e-6, rtol=0)
	assert (dict(physics.body_uids), dict(physics.joint_ids)) == ids
	assert physics.world.stats()["rejected"] == 0


def test_checkpoint_restores_sleep_and_does_not_redeliver_events(setup):
	ecs, physics, owners = setup
	tick(physics, 90)
	assert not body_state(physics, owners[:2]).awake.any()
	checkpoint = physics.save_state()
	physics.add_impulse(owners[:2], (2, 0, 0))
	tick(physics)
	assert body_state(physics, owners[:2]).awake.all()
	seen = []
	physics.subscribe(lambda world, commands, events: seen.append(events.copy()))
	physics.restore_state(checkpoint)
	assert not body_state(physics, owners[:2]).awake.any()
	physics.dispatch_events()
	assert not seen


def test_checkpoint_preconditions_and_damaged_native_bytes_are_nonmutating(setup):
	ecs, physics, owners = setup
	with pytest.raises(RuntimeError, match="between ticks"): physics.save_state()
	tick(physics)
	checkpoint = physics.save_state()
	before = body_state(physics, owners).position.copy()
	with pytest.raises(ValueError): physics.restore_state(replace(checkpoint, native=checkpoint.native[:-1]))
	np.testing.assert_array_equal(body_state(physics, owners).position, before)
	physics.world.add_impulses([physics.body_uids[int(owners[1])]], [[1, 0, 0]], at_tick=physics.world.tick + 10)
	with pytest.raises(RuntimeError, match="applied native"): physics.save_state()
	with pytest.raises(RuntimeError, match="applied native"): physics.restore_state(checkpoint)
	with pytest.raises(RuntimeError, match="apply native"): physics.reset_bodies([owners[0]], (0, 5, 0))


@pytest.mark.parametrize("change", ("rebuild", "configuration", "generation", "unreconciled"))
def test_checkpoint_rejects_changed_declarations_and_recycled_owners(setup, change):
	ecs, physics, owners = setup
	box = owners[0]
	tick(physics)
	checkpoint = physics.save_state()
	if change == "rebuild": ecs.get(MaterialTransform)[[box]].scale = (2, 1, 1)
	elif change == "configuration": ecs.get(RigidBody)[[box]].friction = 0.8
	elif change == "generation":
		ecs.delete(box)
		replacement = ecs.create(1)[0]
		ecs.add(replacement, Collider(0), RigidBody(), MaterialTransform(position=(0, 8, 0)))
	else: ecs.get(RigidBody)[[box]].mass = 2
	if change != "unreconciled": tick(physics)
	before = physics.world.latest().position.copy()
	with pytest.raises((RuntimeError, ValueError)): physics.restore_state(checkpoint)
	np.testing.assert_array_equal(physics.world.latest().position, before)


def test_checkpoint_rejects_foreign_adapter_and_unowned_native_bodies(setup, pj):
	ecs, physics, owners = setup
	tick(physics)
	checkpoint = physics.save_state()
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box(), 1: pj.Shape.box()}, joint_type=PhysicsJoint) as other:
		with pytest.raises(ValueError, match="different physics adapter"): other.restore_state(checkpoint)
	physics.world.add_bodies(pj.BodyBatch.boxes([[10, 10, 0]]))
	tick(physics)
	with pytest.raises(ValueError, match="exclusively adapter-owned"): physics.save_state()


@pytest.mark.parametrize("control", ("character", "ragdoll"))
def test_checkpoints_reject_native_control_state_that_cannot_be_restored(pj, control):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsCharacter, CharacterState, PhysicsRagdoll)
	owner = ecs.create(1)[0]
	ecs.add(owner, MaterialTransform())
	kwargs = {}
	if control == "character":
		ecs.add(owner, PhysicsCharacter(), CharacterState())
		kwargs["characters"] = {0: pj.CharacterCapabilities()}
	else:
		rig = Skeleton([-1], [[0, 1, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
		kwargs["ragdolls"] = {0: RagdollAsset(rig)}
		ecs.add(owner, PhysicsRagdoll(0))
	with EcsPhysicsAdapter(ecs, {}, settings=pj.WorldSettings(threads=1), **kwargs) as physics:
		tick(physics)
		with pytest.raises(NotImplementedError, match="native control state"): physics.save_state()


def test_selective_reset_discards_selected_inputs_and_preserves_other_history(setup):
	ecs, physics, owners = setup
	box, sibling, _, _ = owners
	physics.set_velocity([box, sibling], linear=(1, 0, 0), angular=(0, 1, 0))
	tick(physics, 3)
	previous = body_state(physics, [sibling]).position.copy()
	physics.add_impulse([box, sibling], [[1000, 0, 0], [2, 0, 0]])
	physics.reset_bodies([box], (0, 8, 0))
	tick(physics)
	state = body_state(physics, [box, sibling])
	np.testing.assert_allclose(state.position[0], [0, 8, 0])
	np.testing.assert_allclose(state.linear_velocity, [[0, 0, 0], [3, 0, 0]])
	np.testing.assert_allclose(state.angular_velocity[0], [0, 0, 0])
	np.testing.assert_allclose(physics.render_transforms([box, sibling], alpha=0)[0], np.concatenate(([[0, 8, 0]], previous)), atol=1e-6)


@pytest.mark.parametrize("bad", ("rotation", "position", "duplicate", "deleted"))
def test_invalid_reset_preserves_queued_inputs(setup, bad):
	ecs, physics, owners = setup
	box, sibling, _, _ = owners
	tick(physics)
	physics.add_impulse([sibling], (2, 0, 0))
	kwargs = {}
	selected = [box]
	position = (0, 8, 0)
	if bad == "rotation": kwargs["orientations"] = (0, 0, 0, 0)
	elif bad == "position": position = (np.nan, 0, 0)
	elif bad == "duplicate": selected = [box, box]
	else: ecs.delete(box)
	with pytest.raises(ValueError): physics.reset_bodies(selected, position, **kwargs)
	tick(physics)
	np.testing.assert_allclose(body_state(physics, [sibling]).linear_velocity, [[2, 0, 0]])


def test_reset_static_and_kinematic_bodies_updates_targets_without_advancing_time(setup):
	ecs, physics, owners = setup
	platform, floor = owners[2:]
	tick(physics)
	now = physics.world.tick
	physics.reset_bodies([platform, floor], [[9, 3, 0], [0, -5, 0]])
	assert physics.world.tick == now
	tick(physics, 2)
	np.testing.assert_allclose(body_state(physics, [platform, floor]).position, [[9, 3, 0], [0, -5, 0]])
	np.testing.assert_allclose(ecs.get(MaterialTransform)[[platform, floor]].position, [[9, 3, 0], [0, -5, 0]])


@pytest.mark.parametrize("configuration", ("motor", "filter"))
def test_checkpoint_rejects_imperative_configuration_changes(setup, configuration):
	ecs, physics, owners = setup
	box, sibling, _, _ = owners
	ecs.add(sibling, PhysicsJoint(key=1, kind=2, anchor=(3, 5, 0)))
	tick(physics)
	checkpoint = physics.save_state()
	if configuration == "motor": physics.set_joint_motor([sibling], [1], mode=2, target=0.2, max_force=10)
	else: physics.set_filter_table(np.ones((1, 1), dtype=bool))
	tick(physics)
	with pytest.raises(ValueError, match="configuration"): physics.restore_state(checkpoint)


def test_native_save_does_not_restore_character_motion_or_clock(pj):
	with pj.World(pj.WorldSettings(threads=1, gravity=(0, 0, 0))) as world:
		character = world.add_character((0, 2, 0), pj.CharacterCapabilities())
		world.run_ticks(1)
		data = world.save_state()
		before = character.position.copy()
		character.set_move((1, 0, 0), speed=2)
		world.run_ticks(20)
		after = character.position.copy()
		assert after[0] > before[0]
		now = world.tick
		world.restore_state(data)
		world.flush()
		assert world.tick == now
		np.testing.assert_array_equal(character.position, after)


def test_native_save_does_not_restore_ragdoll_mode_or_strength(pj):
	with pj.World(pj.WorldSettings(threads=1, gravity=(0, 0, 0))) as world:
		ragdoll = world.add_ragdoll(pj.RagdollSettings.humanoid(), mode=pj.RagdollMode.POWERED)
		world.run_ticks(1)
		data = world.save_state()
		ragdoll.set_mode(pj.RagdollMode.LIMP)
		ragdoll.set_strength(0)
		world.run_ticks(1)
		world.restore_state(data)
		world.flush()
		assert world.latest().ragdolls.mode[0] == pj.RagdollMode.LIMP
		assert world.latest().ragdolls.strength[0] == 0
