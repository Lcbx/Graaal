"""Native character authority, inputs, ownership and shared capacity."""

import numpy as np
import pytest

from graaal.ECS import ECS, Scheduler, entity_index
from graaal.EcsPhysics import CharacterState, Collider, EcsPhysicsAdapter, PhysicsCharacter, PhysicsRagdoll, RigidBody, WORLD_ENTITY
from graaal.MaterialTransform import MaterialTransform
from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import RagdollAsset
from graaal.SkeletalAnimation import Skeleton


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


@pytest.fixture
def setup(pj):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsCharacter, CharacterState, PhysicsRagdoll)
	character, floor = ecs.create(2)
	ecs.add(character, PhysicsCharacter(), CharacterState(), MaterialTransform())
	ecs.add(floor, Collider(0), MaterialTransform(position=(0, -0.5, 0)))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box((30, 1, 30)), 1: pj.Shape.box((1, 4, 8)), 2: pj.Shape.box((1, 0.2, 4))}, characters={0: pj.CharacterCapabilities()}, ragdolls={}, settings=pj.WorldSettings(threads=1), interpolate=True) as physics:
		yield ecs, character, floor, physics


def tick(physics, count=1):
	for _ in range(count):
		physics.reconcile()
		physics.prepare_inputs()
		physics.step()
		physics.pull()
		physics.dispatch_events()
	return physics.events


def test_feet_authority_ground_state_and_held_batched_inputs(setup):
	ecs, owner, floor, physics = setup
	tick(physics)
	output = ecs.get(CharacterState)
	assert output[[owner]].grounded[0]
	assert output[[owner]].ground[0] == floor
	np.testing.assert_allclose(ecs.get(MaterialTransform)[[owner]].position, [[0, 0, 0]], atol=1e-5)
	ecs.get(MaterialTransform)[[owner]].position = (100, 50, 0)
	physics.move_characters([owner], [1, 0, 0], speed=2)
	tick(physics, 60)
	position = ecs.get(MaterialTransform)[[owner]].position[0]
	assert 0.9 < position[0] < 1.1
	assert abs(position[1]) < 1e-4
	assert output[[owner]].velocity[0, 0] == pytest.approx(2, abs=0.01)
	physics.move_characters([owner], [0, 0, 0])
	tick(physics)
	assert abs(output[[owner]].velocity[0, 0]) < 0.01
	assert physics.world.latest().count == 1  # Proxies are outside regular snapshot rows.
	assert physics.world.reserved_bodies == 2


def test_wall_blocks_requested_motion_and_low_step_is_walkable(setup):
	ecs, owner, floor, physics = setup
	wall = ecs.create(1)[0]
	ecs.add(wall, Collider(1), MaterialTransform(position=(3, 2, 0)))
	step = ecs.create(1)[0]
	ecs.add(step, Collider(2), MaterialTransform(position=(1, 0.1, 0)))
	physics.reconcile()
	physics.move_characters([owner], [1, 0, 0], speed=3)
	heights = []
	for _ in range(180):
		tick(physics)
		heights.append(ecs.get(MaterialTransform)[[owner]].position[0, 1])
	position = ecs.get(MaterialTransform)[[owner]].position[0]
	assert max(heights) > 0.15
	assert 2.1 < position[0] < 2.3
	assert abs(position[1]) < 0.02
	assert np.linalg.norm(ecs.get(CharacterState)[[owner]].velocity) < 0.01


def test_jump_crouch_and_character_events_on_scheduler_thread(setup, pj):
	ecs, owner, floor, physics = setup
	seen = []
	physics.subscribe(lambda world, commands, events: seen.extend(events.copy()), kind=pj.EventKind.CHARACTER, entity=owner)
	tick(physics)
	physics.jump_characters([owner])
	tick(physics)
	assert not ecs.get(CharacterState)[[owner]].grounded[0]
	assert ecs.get(CharacterState)[[owner]].velocity[0, 1] > 4
	tick(physics, 150)
	assert ecs.get(CharacterState)[[owner]].grounded[0]
	events = np.array(seen, dtype=physics.event_dtype)
	assert {pj.EventPhase.LAND, pj.EventPhase.JUMP} <= set(events["phase"])
	assert np.all(events["entity_a"] == owner)
	land = events[events["phase"] == pj.EventPhase.LAND]
	assert np.all(land["entity_b"] == floor)
	assert np.all(events["bone_a"] == -1)
	physics.character_commands([owner], pj.CharacterOp.CROUCH, code=True)
	tick(physics)
	assert ecs.get(CharacterState)[[owner]].height[0] == pytest.approx(1)
	assert abs(ecs.get(MaterialTransform)[[owner]].position[0, 1]) < 0.01


def test_proxy_query_exclusion_and_trigger_owner_identity(setup, pj):
	ecs, owner, floor, physics = setup
	sensor = ecs.create(1)[0]
	ecs.add(sensor, Collider(1, sensor=True), MaterialTransform(position=(0, 2, 0)))
	events = tick(physics)
	uid = physics.characters[int(owner)].id
	assert physics.entities_of(uid) == owner
	assert physics.bone_indices_of(uid) == -1
	hit = physics.raycast([0, 5, 0], [0, -1, 0])
	assert hit.uid == uid
	assert physics.entities_of(physics.raycast([0, 5, 0], [0, -1, 0], exclude=[owner]).uid) == floor
	assert physics.entities_of(physics.raycast([0, 5, 0], [0, -1, 0], filter=physics.query_filter(characters=False)).uid) == floor
	triggers = events[events["kind"] == pj.EventKind.TRIGGER]
	assert len(triggers)
	assert owner in np.concatenate((triggers["entity_a"], triggers["entity_b"]))
	ecs.get(PhysicsCharacter)[[owner]].enabled = False
	events = tick(physics)
	ends = events[(events["kind"] == pj.EventKind.TRIGGER) & (events["phase"] == pj.EventPhase.END)]
	assert len(ends)
	assert owner in np.concatenate((ends["entity_a"], ends["entity_b"]))
	assert not ecs.get(CharacterState)[[owner]].grounded[0]


def test_rebuild_preserves_native_feet_velocity_and_deleted_inputs_do_not_alias(setup, pj):
	ecs, owner, floor, physics = setup
	tick(physics)
	physics.character_commands([owner], pj.CharacterOp.IMPULSE, values=(1, 2, 0, 0))
	tick(physics)
	old = physics.characters[int(owner)].id
	before = ecs.get(MaterialTransform)[[owner]].position.copy()
	physics._character_assets[1] = pj.CharacterCapabilities(standHeight=2)
	ecs.get(PhysicsCharacter)[[owner]].capabilities = 1
	ecs.get(MaterialTransform)[[owner]].position = (50, 50, 50)
	tick(physics)
	assert physics.characters[int(owner)].id != old
	np.testing.assert_allclose(ecs.get(MaterialTransform)[[owner]].position, before, atol=0.03)
	assert ecs.get(CharacterState)[[owner]].velocity[0, 1] > 1
	physics.move_characters([owner], [1, 0, 0], speed=50)
	ecs.delete(owner)
	replacement = ecs.create(1)[0]
	assert entity_index(replacement) == entity_index(owner)
	ecs.add(replacement, PhysicsCharacter(), CharacterState(), MaterialTransform())
	tick(physics)
	assert old not in physics.query_filter(exclude=[replacement]).exclude
	assert physics.entities_of(old) == WORLD_ENTITY
	assert not np.any(physics.events["entity_a"] == owner)
	assert ecs.get(MaterialTransform)[[replacement]].position[0, 0] == 0
	assert len(physics.characters) == 1
	physics.close()
	assert not len(physics.world.latest().characters)
	assert physics.world.reserved_bodies == 0


def test_per_character_batches_and_optional_output_components(setup):
	ecs, owner, floor, physics = setup
	other = ecs.create(1)[0]
	ecs.add(other, PhysicsCharacter(proxy=False), MaterialTransform(position=(0, 0, 3)))
	physics.reconcile()
	physics.move_characters([other, owner], [[1, 0, 0], [-1, 0, 0]], speed=[1, 2])
	tick(physics, 30)
	positions = ecs.get(MaterialTransform)[[other, owner]].position
	assert positions[0, 0] > 0.2 and positions[1, 0] < -0.4
	assert physics.characters[int(other)].id not in physics.overlap(physics.pj.Shape.box((1, 3, 1)), positions[0] + [0, 1, 0])


def test_character_interpolation_teleport_and_same_tick_pull(setup, pj):
	ecs, owner, floor, physics = setup
	tick(physics)
	before = physics.render_transforms([owner], alpha=1)[0]
	physics.move_characters([owner], [1, 0, 0], speed=2)
	tick(physics)
	after = physics.render_transforms([owner], alpha=1)[0]
	np.testing.assert_allclose(physics.render_transforms([owner, owner], alpha=0)[0], np.repeat(before, 2, axis=0))
	physics.pull()
	np.testing.assert_allclose(physics.render_transforms([owner], alpha=0.5)[0], (before + after) / 2)
	physics.character_commands([owner], pj.CharacterOp.TELEPORT, values=(5, 0.9, 0, 0))
	tick(physics)
	np.testing.assert_allclose(physics.render_transforms([owner], alpha=0)[0], physics.render_transforms([owner], alpha=1)[0])
	physics.world.run_ticks(3)
	physics.pull()
	np.testing.assert_allclose(physics.render_transforms([owner], alpha=0)[0], physics.render_transforms([owner], alpha=1)[0])


@pytest.mark.parametrize("proxy", (False, True))
def test_character_body_ragdoll_shared_capacity_and_unpublished_transitions(pj, proxy):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsCharacter, CharacterState, PhysicsRagdoll)
	owner = ecs.create(1)[0]
	ecs.add(owner, PhysicsCharacter(proxy=proxy), CharacterState(), MaterialTransform())
	rig = Skeleton([-1], [[0, 1, 0]], [[0, 0, 0, 1]], [[1, 1, 1]], [np.eye(4)])
	asset = RagdollAsset(rig, gravity_factor=0)
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, characters={0: pj.CharacterCapabilities()}, ragdolls={0: asset}, settings=pj.WorldSettings(threads=1, max_bodies=1)) as physics:
		physics.reconcile()
		assert physics.world.reserved_bodies == 1
		ecs.get(PhysicsCharacter)[[owner]].enabled = False
		ecs.add(owner, PhysicsRagdoll(0))
		physics.reconcile()
		assert physics.world.tick == 0
		tick(physics)
		assert not len(physics.world.latest().characters)
		assert len(physics.world.latest().ragdolls) == 1
		ecs.remove(owner, PhysicsRagdoll)
		ecs.get(PhysicsCharacter)[[owner]].enabled = True
		tick(physics)
		assert len(physics.world.latest().characters) == 1
		assert not len(physics.world.latest().ragdolls)
		ecs.remove(owner, PhysicsCharacter)
		ecs.add(owner, Collider(0))
		tick(physics)
		assert physics.world.latest().count == 1
		ecs.remove(owner, Collider)
		ecs.add(owner, PhysicsCharacter(proxy=proxy))
		tick(physics)
		assert len(physics.world.latest().characters) == 1


@pytest.mark.parametrize("field,value", (("capabilities", 99), ("gravity", -1), ("gravity", np.inf)))
def test_invalid_character_declarations_leave_live_controller_untouched(setup, field, value):
	ecs, owner, floor, physics = setup
	tick(physics)
	old = physics.characters[int(owner)]
	setattr(ecs.get(PhysicsCharacter)[[owner]], field, value)
	with pytest.raises(ValueError): physics.reconcile()
	assert physics.characters[int(owner)] is old


def test_mixed_authority_scaled_characters_and_bad_inputs_rejected(setup):
	ecs, owner, floor, physics = setup
	ecs.add(owner, RigidBody())
	with pytest.raises(ValueError, match="cannot also"): physics.reconcile()
	ecs.remove(owner, RigidBody)
	ecs.get(MaterialTransform)[[owner]].scale = 2
	with pytest.raises(ValueError, match="scale"): physics.reconcile()
	ecs.get(MaterialTransform)[[owner]].scale = 1
	with pytest.raises(ValueError): physics.move_characters([owner], [1, 0, 0], speed=-1)
	with pytest.raises(ValueError): physics.move_characters([owner], [1, 0, 0], mode=8)
	with pytest.raises(ValueError): physics.character_commands([owner], 99)
	with pytest.raises(ValueError): physics.character_commands([owner], 1, values=(np.nan, 0, 0, 0))


def test_scheduler_event_commands_take_effect_on_next_tick(setup, pj):
	ecs, owner, floor, physics = setup
	schedule = Scheduler(ecs)
	for name, callback in (("sync", physics.reconcile), ("input", physics.prepare_inputs), ("step", physics.step), ("pull", physics.pull), ("events", physics.dispatch_events)):
		schedule.add(name, callback, phase="fixed_update", after=() if name == "sync" else previous)
		previous = name
	physics.subscribe(lambda world, commands, events: physics.jump_characters([owner]) if np.any(events["phase"] == pj.EventPhase.LAND) else None, kind=pj.EventKind.CHARACTER)
	schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
	assert ecs.get(CharacterState)[[owner]].grounded[0]
	schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
	assert ecs.get(CharacterState)[[owner]].velocity[0, 1] > 4


@pytest.mark.parametrize("max_slope,walkable", ((20, False), (50, True)))
def test_authored_slope_capability_controls_walkable_ground(pj, max_slope, walkable):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsCharacter, CharacterState)
	owner, slope = ecs.create(2)
	ecs.add(owner, PhysicsCharacter(), CharacterState(), MaterialTransform(position=(0, 2, 0)))
	ecs.add(slope, Collider(0), MaterialTransform(position=(0, -0.2, 0), rotation=Quaternion.from_z_rotation(np.pi / 6)))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box((20, 0.4, 10))}, characters={0: pj.CharacterCapabilities(maxSlope=max_slope)}, settings=pj.WorldSettings(threads=1)) as physics:
		grounded = []
		for _ in range(180):
			tick(physics)
			grounded.append(ecs.get(CharacterState)[[owner]].grounded[0])
		assert bool(np.any(grounded)) == walkable
		assert ecs.get(CharacterState)[[owner]].ground_normal[0, 1] == pytest.approx(np.cos(np.pi / 6), abs=1e-4)


def test_character_masks_input_copies_and_transform_removal(setup):
	ecs, owner, floor, physics = setup
	ecs.get(PhysicsCharacter)[[owner]].mask = 0
	physics.reconcile()
	direction = np.array([1, 0, 0])
	physics.move_characters([owner], direction)
	direction[:] = -1
	tick(physics, 90)
	position = ecs.get(MaterialTransform)[[owner]].position[0]
	assert position[0] > 0
	assert position[1] < -2
	assert not ecs.get(CharacterState)[[owner]].grounded[0]
	ecs.remove(owner, MaterialTransform)
	physics.reconcile()
	physics.close()
	assert not physics._character_history
	assert physics.world.reserved_bodies == 0


def test_character_removed_events_keep_old_generation_unresolved(setup, pj):
	ecs, owner, floor, physics = setup
	sensor = ecs.create(1)[0]
	ecs.add(sensor, Collider(1, sensor=True), MaterialTransform(position=(0, 2, 0)))
	tick(physics)
	old_uid = physics.characters[int(owner)].id
	ecs.delete(owner)
	replacement = ecs.create(1)[0]
	ecs.add(replacement, PhysicsCharacter(), CharacterState(), MaterialTransform(position=(10, 0, 0)))
	events = tick(physics)
	ends = events[(events["kind"] == pj.EventKind.TRIGGER) & (events["phase"] == pj.EventPhase.END)]
	assert len(ends)
	assert not np.any(ends["entity_a"] == replacement) and not np.any(ends["entity_b"] == replacement)
	for side in ("a", "b"):
		assert np.all(ends[f"entity_{side}"][ends[side] == old_uid] == WORLD_ENTITY)


def test_character_overflow_preserves_current_owners_and_disabled_render_uses_ecs(pj):
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsCharacter, CharacterState)
	owner = ecs.create(1)[0]
	ecs.add(owner, PhysicsCharacter(), CharacterState(), MaterialTransform())
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, characters={0: pj.CharacterCapabilities()}, settings=pj.WorldSettings(threads=1, max_bodies=1), interpolate=True) as physics:
		tick(physics)
		handle = physics.characters[int(owner)]
		extra = ecs.create(1)[0]
		ecs.add(extra, Collider(0), MaterialTransform())
		with pytest.raises(ValueError, match="max_bodies"): physics.reconcile()
		assert physics.characters[int(owner)] is handle
		assert physics.world.latest().characters.rows[handle.id] == 0
		ecs.get(PhysicsCharacter)[[owner]].enabled = False
		ecs.get(MaterialTransform)[[owner]].position = (5, 5, 5)
		np.testing.assert_array_equal(physics.render_transforms([owner], alpha=0)[0], [[5, 5, 5]])


def test_batched_character_history_survives_row_compaction_generation_reuse_and_rebuild(setup):
	ecs, owner, floor, physics = setup
	other, third = ecs.create(2)
	for entity, z in ((other, 3), (third, 6)):
		ecs.add(entity, PhysicsCharacter(), MaterialTransform(position=(0, 0, z)))
	tick(physics)
	physics.move_characters([owner, other, third], [1, 0, 0], speed=[1, 2, 3])
	tick(physics)
	before = physics.render_transforms([third, owner, third], alpha=1)[0]
	ecs.delete(other)
	replacement = ecs.create(1)[0]
	ecs.add(replacement, PhysicsCharacter(), MaterialTransform(position=(10, 0, 3)))
	tick(physics)
	np.testing.assert_allclose(physics.render_transforms([third, owner, third], alpha=0)[0], before)
	np.testing.assert_allclose(physics.render_transforms([replacement], alpha=0)[0], physics.render_transforms([replacement], alpha=1)[0])
	before = physics.render_transforms([owner], alpha=1)[0]
	ecs.get(PhysicsCharacter)[[third]].gravity = 0
	tick(physics)
	np.testing.assert_allclose(physics.render_transforms([owner], alpha=0)[0], before)
	np.testing.assert_allclose(physics.render_transforms([third], alpha=0)[0], physics.render_transforms([third], alpha=1)[0])
