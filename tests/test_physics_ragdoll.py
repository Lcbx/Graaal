"""Native ragdolls, skeleton frames, sparse mapping and ECS ownership."""

import numpy as np
import pytest

from graaal.ECS import ECS, entity_index
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, PhysicsRagdoll, RigidBody, WORLD_ENTITY
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import RagdollAsset
from graaal.Animation.SkeletalAnimation import Skeleton, model_matrices, skin_matrices, skin_vertices


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


def skeleton(parents=(-1, 0, 1, 2)):
	count = len(parents)
	translation = np.array([[0, 1, 0]] + [[0, 0.3, 0]] * (count - 1), dtype=np.float32)
	rotation = np.tile([0, 0, 0, 1], (count, 1))
	scale = np.ones((count, 3))
	rig = Skeleton(parents, translation, rotation, scale, np.tile(np.eye(4), (count, 1, 1)))
	return Skeleton(parents, translation, rotation, scale, np.linalg.inv(model_matrices(rig, translation[None], rotation[None], scale[None])[0]))


def rest(rig, count=1):
	return tuple(np.tile(value, (count, 1, 1)) for value in (rig.translation, rig.rotation, rig.scale))


def tick(physics):
	physics.reconcile()
	physics.prepare_inputs()
	physics.step()
	physics.pull()
	return physics.dispatch_events()


@pytest.fixture
def setup(pj):
	rig = skeleton()
	asset = RagdollAsset(rig, bones=[0, 2], gravity_factor=0, self_collision=False)
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owner = ecs.create(1)[0]
	ecs.add(owner, PhysicsRagdoll(0), MaterialTransform(position=(4, 2, -3), rotation=Quaternion.from_y_rotation(0.8)))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box((20, 1, 20))}, settings=pj.WorldSettings(threads=1), ragdolls={0: asset}) as physics:
		yield ecs, owner, physics, asset


def test_sparse_targets_and_native_skinning_with_rotated_entity(setup):
	ecs, owner, physics, asset = setup
	tick(physics)
	pose = rest(asset.skeleton)
	pose[1][:, 1] = Quaternion.from_y_rotation(0.5)
	pose[1][:, 2] = Quaternion.from_x_rotation(0.25)
	pose[0][:, 3, 0] = 0.12
	physics.set_ragdoll_pose([owner], pose)
	tick(physics)
	skin = physics.ragdoll_skin([owner])
	expected = skin_matrices(asset.skeleton, model_matrices(asset.skeleton, *pose))
	np.testing.assert_allclose(skin, expected, atol=2e-5)
	np.testing.assert_allclose(physics.ragdoll_skin([owner], weight=0), expected, atol=2e-5)
	# Shape offsets must not be applied a second time to the joint snapshot.
	handle = physics.ragdolls[int(owner)]
	positions, _ = handle.pose()
	snapshot = physics.world.latest()
	np.testing.assert_allclose(snapshot.position[snapshot.rows_of(handle.bones)], positions)
	hit = physics.raycast([4, 5, -3], [0, -1, 0])
	assert hit.point[1] > np.max(positions[:, 1]) + 0.1
	assert np.all(physics.entities_of(handle.bones) == owner)
	np.testing.assert_array_equal(physics.bone_indices_of(handle.bones), [0, 1])
	assert set(physics.query_filter(exclude=[owner]).exclude) == set(handle.bones)
	# Existing CPU skinning consumes this exact palette and inverse bind convention.
	vertices = model_matrices(asset.skeleton, *rest(asset.skeleton))[0, :, :3, 3]
	joints = np.repeat(np.arange(len(vertices))[:, None], 4, axis=1)
	weights = np.tile([1, 0, 0, 0], (len(vertices), 1))
	actual, _ = skin_vertices(vertices, np.tile([0, 1, 0], (len(vertices), 1)), joints, weights, skin[0])
	np.testing.assert_allclose(actual, model_matrices(asset.skeleton, *pose)[0, :, :3, 3], atol=2e-5)


def test_roundtrip_batches_unmapped_bones_and_large_world_coordinates(pj):
	rig = skeleton()
	asset = RagdollAsset(rig, bones=[0, 2])
	pose = rest(rig, 2)
	pose[1][:, 1] = Quaternion.from_z_rotation(0.4)
	model = model_matrices(rig, *pose)
	frame = np.tile(np.eye(4), (2, 1, 1))
	frame[:, :3, :3] = Quaternion.from_x_rotation(0.9).matrix33
	frame[:, :3, 3] = [1e8, 3, -1e8]
	world = frame[:, None] @ model[:, asset.bones]
	turns = Quaternion.from_matrices_batch(world[..., :3, :3].reshape(-1, 3, 3)).reshape(2, 2, 4)
	result = asset.local_pose(world[..., :3, 3], turns, pose, position=frame[:, :3, 3], rotation=Quaternion.from_x_rotation(0.9))
	np.testing.assert_allclose(model_matrices(rig, *result), model, atol=2e-6)
	assert not any(np.shares_memory(a, b) for a, b in zip(result, pose))


@pytest.mark.parametrize("bones", ([0, 0], [2, 0], [1, 3, 0], [-1, 2], [0, 9], [0.0, 2.0]))
def test_invalid_mapping_rejected(pj, bones):
	with pytest.raises(ValueError): RagdollAsset(skeleton(), bones=bones)


@pytest.mark.parametrize("weight", (0, 0.5, 1))
def test_sparse_skinning_normalizes_animation_only_parent_rotations(setup, weight):
	ecs, owner, physics, asset = setup
	tick(physics)
	pose = rest(asset.skeleton)
	pose[1][:, 1] = Quaternion.from_z_rotation(0.7)
	pose[1][:, 3] = Quaternion.from_x_rotation(0.3)
	physics.set_ragdoll_pose([owner], pose)
	tick(physics)
	expected = physics.ragdoll_skin([owner], weight=weight)
	pose[1][:, 1] *= 2
	pose[1][:, 3] *= 3
	original = pose[1].copy()
	physics.set_ragdoll_pose([owner], pose)
	tick(physics)
	np.testing.assert_allclose(physics.ragdoll_skin([owner], weight=weight), expected, atol=2e-5)
	np.testing.assert_array_equal(pose[1], original)


def test_batched_held_targets_preserve_mixed_assets_owner_order_and_current_frames(pj):
	rig = skeleton()
	assets = {0: RagdollAsset(rig, bones=[0, 2], gravity_factor=0), 1: RagdollAsset(rig, bones=[0, 1, 2], gravity_factor=0)}
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owners = ecs.create(3)
	ecs.add(owners, PhysicsRagdoll(0), MaterialTransform())
	ecs.get(PhysicsRagdoll)[owners[1:2]].asset = 1
	with EcsPhysicsAdapter(ecs, {}, settings=pj.WorldSettings(threads=1), ragdolls=assets) as physics:
		tick(physics)
		order = owners[[2, 0, 1, 2]]
		pose = rest(rig, len(order))
		pose[0][:, 0, 0] = [1, 2, 3, 4]
		pose[1][:, 1] = Quaternion.from_z_rotation(0.4)
		pose[1][:, 1] *= 2
		pose[1][:, 2] = Quaternion.from_x_rotation(0.2)
		pose[0][:, 3, 0] = 0.15
		expected_pose = tuple(value[[1, 2, 3]].copy() for value in pose)
		physics.set_ragdoll_pose(order, pose)
		# Caller buffers may be reused immediately after submission.
		pose[0][:] = 100
		pose[1][:] = 0
		for offset in (0, 2):
			ecs.get(MaterialTransform)[owners].assign(position=np.array([[10, 1, 0], [0, 2, 10], [-10, 3, 0]]) + offset, rotation=Quaternion.from_y_rotation(0.6 + offset))
			tick(physics)
			expected = skin_matrices(rig, model_matrices(rig, *expected_pose))
			np.testing.assert_allclose(physics.ragdoll_skin(owners[[0, 2]]), expected[[0, 2]], atol=3e-5)
			np.testing.assert_allclose(physics.ragdoll_skin(owners[1:2]), expected[1:2], atol=3e-5)
		# Updating one owner retains the other owners' independently held targets.
		replacement = rest(rig)
		replacement[0][:, 0, 0] = -2
		physics.set_ragdoll_pose(owners[:1], replacement)
		tick(physics)
		for output, value in zip(expected_pose, replacement): output[:1] = value
		expected = skin_matrices(rig, model_matrices(rig, *expected_pose))
		np.testing.assert_allclose(physics.ragdoll_skin(owners[[2, 0, 2]]), expected[[2, 0, 2]], atol=3e-5)
		np.testing.assert_allclose(physics.ragdoll_skin(owners[1:2]), expected[1:2], atol=3e-5)


def test_invalid_batched_targets_preserve_all_held_poses(pj):
	rig = skeleton()
	assets = {0: RagdollAsset(rig, bones=[0, 2], gravity_factor=0), 1: RagdollAsset(rig, gravity_factor=0)}
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owners = ecs.create(2)
	ecs.add(owners, MaterialTransform())
	ecs.add(owners[0], PhysicsRagdoll(0))
	ecs.add(owners[1], PhysicsRagdoll(1))
	with EcsPhysicsAdapter(ecs, {}, settings=pj.WorldSettings(threads=1), ragdolls=assets) as physics:
		tick(physics)
		pose = rest(rig, 2)
		pose[0][:, 0, 0] = [1, 2]
		physics.set_ragdoll_pose(owners, pose)
		tick(physics)
		before = [physics.ragdoll_skin([owner]) for owner in owners]
		pose[0][:, 0, 0] = [5, 6]
		pose[2][1, 1] = 2
		with pytest.raises(ValueError, match="unit local scale"): physics.set_ragdoll_pose(owners, pose)
		with pytest.raises(ValueError, match="rows must match"): physics.set_ragdoll_pose(owners, rest(rig))
		tick(physics)
		for owner, expected in zip(owners, before): np.testing.assert_allclose(physics.ragdoll_skin([owner]), expected, atol=2e-5)


@pytest.mark.parametrize("publish_initial", (False, True))
def test_shared_capacity_body_ragdoll_transitions_both_directions(pj, publish_initial):
	asset = RagdollAsset(skeleton(), bones=[0, 2], gravity_factor=0)
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owners = ecs.create(2)
	ecs.add(owners, Collider(0), RigidBody(gravity_factor=0), MaterialTransform())
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, settings=pj.WorldSettings(threads=1, max_bodies=2), ragdolls={0: asset}) as physics:
		physics.reconcile()
		if publish_initial:
			physics.step()
			physics.pull()
		initial_tick = physics.world.tick
		old_bodies = np.array(list(physics.body_uids.values()), dtype=np.uint32)
		ecs.remove(owners, Collider)
		ecs.remove(owners, RigidBody)
		ecs.add(owners[0], PhysicsRagdoll(0))
		assert physics.reconcile()
		assert physics.world.tick == initial_tick
		physics.prepare_inputs()
		physics.step()
		physics.pull()
		handle = physics.ragdolls[int(owners[0])]
		assert physics.world.latest().count == 2
		assert not physics.body_uids
		assert np.all(physics.world.latest().rows_of(old_bodies) == -1)
		ecs.remove(owners[0], PhysicsRagdoll)
		ecs.add(owners, Collider(0), RigidBody(gravity_factor=0))
		assert physics.reconcile()
		assert physics.world.tick == initial_tick + 1
		physics.prepare_inputs()
		physics.step()
		physics.pull()
		assert physics.world.latest().count == 2
		assert not len(physics.world.latest().ragdolls)
		assert len(physics.body_uids) == 2
		assert np.all(physics.world.latest().rows_of(handle.bones) == -1)


def test_shared_capacity_simultaneous_exchanges_and_overflow_validation(pj):
	asset = RagdollAsset(skeleton(), bones=[0, 2], gravity_factor=0)
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owners = ecs.create(3)
	ecs.add(owners, MaterialTransform())
	ecs.add(owners[:2], Collider(0))
	ecs.add(owners[2], PhysicsRagdoll(0))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, settings=pj.WorldSettings(threads=1, max_bodies=4), ragdolls={0: asset}) as physics:
		tick(physics)
		old_uids = np.concatenate(([physics.body_uids[int(owners[0])]], physics.ragdolls[int(owners[2])].bones)).astype(np.uint32)
		ecs.remove(owners[:2], Collider)
		ecs.remove(owners[2], PhysicsRagdoll)
		ecs.add(owners[0], PhysicsRagdoll(0))
		ecs.add(owners[1:], Collider(0))
		physics.reconcile()
		assert physics.world.tick == 1
		tick(physics)
		assert physics.world.latest().count == 4
		assert np.all(physics.world.latest().rows_of(old_uids) == -1)
		kept_uids = physics.world.latest().uid.copy()
		ecs.remove(owners[1:], Collider)
		ecs.add(owners[1:], PhysicsRagdoll(0))
		with pytest.raises(ValueError, match="max_bodies"): physics.reconcile()
		physics.world.flush()
		assert physics.world.latest().count == 4
		assert np.all(physics.world.latest().rows_of(kept_uids) >= 0)


@pytest.mark.parametrize("invalid_first", (False, True))
def test_shared_capacity_unpublished_ragdoll_to_bodies_validates_before_removal(pj, invalid_first):
	asset = RagdollAsset(skeleton(), bones=[0, 2], gravity_factor=0)
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owners = ecs.create(2)
	ecs.add(owners, MaterialTransform())
	ecs.add(owners[0], PhysicsRagdoll(0))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, settings=pj.WorldSettings(threads=1, max_bodies=2), ragdolls={0: asset}) as physics:
		physics.reconcile()
		handle = physics.ragdolls[int(owners[0])]
		ecs.remove(owners[0], PhysicsRagdoll)
		ecs.add(owners, Collider(0), RigidBody(mass=-1 if invalid_first else 1))
		if invalid_first:
			with pytest.raises(ValueError, match="mass"): physics.reconcile()
			assert physics.ragdolls[int(owners[0])] is handle
			physics.world.flush()
			assert np.all(physics.world.latest().rows_of(handle.bones) >= 0)
		ecs.get(RigidBody)[owners].mass = 1
		physics.reconcile()
		assert physics.world.tick == 0
		physics.prepare_inputs()
		physics.step()
		physics.pull()
		assert physics.world.latest().count == 2
		assert not physics.ragdolls
		assert np.all(physics.world.latest().rows_of(handle.bones) == -1)


def test_authored_settings_frames_and_scale_validation(pj):
	rig = skeleton()
	fallback = RagdollAsset(rig, bones=[0, 2])
	settings = pj.RagdollSettings.from_dict(fallback.settings.as_dict())
	asset = RagdollAsset(rig, settings, bones=[0, 2])
	assert asset.settings is settings
	settings.bones[1].translation = (0, 8, 0)
	with pytest.raises(ValueError, match="rest joint frames"): RagdollAsset(rig, settings, bones=[0, 2])
	pose = rest(rig)
	pose[2][:, 0] = 2
	with pytest.raises(ValueError, match="unit local scale"): asset.target_pose(pose)
	forest = skeleton((-1, -1))
	with pytest.raises(ValueError, match="one parent"): RagdollAsset(forest)


def test_lifecycle_rebuild_recycled_handle_capacity_and_close(setup):
	ecs, owner, physics, asset = setup
	tick(physics)
	pose = rest(asset.skeleton)
	pose[0][:, 0, 0] = 2
	physics.set_ragdoll_pose([owner], pose)
	tick(physics)
	handle = physics.ragdolls[int(owner)]
	uids = handle.bones.copy()
	physics._ragdoll_assets[1] = RagdollAsset(asset.skeleton, bones=[0, 1, 2])
	ecs.get(PhysicsRagdoll)[[owner]].asset = 1
	tick(physics)
	assert physics.ragdolls[int(owner)].id != handle.id
	assert np.all(physics.world.latest().rows_of(uids) == -1)
	assert physics.world.latest().count == 3
	np.testing.assert_allclose(physics.ragdoll_skin([owner]), np.tile(np.eye(4)[:3], (1, len(asset.skeleton.parents), 1, 1)), atol=3e-5)
	ecs.delete(owner)
	replacement = ecs.create(1)[0]
	assert entity_index(owner) == entity_index(replacement)
	ecs.add(replacement, PhysicsRagdoll(0), MaterialTransform())
	tick(physics)
	assert owner not in physics.ragdolls
	assert np.all(physics.entities_of(uids) == WORLD_ENTITY)
	assert np.all(physics.bone_indices_of(uids) == -1)
	assert physics.world.latest().count == 2
	ecs.remove(replacement, PhysicsRagdoll)
	physics.reconcile()
	physics.close()
	assert physics.world.latest().count == 0
	assert not len(physics.world.latest().ragdolls)
	assert not physics.ragdolls
	assert not physics._ragdoll_targets
	physics.close()


@pytest.mark.parametrize("change", ("delete", "remove_ragdoll", "remove_transform"))
def test_held_target_is_not_sent_to_an_owner_removed_before_prepare(setup, change):
	ecs, owner, physics, asset = setup
	tick(physics)
	handle = physics.ragdolls[int(owner)]
	before = physics.world.latest().position.copy()
	pose = rest(asset.skeleton)
	pose[0][:, 0, 0] = 5
	physics.set_ragdoll_pose([owner], pose)
	if change == "delete":
		ecs.delete(owner)
		replacement = ecs.create(1)[0]
		ecs.add(replacement, PhysicsRagdoll(0), MaterialTransform(position=(30, 0, 0)))
	else:
		ecs.remove(owner, PhysicsRagdoll if change == "remove_ragdoll" else MaterialTransform)
	physics.prepare_inputs()
	physics.step()
	physics.pull()
	physics.dispatch_events()
	np.testing.assert_allclose(physics.world.latest().position, before, atol=1e-6)
	tick(physics)
	assert owner not in physics._ragdoll_targets
	assert np.all(physics.world.latest().rows_of(handle.bones) == -1)
	assert physics.world.stats()["rejected"] == 0


def test_modes_strength_impulse_and_recovery(setup, pj):
	ecs, owner, physics, asset = setup
	tick(physics)
	component = ecs.get(PhysicsRagdoll)
	handle = physics.ragdolls[int(owner)]
	component[[owner]].assign(mode=1, strength=0.4, blend=0.05)
	for _ in range(12): tick(physics)
	assert physics.world.latest().ragdolls.mode[0] == pj.RagdollMode.POWERED
	handle.hit(1, [3, 0, 0])
	tick(physics)
	assert np.max(np.abs(physics.world.bodies_state(handle.bones).linear_velocity)) > 0.05
	component[[owner]].assign(mode=2, blend=0)
	for _ in range(3): tick(physics)
	assert physics.world.latest().ragdolls.mode[0] == pj.RagdollMode.LIMP
	assert physics.world.latest().ragdolls.strength[0] == 0
	component[[owner]].assign(mode=0, blend=0.1)
	for _ in range(30): tick(physics)
	np.testing.assert_allclose(physics.ragdoll_skin([owner]), np.tile(np.eye(4)[:3], (1, len(asset.skeleton.parents), 1, 1)), atol=3e-5)


def test_bone_events_removal_and_query_exclusion(setup, pj):
	ecs, owner, physics, asset = setup
	floor = ecs.create(1)[0]
	ecs.add(floor, Collider(0), MaterialTransform(position=(4, 2.65, -3)))
	ecs.get(PhysicsRagdoll)[[owner]].mode = 2
	events = tick(physics)
	contacts = events[events["kind"] == pj.EventKind.CONTACT]
	assert len(contacts)
	for side in ("a", "b"):
		owned = contacts[f"entity_{side}"] == owner
		assert np.all(contacts[f"bone_{side}"][owned] >= 0)
		assert np.all(contacts[f"bone_{side}"][~owned] == -1)
	seen = []
	physics.subscribe(lambda world, commands, records: seen.extend(records), kind=pj.EventKind.CONTACT, entity=owner)
	bones = physics.ragdolls[int(owner)].bones.copy()
	ecs.remove(owner, PhysicsRagdoll)
	events = tick(physics)
	assert len(seen)
	ends = events[(events["kind"] == pj.EventKind.CONTACT) & (events["phase"] == pj.EventPhase.END)]
	assert np.any((ends["entity_a"] == owner) | (ends["entity_b"] == owner))
	for side in ("a", "b"):
		owned = ends[f"entity_{side}"] == owner
		indices = ends[f"bone_{side}"][owned]
		assert np.all(indices >= 0)
		np.testing.assert_array_equal(bones[indices], ends[side][owned])
	assert physics.world.latest().count == 1
	assert physics.raycast([4, 10, -3], [0, -1, 0], exclude=[floor]) is None


@pytest.mark.parametrize("field,value,match", (("mode", 3, "mode"), ("blend", -1, "blend"), ("strength", np.nan, "strength"), ("asset", 999, "unknown")))
def test_bad_declarations_are_rejected_before_creation(setup, field, value, match):
	ecs, owner, physics, asset = setup
	setattr(ecs.get(PhysicsRagdoll)[[owner]], field, value)
	with pytest.raises(ValueError, match=match): physics.reconcile()
	assert not physics.ragdolls


def test_mixed_body_and_scaled_entity_rejected(setup):
	ecs, owner, physics, asset = setup
	ecs.add(owner, Collider(0))
	with pytest.raises(ValueError, match="cannot also"): physics.reconcile()
	ecs.remove(owner, Collider)
	ecs.get(MaterialTransform)[[owner]].scale = 2
	with pytest.raises(ValueError, match="entity scale"): physics.reconcile()


def test_native_humanoid_limits_and_shape_offsets(pj):
	settings = pj.RagdollSettings.humanoid(root_force=1000, root_torque=500)
	rig = Skeleton(settings.parents, settings.rest_translations, settings.rest_rotations, np.ones((len(settings), 3)), np.tile(np.eye(4), (len(settings), 1, 1)))
	asset = RagdollAsset(rig, settings)
	with pj.World(pj.WorldSettings(threads=1)) as world:
		handle = world.add_ragdoll(asset.settings, mode=pj.RagdollMode.POWERED)
		world.run_ticks(1)
		handle.hit(4, [20, 10, 10])
		world.run_ticks(120)
		positions, orientations = handle.pose()
		assert np.isfinite(positions).all() and np.isfinite(orientations).all()
		# Native joint anchors stay attached under an impulse; limits remain authored.
		lengths = np.linalg.norm(positions[1:] - positions[settings.parents[1:]], axis=1)
		np.testing.assert_allclose(lengths, np.linalg.norm(settings.rest_translations[1:], axis=1), atol=0.04)
		assert settings.bones[4].swing == (0.7, 0.05)


def test_native_limits_constrain_an_out_of_range_motor_target(pj):
	settings = pj.RagdollSettings.from_skeleton(["root", "child"], [-1, 0], [[0, 1, 0], [0, 0.6, 0]], self_collision=False, root_force=10000, root_torque=10000, gravity_factor=0)
	settings.bones[1].twist = (-0.1, 0.1)
	settings.bones[1].swing = (0.1, 0.1)
	settings.bones[1].strength = 100
	with pj.World(pj.WorldSettings(threads=1)) as world:
		handle = world.add_ragdoll(settings, mode=pj.RagdollMode.POWERED)
		handle.set_pose(np.array([[0, 0, 0, 1], Quaternion.from_x_rotation(1.3)]))
		world.run_ticks(240)
		_, rotation = handle.pose()
		relative = Quaternion.multiply_batch(rotation[0] * [-1, -1, -1, 1], rotation[1])
		angle = 2 * np.arccos(np.clip(abs(relative[3]), 0, 1))
		assert 0.08 < angle < 0.13


def test_capacity_rebuild_applies_removal_without_advancing_time(pj):
	asset = RagdollAsset(skeleton(), bones=[0, 2], gravity_factor=0)
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owner = ecs.create(1)[0]
	ecs.add(owner, PhysicsRagdoll(0), MaterialTransform())
	with EcsPhysicsAdapter(ecs, {}, settings=pj.WorldSettings(threads=1, max_bodies=2), ragdolls={0: asset, 1: asset}) as physics:
		tick(physics)
		old = physics.ragdolls[int(owner)].bones.copy()
		ecs.get(PhysicsRagdoll)[[owner]].asset = 1
		physics.reconcile()
		assert physics.world.tick == 1
		physics.prepare_inputs()
		physics.step()
		physics.pull()
		assert physics.world.latest().count == 2
		assert np.all(physics.world.latest().rows_of(old) == -1)
		extra = ecs.create(1)[0]
		ecs.add(extra, PhysicsRagdoll(0), MaterialTransform())
		with pytest.raises(ValueError, match="max_bodies"): physics.reconcile()
		assert physics.world.latest().count == 2


def test_ragdoll_history_per_bone_repeated_owners_same_tick_and_skipped_ticks(pj):
	asset = RagdollAsset(skeleton(), bones=[0, 2], gravity_factor=0)
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owner = ecs.create(1)[0]
	ecs.add(owner, PhysicsRagdoll(0), MaterialTransform())
	with EcsPhysicsAdapter(ecs, {}, settings=pj.WorldSettings(threads=1), ragdolls={0: asset}, interpolate=True) as physics:
		tick(physics)
		before = physics.ragdoll_skin([owner])
		pose = rest(asset.skeleton)
		pose[0][:, 0, 0] += 2
		physics.set_ragdoll_pose([owner], pose)
		tick(physics)
		after = physics.ragdoll_skin([owner])
		assert np.max(abs(after - before)) > 1
		np.testing.assert_allclose(physics.ragdoll_skin([owner], alpha=0), before, atol=1e-6)
		np.testing.assert_allclose(physics.ragdoll_skin([owner], alpha=1), after, atol=1e-6)
		np.testing.assert_allclose(physics.ragdoll_skin([owner], alpha=0.5), (before + after) / 2, atol=1e-6)
		np.testing.assert_allclose(physics.ragdoll_skin([owner, owner], alpha=0), np.repeat(before, 2, axis=0), atol=1e-6)
		physics.pull()
		np.testing.assert_allclose(physics.ragdoll_skin([owner], alpha=0), before, atol=1e-6)
		physics.world.run_ticks(3)
		physics.pull()
		np.testing.assert_allclose(physics.ragdoll_skin([owner], alpha=0), physics.ragdoll_skin([owner], alpha=1), atol=1e-6)
		with pytest.raises(ValueError, match="alpha"): physics.ragdoll_skin([owner], alpha=1.1)
		# Rendering never moves the gameplay frame or the held animation target.
		np.testing.assert_array_equal(ecs.get(MaterialTransform)[[owner]].position, [[0, 0, 0]])
		np.testing.assert_array_equal(physics._ragdoll_poses[int(owner)][0], pose[0])
		ecs.remove(owner, MaterialTransform)
		tick(physics)
		assert not physics._ragdoll_history
		assert physics.world.latest().count == 0


def test_deleted_ragdoll_events_do_not_alias_reused_entity(setup, pj):
	ecs, owner, physics, asset = setup
	floor = ecs.create(1)[0]
	ecs.add(floor, Collider(0), MaterialTransform(position=(4, 2.65, -3)))
	ecs.get(PhysicsRagdoll)[[owner]].mode = 2
	tick(physics)
	ecs.delete(owner)
	replacement = ecs.create(1)[0]
	ecs.add(replacement, PhysicsRagdoll(0), MaterialTransform(position=(50, 0, 0)))
	events = tick(physics)
	ends = events[(events["kind"] == pj.EventKind.CONTACT) & (events["phase"] == pj.EventPhase.END)]
	assert len(ends)
	for side in ("a", "b"):
		assert not np.any(ends[f"entity_{side}"] == replacement)
		assert np.all(ends[f"bone_{side}"] == -1)
