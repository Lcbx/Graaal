"""Optional ECS rigid-body ownership over a standalone pyjoltphysics world."""

from dataclasses import fields
from types import MappingProxyType
from typing import get_type_hints

import numpy as np

from .ecs_types import as_entities, component, _synchronous_callable
from .Math import Vec3, Vec4, Quaternion


@component
class RigidBody:
	"""Declarative settings; motion codes are static=0, kinematic=1, dynamic=2."""
	motion: np.uint8 = 2
	mass: np.float32 = 1.0
	friction: np.float32 = 0.5
	restitution: np.float32 = 0.0
	linear_damping: np.float32 = 0.0
	angular_damping: np.float32 = 0.0
	gravity_factor: np.float32 = 1.0
	group: np.uint32 = 1
	mask: np.uint32 = 0xffffffff
	flags: np.uint32 = 0
	material: np.uint16 = 0
	filter: np.uint16 = 0


@component(multiples=True)
class Collider:
	"""Registry shape key and local pose; geometry is separate from render meshes."""
	shape: np.uint32
	position: Vec3 = (0, 0, 0)
	rotation: Vec4 = (0, 0, 0, 1)
	scale: Vec3 = (1, 1, 1)
	sensor: bool = False


WORLD_ENTITY = np.uint64(0xffffffffffffffff)


@component(multiples=True)
class PhysicsJoint:
	"""A keyed constraint from its owner's body to a target body or WORLD_ENTITY.

	Anchors and axes are world-space at creation. Kind codes match native
	fixed=0, ball=1, hinge=2, slider=3, distance=4, six-DoF=5.
	"""
	key: np.uint32 = 0
	target: np.uint64 = WORLD_ENTITY
	kind: np.uint8 = 0
	anchor: Vec3 = (0, 0, 0)
	to_anchor: Vec3 = (0, 0, 0)
	axis: Vec3 = (1, 0, 0)
	axis2: Vec3 = (0, 1, 0)
	lower: np.dtype((np.float64, 6)) = (-np.inf,) * 6
	upper: np.dtype((np.float64, 6)) = (np.inf,) * 6
	limit_spring: np.dtype((np.float32, 2)) = (0, 0)
	friction: np.float32 = 0
	motor_mode: np.dtype((np.uint8, 6)) = (0,) * 6
	motor_target: np.dtype((np.float64, 6)) = (0,) * 6
	motor_max: np.dtype((np.float64, 6)) = (np.inf,) * 6
	motor_frequency: np.float32 = 2
	motor_damping: np.float32 = 1
	break_force: np.float32 = 0
	break_torque: np.float32 = 0
	collide: bool = False
	enabled: bool = True


_JOINT_FIELDS = tuple(field.name for field in fields(PhysicsJoint))
_JOINT_WRITABLE = ("motor_mode", "motor_target", "motor_max", "enabled")


_BODY_FIELDS = tuple(field.name for field in fields(RigidBody))
_BODY_DTYPE = np.dtype([(field.name, field.type) for field in fields(RigidBody)])


def _vectors(value, count, width, name):
	array = np.asarray(value, dtype=np.float64)
	if array.shape not in ((width,), (count, width)) or not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must be finite ({width},) or ({count}, {width}) values")
	return np.broadcast_to(array, (count, width)).copy()


def _rotations(value):
	if not np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-5):
		raise ValueError("rotation must be a unit xyzw quaternion")


class EcsPhysicsAdapter:
	"""Own a lockstep Jolt world and full-handle ECS/body UID mappings.

	Register Collider, RigidBody and the transform type before construction.
	Shapes is a mapping of integer asset keys to immutable pyjoltphysics.Shape
	objects. Call reconcile, prepare_inputs, step, pull in each fixed tick.
	The caller owns fixed time; dynamic ECS poses never become implicit inputs.
	"""
	def __init__(self, ecs, shapes, *, transform_type=None, settings=None, interpolate=False, joint_type=None):
		import pyjoltphysics as pj
		if transform_type is None:
			from .MaterialTransform import MaterialTransform
			transform_type = MaterialTransform
		self.ecs, self.pj, self.transform_type = ecs, pj, transform_type
		self.colliders, self.bodies, self.transforms = ecs.get(Collider), ecs.get(RigidBody), ecs.get(transform_type)
		self._has_scale = "scale" in get_type_hints(transform_type)
		self.query = ecs.query(Collider, transform_type, optional=RigidBody)
		self._shapes = dict(shapes)
		self.joints = None if joint_type is None else ecs.get(joint_type)
		self._joint_type = joint_type
		self._joint_query = None if joint_type is None else ecs.query(joint_type)
		self._joint_records = {}
		self._joint_token = None
		self._retired_bodies, self._retired_joints = {}, {}
		self._subscriptions = {}
		self._next_subscription = 1
		self._events_tick = -1
		self.event_dtype = np.dtype(pj.EVENT_DTYPE.descr + [("entity_a", np.uint64), ("entity_b", np.uint64), ("joint_key", np.uint32)])
		self.events = np.empty(0, dtype=self.event_dtype)
		if any(not isinstance(shape, pj.Shape) for shape in self._shapes.values()):
			raise TypeError("shape registry values must be pyjoltphysics.Shape objects")
		self.world = pj.World(settings if settings is not None else pj.WorldSettings(tick_hz=120))
		self._shape_cache = {}
		self._entities = np.empty(0, dtype=np.uint64)
		self._uids = np.empty(0, dtype=np.uint32)
		self._values = np.empty(0, dtype=_BODY_DTYPE)
		self._geometry = []
		self._owner_by_uid, self._uid_by_owner = {}, {}
		self._pending = []
		self._token = None
		self.last_seq = 0
		self._closed = False
		self._interpolate = interpolate
		self._pose_tick = self.world.tick
		self._position = self._previous_position = np.empty((0, 3), dtype=np.float64)
		self._rotation = self._previous_rotation = np.empty((0, 4), dtype=np.float32)
		self._entity_order = np.empty(0, dtype=np.intp)
		self._teleported = np.empty(0, dtype=np.uint32)

	def __enter__(self):
		return self

	def __exit__(self, kind, value, traceback):
		self.close()

	@property
	def body_uids(self):
		"""Read-only full entity handle -> UID map, including pending creations."""
		return MappingProxyType(self._uid_by_owner)

	@property
	def body_owners(self):
		return MappingProxyType(self._owner_by_uid)

	def _check(self, ecs=None, dt=None):
		if self._closed: raise RuntimeError("physics adapter is closed")
		if ecs is not None and ecs is not self.ecs: raise ValueError("physics adapter belongs to a different ECS")
		if dt is not None and (not np.isfinite(dt) or not np.isclose(dt, 1 / self.world.settings.tick_hz, rtol=1e-7, atol=0)):
			raise ValueError("fixed dt must equal 1 / physics tick_hz")

	def _seq(self, seq):
		self.last_seq = max(self.last_seq, seq)
		return seq

	def _tokens(self):
		return (self.colliders.version(), self.bodies.version(), self.transforms.version("scale") if self._has_scale else (self.transforms.membership_version,))

	def _shape(self, key):
		parts, scale = key
		cached = self._shape_cache.get(parts)
		if cached is not None: return cached
		shapes = []
		for asset, position, rotation, local_scale in parts:
			shape = self._shapes[asset]
			if shape.mutable: raise ValueError("physics collider shapes must be immutable")
			shapes.append(shape)
		if len(parts) == 1:
			shape = shapes[0]
		else:
			if any(local_scale != (1, 1, 1) for _, _, _, local_scale in parts):
				raise ValueError("multiple colliders require unit local scale; author scaled shape assets")
			shape = self.pj.Shape.compound([(shape, part[1], part[2]) for shape, part in zip(shapes, parts)])
		shape_id = self.world.add_shape(shape)
		self._shape_cache[parts] = (shape_id, self.world.shape_info(shape_id).dynamic)
		return self._shape_cache[parts]

	def reconcile(self, ecs=None, commands=None, dt=None):
		"""Reconcile bodies first, then constraints referencing their current UIDs."""
		changed = self._reconcile_bodies(ecs, commands, dt)
		return self._reconcile_joints(changed) or changed

	def _reconcile_bodies(self, ecs=None, commands=None, dt=None):
		"""Validate declarations, batch create/remove/rebuild, and update settings.

		Static poses change only through teleport. Shape/scale/motion/filter-bit
		changes rebuild bodies; dynamic rebuilds retain native pose and velocity.
		"""
		self._check(ecs, dt)
		token = self._tokens()
		if token == self._token: return False
		result = self.query.fetch()
		entities = result.entities
		count = len(entities)
		pose = result[self.transform_type].selection
		position, rotation = pose.position, pose.rotation
		if position.shape != (count, 3) or rotation.shape != (count, 4) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
			raise ValueError("physics transforms require finite position (3) and rotation (4)")
		_rotations(rotation)
		scale = pose.scale if self._has_scale else np.ones((count, 3))
		if not np.all(np.isfinite(scale)) or np.any(scale <= 0): raise ValueError("physics scale must be finite and positive")
		values = np.empty(count, dtype=_BODY_DTYPE)
		defaults = RigidBody(motion=0)
		body_term = result[RigidBody]
		for name in _BODY_FIELDS:
			values[name] = getattr(defaults, name)
			values[name][body_term.present] = getattr(body_term.selection, name)
		if np.any(values["motion"] > 2): raise ValueError("invalid rigid body motion")
		for name in ("mass", "friction", "restitution", "linear_damping", "angular_damping", "gravity_factor"):
			if not np.all(np.isfinite(values[name])): raise ValueError(f"{name} must be finite")
		for name in ("friction", "restitution", "linear_damping", "angular_damping"):
			if np.any(values[name] < 0): raise ValueError(f"{name} must be nonnegative")
		if np.any((values["motion"] != 0) & (values["mass"] <= 0)): raise ValueError("moving bodies require positive mass")
		collider_term = result[Collider]
		colliders = collider_term.selection
		assets, offsets, rotations, scales, sensors = (getattr(colliders, name) for name in ("shape", "position", "rotation", "scale", "sensor"))
		if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(rotations)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
			raise ValueError("collider poses must be finite and scales positive")
		_rotations(rotations)
		geometry, shape_ids, dynamic = [], [], []
		# Iterate authored topology only; native body creation and state I/O are batches.
		for i, (start, stop) in enumerate(zip(collider_term.offsets[:-1], collider_term.offsets[1:])):
			if np.any(sensors[start:stop] != sensors[start]): raise ValueError("mixed solid and sensor colliders require separate entities")
			if sensors[start]: values["flags"][i] |= int(self.pj.BodyFlag.SENSOR)
			parts = tuple((int(assets[j]), tuple(offsets[j]), tuple(rotations[j]), tuple(scales[j])) for j in range(start, stop))
			key = (parts, tuple(scale[i]))
			shape_id, movable = self._shape(key)
			geometry.append(key)
			shape_ids.append(shape_id)
			dynamic.append(movable)
		if np.any((values["motion"] == 2) & ~np.asarray(dynamic, dtype=bool)):
			raise ValueError("dynamic bodies require convex or primitive shapes; meshes/heightfields are not dynamic")
		_, old, new = np.intersect1d(self._entities, entities, return_indices=True)
		uids = np.zeros(count, dtype=np.uint32)
		uids[new] = self._uids[old]
		mutable_flags = int(self.pj.BodyFlag.REPORT | self.pj.BodyFlag.NO_SLEEP | self.pj.BodyFlag.FILTER_CONTACTS)
		rebuild = np.array([self._geometry[a] != geometry[b] for a, b in zip(old, new)], dtype=bool)
		for name in ("motion", "group", "mask"):
			rebuild |= self._values[name][old] != values[name][new]
		rebuild |= ((self._values["flags"][old] ^ values["flags"][new]) & ~np.uint32(mutable_flags)) != 0
		uids[new[rebuild]] = 0
		removed = self._uids[~np.isin(self._uids, uids)]
		spawn = np.flatnonzero(uids == 0)
		linear, angular = np.zeros((count, 3)), np.zeros((count, 3))
		# Only kinematic targets take their replacement pose from ECS.
		preserve = rebuild & (self._values["motion"][old] != 1) & (values["motion"][new] != 1)
		if np.any(preserve):
			state = self.world.bodies_state(self._uids[old[preserve]])
			kept = new[preserve][state.found]
			position[kept], rotation[kept] = state.position[state.found], state.orientation[state.found]
			linear[kept], angular[kept] = state.linear_velocity[state.found], state.angular_velocity[state.found]
		kwargs = {name: values[name][spawn] for name in _BODY_FIELDS}
		shape_scale, shape_offset, shape_rotation = scale[spawn].copy(), np.zeros((len(spawn), 3)), np.tile((0, 0, 0, 1), (len(spawn), 1))
		for row, target in enumerate(spawn):
			parts = geometry[target][0]
			if len(parts) == 1:
				_, offset, orient, local_scale = parts[0]
				if not np.allclose(scale[target], scale[target, 0]) and not np.allclose(orient, (0, 0, 0, 1)):
					raise ValueError("rotated local colliders require uniform entity scale")
				shape_scale[row] *= local_scale
				shape_offset[row], shape_rotation[row] = np.asarray(offset) * scale[target], orient
		batch = self.pj.BodyBatch.of_shape(
			position[spawn], np.asarray(shape_ids, dtype=np.uint32)[spawn], **kwargs,
			orientations=rotation[spawn], scale=shape_scale, shape_offset=shape_offset,
			shape_rotation=shape_rotation, linear_velocity=linear[spawn], angular_velocity=angular[spawn],
		)
		# Validate creation arrays before submitting any destructive changes.
		if len(spawn): batch.arrays(shape_id=self.world.add_shape)
		self._retired_bodies.update((int(uid), self._owner_by_uid[int(uid)]) for uid in removed)
		if self.world.reserved_bodies + len(spawn) > self.world.settings.max_bodies:
			if self.world.reserved_bodies - len(removed) + len(spawn) > self.world.settings.max_bodies:
				raise ValueError("physics declarations exceed max_bodies")
			# The pinned backend reserves additions before releasing swap removals.
			# Flush removal without advancing time when a rebuild needs that capacity.
			self._seq(self.world.remove_bodies(removed))
			self.world.flush()
			removed = np.empty(0, dtype=np.uint32)
		if len(spawn):
			created, seq = self.world.swap_bodies(removed, batch) if len(removed) else self.world.add_bodies(batch)
			uids[spawn] = created
			self._seq(seq)
		elif len(removed): self._seq(self.world.remove_bodies(removed))
		kept_old, kept_new = old[~rebuild], new[~rebuild]
		self._update_settings(uids[kept_new], self._values[kept_old], values[kept_new])
		if np.any(preserve): self.transforms[entities[kept]].assign(position=position[kept], rotation=rotation[kept])
		if self._interpolate:
			current_position, current_rotation = position.astype(np.float64), rotation.astype(np.float32)
			previous_position, previous_rotation = current_position.copy(), current_rotation.copy()
			current_position[kept_new], current_rotation[kept_new] = self._position[kept_old], self._rotation[kept_old]
			previous_position[kept_new], previous_rotation[kept_new] = self._previous_position[kept_old], self._previous_rotation[kept_old]
			self._position, self._rotation = current_position, current_rotation
			self._previous_position, self._previous_rotation = previous_position, previous_rotation
			self._entity_order = np.argsort(entities)
		self._entities, self._uids, self._values, self._geometry = entities, uids, values, geometry
		self._uid_by_owner = dict(zip(map(int, entities), map(int, uids)))
		self._owner_by_uid = dict(zip(map(int, uids), map(int, entities)))
		self._token = token
		return True

	@property
	def joint_ids(self):
		"""(full owner handle, stable component key) -> native joint ID."""
		return MappingProxyType({key: record[0] for key, record in self._joint_records.items()})

	def _reconcile_joints(self, bodies_changed):
		if self.joints is None: return False
		token = self.joints.version()
		if not bodies_changed and token == self._joint_token: return False
		result = self._joint_query.fetch()
		selection = result[self._joint_type].selection
		owners = self.joints.owner_ids(selection)
		values = {name: getattr(selection, name) for name in _JOINT_FIELDS}
		if np.any(values["kind"] > 5): raise ValueError("unsupported joint kind")
		for name in ("anchor", "to_anchor", "axis", "axis2", "motor_target", "limit_spring", "friction", "motor_frequency", "motor_damping", "break_force", "break_torque"):
			if not np.all(np.isfinite(values[name])): raise ValueError(f"joint {name} must be finite")
		if np.any(np.isnan(values["lower"])) or np.any(np.isnan(values["upper"])) or np.any(values["lower"] > values["upper"]):
			raise ValueError("joint limits require lower <= upper")
		if np.any(values["motor_mode"] > 2) or np.any(np.isnan(values["motor_max"])) or np.any(values["motor_max"] < 0):
			raise ValueError("invalid joint motor settings")
		for name in ("limit_spring", "friction", "motor_frequency", "motor_damping", "break_force", "break_torque"):
			if np.any(values[name] < 0): raise ValueError(f"joint {name} must be nonnegative")
		axial = np.isin(values["kind"], (2, 3, 5))
		if np.any(np.linalg.norm(values["axis"][axial], axis=1) < 1e-6): raise ValueError("joint axis must have direction")
		six = values["kind"] == 5
		if np.any(np.linalg.norm(np.cross(values["axis"][six], values["axis2"][six]), axis=1) < 1e-6):
			raise ValueError("six-DoF axes must not be parallel")
		desired = {}
		keys = list(zip(map(int, owners), map(int, values["key"])))
		if len(set(keys)) != len(keys): raise ValueError("joint keys must be unique within each owner")
		for i, key in enumerate(keys):
			body = self._uid_by_owner.get(key[0])
			target = int(values["target"][i])
			to = 0 if target == WORLD_ENTITY else self._uid_by_owner.get(target)
			if body is None or to is None: continue
			if body == to: raise ValueError("joint cannot connect a body to itself")
			row = {name: tuple(value[i]) if value.ndim > 1 else value[i].item() for name, value in values.items()}
			signature = (body, to, *(row[name] for name in _JOINT_FIELDS if name not in _JOINT_WRITABLE))
			desired[key] = (signature, row)
		removed = [key for key, (_, signature, _) in self._joint_records.items() if key not in desired or signature != desired[key][0]]
		created = [key for key, (signature, _) in desired.items() if key not in self._joint_records or self._joint_records[key][1] != signature]
		descriptors = []
		for key in created:
			signature, row = desired[key]
			motors = tuple(self.pj.Motor(mode=mode, target=target, max=limit, frequency=row["motor_frequency"], damping=row["motor_damping"]) if mode else None for mode, target, limit in zip(row["motor_mode"], row["motor_target"], row["motor_max"]))
			arguments = {name: row[name] for name in ("anchor", "to_anchor", "axis", "axis2", "lower", "upper", "limit_spring", "friction", "break_force", "break_torque", "collide")}
			descriptors.append(self.pj.Joint(row["kind"], signature[0], signature[1], **arguments, motors=motors))
		if removed:
			self._retired_joints.update((self._joint_records[key][0], (key, self._joint_records[key][2]["target"])) for key in removed)
			self._seq(self.world.remove_joints([self._joint_records[key][0] for key in removed]))
			for key in removed: del self._joint_records[key]
		if created:
			ids, seq = self.world.add_joints(descriptors)
			self._seq(seq)
			self._joint_records.update((key, (int(uid), *desired[key])) for key, uid in zip(created, ids))
		motor_updates = {mode: [] for mode in range(3)}
		enabled_ids, enabled = [], []
		for key, (signature, row) in desired.items():
			uid, _, old = self._joint_records[key]
			if key in created or old["enabled"] != row["enabled"]:
				enabled_ids.append(uid)
				enabled.append(row["enabled"])
			if key not in created:
				for axis, mode in enumerate(row["motor_mode"]):
					if any(old[name][axis] != row[name][axis] for name in ("motor_mode", "motor_target", "motor_max")):
						motor_updates[mode].append((uid, axis, row["motor_target"][axis], row["motor_max"][axis]))
			self._joint_records[key] = (uid, signature, row)
		for mode, changes in motor_updates.items():
			if changes:
				ids, axes, targets, limits = map(list, zip(*changes))
				self._seq(self.world.set_joint_motors(ids, mode, target=targets, max=limits, axis=axes))
		if enabled_ids: self._seq(self.world.set_joints_enabled(enabled_ids, enabled))
		self._joint_token = token
		return True

	def set_joint_motor(self, entities, keys, *, mode=None, target=np.nan, max_force=np.nan, axis=0):
		self._check()
		entities = as_entities(entities)
		keys = np.broadcast_to(np.asarray(keys), entities.shape)
		if keys.dtype.kind not in "iu" or np.any(keys < 0) or np.any(keys > 0xffffffff): raise ValueError("joint keys must be uint32 integers")
		ids = [self._joint_records[(int(owner), int(key))][0] for owner, key in zip(entities, keys)]
		return self._seq(self.world.set_joint_motors(ids, mode, target=target, max=max_force, axis=axis))

	def entities_of(self, uids):
		"""Resolve current body UIDs to live full handles; WORLD_ENTITY means absent."""
		self._check()
		return self._resolve_owners(uids, self._owner_by_uid)

	def _resolve_owners(self, uids, mapping):
		wanted = np.asarray(uids)
		if wanted.size == 0: return np.full(wanted.shape, WORLD_ENTITY, dtype=np.uint64)
		if wanted.dtype.kind not in "iu": raise TypeError("body UIDs must be integers")
		if np.any(wanted < 0) or np.any(wanted > 0xffffffff): raise ValueError("body UIDs must fit uint32")
		result = np.full(wanted.shape, WORLD_ENTITY, dtype=np.uint64)
		if mapping:
			keys = np.array(sorted(mapping), dtype=np.uint32)
			owners = np.array([mapping[int(key)] for key in keys], dtype=np.uint64)
			indices = np.searchsorted(keys, wanted)
			rows = np.minimum(indices, len(keys) - 1)
			valid = (indices < len(keys)) & (keys[rows] == wanted)
			result[valid] = owners[rows[valid]]
		result[~self.ecs.is_alive(result.reshape(-1)).reshape(result.shape)] = WORLD_ENTITY
		return result

	def query_filter(self, *, exclude=(), group=0xffffffff, mask=0xffffffff, triggers=False, characters=True):
		self._check()
		uids = [self._uid_by_owner[int(owner)] for owner in as_entities(exclude) if int(owner) in self._uid_by_owner]
		return self.pj.QueryFilter(group=group, mask=mask, exclude=uids, triggers=triggers, characters=characters)

	def _query_filter(self, filter, exclude):
		if filter is None: return self.query_filter(exclude=exclude)
		extra = self.query_filter(exclude=exclude).exclude
		return self.pj.QueryFilter(group=filter.group, mask=filter.mask, exclude=(*filter.exclude, *extra), triggers=filter.triggers, characters=filter.characters)

	def raycast(self, origin, direction, max_distance=1000.0, *, filter=None, exclude=()):
		self._check()
		return self.world.raycast(origin, direction, max_distance, self._query_filter(filter, exclude))

	def raycast_batch(self, origins, directions, max_distance=1000.0, *, filter=None, exclude=()):
		self._check()
		return self.world.raycast_batch(origins, directions, max_distance, self._query_filter(filter, exclude))

	def sweep(self, shape, start, end, *, orientation=(0, 0, 0, 1), filter=None, exclude=(), all_hits=False):
		self._check()
		shape = shape if isinstance(shape, self.pj.Shape) else self._shapes[int(shape)]
		return self.world.sweep(shape, start, end, orientation, self._query_filter(filter, exclude), all_hits)

	def overlap(self, shape, position, *, orientation=(0, 0, 0, 1), filter=None, exclude=()):
		self._check()
		shape = shape if isinstance(shape, self.pj.Shape) else self._shapes[int(shape)]
		return self.world.overlap(shape, position, orientation, self._query_filter(filter, exclude))

	def set_filter_table(self, collides):
		self._check()
		return self._seq(self.world.set_filter_table(collides))

	@property
	def materials(self):
		return self.world.materials

	@property
	def events_dropped(self):
		return self.world.stats()["events_dropped"]

	def subscribe(self, callback, *, kind=None, entity=None):
		"""Call callback(ecs, commands, event_batch) on the dispatching thread."""
		self._check()
		_synchronous_callable(callback, "physics event callback")
		if kind is not None: kind = self.pj.EventKind(kind)
		if entity is not None:
			owners = as_entities(entity)
			if len(owners) != 1: raise ValueError("subscription entity must be a single handle")
			entity = int(owners[0])
		token = self._next_subscription
		self._next_subscription += 1
		self._subscriptions[token] = (callback, kind, entity)
		return token

	def unsubscribe(self, token):
		self._subscriptions.pop(token, None)

	def dispatch_events(self, ecs=None, commands=None, dt=None):
		"""Drain once per tick and route native records, preserving removal identity."""
		self._check(ecs, dt)
		if self._events_tick == self.world.tick: return self.events
		raw = self.world.drain_events()
		events = np.zeros(len(raw), dtype=self.event_dtype)
		for name in raw.dtype.names: events[name] = raw[name]
		events["entity_a"] = events["entity_b"] = WORLD_ENTITY
		body_events = np.isin(raw["kind"], (self.pj.EventKind.CONTACT, self.pj.EventKind.TRIGGER, self.pj.EventKind.STATE))
		owners = self._retired_bodies | self._owner_by_uid
		events["entity_a"][body_events] = self._resolve_owners(raw["a"][body_events], owners)
		pair = np.isin(raw["kind"], (self.pj.EventKind.CONTACT, self.pj.EventKind.TRIGGER, self.pj.EventKind.CHARACTER))
		events["entity_b"][pair] = self._resolve_owners(raw["b"][pair], owners)
		joint_owners = self._retired_joints | {record[0]: (key, record[2]["target"]) for key, record in self._joint_records.items()}
		joint = raw["kind"] == self.pj.EventKind.JOINT
		if joint_owners:
			events["entity_a"][joint] = self._resolve_owners(raw["a"][joint], {uid: key[0] for uid, (key, _) in joint_owners.items()})
			events["entity_b"][joint] = self._resolve_owners(raw["a"][joint], {uid: target for uid, (_, target) in joint_owners.items()})
			events["joint_key"][joint] = np.array([joint_owners.get(int(uid), ((0, 0), 0))[0][1] for uid in raw["a"][joint]], dtype=np.uint32)
		events.flags.writeable = False
		self.events = events
		self._events_tick = self.world.tick
		self._retired_bodies.clear()
		self._retired_joints.clear()
		own_commands = commands is None
		commands = self.ecs.commands() if own_commands else commands
		for callback, kind, entity in tuple(self._subscriptions.values()):
			# A prior callback may delete an owner; never route to its recycled slot.
			live_a, live_b = self.ecs.is_alive(events["entity_a"]), self.ecs.is_alive(events["entity_b"])
			selected = live_a | live_b
			if kind is not None: selected &= events["kind"] == kind
			if entity is not None: selected &= (live_a & (events["entity_a"] == entity)) | (live_b & (events["entity_b"] == entity))
			if np.any(selected): callback(self.ecs, commands, events[selected])
		if own_commands: commands.flush()
		return self.events

	def _update_settings(self, uids, previous, values):
		def changed(*names):
			return np.logical_or.reduce([previous[name] != values[name] for name in names])
		mask = changed("friction", "restitution")
		if np.any(mask): self._seq(self.world.set_materials(uids[mask], values["friction"][mask], values["restitution"][mask]))
		mask = changed("mass", "linear_damping", "angular_damping", "gravity_factor") & (values["motion"] != 0)
		if np.any(mask): self._seq(self.world.set_motion(uids[mask], **{name: values[name][mask] for name in ("mass", "linear_damping", "angular_damping", "gravity_factor")}))
		for name, setter in (("material", self.world.set_body_materials), ("filter", self.world.set_body_filters)):
			mask = changed(name)
			if np.any(mask): self._seq(setter(uids[mask], values[name][mask]))
		for flag in (self.pj.BodyFlag.REPORT, self.pj.BodyFlag.NO_SLEEP, self.pj.BodyFlag.FILTER_CONTACTS):
			for on in (False, True):
				mask = changed("flags") & ((values["flags"] & int(flag) != 0) == on) & ((previous["flags"] & int(flag) != 0) != on)
				if np.any(mask): self._seq(self.world.set_flags(uids[mask], flag, on))

	def _queue(self, kind, entities, **values):
		self._check()
		entities = as_entities(entities).copy()
		arrays = {name: None if value is None else _vectors(value, len(entities), 4 if name == "orientations" else 3, name) for name, value in values.items()}
		if "orientations" in arrays: _rotations(arrays["orientations"])
		self._pending.append((kind, entities, arrays))

	def teleport(self, entities, positions, orientations=(0, 0, 0, 1)):
		"""Queue an explicit pose move; velocities are preserved."""
		self._queue("set_transforms", entities, positions=positions, orientations=orientations)

	def set_velocity(self, entities, linear=None, angular=None):
		self._queue("set_velocities", entities, linear=linear, angular=angular)

	def add_impulse(self, entities, impulses, points=None):
		self._queue("add_impulses", entities, impulses=impulses, points=points)

	def add_force(self, entities, forces, points=None):
		self._queue("add_forces", entities, forces=forces, points=points)

	def prepare_inputs(self, ecs=None, commands=None, dt=None):
		"""Send current kinematic targets and queued commands for the next tick."""
		self._check(ecs, dt)
		kinematic = (self._values["motion"] == 1) & self.ecs.is_alive(self._entities) & self.ecs.has(self._entities, self.transform_type)
		kinematic &= self.ecs.has(self._entities, Collider)
		if np.any(kinematic):
			pose = self.transforms[self._entities[kinematic]]
			self._seq(self.world.move_kinematic(self._uids[kinematic], pose.position, pose.rotation))
		while self._pending:
			kind, entities, values = self._pending[0]
			live = self.ecs.is_alive(entities) & self.ecs.has(entities, Collider) & self.ecs.has(entities, self.transform_type)
			uids = np.array([self._uid_by_owner.get(int(entity), 0) for entity in entities], dtype=np.uint32)
			if np.any(live & (uids == 0)): raise ValueError("reconcile bodies before submitting physics commands")
			if np.any(live):
				self._seq(getattr(self.world, kind)(uids[live], **{name: None if array is None else array[live] for name, array in values.items()}))
				if kind == "set_transforms":
					self.transforms[entities[live]].assign(position=values["positions"][live], rotation=values["orientations"][live])
					if self._interpolate: self._teleported = np.union1d(self._teleported, uids[live])
			self._pending.pop(0)

	def step(self, ecs=None, commands=None, dt=None):
		self._check(ecs, dt)
		self.world.run_ticks(1)

	def pull(self, ecs=None, commands=None, dt=None):
		"""Mirror changed dynamic poses; never keep native snapshot views."""
		self._check(ecs, dt)
		snapshot = self.world.latest()
		if snapshot is None: return
		rows = snapshot.rows_of(self._uids)
		if snapshot.applied_seq >= self.last_seq and np.any(rows < 0):
			raise RuntimeError("owned physics body disappeared; do not remove adapter bodies through world")
		if self._interpolate:
			advanced = snapshot.tick != self._pose_tick
			if advanced:
				self._previous_position[:] = self._position
				self._previous_rotation[:] = self._rotation
			found = rows >= 0
			self._position[found], self._rotation[found] = snapshot.position[rows[found]], snapshot.orientation[rows[found]]
			if advanced:
				snap = np.isin(self._uids, self._teleported) if snapshot.tick == self._pose_tick + 1 else np.ones(len(rows), dtype=bool)
				self._previous_position[snap], self._previous_rotation[snap] = self._position[snap], self._rotation[snap]
				self._teleported = np.empty(0, dtype=np.uint32)
				self._pose_tick = snapshot.tick
		mask = (rows >= 0) & (self._values["motion"] == 2) & np.isin(rows, snapshot.changed)
		mask &= self.ecs.is_alive(self._entities) & self.ecs.has(self._entities, self.transform_type) & self.ecs.has(self._entities, Collider)
		if np.any(mask):
			self.transforms[self._entities[mask]].assign(position=snapshot.position[rows[mask]], rotation=snapshot.orientation[rows[mask]])

	def render_transforms(self, entities, *, alpha):
		"""Return interpolated TRS copies in requested entity order, without ECS writes.

		Requires interpolate=True. Alpha is the caller's accumulator / fixed_dt
		in [0, 1]. Static and non-physics owners use their ordinary ECS transform.
		"""
		self._check()
		if not self._interpolate: raise RuntimeError("enable interpolate=True to read render transforms")
		if np.ndim(alpha) != 0 or not np.isfinite(alpha) or not 0 <= alpha <= 1:
			raise ValueError("interpolation alpha must be finite and in [0, 1]")
		entities = as_entities(entities)
		pose = self.transforms[entities]
		if len(pose) != len(entities): raise ValueError("render owners must have live transform components")
		position, rotation = pose.position, pose.rotation
		scale = pose.scale if self._has_scale else np.ones((len(entities), 3), dtype=position.dtype)
		if len(self._entities):
			indices = np.searchsorted(self._entities[self._entity_order], entities)
			rows = self._entity_order[np.minimum(indices, len(self._entity_order) - 1)]
			moving = (self._entities[rows] == entities) & (self._values["motion"][rows] != 0) & self.ecs.has(entities, Collider)
			selected = rows[moving]
			position[moving] = self._previous_position[selected] * (1 - alpha) + self._position[selected] * alpha
			rotation[moving] = Quaternion.slerp_batch(self._previous_rotation[selected], self._rotation[selected], alpha)
		return position, rotation, scale

	def close(self):
		"""Join the native thread and clear ownership and queued commands; idempotent."""
		if self._closed: return
		try:
			if self._joint_records and self.world.error is None:
				self.world.remove_joints([record[0] for record in self._joint_records.values()])
			if len(self._uids) and self.world.error is None:
				self.world.remove_bodies(self._uids)
			if self.world.error is None:
				self.world.flush()
		finally:
			self.world.stop()
			self._pending.clear()
			self._shape_cache.clear()
			self._uid_by_owner.clear()
			self._owner_by_uid.clear()
			self._joint_records.clear()
			self._retired_bodies.clear()
			self._retired_joints.clear()
			self._subscriptions.clear()
			self.events = np.empty(0, dtype=self.event_dtype)
			self._position = self._previous_position = np.empty((0, 3), dtype=np.float64)
			self._rotation = self._previous_rotation = np.empty((0, 4), dtype=np.float32)
			self._entity_order = np.empty(0, dtype=np.intp)
			self._teleported = np.empty(0, dtype=np.uint32)
			self._closed = True
