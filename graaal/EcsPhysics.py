"""Optional ECS rigid-body ownership over a standalone pyjoltphysics world."""

from dataclasses import fields
from types import MappingProxyType
from typing import get_type_hints

import numpy as np

from .ecs_types import as_entities, component
from .Math import Vec3, Vec4


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
	def __init__(self, ecs, shapes, *, transform_type=None, settings=None):
		import pyjoltphysics as pj
		if transform_type is None:
			from .MaterialTransform import MaterialTransform
			transform_type = MaterialTransform
		self.ecs, self.pj, self.transform_type = ecs, pj, transform_type
		self.colliders, self.bodies, self.transforms = ecs.get(Collider), ecs.get(RigidBody), ecs.get(transform_type)
		self._has_scale = "scale" in get_type_hints(transform_type)
		self.query = ecs.query(Collider, transform_type, optional=RigidBody)
		self._shapes = dict(shapes)
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
		self._entities, self._uids, self._values, self._geometry = entities, uids, values, geometry
		self._uid_by_owner = dict(zip(map(int, entities), map(int, uids)))
		self._owner_by_uid = dict(zip(map(int, uids), map(int, entities)))
		self._token = token
		return True

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
		mask = (rows >= 0) & (self._values["motion"] == 2) & np.isin(rows, snapshot.changed)
		mask &= self.ecs.is_alive(self._entities) & self.ecs.has(self._entities, self.transform_type) & self.ecs.has(self._entities, Collider)
		if np.any(mask):
			self.transforms[self._entities[mask]].assign(position=snapshot.position[rows[mask]], rotation=snapshot.orientation[rows[mask]])

	def close(self):
		"""Join the native thread and clear ownership and queued commands; idempotent."""
		if self._closed: return
		try:
			if len(self._uids) and self.world.error is None:
				self.world.remove_bodies(self._uids)
				self.world.flush()
		finally:
			self.world.stop()
			self._pending.clear()
			self._shape_cache.clear()
			self._uid_by_owner.clear()
			self._owner_by_uid.clear()
			self._closed = True
