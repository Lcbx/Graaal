"""Optional ECS body, character and ragdoll ownership over standalone Jolt."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import field, dataclass, fields
from types import TracebackType, MappingProxyType
from typing import cast, Any, Callable, Generic, Mapping, Self, Sequence, TYPE_CHECKING, TypeVar, get_type_hints

from numpy.typing import ArrayLike
import numpy as np

from ..ECS.ecs_types import as_entities, component, _synchronous_callable
from ..Math import Vec3, Vec4, Quaternion
from ..PhysicsRagdoll import RagdollAsset

if TYPE_CHECKING:
	import pyjoltphysics as pj


if TYPE_CHECKING:
	from ..ECS import ECS, CommandBuffer
	from ..ECS.ecs_types import EntityLike


@component
class RigidBody:
	"""Declarative settings; motion codes are static=0, kinematic=1, dynamic=2."""
	motion: np.uint8 = cast(np.uint8, 2)
	mass: np.float32 = cast(np.float32, 1.0)
	friction: np.float32 = cast(np.float32, 0.5)
	restitution: np.float32 = cast(np.float32, 0.0)
	linear_damping: np.float32 = cast(np.float32, 0.0)
	angular_damping: np.float32 = cast(np.float32, 0.0)
	gravity_factor: np.float32 = cast(np.float32, 1.0)
	group: np.uint32 = cast(np.uint32, 1)
	mask: np.uint32 = cast(np.uint32, 0xffffffff)
	flags: np.uint32 = cast(np.uint32, 0)
	material: np.uint16 = cast(np.uint16, 0)
	filter: np.uint16 = cast(np.uint16, 0)


@component(multiples=True)
class Collider:
	"""Registry shape key and local pose; geometry is separate from render meshes."""
	shape: np.uint32
	position: Vec3 = cast(Vec3, (0, 0, 0))
	rotation: Vec4 = cast(Vec4, (0, 0, 0, 1))
	scale: Vec3 = cast(Vec3, (1, 1, 1))
	sensor: bool = False


WORLD_ENTITY = np.uint64(0xffffffffffffffff)


@component
class PhysicsCharacter:
	"""Capability asset and native controller settings; ECS position is at the feet."""
	capabilities: np.uint32 = cast(np.uint32, 0)
	gravity: np.float64 = cast(np.float64, np.nan)
	pass_group: np.uint32 = cast(np.uint32, 0)
	group: np.uint32 = cast(np.uint32, 1)
	mask: np.uint32 = cast(np.uint32, 0xffffffff)
	proxy: bool = True
	enabled: bool = True


@component
class CharacterState:
	"""Physics output only; ground is a full entity handle or WORLD_ENTITY."""
	velocity: Vec3 = cast(Vec3, (0, 0, 0))
	grounded: bool = False
	ground: np.uint64 = cast(np.uint64, WORLD_ENTITY)
	ground_normal: Vec3 = cast(Vec3, (0, 0, 0))
	state: np.uint8 = cast(np.uint8, 3)
	height: np.float32 = cast(np.float32, 0)
	stuck: bool = False


_CHARACTER_FIELDS = ("capabilities", "gravity", "pass_group", "group", "mask", "proxy")


@component
class PhysicsRagdoll:
	"""Native ragdoll asset and mode (kinematic=0, powered=1, limp=2)."""
	asset: np.uint32
	mode: np.uint8 = cast(np.uint8, 0)
	blend: np.float32 = cast(np.float32, 0)
	strength: np.float32 = cast(np.float32, 1)


@component(multiples=True)
class PhysicsJoint:
	"""A keyed constraint from its owner's body to a target body or WORLD_ENTITY.

	Anchors and axes are world-space at creation. Kind codes match native
	fixed=0, ball=1, hinge=2, slider=3, distance=4, six-DoF=5.
	"""
	key: np.uint32 = cast(np.uint32, 0)
	target: np.uint64 = cast(np.uint64, WORLD_ENTITY)
	kind: np.uint8 = cast(np.uint8, 0)
	anchor: Vec3 = cast(Vec3, (0, 0, 0))
	to_anchor: Vec3 = cast(Vec3, (0, 0, 0))
	axis: Vec3 = cast(Vec3, (1, 0, 0))
	axis2: Vec3 = cast(Vec3, (0, 1, 0))
	lower: tuple[float, ...] = field(default=(-np.inf,) * 6, metadata={"graaal.dtype": np.dtype((np.float64, 6))})
	upper: tuple[float, ...] = field(default=(np.inf,) * 6, metadata={"graaal.dtype": np.dtype((np.float64, 6))})
	limit_spring: tuple[float, ...] = field(default=(0, 0), metadata={"graaal.dtype": np.dtype((np.float32, 2))})
	friction: np.float32 = cast(np.float32, 0)
	motor_mode: tuple[int, ...] = field(default=(0,) * 6, metadata={"graaal.dtype": np.dtype((np.uint8, 6))})
	motor_target: tuple[float, ...] = field(default=(0,) * 6, metadata={"graaal.dtype": np.dtype((np.float64, 6))})
	motor_max: tuple[float, ...] = field(default=(np.inf,) * 6, metadata={"graaal.dtype": np.dtype((np.float64, 6))})
	motor_frequency: np.float32 = cast(np.float32, 2)
	motor_damping: np.float32 = cast(np.float32, 1)
	break_force: np.float32 = cast(np.float32, 0)
	break_torque: np.float32 = cast(np.float32, 0)
	collide: bool = False
	enabled: bool = True


_JOINT_FIELDS = tuple(field.name for field in fields(PhysicsJoint))
_JOINT_WRITABLE = ("motor_mode", "motor_target", "motor_max", "enabled")


_BODY_FIELDS = tuple(field.name for field in fields(RigidBody))
_BODY_DTYPE = np.dtype(list(get_type_hints(RigidBody).items()))


def _vectors(value: ArrayLike, count: int, width: int, name: str) -> np.ndarray:
	array = np.asarray(value, dtype=np.float64)
	if array.shape not in ((width,), (count, width)) or not np.all(np.isfinite(array)):
		raise ValueError(f"{name} must be finite ({width},) or ({count}, {width}) values")
	return np.broadcast_to(array, (count, width)).copy()


def _rotations(value: np.ndarray) -> None:
	if not np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-5):
		raise ValueError("rotation must be a unit xyzw quaternion")


def _lookup_uids(uids: ArrayLike, mapping: Mapping[int, int], missing: np.uint64 | np.int32) -> np.ndarray:
	wanted = np.asarray(uids)
	result = np.full(wanted.shape, missing, dtype=missing.dtype)
	if wanted.size == 0: return result
	if wanted.dtype.kind not in "iu": raise TypeError("body UIDs must be integers")
	if np.any(wanted < 0) or np.any(wanted > 0xffffffff): raise ValueError("body UIDs must fit uint32")
	if mapping:
		keys = np.array(sorted(mapping), dtype=np.uint32)
		values = np.array([mapping[int(key)] for key in keys], dtype=result.dtype)
		indices = np.searchsorted(keys, wanted)
		rows = np.minimum(indices, len(keys) - 1)
		valid = (indices < len(keys)) & (keys[rows] == wanted)
		result[valid] = values[rows[valid]]
	return result


def _character_feet(centers: np.ndarray, heights: ArrayLike) -> np.ndarray:
	feet = centers.copy()
	feet[..., 1] -= np.asarray(heights) * 0.5
	return feet


_Removals = TypeVar("_Removals")


@dataclass(frozen=True)
class _ReconcilePlan(Generic[_Removals]):
	removals: _Removals
	required_bodies: int
	commit: Callable[[], bool]


@dataclass(frozen=True)
class _PhysicsState:
	"""Opaque same-adapter checkpoint; contains no borrowed native snapshot views."""
	adapter: object
	tick: int
	native: bytes
	uids: np.ndarray
	entities: np.ndarray
	values: np.ndarray
	geometry: tuple
	joints: tuple
	revision: int
	position: np.ndarray
	rotation: np.ndarray
	pending: list


class EcsPhysicsAdapter:
	"""Own a lockstep Jolt world and full-handle ECS/body UID mappings.

	Register Collider, RigidBody and the transform type before construction.
	Shapes is a mapping of integer asset keys to immutable pyjoltphysics.Shape
	objects. Call reconcile, prepare_inputs, step, pull in each fixed tick.
	The caller owns fixed time; dynamic ECS poses never become implicit inputs.
	"""
	def __init__(self, ecs: ECS, shapes: Mapping[int, pj.Shape], *, transform_type: type[Any] | None=None, settings: pj.WorldSettings | None=None, interpolate: bool=False, joint_type: type[Any] | None=None, ragdolls: Mapping[int, RagdollAsset] | None=None, characters: Mapping[int, pj.CharacterCapabilities] | None=None) -> None:
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
		self._joint_records: dict[tuple[int, int], tuple[int, Any, dict[str, Any]]]
		self._joint_records = {}
		self._joint_token: tuple[Any, ...] | None
		self._joint_token = None
		self._ragdoll_assets = {} if ragdolls is None else dict(ragdolls)
		if any(not isinstance(asset, RagdollAsset) for asset in self._ragdoll_assets.values()): raise TypeError("ragdoll registry values must be RagdollAsset objects")
		self._ragdoll_query = None if ragdolls is None else ecs.query(PhysicsRagdoll, transform_type)
		self._ragdolls: dict[int, tuple[int, pj.Ragdoll, int, float, float]]
		self._ragdoll_poses: dict[int, tuple[np.ndarray, ...]]
		self._ragdolls, self._ragdoll_poses = {}, {}
		self._ragdoll_targets: dict[int, tuple[np.ndarray, np.ndarray]]
		self._ragdoll_targets = {}
		self._ragdoll_history: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
		self._ragdoll_history = {}
		self._ragdoll_owners: dict[int, int]
		self._bone_by_uid: dict[int, int]
		self._retired_bones: dict[int, int]
		self._ragdoll_owners, self._bone_by_uid, self._retired_bones = {}, {}, {}
		self._character_assets = {} if characters is None else dict(characters)
		for caps in self._character_assets.values():
			if not isinstance(caps, pj.CharacterCapabilities): raise TypeError("character registry values must be CharacterCapabilities objects")
			if not np.isfinite((caps.radius, caps.standHeight, caps.crouchHeight)).all() or caps.radius <= 0 or min(caps.standHeight, caps.crouchHeight) < 2 * caps.radius:
				raise ValueError("character capsule heights must be finite and at least twice the positive radius")
		self._character_query = None if characters is None else ecs.query(PhysicsCharacter, transform_type)
		self._character_states = None if characters is None else ecs.get(CharacterState)
		self._characters: dict[int, tuple[tuple[Any, ...], pj.Character]]
		self._character_owners: dict[int, int]
		self._retired_characters: dict[int, int]
		self._characters, self._character_owners, self._retired_characters = {}, {}, {}
		self._character_pending: list[tuple[Any, ...]]
		self._character_history: Any
		self._character_snap: set[int]
		self._character_pending, self._character_history, self._character_snap = [], None, set()
		self._retired_bodies: dict[int, int]
		self._retired_joints: dict[int, tuple[tuple[int, int], int]]
		self._retired_bodies, self._retired_joints = {}, {}
		self._subscriptions: dict[int, tuple[Callable[[ECS, CommandBuffer, np.ndarray], None], int | None, int | None]]
		self._subscriptions = {}
		self._next_subscription = 1
		self._events_tick = -1
		self.event_dtype = np.dtype(pj.EVENT_DTYPE.descr + [("entity_a", np.uint64), ("entity_b", np.uint64), ("joint_key", np.uint32), ("bone_a", np.int32), ("bone_b", np.int32)])
		self.events = np.empty(0, dtype=self.event_dtype)
		if any(not isinstance(shape, pj.Shape) for shape in self._shapes.values()):
			raise TypeError("shape registry values must be pyjoltphysics.Shape objects")
		self.world = pj.World(settings if settings is not None else pj.WorldSettings(tick_hz=120))
		self._shape_cache: dict[Any, tuple[int, bool]]
		self._shape_cache = {}
		self._entities = np.empty(0, dtype=np.uint64)
		self._uids = np.empty(0, dtype=np.uint32)
		self._values = np.empty(0, dtype=_BODY_DTYPE)
		self._geometry: list[Any]
		self._geometry = []
		self._owner_by_uid: dict[int, int]
		self._uid_by_owner: dict[int, int]
		self._owner_by_uid, self._uid_by_owner = {}, {}
		self._pending: list[tuple[str, np.ndarray, dict[str, np.ndarray | None]]]
		self._pending = []
		self._token: tuple[Any, ...] | None
		self._token = None
		self.last_seq = 0
		self._checkpoint_revision = 0
		self._closed = False
		self._interpolate = interpolate
		self._pose_tick = self.world.tick
		self._position = self._previous_position = np.empty((0, 3), dtype=np.float64)
		self._rotation = self._previous_rotation = np.empty((0, 4), dtype=np.float32)
		self._entity_order = np.empty(0, dtype=np.intp)
		self._teleported = np.empty(0, dtype=np.uint32)

	def __enter__(self) -> Self:
		return self

	def __exit__(self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None) -> None:
		self.close()

	@property
	def body_uids(self) -> Mapping[int, int]:
		"""Read-only full entity handle -> UID map, including pending creations."""
		return MappingProxyType(self._uid_by_owner)

	@property
	def body_owners(self) -> Mapping[int, int]:
		return MappingProxyType(self._owner_by_uid)

	def _check(self, ecs: ECS | None=None, dt: float | None=None) -> None:
		if self._closed: raise RuntimeError("physics adapter is closed")
		if ecs is not None and ecs is not self.ecs: raise ValueError("physics adapter belongs to a different ECS")
		if dt is not None and (not np.isfinite(dt) or not np.isclose(dt, 1 / self.world.settings.tick_hz, rtol=1e-7, atol=0)):
			raise ValueError("fixed dt must equal 1 / physics tick_hz")

	def _seq(self, seq: int) -> int:
		self.last_seq = max(self.last_seq, seq)
		return seq

	def _tokens(self) -> tuple[Any, ...]:
		return (self.colliders.version(), self.bodies.version(), self.transforms.version("scale") if self._has_scale else (self.transforms.membership_version,))

	def _shape(self, key: Any) -> tuple[int, bool]:
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

	def reconcile(self, ecs: ECS | None=None, commands: CommandBuffer | None=None, dt: float | None=None) -> bool:
		"""Validate shared capacity, release outgoing resources, then create replacements."""
		self._check(ecs, dt)
		characters = self._plan_characters()
		ragdolls = self._plan_ragdolls()
		bodies = self._plan_bodies()
		removed = np.empty(0, dtype=np.uint32) if bodies is None else bodies.removals
		handles = [] if ragdolls is None else ragdolls.removals
		character_ids = [] if characters is None else characters.removals
		needed = sum(plan.required_bodies for plan in (bodies, ragdolls, characters) if plan is not None)
		freed = len(removed) + sum(len(handle.bones) for handle in handles) + len(character_ids)
		capacity = self.world.settings.max_bodies
		if self.world.reserved_bodies - freed + needed > capacity:
			# Prior reconciliations may still have queued removals reserving space.
			self.world.flush()
			if self.world.reserved_bodies - freed + needed > capacity: raise ValueError("physics declarations exceed max_bodies")
		if len(removed): self._seq(self.world.remove_bodies(removed))
		if handles: self._seq(self.world.remove_ragdolls([handle.id for handle in handles]))
		if character_ids: self._seq(self.world.remove_characters(character_ids))
		# Native additions reserve before queued removals free capacity.
		if self.world.reserved_bodies + needed > capacity: self.world.flush()
		ragdolls_changed = ragdolls is not None and ragdolls.commit()
		bodies_changed = bodies is not None and bodies.commit()
		characters_changed = characters is not None and characters.commit()
		return self._reconcile_joints(bodies_changed) or bodies_changed or ragdolls_changed or characters_changed

	@property
	def characters(self) -> Mapping[int, pj.Character]:
		"""Read-only full owner -> native Character handles, including pending creations."""
		return MappingProxyType({owner: record[1] for owner, record in self._characters.items()})

	def _active_character_mask(self, owners: np.ndarray) -> np.ndarray:
		active = self.ecs.has(owners, PhysicsCharacter) & self.ecs.has(owners, self.transform_type)
		active[active] = self.ecs.get(PhysicsCharacter)[owners[active]].enabled
		return active

	def _plan_characters(self) -> _ReconcilePlan[list[int]] | None:
		"""Validate controllers and return removed IDs, reservation count and commit."""
		if self._character_query is None: return None
		result = self._character_query.fetch()
		values = result[PhysicsCharacter].selection
		active = values.enabled
		owners = result.entities[active]
		pose = result[self.transform_type].selection
		positions, rotations = pose.position[active].astype(np.float64), pose.rotation[active]
		if np.any(self.ecs.has(owners, Collider) | self.ecs.has(owners, RigidBody)) or (self._ragdoll_query is not None and np.any(self.ecs.has(owners, PhysicsRagdoll))):
			raise ValueError("active character owners cannot also declare bodies or ragdolls")
		if not np.isfinite(positions).all(): raise ValueError("character feet positions must be finite")
		_rotations(rotations)
		if self._has_scale and not np.allclose(pose.scale[active], 1, atol=1e-6): raise ValueError("character entity scale must be unit")
		gravity = values.gravity[active]
		if np.any(np.isinf(gravity)) or np.any(gravity < 0): raise ValueError("character gravity must be nonnegative or NaN for world gravity")
		columns = [getattr(values, name)[active] for name in _CHARACTER_FIELDS]
		desired: dict[int, tuple[tuple[Any, ...], int]] = {}
		for row, owner in enumerate(map(int, owners)):
			key = int(columns[0][row])
			if key not in self._character_assets: raise ValueError(f"unknown character capabilities {key}")
			signature = tuple(None if name == "gravity" and np.isnan(column[row]) else column[row].item() for name, column in zip(_CHARACTER_FIELDS, columns)) + (self._character_assets[key],)
			desired[owner] = (signature, row)
		removed = [owner for owner, record in self._characters.items() if owner not in desired or record[0] != desired[owner][0]]
		added = [owner for owner in desired if owner not in self._characters or owner in removed]
		velocities = {}
		snapshot = self.world.latest()
		if snapshot is not None:
			rows = snapshot.characters
			for owner in added:
				if owner not in self._characters: continue
				row = cast(Any, rows.rows.get(self._characters[owner][1].id))
				if row is not None:
					positions[desired[owner][1]] = _character_feet(rows.position[row], rows.height[row])
					velocities[owner] = rows.velocity[row].copy()
		def apply() -> bool:
			if removed and self._character_history is not None:
				tick, old_owners, old_ids, previous, current = self._character_history
				keep = ~np.isin(old_ids, [self._characters[owner][1].id for owner in removed])
				self._character_history = (tick, old_owners[keep], old_ids[keep], previous[keep], current[keep]) if np.any(keep) else None
			for owner in removed:
				_, handle = self._characters.pop(owner)
				self._retired_characters[handle.id] = owner
				self._retired_bodies[handle.id] = owner
				if self.ecs.is_alive([owner])[0] and self.ecs.has([owner], CharacterState)[0]:
					cast(Any, self._character_states)[[owner]].assign(velocity=(0, 0, 0), grounded=False, ground=WORLD_ENTITY, ground_normal=(0, 0, 0), state=3, height=0, stuck=False)
			for owner in added:
				signature, row = desired[owner]
				key, gravity, pass_group, group, mask, proxy, caps = signature
				handle = self.world.add_character(positions[row] + [0, caps.standHeight * 0.5, 0], caps, gravity, pass_group, group, mask, proxy=proxy)
				self._characters[owner] = (signature, handle)
				if owner in velocities: self._seq(handle.apply_impulse(velocities[owner]))
				else: self._seq(handle.set_move((0, 0, 0)))
			self._character_owners = {handle.id: owner for owner, (_, handle) in self._characters.items()}
			return bool(removed or added)
		return _ReconcilePlan([self._characters[owner][1].id for owner in removed], len(added), apply)

	@property
	def ragdolls(self) -> Mapping[int, pj.Ragdoll]:
		"""Read-only full owner -> native Ragdoll handles; use commands, never remove bones."""
		return MappingProxyType({owner: record[1] for owner, record in self._ragdolls.items()})

	def _plan_ragdolls(self) -> _ReconcilePlan[list[pj.Ragdoll]] | None:
		"""Validate ragdolls and return removed handles, addition count and commit."""
		if self._ragdoll_query is None: return None
		result = self._ragdoll_query.fetch()
		owners = result.entities
		values = result[PhysicsRagdoll].selection
		pose = result[self.transform_type].selection
		if np.any(self.ecs.has(owners, Collider) | self.ecs.has(owners, RigidBody)):
			raise ValueError("ragdoll owners cannot also declare Collider or RigidBody")
		if np.any(values.mode > 2) or not np.all(np.isfinite(values.blend)) or np.any(values.blend < 0) or not np.all(np.isfinite(values.strength)) or np.any(values.strength < 0):
			raise ValueError("ragdoll mode must be 0..2; blend and strength must be finite and nonnegative")
		if not np.all(np.isfinite(pose.position)): raise ValueError("ragdoll frame position must be finite")
		_rotations(pose.rotation)
		if self._has_scale and not np.allclose(pose.scale, 1, atol=1e-6):
			raise ValueError("ragdoll entity scale must be unit; bake scale into skeleton and shapes")
		declarations = {int(owner): (int(asset), int(mode), float(blend), float(strength)) for owner, asset, mode, blend, strength in zip(owners, values.asset, values.mode, values.blend, values.strength)}
		for asset, _, _, _ in declarations.values():
			if asset not in self._ragdoll_assets: raise ValueError(f"unknown ragdoll asset {asset}")
		removed = [owner for owner, record in self._ragdolls.items() if owner not in declarations or record[0] != declarations[owner][0]]
		added = [owner for owner in declarations if owner not in self._ragdolls or owner in removed]
		needed = sum(len(self._ragdoll_assets[declarations[owner][0]].bones) for owner in added)
		def apply() -> bool:
			for owner in removed:
				_, handle, *_ = self._ragdolls.pop(owner)
				self._retired_bodies.update((int(uid), owner) for uid in handle.bones)
				self._retired_bones.update((int(uid), i) for i, uid in enumerate(handle.bones))
				self._ragdoll_poses.pop(owner, None)
				self._ragdoll_targets.pop(owner, None)
				self._ragdoll_history.pop(owner, None)
			for row, owner in enumerate(map(int, owners)):
				asset_key, mode, blend, strength = declarations[owner]
				asset = self._ragdoll_assets[asset_key]
				if owner in added:
					handle = self.world.add_ragdoll(asset.settings, pose.position[row], pose.rotation[row], self.pj.RagdollMode(mode))
					self._ragdoll_poses[owner] = tuple(value[None].copy() for value in (asset.skeleton.translation, asset.skeleton.rotation, asset.skeleton.scale))
					self._ragdoll_targets[owner] = asset.target_pose(self._ragdoll_poses[owner])
				else:
					_, handle, old_mode, old_blend, old_strength = self._ragdolls[owner]
					if (mode, blend) != (old_mode, old_blend): self._seq(handle.set_mode(self.pj.RagdollMode(mode), blend))
				if owner in added or strength != old_strength:
					self._seq(handle.set_strength(np.array([bone.strength for bone in asset.settings.bones]) * strength))
				self._ragdolls[owner] = (asset_key, handle, mode, blend, strength)
			self._ragdoll_owners = {int(uid): owner for owner, record in self._ragdolls.items() for uid in record[1].bones}
			self._bone_by_uid = {int(uid): i for record in self._ragdolls.values() for i, uid in enumerate(record[1].bones)}
			return bool(removed or added)
		return _ReconcilePlan([self._ragdolls[owner][1] for owner in removed], needed, apply)

	def set_ragdoll_pose(self, entities: EntityLike, pose: Sequence[np.ndarray]) -> None:
		"""Copy render-local animation targets in owner order, before prepare_inputs."""
		self._check()
		owners = as_entities(entities)
		if any(int(owner) not in self._ragdolls for owner in owners): raise ValueError("reconcile ragdolls before setting poses")
		copies = tuple(np.array(value, dtype=np.float32, copy=True) for value in pose)
		if any(value.ndim == 0 or len(value) != len(owners) for value in copies): raise ValueError("pose rows must match owners")
		groups: dict[int, list[int]]
		groups, targets = {}, {}
		for row, owner in enumerate(map(int, owners)):
			groups.setdefault(self._ragdolls[owner][0], []).append(row)
		# Convert compatible owners together, committing only after all assets validate.
		for asset_key, rows in groups.items():
			translation, rotation = self._ragdoll_assets[asset_key].target_pose(tuple(value[rows] for value in copies))
			for index, row in enumerate(rows):
				targets[int(owners[row])] = (translation[index:index + 1].copy(), rotation[index:index + 1].copy())
		for row, owner in enumerate(map(int, owners)):
			self._ragdoll_poses[owner] = tuple(value[row:row + 1].copy() for value in copies)
			self._ragdoll_targets[owner] = targets[owner]

	def ragdoll_skin(self, entities: EntityLike, *, weight: ArrayLike=1, alpha: float | None=None) -> np.ndarray:
		"""Read one snapshot and return a skin palette for owners sharing one asset."""
		self._check()
		owners = as_entities(entities)
		if not len(owners): raise ValueError("ragdoll_skin requires at least one owner")
		records = [self._ragdolls[int(owner)] for owner in owners]
		if len({record[0] for record in records}) != 1: raise ValueError("skin owners must share one ragdoll asset")
		if not np.all(self.ecs.is_alive(owners) & self.ecs.has(owners, self.transform_type)): raise ValueError("ragdoll skin owners must have live transforms")
		asset = self._ragdoll_assets[records[0][0]]
		snapshot = self.world.latest()
		if snapshot is None: raise LookupError("step physics before reading ragdoll skin")
		rows = snapshot.ragdolls
		lookup = rows.rows()
		if any(record[1].id not in lookup for record in records): raise LookupError("owned ragdoll is absent from snapshot")
		spans = [rows.bones(lookup[record[1].id]) for record in records]
		positions = np.stack([rows.bone_position[span] for span in spans])
		orientations = np.stack([rows.bone_orientation[span] for span in spans])
		if alpha is not None:
			if not self._interpolate: raise RuntimeError("enable interpolate=True to sample ragdoll history")
			if np.ndim(alpha) != 0 or not np.isfinite(alpha) or not 0 <= alpha <= 1: raise ValueError("interpolation alpha must be finite and in [0, 1]")
			for row, owner in enumerate(map(int, owners)):
				if owner not in self._ragdoll_history or self._ragdoll_history[owner][0] != snapshot.tick: raise RuntimeError("pull the latest ragdoll tick before interpolation")
				_, previous_position, previous_rotation, current_position, current_rotation = self._ragdoll_history[owner]
				positions[row] = previous_position * (1 - alpha) + current_position * alpha
				orientations[row] = Quaternion.slerp_batch(previous_rotation, current_rotation, alpha)
		base = tuple(np.concatenate([self._ragdoll_poses[int(owner)][i] for owner in owners]) for i in range(3))
		frame = self.transforms[owners]
		return asset.skin(positions, orientations, base, position=frame.position, rotation=frame.rotation, weight=weight)

	def _plan_bodies(self) -> _ReconcilePlan[np.ndarray] | None:
		"""Validate bodies and return removed UIDs, addition count and commit.

		Static poses change only through teleport. Shape/scale/motion/filter-bit
		changes rebuild bodies; dynamic rebuilds retain native pose and velocity.
		"""
		token = self._tokens()
		if token == self._token: return None
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
		defaults = RigidBody(motion=cast(np.uint8, 0))
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
			position[spawn], cast(Sequence[int], np.asarray(shape_ids, dtype=np.uint32)[spawn]), **kwargs,
			orientations=rotation[spawn], scale=shape_scale, shape_offset=shape_offset,
			shape_rotation=shape_rotation, linear_velocity=linear[spawn], angular_velocity=angular[spawn],
		)
		# Validate creation arrays before submitting any destructive changes.
		if len(spawn): batch.arrays(shape_id=self.world.add_shape)
		def apply() -> bool:
			self._retired_bodies.update((int(uid), self._owner_by_uid[int(uid)]) for uid in removed)
			if len(spawn):
				created, seq = self.world.add_bodies(batch)
				uids[spawn] = created
				self._seq(seq)
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
		return _ReconcilePlan(removed, len(spawn), apply)

	@property
	def joint_ids(self) -> Mapping[tuple[int, int], int]:
		"""(full owner handle, stable component key) -> native joint ID."""
		return MappingProxyType({key: record[0] for key, record in self._joint_records.items()})

	def _reconcile_joints(self, bodies_changed: bool) -> bool:
		if self.joints is None: return False
		token = self.joints.version()
		if not bodies_changed and token == self._joint_token: return False
		assert self._joint_query is not None and self._joint_type is not None
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
			row: dict[str, Any] = {name: tuple(value[i]) if value.ndim > 1 else value[i].item() for name, value in values.items()}
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
			ids: Any
			ids, seq = self.world.add_joints(descriptors)
			self._seq(seq)
			self._joint_records.update((key, (int(uid), *desired[key])) for key, uid in zip(created, ids))
		motor_updates: dict[int, list[tuple[int, int, Any, Any]]]
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

	def set_joint_motor(self, entities: EntityLike, keys: ArrayLike, *, mode: Any=None, target: ArrayLike=np.nan, max_force: ArrayLike=np.nan, axis: ArrayLike=0) -> int:
		self._check()
		entities = as_entities(entities)
		keys = np.broadcast_to(np.asarray(keys), entities.shape)
		if keys.dtype.kind not in "iu" or np.any(keys < 0) or np.any(keys > 0xffffffff): raise ValueError("joint keys must be uint32 integers")
		ids = [self._joint_records[(int(owner), int(key))][0] for owner, key in zip(entities, keys)]
		seq = self._seq(self.world.set_joint_motors(ids, mode, target=target, max=max_force, axis=axis))
		self._checkpoint_revision += 1
		return seq

	def entities_of(self, uids: ArrayLike) -> np.ndarray:
		"""Resolve current body UIDs to live full handles; WORLD_ENTITY means absent."""
		self._check()
		return self._resolve_owners(uids, self._owner_by_uid | self._ragdoll_owners | self._character_owners)

	def bone_indices_of(self, uids: ArrayLike) -> np.ndarray:
		"""Resolve native physical bone indices, or -1 for other/unowned bodies."""
		self._check()
		bones = _lookup_uids(uids, self._bone_by_uid, np.int32(-1))
		owners = self._resolve_owners(uids, self._ragdoll_owners)
		bones[owners == WORLD_ENTITY] = -1
		return bones

	def _resolve_owners(self, uids: ArrayLike, mapping: Mapping[int, int]) -> np.ndarray:
		result = _lookup_uids(uids, mapping, WORLD_ENTITY)
		if result.size:
			result[~self.ecs.is_alive(result.reshape(-1)).reshape(result.shape)] = WORLD_ENTITY
		return result

	def query_filter(self, *, exclude: EntityLike=(), group: int=0xffffffff, mask: int=0xffffffff, triggers: bool=False, characters: bool=True) -> pj.QueryFilter:
		self._check()
		uids = [self._uid_by_owner[int(owner)] for owner in as_entities(exclude) if int(owner) in self._uid_by_owner]
		ragdolls = [self._ragdolls[int(owner)][1].bones for owner in as_entities(exclude) if int(owner) in self._ragdolls]
		if ragdolls: uids.extend(np.concatenate(ragdolls).tolist())
		uids.extend(self._characters[int(owner)][1].id for owner in as_entities(exclude) if int(owner) in self._characters)
		return self.pj.QueryFilter(group=group, mask=mask, exclude=uids, triggers=triggers, characters=characters)

	def _query_filter(self, filter: pj.QueryFilter | None, exclude: EntityLike) -> pj.QueryFilter:
		if filter is None: return self.query_filter(exclude=exclude)
		extra = self.query_filter(exclude=exclude).exclude
		return self.pj.QueryFilter(group=filter.group, mask=filter.mask, exclude=(*filter.exclude, *extra), triggers=filter.triggers, characters=filter.characters)

	def raycast(self, origin: ArrayLike, direction: ArrayLike, max_distance: float=1000.0, *, filter: pj.QueryFilter | None=None, exclude: EntityLike=()) -> pj.RayHit | None:
		self._check()
		return self.world.raycast(cast(Sequence[float], origin), cast(Sequence[float], direction), max_distance, self._query_filter(filter, exclude))

	def raycast_batch(self, origins: ArrayLike, directions: ArrayLike, max_distance: float=1000.0, *, filter: pj.QueryFilter | None=None, exclude: EntityLike=()) -> pj.RayHits:
		self._check()
		return self.world.raycast_batch(origins, directions, max_distance, self._query_filter(filter, exclude))

	def sweep(self, shape: pj.Shape | int, start: ArrayLike, end: ArrayLike, *, orientation: ArrayLike=(0, 0, 0, 1), filter: pj.QueryFilter | None=None, exclude: EntityLike=(), all_hits: bool=False) -> pj.ShapeHit | list[pj.ShapeHit] | None:
		self._check()
		shape = shape if isinstance(shape, self.pj.Shape) else self._shapes[int(shape)]
		return self.world.sweep(shape, cast(Sequence[float], start), cast(Sequence[float], end), cast(Sequence[float], orientation), self._query_filter(filter, exclude), all_hits)

	def overlap(self, shape: pj.Shape | int, position: ArrayLike, *, orientation: ArrayLike=(0, 0, 0, 1), filter: pj.QueryFilter | None=None, exclude: EntityLike=()) -> np.ndarray:
		self._check()
		shape = shape if isinstance(shape, self.pj.Shape) else self._shapes[int(shape)]
		return self.world.overlap(shape, cast(Sequence[float], position), cast(Sequence[float], orientation), self._query_filter(filter, exclude))

	def set_filter_table(self, collides: ArrayLike) -> int:
		self._check()
		seq = self._seq(self.world.set_filter_table(collides))
		self._checkpoint_revision += 1
		return seq

	@property
	def materials(self) -> pj.MaterialTable:
		return self.world.materials

	@property
	def events_dropped(self) -> int:
		return int(self.world.stats()["events_dropped"])

	def subscribe(self, callback: Callable[[ECS, CommandBuffer, np.ndarray], None], *, kind: Any=None, entity: EntityLike | None=None) -> int:
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

	def unsubscribe(self, token: int) -> None:
		self._subscriptions.pop(token, None)

	def dispatch_events(self, ecs: ECS | None=None, commands: CommandBuffer | None=None, dt: float | None=None) -> np.ndarray:
		"""Drain once per tick and route native records, preserving removal identity."""
		self._check(ecs, dt)
		if self._events_tick == self.world.tick: return self.events
		raw = self.world.drain_events()
		events = np.zeros(len(raw), dtype=self.event_dtype)
		for name in raw.dtype.names or (): events[name] = raw[name]
		events["entity_a"] = events["entity_b"] = WORLD_ENTITY
		events["bone_a"] = events["bone_b"] = -1
		body_events = np.isin(raw["kind"], (self.pj.EventKind.CONTACT, self.pj.EventKind.TRIGGER, self.pj.EventKind.STATE))
		owners = self._retired_bodies | self._owner_by_uid | self._ragdoll_owners | self._character_owners
		events["entity_a"][body_events] = self._resolve_owners(raw["a"][body_events], owners)
		character = raw["kind"] == self.pj.EventKind.CHARACTER
		events["entity_a"][character] = self._resolve_owners(raw["a"][character], self._retired_characters | self._character_owners)
		pair = np.isin(raw["kind"], (self.pj.EventKind.CONTACT, self.pj.EventKind.TRIGGER, self.pj.EventKind.CHARACTER))
		events["entity_b"][pair] = self._resolve_owners(raw["b"][pair], owners)
		bones = self._retired_bones | self._bone_by_uid
		if bones:
			live_body_a = body_events & (events["entity_a"] != WORLD_ENTITY)
			live_body_b = pair & (events["entity_b"] != WORLD_ENTITY)
			events["bone_a"][live_body_a] = _lookup_uids(raw["a"][live_body_a], bones, np.int32(-1))
			events["bone_b"][live_body_b] = _lookup_uids(raw["b"][live_body_b], bones, np.int32(-1))
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
		self._retired_bones.clear()
		self._retired_characters.clear()
		own_commands = commands is None
		commands = self.ecs.commands() if commands is None else commands
		for callback, kind, entity in tuple(self._subscriptions.values()):
			# A prior callback may delete an owner; never route to its recycled slot.
			live_a, live_b = self.ecs.is_alive(events["entity_a"]), self.ecs.is_alive(events["entity_b"])
			selected = live_a | live_b
			if kind is not None: selected &= events["kind"] == kind
			if entity is not None: selected &= (live_a & (events["entity_a"] == entity)) | (live_b & (events["entity_b"] == entity))
			if np.any(selected): callback(self.ecs, commands, events[selected])
		if own_commands: commands.flush()
		return self.events

	def _update_settings(self, uids: np.ndarray, previous: np.ndarray, values: np.ndarray) -> None:
		def changed(*names: str) -> np.ndarray:
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

	def _queue(self, kind: str, entities: EntityLike, **values: ArrayLike | None) -> None:
		self._check()
		entities = as_entities(entities).copy()
		arrays = {name: None if value is None else _vectors(value, len(entities), 4 if name == "orientations" else 3, name) for name, value in values.items()}
		if "orientations" in arrays:
			assert arrays["orientations"] is not None
			_rotations(arrays["orientations"])
		self._pending.append((kind, entities, arrays))

	def teleport(self, entities: EntityLike, positions: ArrayLike, orientations: ArrayLike=(0, 0, 0, 1)) -> None:
		"""Queue an explicit pose move; velocities are preserved."""
		self._queue("set_transforms", entities, positions=positions, orientations=orientations)

	def set_velocity(self, entities: EntityLike, linear: ArrayLike | None=None, angular: ArrayLike | None=None) -> None:
		self._queue("set_velocities", entities, linear=linear, angular=angular)

	def reset_bodies(self, entities: EntityLike, positions: ArrayLike, orientations: ArrayLike=(0, 0, 0, 1), *, linear: ArrayLike=(0, 0, 0), angular: ArrayLike=(0, 0, 0)) -> None:
		"""Queue selected regular bodies' poses/velocities, replacing their pending inputs.

		Call between ticks after native submissions have applied. Other owners'
		commands and interpolation history are preserved; constraints are not reset.
		"""
		self._check()
		owners = as_entities(entities).copy()
		if len(np.unique(owners)) != len(owners): raise ValueError("reset owners must be distinct")
		if any(int(owner) not in self._uid_by_owner for owner in owners) or not np.all(self.ecs.is_alive(owners) & self.ecs.has(owners, Collider) & self.ecs.has(owners, self.transform_type)):
			raise ValueError("reset requires live reconciled regular body owners")
		position = _vectors(positions, len(owners), 3, "positions")
		rotation = _vectors(orientations, len(owners), 4, "orientations")
		_rotations(rotation)
		velocity = _vectors(linear, len(owners), 3, "linear")
		spin = _vectors(angular, len(owners), 3, "angular")
		snapshot = self.world.latest()
		if not self.world.lockstep or snapshot is None or snapshot.applied_seq != self.world.submitted_seq:
			raise RuntimeError("apply native submissions before resetting bodies")
		pending = []
		for kind, queued, values in self._pending:
			keep = ~np.isin(queued, owners)
			if np.any(keep): pending.append((kind, queued[keep], {name: None if value is None else value[keep] for name, value in values.items()}))
		pending.append(("set_transforms", owners, dict(positions=position, orientations=rotation)))
		pending.append(("set_velocities", owners.copy(), dict(linear=velocity, angular=spin)))
		self._pending = pending

	def _checkpoint_joints(self) -> tuple[Any, ...]:
		return tuple((key, record[0], tuple(np.asarray(record[2][name]).tobytes() for name in _JOINT_FIELDS)) for key, record in sorted(self._joint_records.items()))

	def _checkpoint_snapshot(self) -> pj.Snapshot:
		self._check()
		snapshot = self.world.latest()
		if not self.world.lockstep or snapshot is None or snapshot.applied_seq != self.world.submitted_seq or self._events_tick != self.world.tick:
			raise RuntimeError("checkpoints require applied native submissions and dispatched events between ticks")
		stats = self.world.stats()
		if self._ragdolls or self._characters or self._character_pending or any(stats[name] for name in ("ragdolls", "characters", "vehicles", "soft_bodies")):
			raise NotImplementedError("checkpoints support regular bodies and joints; native control state is not captured")
		if self._tokens() != self._token or (self.joints is not None and self.joints.version() != self._joint_token):
			raise RuntimeError("reconcile physics declarations before checkpoint operations")
		if not np.array_equal(np.sort(snapshot.uid), np.sort(self._uids)) or set(snapshot.joints.id) != {record[0] for record in self._joint_records.values()}:
			raise ValueError("checkpoints require exclusively adapter-owned bodies and joints")
		return snapshot

	def save_state(self) -> _PhysicsState:
		"""Capture regular body/joint state, ECS targets and queued adapter inputs.

		Requires a completed tick and unchanged configuration/topology on restore.
		Native ticks, event delivery, external clocks and command buffers are not
		rewound. Characters and ragdolls require unavailable native control state.
		"""
		snapshot = self._checkpoint_snapshot()
		pose = self.transforms[self._entities]
		return _PhysicsState(
			self, self.world.tick, self.world.save_state(), snapshot.uid.copy(), self._entities.copy(), self._values.copy(),
			tuple(self._geometry), self._checkpoint_joints(), self._checkpoint_revision, pose.position, pose.rotation, deepcopy(self._pending),
		)

	def restore_state(self, state: _PhysicsState) -> int:
		"""Restore a same-adapter checkpoint without stepping; return its saved tick.

		Replace Python input queues and snap render history to restored poses.
		The caller restores its own animation/gameplay clocks using the returned tick.
		"""
		if not isinstance(state, _PhysicsState) or state.adapter is not self: raise ValueError("checkpoint belongs to a different physics adapter")
		snapshot = self._checkpoint_snapshot()
		if not np.array_equal(snapshot.uid, state.uids) or not np.array_equal(self._entities, state.entities) or not np.array_equal(self._values, state.values) or tuple(self._geometry) != state.geometry or self._checkpoint_joints() != state.joints or self._checkpoint_revision != state.revision:
			raise ValueError("checkpoint body/joint topology and configuration must remain unchanged")
		rejected = self.world.stats()["rejected"]
		self._seq(self.world.restore_state(state.native))
		self.world.flush()
		if self.world.stats()["rejected"] != rejected: raise RuntimeError("native physics checkpoint was rejected")
		self.transforms[self._entities].assign(position=state.position, rotation=state.rotation)
		self._pending = deepcopy(state.pending)
		self._pose_tick = -1
		self._teleported = np.empty(0, dtype=np.uint32)
		self.pull()
		self.world.drain_events()
		self.events = np.empty(0, dtype=self.event_dtype)
		self.events.flags.writeable = False
		self._events_tick = self.world.tick
		return state.tick

	def add_impulse(self, entities: EntityLike, impulses: ArrayLike, points: ArrayLike | None=None) -> None:
		self._queue("add_impulses", entities, impulses=impulses, points=points)

	def add_force(self, entities: EntityLike, forces: ArrayLike, points: ArrayLike | None=None) -> None:
		self._queue("add_forces", entities, forces=forces, points=points)

	def character_commands(self, entities: EntityLike, op: Any, *, code: int=0, values: ArrayLike=(0, 0, 0, 0)) -> None:
		"""Queue native CharacterOp rows by full entity handle; TELEPORT values use centres."""
		self._check()
		owners = as_entities(entities).copy()
		ops, codes = np.broadcast_to(op, (len(owners),)).copy(), np.broadcast_to(code, (len(owners),)).copy()
		if ops.dtype.kind not in "iu" or not np.isin(ops, list(self.pj.CharacterOp)).all(): raise ValueError("invalid character operation")
		if codes.dtype.kind not in "iub" or np.any(codes < 0) or np.any(codes > 255): raise ValueError("character codes must fit uint8")
		data = np.array(np.broadcast_to(values, (len(owners), 4)), dtype=np.float64, copy=True)
		move = ops == self.pj.CharacterOp.MOVE
		allowed_nan = np.zeros(data.shape, dtype=bool)
		allowed_nan[move, 3] = True
		if np.any(np.isinf(data)) or np.any(np.isnan(data) & ~allowed_nan): raise ValueError("character values must be finite except an optional MOVE speed")
		if np.any(codes[move] > 3) or np.any(data[move, 3] < 0): raise ValueError("character MOVE mode must be 0..3 and speed nonnegative")
		if np.any(data[ops == self.pj.CharacterOp.TELEPORT, 3] < 0): raise ValueError("teleport search must be nonnegative")
		self._character_pending.append((owners, ops.astype(np.uint8), codes.astype(np.uint8), data))

	def move_characters(self, entities: EntityLike, directions: ArrayLike, *, speed: ArrayLike | None=None, mode: Any=0) -> None:
		"""Queue held world-space movement; zero direction stops the native controller."""
		owners = as_entities(entities)
		data = np.empty((len(owners), 4), dtype=np.float64)
		data[:, :3] = _vectors(directions, len(owners), 3, "directions")
		data[:, 3] = np.nan if speed is None else speed
		self.character_commands(owners, self.pj.CharacterOp.MOVE, code=mode, values=data)

	def jump_characters(self, entities: EntityLike) -> None:
		self.character_commands(entities, self.pj.CharacterOp.JUMP)

	def prepare_inputs(self, ecs: ECS | None=None, commands: CommandBuffer | None=None, dt: float | None=None) -> None:
		"""Send current kinematic targets and queued commands for the next tick."""
		self._check(ecs, dt)
		while self._character_pending:
			owners, ops, codes, data = self._character_pending[0]
			if self._character_query is None: raise ValueError("enable characters before submitting character commands")
			live = self._active_character_mask(owners)
			ids = np.array([self._characters[int(owner)][1].id if int(owner) in self._characters else 0 for owner in owners], dtype=np.uint32)
			if np.any(live & (ids == 0)): raise ValueError("reconcile characters before submitting inputs")
			if np.any(live):
				self._seq(self.world.character_ops(ids[live], ops[live], codes[live], data[live]))
				self._character_snap.update(map(int, ids[live & (ops == self.pj.CharacterOp.TELEPORT)]))
			self._character_pending.pop(0)
		if self._ragdolls:
			owners = np.fromiter(self._ragdolls, dtype=np.uint64)
			owners = owners[self.ecs.has(owners, PhysicsRagdoll) & self.ecs.has(owners, self.transform_type)]
			frame = self.transforms[owners]
			if self._has_scale and not np.allclose(frame.scale, 1, atol=1e-6): raise ValueError("ragdoll entity scale must be unit")
			positions, rotations = frame.position, frame.rotation
			for row, owner in enumerate(map(int, owners)):
				translation, rotation = self._ragdoll_targets[owner]
				self._seq(self._ragdolls[owner][1].set_pose(rotation[0], translation[0], positions[row], rotations[row]))
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
					self.transforms[entities[live]].assign(position=cast(np.ndarray, values["positions"])[live], rotation=cast(np.ndarray, values["orientations"])[live])
					if self._interpolate: self._teleported = np.union1d(self._teleported, uids[live])
			self._pending.pop(0)

	def step(self, ecs: ECS | None=None, commands: CommandBuffer | None=None, dt: float | None=None) -> None:
		self._check(ecs, dt)
		self.world.run_ticks(1)

	def pull(self, ecs: ECS | None=None, commands: CommandBuffer | None=None, dt: float | None=None) -> None:
		"""Mirror changed dynamic poses; never keep native snapshot views."""
		self._check(ecs, dt)
		snapshot = self.world.latest()
		if snapshot is None: return
		if self._characters: self._pull_characters(snapshot)
		rows = snapshot.rows_of(self._uids)
		if snapshot.applied_seq >= self.last_seq and np.any(rows < 0):
			raise RuntimeError("owned physics body disappeared; do not remove adapter bodies through world")
		if self._interpolate and self._ragdolls:
			ragdolls = snapshot.ragdolls
			lookup = ragdolls.rows()
			for owner, record in self._ragdolls.items():
				if record[1].id not in lookup: continue
				span = ragdolls.bones(lookup[record[1].id])
				position, rotation = ragdolls.bone_position[span].copy(), ragdolls.bone_orientation[span].copy()
				previous = self._ragdoll_history.get(owner)
				if previous is None or snapshot.tick not in (previous[0], previous[0] + 1):
					previous_position, previous_rotation = position, rotation
				elif snapshot.tick == previous[0]:
					previous_position, previous_rotation = previous[1:3]
				else:
					previous_position, previous_rotation = previous[3:5]
				self._ragdoll_history[owner] = (snapshot.tick, previous_position, previous_rotation, position, rotation)
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

	def _pull_characters(self, snapshot: pj.Snapshot) -> None:
		rows = snapshot.characters
		owners = np.array(list(self._characters), dtype=np.uint64)
		ids = np.array([self._characters[int(owner)][1].id for owner in owners], dtype=np.uint32)
		indices = np.searchsorted(rows.id, ids)
		found = indices < len(rows)
		if len(rows): found &= rows.id[np.minimum(indices, len(rows) - 1)] == ids
		if snapshot.applied_seq >= self.last_seq and not found.all(): raise RuntimeError("owned character disappeared; remove controllers through ECS")
		selected = found & self._active_character_mask(owners)
		owners, ids, indices = owners[selected], ids[selected], indices[selected]
		feet = _character_feet(rows.position[indices], rows.height[indices])
		self.transforms[owners].position = feet
		state = self.ecs.has(owners, CharacterState)
		if np.any(state):
			selected_rows = indices[state]
			cast(Any, self._character_states)[owners[state]].assign(velocity=rows.velocity[selected_rows], grounded=rows.grounded[selected_rows], ground=self.entities_of(rows.ground_uid[selected_rows]), ground_normal=rows.ground_normal[selected_rows], state=rows.state[selected_rows], height=rows.height[selected_rows], stuck=rows.stuck[selected_rows])
		if self._interpolate:
			previous = feet.copy()
			old = self._character_history
			if old is not None and snapshot.tick in (old[0], old[0] + 1):
				_, before, now = np.intersect1d(old[2], ids, return_indices=True)
				previous[now] = old[3 if snapshot.tick == old[0] else 4][before]
			previous[np.isin(ids, list(self._character_snap))] = feet[np.isin(ids, list(self._character_snap))]
			order = np.argsort(owners)
			self._character_history = (snapshot.tick, owners[order].copy(), ids[order].copy(), previous[order], feet[order]) if len(owners) else None
		self._character_snap.clear()

	def render_transforms(self, entities: EntityLike, *, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
		if self._character_history is not None:
			_, owners, _, previous, current = self._character_history
			indices = np.searchsorted(owners, entities)
			rows = np.minimum(indices, len(owners) - 1)
			selected = (indices < len(owners)) & (owners[rows] == entities)
			selected[selected] = self._active_character_mask(entities[selected])
			if np.any(selected): position[selected] = previous[rows[selected]] * (1 - alpha) + current[rows[selected]] * alpha
		return position, rotation, scale

	def close(self) -> None:
		"""Join the native thread and clear ownership and queued commands; idempotent."""
		if self._closed: return
		try:
			if self._characters and self.world.error is None:
				self.world.remove_characters([handle.id for _, handle in self._characters.values()])
			if self._ragdolls and self.world.error is None:
				self.world.remove_ragdolls([record[1].id for record in self._ragdolls.values()])
			if self._joint_records and self.world.error is None:
				self.world.remove_joints([record[0] for record in self._joint_records.values()])
			if len(self._uids) and self.world.error is None:
				self.world.remove_bodies(self._uids)
			if self.world.error is None:
				self.world.flush()
		finally:
			self.world.stop()
			self._pending.clear()
			self._character_pending.clear()
			self._characters.clear()
			self._character_owners.clear()
			self._retired_characters.clear()
			self._character_history = None
			self._character_snap.clear()
			self._shape_cache.clear()
			self._uid_by_owner.clear()
			self._owner_by_uid.clear()
			self._joint_records.clear()
			self._ragdolls.clear()
			self._ragdoll_poses.clear()
			self._ragdoll_targets.clear()
			self._ragdoll_history.clear()
			self._ragdoll_owners.clear()
			self._bone_by_uid.clear()
			self._retired_bones.clear()
			self._retired_bodies.clear()
			self._retired_joints.clear()
			self._subscriptions.clear()
			self.events = np.empty(0, dtype=self.event_dtype)
			self._position = self._previous_position = np.empty((0, 3), dtype=np.float64)
			self._rotation = self._previous_rotation = np.empty((0, 4), dtype=np.float32)
			self._entity_order = np.empty(0, dtype=np.intp)
			self._teleported = np.empty(0, dtype=np.uint32)
			self._closed = True
