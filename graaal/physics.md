# Optional ECS physics

Install `python -m pip install -e ".[physics]"` from the repository root.
The extra pins `pyjoltphysics==0.1.0a3`. Core ECS and physics component imports
do not require this package or rendering dependencies. Constructing the adapter
loads the native backend; physics updates reuse the stored module reference.
Non-ECS applications use `pyjoltphysics.World` directly;
see [the standalone example](../scenes/physics_headless_example.py).

```python
import pyjoltphysics as pj
from graaal.ECS import ECS, Scheduler
from graaal.MaterialTransform import MaterialTransform
from graaal.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody

ecs = ECS()
ecs.register(MaterialTransform, Collider, RigidBody)
floor, box = ecs.create(2)
ecs.add(floor, MaterialTransform(position=(0, -0.5, 0), scale=(20, 1, 20)), Collider(0))
ecs.add(box, MaterialTransform(position=(0, 3, 0)), Collider(0), RigidBody())

with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, settings=pj.WorldSettings(threads=1)) as physics:
	schedule = Scheduler(ecs)
	schedule.add("physics_sync", physics.reconcile, phase="fixed_update")
	schedule.add("physics_input", physics.prepare_inputs, phase="fixed_update", after="physics_sync")
	schedule.add("physics_step", physics.step, phase="fixed_update", after="physics_input")
	schedule.add("physics_output", physics.pull, phase="fixed_update", after="physics_step")
	schedule.add("physics_events", physics.dispatch_events, phase="fixed_update", after="physics_output")
	schedule.run(1 / 60, fixed_steps=2, fixed_dt=1 / 120)
	physics.add_impulse([box], [2, 0, 0])
	schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
	print(ecs.get(MaterialTransform)[[box]].position)
```

The caller owns the accumulator and fixed-step count. Each `step` runs exactly
one native tick; scheduler `dt` must equal `1 / settings.tick_hz`. Default rate
is 120 Hz. Callbacks also support direct calls without scheduler arguments.
An earlier system that queues component changes needs `flush_after=True` for
`reconcile` to see them in the same tick. Dependencies alone do not flush.

`RigidBody` stores motion (`0` static, `1` kinematic, `2` dynamic), positive
mass for moving bodies, friction, restitution, linear/angular damping, gravity
factor, group/mask bitsets, native flags, and material/filter indices.
Collider-only entities create static bodies. Entities require a transform and
at least one collider to own a body. Native handles and objects are adapter
resources; components contain only settings and numeric shape asset keys.

`Collider(shape, position=..., rotation=..., scale=..., sensor=False)` names an
immutable `pj.Shape` in the constructor's shape mapping. Share one shape object
across entities. The adapter caches native shapes and compounds. Geometry is
independent of render meshes; mesh and height-field shapes cannot be dynamic.
All dimensions and translations are metres; quaternions use **xyzw**.

Use `transform_type=YourTransform` to bind another component with numeric
`position` (3) and `rotation` (4) fields. Optional `scale` (3) defaults to unit
scale. Transform and collider scales must be positive and finite. The default
is `MaterialTransform`, defined in `graaal.MaterialTransform`; importing it
requires no shader or graphics package.

- Dynamic bodies own their native pose and velocity. `pull` writes changed
  dynamic poses to ECS; editing the mirror never implicitly teleports a body.
- Kinematic transforms are targets sent each tick by `prepare_inputs`.
- Static transforms initialize the body. Move them with `teleport`, which also
  updates the ECS pose. Subsequent ordinary position/rotation writes have no
  physics effect. Teleport preserves velocity; reset it explicitly if needed.

Commands `teleport(entities, positions, orientations=...)`,
`set_velocity(entities, linear=None, angular=None)`,
`add_impulse(entities, impulses, points=None)`, and
`add_force(entities, forces, points=None)` copy their inputs and queue full ECS
handles. Vectors broadcast from one row or accept one row per entity. Optional
points are world positions. `prepare_inputs` resolves current UIDs and submits
NumPy batches; deleted owners and owners that lost their collider/transform are
dropped. An active eligible owner must have been reconciled first. Commands
take effect at the next native tick, in submission order after kinematic targets.

Shape, scale, motion, group/mask and creation-only flag changes rebuild bodies.
Dynamic rebuilds preserve current native pose and velocity when the replacement
is moving. Friction, restitution, damping, mass, gravity factor, material/filter
indices and writable flags use native setters. Membership and field versions
skip unchanged configuration work; in-place object edits are not supported for
shape assets. Local shape offsets/rotations participate in rebuild detection.

Multiple colliders form one compound with one body UID and one sensor flag.
Mixed solid/sensor colliders require separate entities in this version. Multiple
colliders require unit local scale; author scaled shape assets for those cases.
A single rotated local collider requires uniform entity scale to avoid shear.
Native compound restrictions still apply, including no height-field children.

`physics.world` exposes native snapshots, events, queries and advanced APIs.
`body_uids` and `body_owners` are read-only mapping views of current ownership;
reacquire them after reconciliation. Full uint64 entity handles stay distinct
from uint32 body UIDs, including while creations await a tick. `last_seq` is
the latest submitted batch sequence. Do not remove adapter-owned bodies through
the native world; `pull` reports missing owned UIDs rather than indexing row -1.
Removing bodies through ECS also removes their native attached constraints.

Snapshots are read-only native views. Resolve UIDs each snapshot, copy data
needed beyond its lifetime, and release snapshots promptly. Schedule
`dispatch_events` after `pull` once per fixed tick to drain and route events.

## Events and entity identity

`physics.events` is a read-only structured NumPy batch, replaced on every
dispatch tick. It preserves all native fields and adds `entity_a`, `entity_b`
(full uint64 handles) and `joint_key` (uint32, meaningful for joint events).
`WORLD_ENTITY` is the maximum uint64 value and means world/unresolved ownership;
entity zero is a valid owner. Native `a`/`b` IDs retain their original namespace:
body IDs for contacts/triggers/state, joint ID in `a` for joint events, character
ID in `a` for character events. Characters created directly on the native world
have unresolved `entity_a`; their contacted body can still resolve `entity_b`.
Ragdoll body records also resolve `bone_a`/`bone_b` to native physical bone
indices; other or unresolved endpoints use -1. The asset's `bones` array maps
these indices to render bones. Character-owner mapping remains a later phase.

```python
def on_contact(ecs, commands, events):
	for event in events:
		print(event["entity_a"], event["entity_b"], event["phase"])

subscription = physics.subscribe(on_contact, kind=pj.EventKind.CONTACT, entity=box)
# Later: physics.unsubscribe(subscription)
```

Subscriptions accept an optional kind and one full entity handle. Callbacks run
synchronously on the dispatching/scheduler thread and receive matching batches.
CONTACT, TRIGGER, STATE, CHARACTER and JOINT records remain inspectable in
`events`, including records without known owners. Subscriptions deliver rows
with at least one live owner; an entity filter only matches a live endpoint.
The same tick is never delivered twice, even if `dispatch_events` is called again.

Begin/end records are the default; `RigidBody(flags=int(pj.BodyFlag.REPORT))`
opts into persist records. `physics.events_dropped` exposes the native cumulative
overflow counter. Contact `impulse` is an estimate, not the solver's final impulse.

Removal records retain the old UID-to-owner association until dispatch. A live
entity that lost its collider can still receive END(reason=REMOVED). A deleted
owner resolves to `WORLD_ENTITY`, while a live counterpart can receive the end
record. Recycled indices never substitute a new generation into old events.
Joint end records likewise preserve the removed component's owner and key.

Use the supplied command buffer for structural edits. Scheduler commands become
visible at the fixed-tick boundary, or after an explicit `flush_after=True` system;
event-driven physics inputs therefore affect the next tick by default. Direct
dispatch without a command buffer creates one and flushes it after all listeners.
Do not also call `world.drain_events()` while using routed events: the dispatcher
owns the drain and releases its retired-ID metadata there.

## Queries, filters and materials

`raycast(origin, direction, max_distance=...)` and
`raycast_batch(origins, directions, max_distance=...)` return native hit records
with UID, distance, point, normal and sub-shape identity. Use
`entities_of(hit.uid)` or `entities_of(hits.uid)` to resolve live owners; unknown
UIDs and ray misses resolve to `WORLD_ENTITY`. Batch shapes and native IDs remain
unchanged, so advanced native calls can consume the same results.

```python
hit = physics.raycast((0, 10, 0), (0, -1, 0), 20, exclude=[box])
if hit is not None:
	print(physics.entities_of(hit.uid).item(), hit.point, hit.normal)
```

`sweep(shape, start, end, all_hits=False)` returns the first native shape hit
or a list when `all_hits=True`. `overlap(shape, position)` returns ascending
body UIDs. A shape is a `pj.Shape` or an integer **asset key** from the adapter's
shape mapping. Both methods accept an xyzw `orientation` keyword.

All wrappers accept `exclude=entities` and an optional native `filter`.
`query_filter(exclude=..., group=..., mask=..., triggers=False, characters=True)`
builds that filter from current UIDs. A saved filter captures its exclusions at
construction; use each query's `exclude` argument to follow later body rebuilds.
Queries use actual native state from the last completed tick, including for
interpolated renderers. Queued creations and moves become queryable after a tick
or an explicit native `flush()`.

`set_filter_table(bool_matrix)` queues the native symmetric collision-filter
table. Bodies select its rows through `RigidBody.filter`; group/mask bitsets
remain the native collision rules. `physics.materials` exposes the native table:
`material_id = physics.materials.define(friction=0.1, restitution=0.2)` and
`physics.materials.pair(a, b, friction=0.3)` configure it. Bodies select material
indices through `RigidBody.material`. No additional layer/material system is used.

See [physics_queries_example.py](../scenes/physics_queries_example.py) for ray
batches replacing the analytic foot-IK ground query with a static collision mesh.
`foot_ik_example.create_demo(ground_query=...)` accepts world-space foot positions
and expects per-foot heights and normals. Missing/nonfinite hits disable foot
correction for that sample. The example does not add character locomotion physics.

## Joints

Register `PhysicsJoint` and construct the adapter with `joint_type=PhysicsJoint`
to enable constraint reconciliation. The default body-only adapter remains usable
without registering this component. Each joint lives on its source body entity.
`target` is another full entity handle, or `WORLD_ENTITY` for a world attachment.
Creation waits until both referenced bodies exist. Multiple joints on an owner
need distinct explicit `key` values; the default key is zero.

```python
from graaal.EcsPhysics import PhysicsJoint, WORLD_ENTITY

ecs.register(PhysicsJoint)
ecs.add(box, PhysicsJoint(key=1, kind=2, target=WORLD_ENTITY, anchor=(0, 3, 0), axis=(0, 0, 1), lower=(-0.5,) * 6, upper=(0.5,) * 6))
# Construct this world's adapter with joint_type=PhysicsJoint.
```

Kind codes are fixed=0, ball=1, hinge=2, slider=3, distance=4 and six-DoF=5.
Anchors and axes are world-space when the constraint is applied. Limits and motor
columns have six entries: hinge/slider/distance use slot zero; six-DoF uses
translation xyz then rotation xyz. Limits default to free `(-inf, inf)` axes;
`(0, 0)` locks an axis. `limit_spring` is frequency/damping; friction and
break-force/torque thresholds use native units, with zero disabling breakage.

Motor modes are off=0, velocity=1 and position=2. `motor_target`, `motor_max`
and `motor_mode` contain six values; `motor_frequency`/`motor_damping` are shared
spring settings per joint. Changing mode/target/max or `enabled` uses native
setter batches. `set_joint_motor(entities, keys, mode=..., target=..., max_force=...,
axis=0)` sends an explicit batch against current keyed joints; it does not edit
their declarative component columns. Keys can broadcast from one scalar.

Topology, anchors, axes, limits, springs and break thresholds recreate constraints.
A body rebuild also recreates attached constraints against replacement UIDs;
their native angle/position reference is established again at creation. Removal
of either body removes the native joint. `joint_ids` maps `(owner, key)` to current
native IDs; snapshots expose native joint values/state through `world.latest().joints`.
A broken joint remains broken until its component is removed and reconciled, then
created again; it is not silently respawned each tick. Close removes constraints
and bodies and applies pending removals before joining the native thread.

Run [physics_joints_example.py](../scenes/physics_joints_example.py) for hinge and
slider motors, limits and a breakable fixed attachment. Native regression tests
are in [test_ecs_physics_interactions.py](../tests/test_ecs_physics_interactions.py).

## Native ragdolls and skeletons

`graaal.PhysicsRagdoll.RagdollAsset` is a CPU helper independent of ECS and
rendering. Construction loads the optional backend; module import does not.
It binds a render `Skeleton` to authored `pj.RagdollSettings`:

```python
from graaal.PhysicsRagdoll import RagdollAsset

asset = RagdollAsset(rig, authored_settings, bones=physical_render_indices)
# Or generate capsule shapes and generic limits as an authoring fallback:
fallback = RagdollAsset(rig, bones=physical_render_indices, radius=0.05, mass=70)
```

`bones` is one distinct render index per native bone, in native order; omitted
means every render bone. Physical bones must form one parent-before-child tree.
Their nearest mapped ancestors must match native parents, and native rest joint
frames must coincide with mapped render joints. Animation-only intermediate
bones and tips are supported. Native settings retain per-bone shapes, shape
offsets/rotations, mass, joint axes/limits, strength and friction. Treat settings
and mappings as immutable after construction; replace the asset key to rebuild.
The fallback's generic limits require authoring for a particular character.

Skeleton reference poses, animation poses and entity transforms require unit
scale. Bake scale into skeleton translations, inverse binds and physical shapes.
Non-unit scale is rejected rather than producing inconsistent joint frames.
Position units are metres and rotations are xyzw. Shape offsets already belong
to the native body setup; snapshot ragdoll joint poses must not offset them again.
Base animation rotations are normalized consistently with the skeletal evaluator,
including animation-only parents of physical bones.

Register `EcsPhysics.PhysicsRagdoll` and pass an asset mapping through `ragdolls=`:

```python
from graaal.EcsPhysics import PhysicsRagdoll

ecs.register(PhysicsRagdoll)
ecs.add(character, PhysicsRagdoll(asset=0, mode=0), MaterialTransform())
physics = EcsPhysicsAdapter(ecs, shapes, ragdolls={0: asset}, interpolate=True)
# Each fixed tick, after reconcile and local animation/IK evaluation:
physics.set_ragdoll_pose([character], local_pose)  # batched (T, R, S)
# Then prepare_inputs(), step(), pull(), dispatch_events().
matrices = physics.ragdoll_skin([character])
```

`mode` codes are kinematic=0, powered=1, limp=2. `blend` is the native transition
duration in seconds; `strength` multiplies the asset's per-bone torque limits.
Changing the asset rebuilds the native ragdoll; loss of its component, transform
or full entity handle removes it. Ragdoll owners cannot also declare a `Collider`
or `RigidBody`. Register those body component types as usual even in a ragdoll-only
world. `physics.ragdolls` is a read-only owner-to-native-handle mapping. Use
`physics.ragdolls[int(character)].hit(bone, impulse, point=None)` for an impulse
and native `set_strength` for individual bone strengths. Declarative strength
changes overwrite native strengths. Never remove individual ragdoll bones or
remove an adapter-owned ragdoll directly; ECS reconciliation and close use native
`remove_ragdolls` to free the complete structure.

`set_ragdoll_pose` copies local animation targets in owner order, including
animation-only bones. `prepare_inputs` converts them to native-parent joint frames
and queues each held target with the entity's current world frame. The ECS frame
stays caller-controlled in all modes; physical root displacement lives in the
skeletal pose. Apply root-motion requests to that frame explicitly. Do not run a
second native `play` timeline alongside the ECS animation clock.

`ragdoll_skin(owners, weight=1, alpha=None)` returns detached float32 skin matrices
in owner order, with repeated owners supported; all requested owners must share
one asset. Native joint poses are converted from world to entity space, then to
render-local TRS. Animation-only bones retain their local target and follow the
corrected parent. `weight` blends the resulting local pose with animation, as a
scalar or per-owner array. Native mode `blend` controls physical motors/transitions
and is separate from this render weight. With `interpolate=True`, `pull` retains
owned per-bone history; `alpha` interpolates positions and shortest-path rotations
before conversion and skinning. Repeated pulls preserve history, skipped ticks
snap, and rebuild/removal discards old history. Animation-only bones use the
current held animation target. Sampling never changes gameplay state or targets.

Wrap the final matrices in one `EcsAnimation.PoseBatch` per owner and feed the
existing `PosePalette`; omit the ordinary animation finalizer for those owners.
The standalone asset also exposes `target_pose`, `local_pose` and `skin` for
applications without ECS. Render mesh bounds must cover the deformed pose.

`entities_of` resolves both regular bodies and ragdoll bones; `bone_indices_of`
returns native physical indices or -1. Query `exclude=[character]` excludes all
its physical bones. `body_uids`/`body_owners` remain regular one-body ownership
maps. Removal events retain bone identity for a surviving owner until dispatch;
deleted owners resolve to `WORLD_ENTITY` and bone -1.

See [ragdoll_example.py](../scenes/ragdoll_example.py) for the prefab humanoid,
ECS animation clock, powered push, limp fall, blended recovery and the existing
GPU skin palette. Its skinned boxes visualize the authored capsule volumes.
Native frame, limit, mode, ownership and interpolation regressions are in
[test_physics_ragdoll.py](../tests/test_physics_ragdoll.py).

## Render interpolation

Construct the adapter with `interpolate=True` to retain two ticks of owned
pose copies. After every fixed tick, call `pull` even when no body changed:
sleeping bodies still need their interpolation interval advanced. No native
snapshot views are retained by the adapter. Interpolation is off by default
for headless applications that only need the latest gameplay state.

`physics.render_transforms(entities, alpha=...)` returns position, rotation and
scale arrays in the requested full-handle order, including duplicate owners.
`alpha` must be finite and within `[0, 1]`: zero selects the previous tick and
one selects the current tick. Positions use linear interpolation and rotations
use shortest-path spherical interpolation. This introduces the usual one-tick
presentation delay. Gameplay, ECS transforms and native queries remain at the
actual latest tick. Non-physics and static owners use their ECS transforms;
kinematic render interpolation never writes into controller targets.

For an existing renderer and shape mapping:

```python
from graaal.EcsMaterials import EcsMaterialAdapter

physics = EcsPhysicsAdapter(ecs, shapes, interpolate=True)
alpha = 0.0
drawing = EcsMaterialAdapter(renderer, ecs, transform_source=lambda owners: physics.render_transforms(owners, alpha=alpha))
# Each frame: advance the fixed schedule, then set alpha = accumulator / fixed_dt.
# The caller owns this accumulator; Scheduler has no interpolation clock.
drawing.sync()
```

Use the complete accumulator loop in [physics_example.py](../scenes/physics_example.py).
The material adapter samples this source every sync, even on frames with zero
physics ticks, and preserves the renderer's instance order. Material packing
still follows its normal change versions. Other renderers can consume the TRS
arrays directly; the physics adapter imports no rendering tool.

Teleports reset only the affected bodies' history to the applied native pose.
New/rebuilt UIDs start from their creation pose; history follows UIDs through
native row compaction and full entity handles through slot reuse. If native
time skips multiple ticks between pulls, interpolation snaps to the new pose
because intervening tick poses are unavailable. Returned arrays are detached;
release them when no longer needed. Close releases retained pose history.

Run `python -m scenes.benchmarks.benchmark_physics --bodies 100 1000` to measure
each fixed callback, render-pose preparation, and UID/gather/conversion/scatter
costs. This headless benchmark uses isolated moving boxes and reports CPU wall
times; it does not measure GPU frame rate or collision-heavy workloads.

At body capacity, the pinned backend needs removals applied before additions
can reserve space. Reconciliation validates body and ragdoll declarations against
their shared capacity before removing either type, allowing replacements in both
directions within one call. It uses `flush()` when outgoing resources must free
space for additions, without advancing time; native queries must not run
concurrently with reconciliation. Keep world
ownership and ECS callbacks on the scheduler thread. Use the context manager or
`close()` to remove owned bodies and join the native thread; close is idempotent.

Run [the ECS scene](../scenes/physics_example.py) with
`python main.py scenes/physics_example.py`, or headlessly with
`python -m scenes.physics_example --headless --frames 240`.
Native regression coverage lives in [test_ecs_physics.py](../tests/test_ecs_physics.py).
