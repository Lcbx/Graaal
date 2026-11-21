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
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody

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
is `MaterialTransform`, defined in `graaal.EcsAdapters.MaterialTransform`; importing it
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
ID in `a` for character events. Adapter-owned characters resolve `entity_a`;
characters created directly on the native world remain unresolved, while their
contacted body can still resolve `entity_b`.
Ragdoll body records also resolve `bone_a`/`bone_b` to native physical bone
indices; other or unresolved endpoints use -1. The asset's `bones` array maps
these indices to render bones. Character and joint namespaces are resolved
separately from body pairs.

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
from graaal.EcsAdapters.EcsPhysics import PhysicsJoint, WORLD_ENTITY

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
from graaal.EcsAdapters.EcsPhysics import PhysicsRagdoll

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
animation-only bones, and converts compatible owners together to native-parent
joint frames. Validation completes for the entire submission before any held
pose changes; repeated owners use their last input row. Converted targets stay
owned by the adapter until replaced or the ragdoll is rebuilt/removed.
`prepare_inputs` queues each held target with the entity's current world frame,
so frame changes take effect even when no new local pose is supplied. The ECS frame
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

### Passive neutral joint springs

`graaal.PhysicsRagdoll.joint_spring_torques` is a standalone NumPy evaluator;
it does not load the native backend or depend on ECS. Give it parent-before-child
physical bone indices, world orientations `(instances, bones, 4)`, world angular
velocities `(instances, bones, 3)` and neutral parent-local xyzw rotations.
Neutral rotations broadcast to the orientation shape. Required `stiffness`,
`damping` and `max_torque` broadcast to `(instances, bones)` and must be finite
and nonnegative. Their units are N m/rad, N m s/rad and N m respectively.

The result is detached float64 world torque `(instances, bones, 3)`. Each joint
uses shortest-rotation error and relative angular damping, caps its torque
magnitude, and applies an equal opposite reaction to its parent. The root has
no world spring; its input gains/cap are unused. A parent's combined reactions
can exceed one joint's cap. Rotations are normalized; inputs remain unchanged.

For one native ragdoll, apply the result before each fixed step:

```python
from graaal.PhysicsRagdoll import joint_spring_torques

state = world.bodies_state(handle.bones)
torques = joint_spring_torques(handle.settings.parents, state.orientation[None], state.angular_velocity[None], neutral_rotations, stiffness=1, damping=0.05, max_torque=0.5)
world.add_torques(handle.bones, torques[0])
world.run_ticks(1)
```

`neutral_rotations` is an authored physical parent-local pose, such as
`handle.settings.rest_rotations`; render-local poses with unmapped intermediate
bones must first pass through `RagdollAsset.target_pose`. Native torque submission
accepts all ragdolls' flattened bone UIDs in one batch. The helper does not read
motor mode/strength: neutral springs can remain active while native motors are
disabled, including powered mode with zero strength. Author gains for the body's
inertia and fixed timestep; explicit torque control can oscillate with large
gains. Native hard constraints still apply. Add the soft-limit forces below when
needed; balance and control-state resets are separate work.

The [control benchmark](../scenes/benchmarks/README.md#ragdoll-control-prerequisites)
and [tests](../tests/test_physics_joint_springs.py) cover torque reactions,
motor independence, per-bone strengths and native observation costs.

### Passive soft angular limits

`ragdoll_limit_parameters(settings)` prepares detached, read-only arrays from
authored native `RagdollSettings`: parent indices, bone-local constraint frames,
twist intervals and swing half angles. It resolves implicit axes, orthogonalizes
the plane axis, and includes rest rotations and cone `bend`. Prepare it once per
immutable asset; rebuild the parameters after changing that asset. Preparation
uses Python metadata and does not import the native backend.

`joint_limit_torques` consumes the same batched world orientations/angular
velocities as the neutral spring and returns detached float64 world torques.
Low-level callers can supply `parents`, bone-local `parent_frames`/`child_frames`
xyzw arrays, `twist=(min, max)` in `[-pi, pi]`, and `swing=(first, second)` half
angles in `[0, pi]` directly. Frames and limits broadcast to `(instances, bones,
4/2)`. Constraint X is the twist axis, Y the plane axis, Z their cross product.
Swing uses the [Jolt quaternion ellipse](https://jrouwe.github.io/JoltPhysicsDocs/5.6.0/class_swing_twist_constraint_part.html),
including locked axes and unrestricted angles.

Required positive `margin` insets each angular bound to define a comfortable
range. Narrow intervals collapse to their midpoint; narrow swing axes collapse
to zero. Free axes remain free across the quaternion wrap. Outside the comfortable
range, the evaluator radially projects swing onto the inset ellipse and clamps
twist, then uses rotation error toward that feasible pose. For error magnitude
`d`, spring magnitude is `stiffness * d**2 / margin`. Outward-only damping ramps
from zero to `damping` across the margin, allowing inward recovery. Gains and
`max_torque` broadcast to `(instances, bones)`, with the neutral spring's units.
Each soft joint is capped before equal opposite parent reactions accumulate.

```python
from graaal.PhysicsRagdoll import joint_limit_torques, ragdoll_limit_parameters

limit_parameters = ragdoll_limit_parameters(handle.settings)  # Once per asset.
# Each fixed tick, before stepping:
state = world.bodies_state(handle.bones)
soft = joint_limit_torques(orientations=state.orientation[None], angular_velocity=state.angular_velocity[None], margin=0.1, stiffness=1, damping=0.05, max_torque=0.5, **limit_parameters)
world.add_torques(handle.bones, soft[0])
```

Neutral and limit torques may be added before one native torque submission;
their separate caps do not cap the sum. Both remain independent of active motor
strength. These are Python/NumPy forces applied through the existing binding;
Jolt continues to solve hard constraints and contacts. A soft band is not a hard
clamp or a stability guarantee for arbitrary gains, timesteps or inertias.
Native pressure tests cover twist, swing and bent joints with non-identity rest
frames; full humanoid anatomical tuning and fall/impact validation remain pending.

## Character controllers

Register `PhysicsCharacter` and `CharacterState` alongside the usual body and
transform types, and pass immutable native capability assets through `characters=`:

```python
from graaal.EcsAdapters.EcsPhysics import CharacterState, PhysicsCharacter

ecs.register(PhysicsCharacter, CharacterState)
character = ecs.create(1)[0]
ecs.add(character, PhysicsCharacter(capabilities=0), CharacterState(), MaterialTransform())
caps = pj.CharacterCapabilities(stepHeight=0.35, maxSlope=50, jumpHeight=1.2)
physics = EcsPhysicsAdapter(ecs, shapes, characters={0: caps}, interpolate=True)
physics.reconcile()
physics.move_characters([character], [1, 0, 0], speed=2)
physics.prepare_inputs()
physics.step()
physics.pull()
physics.dispatch_events()
print(ecs.get(CharacterState)[[character]].grounded)
```

The component's numeric `capabilities` key selects native settings, including
capsule dimensions, walk/run speeds, acceleration, step height, slope limit,
jump height, air control and pushing behavior. Native capability units are
metres, seconds, kilograms and **degrees** for slope angles. The adapter validates
capsule dimensions; the remaining settings retain their native meaning. Unit
entity scale is required. `gravity` is downward acceleration in m/s², with NaN
selecting the length of world gravity. Character up is world +Y. `group`, `mask`
and `pass_group` retain native collision semantics; a shared nonzero pass group
lets characters pass through each other.

The ECS transform is at the **feet**, suitable for a skinned character's placement.
Native creation, `Character.position`, and native TELEPORT operations use the
capsule **center**. The adapter adds half the standing height on creation and
subtracts half the current snapshot height on readback, including when crouched.
Jolt owns positional motion; editing the ECS mirror does not teleport the
controller. Rotation stays caller-controlled for facing/rendering and does not
rotate the native upright capsule.

`pull` mirrors feet position and, for owners with a `CharacterState` instance,
velocity, grounded/state codes, current height, ground normal, stuck flag and
full ground entity handle. Missing or unowned ground resolves to `WORLD_ENTITY`.
Register the output component type when enabling characters; attaching an
instance is optional. These fields are physics outputs, not movement inputs.
Removing/disabling a controller clears a surviving output instance. Native
`world.latest().characters` exposes additional landing and eye-position data.

`move_characters(owners, directions, speed=None, mode=0)` queues held world-space
input. Native walking flattens direction onto the ground; zero direction stops.
Modes are walk=0, run=1, sprint=2, crouch=3. Omitting speed uses the capability's
mode speed. Directions and speeds may broadcast or supply one row per owner.
`jump_characters(owners)` queues native grounded/coyote/buffered jumping.
Inputs are copied, resolve full handles during `prepare_inputs`, and apply at the
next native tick. Removed, disabled or deleted owners' queued inputs are dropped.
Reconcile active owners before sending inputs.

`character_commands(owners, op, code=0, values=(0, 0, 0, 0))` provides the native
`pj.CharacterOp` batch interface for crouching, slides, gravity, knockback,
swimming, flight and teleports. Ops/codes/values broadcast or align with owners;
the native command order is preserved. MOVE uses xyz direction plus speed;
only its speed can be NaN to select a native default. TELEPORT uses xyz **center**
plus downward search distance. For example, on the default standing capsule:

```python
physics.character_commands([character], pj.CharacterOp.CROUCH, code=True)
physics.character_commands([character], pj.CharacterOp.TELEPORT, values=(5, 0.9, 0, 4))
```

`physics.characters` exposes read-only owner-to-native-handle membership. Do not
remove these handles or their proxies directly; change ECS declarations and
reconcile. Changing capabilities, gravity, groups, masks or proxy settings
rebuilds the controller while retaining its latest native feet and velocity.
Rebuilds reset held input and native action state. `enabled=False`, removal of
the component/transform, and deletion remove it. Active characters cannot share
an owner with a rigid body, collider or ragdoll. A disabled declaration can remain
on a ragdoll owner for a later hand-off.

Every character reserves one slot from shared `max_bodies`, including with
`proxy=False`. The optional proxy appears in queries and trigger records under
the character's native ID; it is not a regular rigid-body snapshot row. The
controller still collides with terrain without a proxy. `entities_of` resolves
owned character/proxy IDs, and `exclude=[owner]` excludes the proxy in queries.
LAND/JUMP and other character records use the separate character namespace;
trigger pairs use the proxy's body identity. Removed/deleted handles follow the
same retirement and generation rules as other owned resources.

With `interpolate=True`, `pull` retains owned feet arrays aligned by native ID;
`render_transforms` samples them using the caller's alpha. Rebuilds and queued
TELEPORT commands reset history; skipped ticks snap and repeated pulls preserve
the interval. Queue teleports through `character_commands` to reset this history.

### Animation and ragdoll hand-off

The [character example](../scenes/character_example.py) compiles idle/walk states
and an in-place pose plan using `AnimationGraphDefinition`. The existing ECS
animation clock supplies `RootMotionBatch` requests; their horizontal distance
over the fixed interval becomes requested controller speed. Gameplay placement
comes from accepted native motion, so a wall can reduce actual velocity to zero
while animation still requests movement. Post-physics batched rays, excluding
the character, drive foot IK only on grounded, valid upward-facing hits. The
example publishes one final skin palette per owner. Facing follows horizontal
movement, with foot queries and IK goals converted through the entity's yaw.
The scripted route jumps a 0.75 m hurdle on approach; this scene uses full native
air control to maintain forward speed through the jump.

Hand-off policy belongs to gameplay. The example captures controller feet and
velocity, disables it, creates a kinematic ragdoll with the held local pose, then
holds that pose for two fixed ticks to settle it before switching to limp and
applying inherited bone velocities. The native API spawns in its asset rest pose;
setting a limp pose target alone does not place its bones in an animated pose.
No controller collides while the ragdoll is active. Recovery projects below the
physical root, checks standing capsule clearance, removes the whole ragdoll,
and recreates the controller at those feet with root-body velocity. It suppresses
new locomotion input for that transfer tick to avoid adding it to inherited
velocity. If standing space is blocked, it remains a ragdoll until recovery is
requested again. This is a standing recovery placement, not continuous active
ragdoll locomotion; the separate active-ragdoll plan covers that control problem.

Run `python -m scenes.character_example --headless --frames 960` for the full
scripted cycle, or use WASD/Space/R/G in the rendered scene. Native regression
coverage is in [test_ecs_characters.py](../tests/test_ecs_characters.py) and
[test_character_example.py](../tests/test_character_example.py).

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
from graaal.EcsAdapters.EcsMaterials import EcsMaterialAdapter

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

## Diagnostics and performance

Collect native statistics explicitly on the world's owning thread:

```python
from graaal.Diagnostics import inspect_physics

report = {"physics": inspect_physics(physics.world)}
print(report["physics"]["stats"])
```

Standalone applications pass their native `pj.World` directly. Inspection copies
the statistics without stepping, flushing commands, draining events, retaining
snapshots or resetting counters. The result contains `tick`, `tick_hz`, `lockstep`
and `stats`, preserving every native statistic name and unit. Native timing
means (`tick_ms`, `step_ms`, `publish_ms`, `events_ms`, etc.) cover ticks since
`world.reset_stats()`; they are not per-render-frame measurements. Body counts,
`events_dropped`, snapshot `pool`, and query count/wait statistics remain visible.

Feed this report to `DiagnosticsVisualizer.update(report)` for a physics-only
panel, or attach `report["physics"]` to a `RenderMonitor` submission report before
updating the widget. The panel shows every native counter and a separate bounded
history of reported mean tick costs. Collection and history updates work without
the UI extra; drawing requires it. Diagnostics do not add callbacks to physics.

Run `python -m scenes.benchmarks.benchmark_physics --bodies 100 1000` to measure
each fixed callback (including event routing), render-pose preparation, and
UID/gather/conversion/scatter costs. The default workload uses isolated moving
boxes. Add `--workloads colliding compounds ragdolls --ragdolls 10 20 --threads 1 2 4`
for gravity-driven stacks, two-shape compounds and passive native humanoids.
Ragdoll final skinning is timed separately. Native timing means exclude warmup;
contact activity and dropped-event counts help validate the measured workload.
See [benchmark contracts and commands](../scenes/benchmarks/README.md#physics-synchronization).
All timings are headless CPU measurements and do not establish GPU frame rate.

## Checkpoints and selective resets

Regular body/joint worlds can capture a same-adapter, in-memory checkpoint after
a complete fixed tick (`pull` and `dispatch_events` included):

```python
checkpoint = physics.save_state()
# Advance fixed ticks, using the same bodies and joint configuration.
checkpoint_tick = physics.restore_state(checkpoint)
```

The opaque checkpoint owns native state bytes, full entity handles, ECS transform
targets and copies of queued adapter body inputs. Restore applies native state
with `flush()` without advancing time, restores targets and pending inputs, mirrors
dynamic poses, and snaps render interpolation to the restored poses. The checkpoint
is reusable. Body UIDs/order, full entity generations, collider geometry, body
settings and joint configuration must remain unchanged. Invalid ownership or
configuration is rejected before native restore; damaged native bytes are rejected
by the backend. Foreign adapters and unreconciled declarations are rejected.

Both operations require lockstep, applied native submissions and dispatched events.
Python body inputs queued through the adapter may remain pending and are captured;
native command queues are not captured. Keep native tracks, scheduled callbacks,
future native commands, world settings, material/filter tables and direct native
configuration changes out of checkpoint replay. Adapter motor/filter changes
invalidate earlier checkpoints. Checkpoints are not portable save-game files and
do not capture the rest of the ECS or application command buffers.

`restore_state` returns the saved native tick as a marker; the native tick/sequence
counters remain monotonic. Capture and restore caller-owned animation/gameplay
clocks, accumulator, random state and other control state separately. Restoration
clears the exposed event batch without invoking subscribers; event delivery and
gameplay side effects are not rewound. This is physical-state replay, not a
deterministic replay of application events.

The pinned backend omits character state and native ragdoll modes/strengths from
its state bytes. Worlds containing ragdolls, characters, vehicles, soft bodies or
unowned native resources are therefore rejected by this checkpoint API. Pose
readback alone cannot recover character jump/slide timers, held controls or
ragdoll transitions. Full control-state replay remains unsupported.

For selected regular bodies, `reset_bodies` replaces pose and velocity inputs:

```python
physics.reset_bodies([box], positions=(0, 3, 0))
# Next prepare_inputs()/step()/pull() applies the reset.
```

Optional `orientations` defaults to identity; `linear` and `angular` default to
zero. Numeric arrays broadcast or align with distinct live, reconciled full
handles. Resetting discards earlier queued adapter inputs for those owners while
preserving other owners' queued inputs and interpolation history. Only selected
history snaps when the reset applies. Static and kinematic ECS targets update too.
Validation completes before queues change. Apply native submissions before queuing
a reset; no tick is advanced by the call. Contacts and constraints can still affect
other bodies when simulation resumes; constraint state and training episodes are
not reset. Characters and ragdolls are not accepted as regular body reset owners.

See [test_physics_replay.py](../tests/test_physics_replay.py) for contact/constraint
trajectories, queued input replay, generation guards and selective history resets.

## Capacity and shutdown

At body capacity, the pinned backend needs removals applied before additions
can reserve space. Reconciliation validates body, ragdoll and character declarations against
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
