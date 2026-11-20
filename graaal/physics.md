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
needed beyond its lifetime, and release snapshots promptly. Drain native events
after stepping; event UIDs are native body IDs and can include removed bodies.
Entity event routing, joint components, and render interpolation are future
phases. Rendering currently consumes the latest fixed-tick ECS pose.

At body capacity, the pinned backend needs removals applied before additions
can reserve space. Reconciliation uses `flush()` for that case without advancing
time; native queries must not run concurrently with reconciliation. Keep world
ownership and ECS callbacks on the scheduler thread. Use the context manager or
`close()` to remove owned bodies and join the native thread; close is idempotent.

Run [the ECS scene](../scenes/physics_example.py) with
`python main.py scenes/physics_example.py`, or headlessly with
`python -m scenes.physics_example --headless --frames 240`.
Native regression coverage lives in [test_ecs_physics.py](../tests/test_ecs_physics.py).
