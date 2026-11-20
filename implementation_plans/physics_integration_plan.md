# PyJoltPhysics integration — Graaal implementation plan

## Goal

Add optional 3D physics to Graaal using **pyjoltphysics** for rigid bodies, colliders, triggers, joints, queries, characters, and native ragdolls. Preserve the existing NumPy ECS, fixed-update scheduler, standalone rendering tools, and skeletal animation pipeline. No custom collision solver or imitation ragdoll implementation.

**Scope:** Phased integration plan and implementation status. Use the existing [`implementation_plans/active_ragdolls.md`](implementation_plans/active_ragdolls.md) for the subsequent passive/active neural-control work.

## Implementation status

- Phase 0 implemented: optional `physics` extra pins `pyjoltphysics==0.1.0a3`; [`scenes/physics_headless_example.py`](../scenes/physics_headless_example.py) uses the native world directly, without ECS or graphics. The backend loads only when the scene runs; context-managed shutdown also covers exceptions.
- [`tests/test_physics_headless.py`](../tests/test_physics_headless.py) covers self-test, public API names, floor contacts, lockstep ticks, command sequences, read-only snapshot formats, xyzw orientation, UID lookup after removal, worker exception propagation, and optional dependency isolation. Native tests skip without the extra.
- Verified on Windows x64, CPython 3.14.10, NumPy 2.5.5, pyjoltphysics 0.1.0a3 (MSVC, SSE4.2, double precision). Both 120 ticks at 120 Hz and 240 ticks at 60 Hz with four threads completed. Keep 120 Hz configurable; engine workload suitability needs Phase 2 measurements. Other platforms and Python/NumPy versions remain unverified.
- Validation after Phase 1: focused physics/import-boundary checks passed (26 tests); material adapter tests also passed; full suite passed (993 tests, 4 subtests) with native graphics access. The sandboxed full run was stopped because GPU tests were substantially slower. `python -m graaal`, the physics guide example, the 240-tick ECS headless scene, two offscreen rendered frames, and `git diff --check` passed. Backend checks ran against the installed pinned release.
- Phase 1 implemented: [`EcsPhysics.py`](../graaal/EcsPhysics.py) adds declarative rigid bodies/multiple colliders, cached shape assets/compounds, full-handle UID ownership, batched lifecycle/setters/commands, dynamic native authority, kinematic targets, and explicit static teleports. Shape/scale changes preserve dynamic native state on rebuild. Mixed solid/sensor owners and non-unit local scales on multi-colliders are explicitly rejected in v1.
- Fixed callbacks (`reconcile`, `prepare_inputs`, `step`, `pull`) provide the minimum usable lifecycle loop without changing the scheduler. `MaterialTransform` lives in its own CPU-only module so physics does not import the material adapter or shader dependencies. `EcsMaterials` re-exports the same class for compatibility and imports `MaterialBinding` at module scope. Consumers and guides use the standalone component module. Native setters support material/filter indices and mutable flags; group/mask and creation-only flag changes rebuild.
- The pinned backend reserves swap additions before releasing removals, so rebuilds at capacity use removal plus `flush()` before adding replacements. This applies commands without advancing simulation time; native queries must not run concurrently with reconciliation. Rebuild and pending-creation capacity cases have native tests.
- [`scenes/physics_example.py`](../scenes/physics_example.py) demonstrates falling cubes, obstacles, a moving platform and sensor through the existing material adapter; its headless run and two offscreen rendered frames passed. [`physics.md`](../graaal/physics.md) documents the supported contract and limitations; [`tests/test_ecs_physics.py`](../tests/test_ecs_physics.py) covers lifecycle and authority.
- The physics scene tracks box occupancy from native trigger transitions each fixed tick and updates its sensor material from green toward red as the count increases. Exits restore the tint, non-box bodies are excluded, and removed box UIDs are pruned. Manual native entry/exit/deletion checks and an offscreen scene smoke check cover this scene behavior.
- Phase 2 implemented: `EcsPhysicsAdapter(..., interpolate=True)` keeps owned copies of two ticks' poses and returns render-only TRS through `render_transforms(owners, alpha=...)`. UID-aligned history survives native row compaction, entity generation reuse, and repeated same-tick pulls; teleports reset affected history and missing intermediate ticks snap to current state. Gameplay and kinematic targets remain authoritative and unmodified by render sampling. Interpolation is opt-in so headless users avoid its copies.
- `EcsMaterialAdapter` accepts a generic transform source in material-owner order, including repeated owners, and refreshes it between fixed ticks while material packing retains version caching. This keeps the standalone renderer independent of physics and supplies smooth presentation in the existing caller-owned accumulator loop without changing the scheduler.
- [`benchmark_physics.py`](../scenes/benchmarks/benchmark_physics.py) measures 100/1,000 isolated moving boxes with interpolation off/on, fixed callback costs and separate UID/gather/float conversion/scatter probes. A Windows x64 / Python 3.14.10 / NumPy 2.5.5 / pinned backend run with 30 warmup and 240 measured ticks found median 0.55 ms per fixed tick and 0.26 ms render-pose preparation at 1,000 bodies with interpolation enabled; conversion alone was about 0.001 ms. No direct snapshot-to-GPU optimization is justified by this workload. This does not measure GPU FPS or collision-heavy/ragdoll performance; see the benchmark guide for reproduction.
- Phase 2 validation: 34 physics/render-source/import-boundary tests and 22 affected material rendering/boundary tests passed. The full suite passed (1,007 tests plus 4 subtests) with native graphics access. Six offscreen scene frames, the 240-tick headless scene, CPU smoke, and `git diff --check` passed. The example clamps accumulator subtraction at zero to avoid a negative render fraction from floating-point roundoff.
- Phase 3 implemented: once-per-tick routed NumPy events and synchronous subscriptions, full-handle removal routing with deleted-owner suppression, dropped-event diagnostics, native-preserving ray/batch/sweep/overlap APIs, entity exclusions, and native filter/material tables. Character records preserve their own namespace; only their owned body counterpart resolves until the later character adapter exists.
- Optional `PhysicsJoint` reconciliation uses stable per-owner keys and numeric declarations for fixed/ball/hinge/slider/distance/six-DoF constraints. Creation waits for both bodies, changes batch native motor/enabled setters or rebuild constraints as needed, and body replacement/deletion updates constraint lifetimes. Broken joints retain their native state instead of respawning.
- The physics scene consumes routed events. `physics_queries_example.py` replaces analytic foot-IK terrain with batched rays over collision geometry; misses disable correction. `physics_joints_example.py` demonstrates motor limits, reversal and break events. The original analytic scene remains independent of physics.
- Phase 3 focused validation: 60 interaction/lifecycle/foot-IK/import-boundary tests passed; headless physics/query/joint examples and six offscreen scene frames passed. Full-suite validation is pending.
- Next: Phase 4 native ragdoll assets, ECS lifecycle and skeletal pose integration. Native advanced APIs remain directly accessible through `physics.world`.
- Import graph cleanup: inline imports are confined to constructors, asset loaders, explicit invariant audits and cached compatibility exports. Physics construction stores its backend module for subsequent updates; component schemas import `ecs_types` without loading the ECS facade. Mesh pools and arenas capture dependencies at construction to break the eager RenderContext cycle. CPU `Diagnostics` lazily re-exports rendering APIs from `RenderDiagnostics`, and `MaterialRenderer` lazily preserves ECS compatibility exports so direct rendering stays independent of ECS. Static inspection found no eager cycles across 58 engine modules; boundary tests enforce cold-path imports and prevent backend loading by CPU imports.
- Validation after import graph cleanup: focused runs passed (121 and 156 tests); CPU/offscreen smoke tests and `git diff --check` passed. The full suite reported 996 passed plus 4 subtests and one statistical transparency GPU failure (`test_band_totals_payload_marginals_and_denominator[0.8-1]`, deviation 0.00535 against 0.005 tolerance). All 30 tests in that GPU module passed on rerun; shader behavior and tolerances were left unchanged.

- Removed nine inline-import exception wrappers that only rewrote dependency errors in asset loading, geometry creation, physics/UI initialization, memory monitoring and the standalone physics scene. Imports remain confined to cold paths; genuine module-level optional-dependency fallbacks remain. Missing-backend tests now check `ModuleNotFoundError.name`, preserving native error details. Validation: all 117 affected subsystem/boundary tests passed, including native physics, ImGui and rendering diagnostics; `git diff --check` passed.

## Current integration points

- `graaal/ecs_schedule.py`: `Scheduler` already has `fixed_update`, `update`, and `render_prep`. The caller owns the accumulator and supplies `fixed_steps`/`fixed_dt`; commands flush at fixed-tick boundaries or after `flush_after=True` systems.
- `graaal/ECS.py`: component stores have field/membership versions, `@component(multiples=True)`, NumPy batch access and `world.query(...)`. ECS entity IDs are generational `uint64` handles; they are not Jolt body IDs.
- `graaal/MaterialTransform.py`: shared CPU-only `MaterialTransform` (`position`, `rotation`, `scale`). `graaal/EcsMaterials.py` supplies `EcsMaterialAdapter` for rendering. `scenes/draw_batches_example.py` defines a different local `Transform`; do **not** assume one engine-wide transform component currently exists.
- `graaal/EcsAnimation.py`: `EcsAnimationAdapter.advance` builds `sample_batches` and `root_motion_batches`, whereas `finalize` produces skin matrices. `graaal/PoseTasks.py` can produce **local translation/rotation/scale**, which a ragdoll motor needs; the final skin matrices alone are not enough.
- `graaal/SkeletalAnimation.py`, `graaal/PosePalette.py`: skeletons have parent-before-child local transforms and inverse-bind matrices; rendering consumes skinning matrices in entity order.
- `implementation_plans/active_ragdolls.md`: broader goals (passive soft limits, neutral-posture forces, active balancing, learning) already exist and should be layered **on top** of the general physics integration.

## Architecture and invariants

1. **Keep the physics backend independent of ECS.** Expose a `pyjoltphysics.World` directly to non-ECS applications. Put the optional ECS bridge in `graaal/EcsPhysics.py`, analogous to existing `EcsMaterialAdapter` and `EcsAnimationAdapter`. Introduce a separate physics helper only where shape construction or pose conversion warrants it.
2. **Start in lockstep.** Use `WorldSettings(tick_hz=120)` (configurable) with `world.run_ticks(1)` from `fixed_update`. The native world thread executes the step, while the synchronous call gives the scheduler one precisely delimited physics tick. Do not call `world.start()` during ordinary ECS simulation; free-running is an optional later mode for independent simulations.
3. **One writer per transform direction.** Dynamic: Jolt owns position/orientation/velocity; ECS stores a mirrored rendering/gameplay pose. Kinematic: ECS/controllers own the target and send it to Jolt. Static: ECS pose initializes the body; later changes are explicit moves. Never automatically send a dynamic body's ECS mirror back into Jolt (feedback/teleport loop).
4. **Keep separate IDs and lifetimes.** `body_uid (uint32) <-> full ECS entity handle (uint64)` and, for ragdolls, `body_uid -> (entity, bone_index)` live in the adapter. An ECS entity can own several Jolt bodies (e.g., ragdoll bones, independently moving trigger). Never rely on snapshot row indices surviving a tick or on an ECS entity index without its generation.
5. **Batched I/O, not shared ownership.** pyjoltphysics snapshots are read-only NumPy views; they are not writable ECS storage. Convert changed body positions from Jolt `float64` to the ECS transform format (`float32` in current rendering paths), and copy/scatter only affected rows. Hold no snapshot views indefinitely; references prevent native buffer reuse.
6. **No import-time physics requirement.** Add an optional dependency group in `pyproject.toml`, initially pinning the tested alpha release (`pyjoltphysics==0.1.0a3`), and load the backend only during adapter construction, retaining its module reference for updates. Existing graphics- and physics-free ECS uses must still import and run.

## Proposed ECS components

Components store **declarative data**, not native Python Jolt objects or body handles. Native resources, UID maps, allocation state and command sequences belong to `EcsPhysicsAdapter`.

| Component | Cardinality | Purpose / fields |
| --- | --- | --- |
| `RigidBody` | One per entity | `motion` (static/kinematic/dynamic), `mass` or density, friction, restitution, linear/angular damping, gravity factor, collision group/mask, flags (sleep/CCD/report), optional material/filter IDs. No `RigidBody` + a collider means a static body. |
| `Collider` | Multiple per entity | Shape resource/descriptor, local translation/rotation/scale, optional stable collider key, `sensor` flag. Multiple solid colliders build one compound shape when possible. Shared immutable `pj.Shape` objects are cached by a shape registry. |
| `PhysicsJoint` | Multiple per entity | Target entity, joint descriptor (fixed/ball/hinge/slider/six-DoF), anchors/axes, limits, motor, break thresholds. Created only after both referenced bodies exist. |
| `PhysicsRagdoll` | One per skinned owner | Ragdoll settings asset, skeleton-to-bone mapping, desired `KINEMATIC`/`POWERED`/`LIMP` mode, motor-strength/blend settings. |
| `PhysicsCharacter` | One per entity (later) | Character capability settings, collision layer and collision-proxy options; inputs are commands and position is read from the physics state. |

Use `MaterialTransform` as the default transform binding for material scenes; allow `transform_type=` for custom component schemas (e.g., `scenes/draw_batches_example.py`). The physics bridge only assumes `position`/`rotation`; shape/mesh scale is read at creation and applied to the collision shape. Changing a dynamic shape's scale requires rebuilding its Jolt body and restoring appropriate state rather than silently changing a display-only scale.

**Important shape limitations:**
- A Jolt **body** has one sensor flag and one collision group/mask. Compound children cannot independently mix a solid and trigger behavior under the API exposed here. Model mixed solids + triggers with separate attached sensor bodies, or reject this case in v1.
- Mesh and height-field shapes are static/kinematic, **not dynamic**. Use convex or primitive shapes for moving colliders; reject invalid configurations explicitly.
- Event records represent **body pairs**, not individual compound-child contact pairs. v1 collision events address the ECS entity (and optionally ragdoll bone); per-collider contact identity requires independent bodies or binding changes.
- Treat collider geometry as an explicit physics asset, separate from the rendered mesh. Start with primitive shapes, convex hulls, static meshes, height fields, compounds; do not assume the rendering mesh is a valid dynamic collision mesh.

## Scheduler integration

A representative order inside `fixed_update` is:

```python
schedule.add("animation_control", animation.advance, phase="fixed_update")
schedule.add("physics_sync", physics.reconcile, phase="fixed_update", after="animation_control")
schedule.add("physics_input", physics.prepare_inputs, phase="fixed_update", after="physics_sync")
schedule.add("physics_step", physics.step, phase="fixed_update", after="physics_input")
schedule.add("physics_output", physics.pull, phase="fixed_update", after="physics_step")
schedule.add("physics_events", physics.dispatch_events, phase="fixed_update", after="physics_output")
schedule.add("animation_pose", compose_final_pose, phase="fixed_update", after="physics_output")
schedule.add("physics_render", physics.render_prep, phase="render_prep")
```

- Register event handling before/after `animation_pose` depending on whether events affect the current animation choice; keep a documented choice (default: events affect the **next** physics tick).
- **Reconcile:** Compare ECS membership/field versions, create/remove/rebuild bodies and constraints, update static/kinematic inputs, and validate component combinations. If a prior scheduler callback queues entity/component changes, that callback must use `flush_after=True` for same-tick visibility. Ordering dependencies alone do **not** flush ECS commands.
- **Prepare inputs:** Apply batched forces/impulses/velocities/teleports, character movement, joint motors and ragdoll pose targets. Enqueued Jolt commands apply at the start of the next Jolt tick; keep track of returned `seq` and the snapshot's `applied_seq`.
- **Step:** `physics.world.run_ticks(1)` once for each scheduler fixed tick, with `fixed_dt == 1 / tick_hz` (reject mismatch). No second hidden time accumulator inside the adapter.
- **Pull:** Read the new `latest()` snapshot. Re-resolve `snapshot.rows_of(uids)` each tick, select changed/active owned bodies and update ECS dynamic transforms/velocities. Do not mistake physics-originated ECS writes for a request to teleport bodies back next tick.
- **Dispatch:** Drain Jolt events once per fixed tick and handle them on the ECS/scheduler thread, not in Jolt's worker-thread callback. Listener writes that change components structurally should use `CommandBuffer`, with explicit visibility timing.
- **Render prep:** Maintain previous/current physics snapshots and interpolate body positions/rotations with the frame's `alpha = accumulator / fixed_dt` for smooth rendering. Existing `Scheduler.run` does not supply `alpha`, so the scene or a small caller-owned render helper must provide it. Physical queries and gameplay always use the actual latest tick, not an interpolated pose.

For skeletal animation, start a fixed tick by evaluating **local** animation pose targets, apply root motion via the character or kinematic/dynamic control path, then step physics. After stepping, composite native ragdoll bone transforms and animation-only bones into one pose and call `skin_matrices`. Feed one final `PoseBatch` per owner to `PosePalette`; do not publish both an animation palette and a ragdoll palette for the same entity.

## Implementation phases

### Phase 0 — Backend compatibility and headless proof

- Add optional dependency/configuration and a smoke test creating a `pj.World`, adding a static floor and dynamic boxes, calling `run_ticks(120)`, reading snapshots and draining events.
- Verify native wheels and `pj.selftest()` on target development platforms (especially Windows), supported Python/NumPy versions, thread shutdown, and useful error propagation.
- Verify `BodyBatch`, `Shape`, `BodyFlag`, `MotionType`, `RagdollSettings`, `RagdollMode` and event enum names against the installed pinned version rather than treating pseudocode as an exact binding signature.
- Confirm 120 Hz is appropriate for the main engine and that there is no accidental free-running thread.

**Deliverable:** `scenes/physics_headless_example.py`; no ECS or graphics needed.

### Phase 1 — ECS rigid bodies and collider lifecycle

- Implement the component schemas, shape registry/cache, body UID ownership maps, and `EcsPhysicsAdapter` lifecycle (`reconcile`, `close`).
- Spawn/delete/rebuild by ECS membership changes. Create a static body for collider-only entities. Batch creations by compatible shape/motion/settings, then use Jolt UID and returned sequence numbers to track pending application.
- Implement the authority rules for static, dynamic and kinematic bodies; support optional transform adapters. Add explicit `teleport`, `set_velocity`, `add_impulse`, `add_force` methods for runtime commands, submitted in NumPy batches.
- Apply writable changes (friction, damping, flags, filters) through Jolt's setter batches. For shape or topology changes, rebuild the physical body, preserve velocity where meaningful, and update UID maps. Deletion must remove attached constraints/trigger proxies and suppress delivery to stale ECS entities.
- Keep a simple v1 rule for multi-colliders: solid sub-shapes combine; sensors remain separate bodies when required. State precisely what combinations are unsupported.

**Deliverables:** `graaal/EcsPhysics.py`, `tests/test_ecs_physics.py`, `scenes/physics_example.py` (falling cubes, floor, static obstacles, movable platform, sensor).

**Tests:** lifecycle including generational-handle reuse, multiple colliders, static/kinematic/dynamic authority, continuous high-speed collider, transform orientation (xyzw), batch create/remove, stale Jolt UID, cleanup and no dependency leakage.

### Phase 2 — Fixed scheduling, state reads and performance

- Expose scheduler callbacks for reconcile/inputs/step/pull and document the canonical ordering. Add a helper for registering those callbacks if repeated boilerplate warrants one; **do not modify** the core scheduler unless a missing contract is demonstrated.
- Leverage ECS membership/field versions to skip unchanged shape/settings work. Beware that ECS versions are per store/field, not row-level dirtiness: on changes, gather the relevant inputs in batches and cache comparisons where helpful; never loop over every entity via individual native calls.
- Integrate physics-to-ECS transform writes with `MaterialTransform` and the alternate `Transform` test scene. For dynamic bodies, use native `Snapshot.changed` plus UID lookup, recognizing that body-row indices change after removals.
- Keep snapshots for at most the immediately needed interpolation history and provide optional direct snapshot reads to gameplay systems to avoid copies when consumers do not require ECS-owned columns.

**Deliverable:** Reproducible 120 Hz lockstep scene using `Scheduler`, with smooth 60+ FPS presentation and measured sync cost.

### Phase 3 — Events, filters, spatial queries and constraints

- Call `world.drain_events()` after each step. Expose a NumPy event batch and optional Python subscriptions/listeners for `CONTACT`, `TRIGGER`, `STATE`, `CHARACTER` and `JOINT` events. Route native UIDs back to entity handles; for ragdoll bones, include bone index. Define lifecycle behavior for `END(reason=REMOVED)` and deleted owners.
- Default events are begin/end. Enable persist records only for bodies opting into `BodyFlag.REPORT`; track `events_dropped`. The reported impulse is an **estimate**, not the actual impulse subsequently applied by Jolt's solver.
- Expose `raycast`, `raycast_batch`, `sweep`, `overlap`, and filter construction with entity-based exclusion. Integrate batched ground queries with `foot_ik_example.py` so the analytic terrain query can be replaced by collision geometry.
- Implement a `PhysicsJoint` reconciler for fixed/hinge/ball/six-axis joints, motor commands, break events and safe removal. Expose group/mask and optional filter tables and material tables without inventing an overlapping collision-layer subsystem.

**Deliverables:** collision/trigger example, raycast and foot-IK example, joint-motor example and event tests (begin, persist, end, removed, sleeping, trigger, sensor overlap, broken joint).

### Phase 4 — Native ragdolls and skeletal rendering

- Implement a ragdoll **asset adapter** that maps a Graaal `Skeleton` and authored physical bone setup to `pj.RagdollSettings` (per-bone collider, joint axes, mass, limits, strength, friction). `RagdollSettings.from_skeleton` is a starting fallback, not production anatomical limits. PyJolt's `humanoid()` prefab is sufficient for the first end-to-end demo.
- Enforce the native ragdoll's single-root, parent-before-child bone hierarchy and build a mapping from native physics bones to Graaal's render skeleton. Nonphysical bones stay animation driven; support many render bones per physical hierarchy. Account for bind-pose joint/shape offsets, coordinate frames, entity root motion, and local scale.
- Spawn with `world.add_ragdoll`; maintain ragdoll ID and each bone UID in the adapter. Implement `KINEMATIC`, `POWERED`, `LIMP`, mode transition/blend, strength changes, `hit(bone, impulse, point)` and cleanup with `remove_ragdolls` (never `remove_bodies` on an individual bone).
- Use local animation poses from `PosePlan`/`sample_clip_pairs` to feed `Ragdoll.set_pose`. Prefer the existing ECS animation clock to `Ragdoll.play` while mixing clips, crossfades and live IK: two independently advancing animation timelines will drift. Native `play` can remain a shortcut for standalone demos and static loops.
- Read `Snapshot.ragdolls` once after the step, reconstruct physical bone world transforms into local skeletal TRS, overlay animation for unmapped bones, build final skin matrices with existing `model_matrices`/`skin_matrices`, and pass a single final pose per owner to the current skinning adapter.
- For render-only blending and interpolation, compose poses per-bone; do not treat Jolt's `set_mode(blend=...)` as a general renderer-side animation blending API.

**Deliverables:** `graaal/PhysicsRagdoll.py` (physics/skeleton helper), ECS integration in `EcsPhysics.py`, `scenes/ragdoll_example.py` (standing animated character -> pushed powered character -> limp fall -> recovery transition).

**Tests:** ragdoll bone mapping, parent/local-frame transforms, native joint limits, proper inverse-bind skinning and placement with translated/rotated root, event-to-bone mapping, mode switches and deletion without leaked physics bodies.

### Phase 5 — Character controller and gameplay integration

- Optionally add `PhysicsCharacter` via pyjoltphysics `add_character`, with ECS input/command batching, jump/step/slope settings and movement readback. Keep it distinct from a dynamic rigid body and from a physical ragdoll.
- Define one authoritative character-motion model: character controller for normal locomotion, ragdoll for physically driven character motion. For simple death/knockdown, explicitly transfer transform and velocity between representations, disable the one that should not collide, then swap back on recovery. Continuous powered ragdoll locomotion belongs to the existing active-ragdoll plan.
- Use batched raycasts for foot placement and terrain responses; feed the accepted physics motion into `RootMotion`-based gameplay rather than blindly applying requested root motion to the render transform.

**Deliverable:** Controllable character with collision-aware locomotion, triggers, foot IK and ragdoll hand-off.

### Phase 6 — Diagnostics, optimization and advanced active-ragdoll bridge

- Surface `world.stats()` alongside `Diagnostics` (tick/step/publish/event/callback times, dropped events, body counts, snapshot buffer pool and query stalls). Benchmark 100/1,000 dynamic objects, colliders with compounds, and 10–20 ragdolls across native thread counts.
- Measure actual costs of ECS row gather/scatter and NumPy `float64 -> float32` conversion; only then consider a direct physics snapshot-to-render staging path for bodies whose ECS transform mirror adds unnecessary overhead.
- Validate `save_state`/`restore_state` and reset behavior for headless tests. Native Jolt save/restore does **not** cover all Python-side ragdoll pose/mode targets, characters, tick counters or queues: save/restore those separately if deterministic replay/training is needed.
- Extend pyjoltphysics **only if measurements or requirements demand it**: passive ragdoll soft-limit torque, always-on neutral-posture spring, per-joint motor strength, bulk joint torque commands, observations and selective resets for high-throughput learning. These are not all provided by the exposed high-level ragdoll API; the existing `active_ragdolls.md` remains the design for that next stage.
- Revisit native-free-run simulation only if a use case benefits from stepping physics independently of Graaal's scheduler; specify temporal synchronization and command latency first.

**Deliverables:** benchmarks, physics diagnostics, backend-gap checklist, API guide `graaal/physics.md`, tests and updated `scenes/README.md`.

## API surface (proposed, not implemented)

```python
physics = EcsPhysicsAdapter(ecs, transform_type=MaterialTransform, settings=...)
physics.reconcile(ecs, commands, dt)
physics.prepare_inputs(ecs, commands, dt)
physics.step(ecs, commands, dt)
physics.pull(ecs, commands, dt)
physics.dispatch_events(ecs, commands, dt)
physics.add_impulse(entities, impulses, points=None)
hit = physics.raycast(origin, direction, max_distance=10)
physics.close()
```

The `EcsPhysicsAdapter` should expose its `world` so non-ECS advanced PyJolt features remain available. The scheduler-compatible names above are proposed public methods, not claims about existing Graaal APIs.

## Definition of done for the first usable physics integration

- A scene can add/remove collider-only, static, dynamic and kinematic entities through ordinary ECS components, without explicitly handling native body UIDs.
- Scheduled physics reliably steps at a caller-controlled fixed rate and dynamic objects update rendering transforms without an input/output feedback loop.
- Contact begin/end, triggers and sleep/wake can be inspected as NumPy event batches and handled by ECS listeners, including entity deletion cases.
- Queries and joints work headlessly; optional physics packages remain optional.
- At least one native Jolt ragdoll follows a Graaal animation target, reacts to an impulse, transitions to limp, and renders correctly through the existing skeletal skinning system.
- Performance and behavior are covered by repeatable headless tests and a graphics example.

## External API references (verified 20 November 2026)

- [pyjoltphysics PyPI — 0.1.0a3, 4 November 2026](https://pypi.org/project/pyjoltphysics/)
- [pyjoltphysics public API documentation](https://github.com/mcfletch/pyjoltphysics/blob/main/docs/API.md)
- [Jolt Physics architecture (sensors, callbacks and filtering)](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md)
