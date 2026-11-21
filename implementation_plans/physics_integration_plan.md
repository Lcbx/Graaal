# Physics integration plan and status

Optional pyjoltphysics/Jolt integration for rigid bodies, colliders, queries,
events, joints, native characters and ragdolls. The supported API and limitations
live in the [physics guide](../graaal/physics.md); subsequent neural/passive
control work belongs to the [active-ragdoll plan](active_ragdolls.md).

## Delivery status

| Phase | Delivered | Canonical examples/tests |
| --- | --- | --- |
| 0: backend proof | Optional pinned `physics` extra, headless lockstep world, snapshots/events and safe shutdown | [standalone scene](../scenes/physics_headless_example.py), [tests](../tests/test_physics_headless.py) |
| 1: rigid bodies | Declarative ECS bodies/colliders, shape cache, compounds, batched lifecycle/setters/commands and authority rules | [ECS scene](../scenes/physics_example.py), [tests](../tests/test_ecs_physics.py) |
| 2: scheduling/presentation | Fixed callbacks, version checks, UID-aligned render interpolation and generic material transform source | [benchmark](../scenes/benchmarks/benchmark_physics.py), [interpolation contract](../graaal/physics.md#render-interpolation) |
| 3: interactions | Once-per-tick routed events, subscriptions, native-preserving queries, exclusions, filters/materials and keyed joints | [queries](../scenes/physics_queries_example.py), [joints](../scenes/physics_joints_example.py), [tests](../tests/test_ecs_physics_interactions.py) |
| 4: ragdolls | Authored/fallback skeleton mapping, modes/strength, bone event identity, held animation targets, final skinning and interpolation | [scene](../scenes/ragdoll_example.py), [tests](../tests/test_physics_ragdoll.py) |
| 5: characters | Capability assets, feet/state readback, held movement/jumps, proxy identity, interpolation and animation/ragdoll hand-off | [scene](../scenes/character_example.py), [controller tests](../tests/test_ecs_characters.py), [scene tests](../tests/test_character_example.py) |
| 6: advanced | Pending | Diagnostics, broader workloads, replay/reset validation and measured binding gaps |

## Architecture and invariants

- Standalone applications use the native world directly. ECS integration lives
  in `EcsPhysics`; skeleton conversion lives in standalone `PhysicsRagdoll`.
  Construction loads the optional backend once; CPU imports remain independent
  of physics and rendering. `physics.world` keeps advanced native APIs accessible.
- Use caller-controlled lockstep, normally 120 Hz. One adapter step is one
  native tick with matching `fixed_dt`; no hidden accumulator or free-running
  native simulation. The caller also owns render interpolation alpha.
- Dynamic bodies own pose/velocity; ECS holds a mirror. Kinematic transforms
  supply targets, static moves require teleport, and character positions come
  from native accepted motion. Never feed mirrored dynamic poses back as input.
- Native UIDs and full uint64 ECS handles are separate namespaces. Resolve rows
  every snapshot; retain retired mappings through event dispatch and suppress
  deleted-owner delivery. Ragdoll bones and characters need their own identity maps.
- Snapshots are read-only native views; promptly release them and copy retained
  history. Version checks skip unchanged configuration, while changed rows use
  NumPy batches and explicit float64-to-float32 conversion where needed.
- Native resources belong to the adapter; components contain declarative numeric
  settings/asset keys. Rebuild shape/topology changes while preserving appropriate
  state. Mixed solid/sensor owners and compound local scaling remain restricted;
  see the guide rather than assuming proposed combinations are supported.
- Body, ragdoll and character changes share capacity preflight. The pinned
  backend may require removals followed by `flush()` before replacements reserve
  slots. This does not advance time; native queries must not run concurrently.

## Scheduling and animation contract

Within each fixed tick: advance animation/control, reconcile declarations,
prepare physics inputs, step once, pull state, dispatch events, then compose the
final pose. Dependencies order callbacks; `flush_after=True` is required when
earlier structural commands must be visible within that tick. Events affect
the next tick by default. Rendering samples owned previous/current poses using
the caller's alpha and never changes gameplay state.

Ragdoll motors consume local animation poses, not final skin matrices. Convert
physical world joints back through the entity frame and mapped skeleton, retain
animation-only bones and publish one final palette per owner. Native ragdolls
require one physical tree, matching rest joint frames and unit scale. Avoid a
second native animation clock alongside ECS playback.

The character example derives speed from requested root motion and uses native
accepted motion for placement, then collision-backed foot IK. Hand-off disables
the controller, settles the held ragdoll pose kinematically for two ticks, enters
limp with inherited velocity, and checks standing clearance before recovery.
Blocked recovery retains the ragdoll. This is a gameplay policy, not continuous
active balancing.

## Remaining work and acceptance criteria

- Expose native stats with engine diagnostics: tick/step/publish/event costs,
  dropped events, body counts, snapshot pools and query stalls.
- Benchmark 100/1,000 colliding bodies, compounds and 10–20 ragdolls across native
  thread counts. The existing isolated-box [benchmark](../scenes/benchmarks/README.md#physics-synchronization)
  measures synchronization, not collision-heavy performance or GPU FPS. Add a
  direct snapshot-to-render path only if measurements justify it.
- Validate save/restore and selective resets. Native state alone omits Python
  ragdoll targets/modes, character state, clocks and queues; capture those for replay.
- Measure binding gaps before extending passive torques, neutral springs,
  per-joint strengths, bulk observations/actions or training resets. Revisit
  free-running simulation only with an explicit latency/synchronization contract.

Retain headless tests for lifecycle/generation reuse, authority, shared capacity,
event retirement, query namespaces, joint limits, ragdoll frames/inverse binds,
character movement/hand-off and interpolation after row compaction. Keep optional
dependency isolation and bounded rendered examples. Test other target platforms;
historical backend verification was Windows x64 only.

Historical validation: the merged Phase 5 suite passed 1,180 tests plus four
subtests; character corrections passed 29 focused checks and bounded headless/
offscreen runs. A separate import-cleanup run had one statistical transparency
failure that passed with its module on rerun; tolerances were unchanged. These
are prior results, not validation performed during this documentation audit.

Documentation audit: replaced incremental progress logs and obsolete proposed
signatures with phase status and links to supported contracts. Outstanding
diagnostics, performance and replay work remains explicit.

## Backend references

- [pyjoltphysics package](https://pypi.org/project/pyjoltphysics/) — integration pins `0.1.0a3`.
- [Backend API](https://github.com/mcfletch/pyjoltphysics/blob/main/docs/API.md).
- [Jolt architecture](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md).
