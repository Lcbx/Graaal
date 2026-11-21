# Physics integration plan and status

Optional pyjoltphysics/Jolt integration for rigid bodies, colliders, queries,
events, joints, native characters and ragdolls. The supported API and limitations
live in the [physics guide](../graaal/physics.md); subsequent neural/passive
control work belongs to the [active-ragdoll plan](active_ragdolls.md).

Control implementation stays in Python and NumPy through the existing pinned
binding. Changes to Jolt or its native bindings are out of scope.

## Delivery status

| Phase | Delivered | Canonical examples/tests |
| --- | --- | --- |
| 0: backend proof | Optional pinned `physics` extra, headless lockstep world, snapshots/events and safe shutdown | [standalone scene](../scenes/physics_headless_example.py), [tests](../tests/test_physics_headless.py) |
| 1: rigid bodies | Declarative ECS bodies/colliders, shape cache, compounds, batched lifecycle/setters/commands and authority rules | [ECS scene](../scenes/physics_example.py), [tests](../tests/test_ecs_physics.py) |
| 2: scheduling/presentation | Fixed callbacks, version checks, UID-aligned render interpolation and generic material transform source | [benchmark](../scenes/benchmarks/benchmark_physics.py), [interpolation contract](../graaal/physics.md#render-interpolation) |
| 3: interactions | Once-per-tick routed events, subscriptions, native-preserving queries, exclusions, filters/materials and keyed joints | [queries](../scenes/physics_queries_example.py), [joints](../scenes/physics_joints_example.py), [tests](../tests/test_ecs_physics_interactions.py) |
| 4: ragdolls | Authored/fallback skeleton mapping, modes/strength, bone event identity, held animation targets, final skinning and interpolation | [scene](../scenes/ragdoll_example.py), [tests](../tests/test_physics_ragdoll.py) |
| 5: characters | Capability assets, feet/state readback, held movement/jumps, proxy identity, interpolation and animation/ragdoll hand-off | [scene](../scenes/character_example.py), [controller tests](../tests/test_ecs_characters.py), [scene tests](../tests/test_character_example.py) |
| 6: advanced | Diagnostics, workload benchmarks and body/joint replay/resets delivered; full control-state replay and binding gaps pending | [Diagnostics](../graaal/physics.md#diagnostics-and-performance), [checkpoints](../graaal/physics.md#checkpoints-and-selective-resets), [replay tests](../tests/test_physics_replay.py) |

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

## Phase 6 diagnostics

Delivered native diagnostics through `Diagnostics.inspect_physics(world)` and
an optional physics section in `DiagnosticsVisualizer`. Explicit collection
copies all native stats without stepping, draining events or retaining snapshots;
a separate bounded history labels native mean tick costs to avoid confusing
them with render-frame times. This keeps standalone native worlds usable and
adds no monitoring work to simulation callbacks.

Validation: 22 focused diagnostics, visualizer and library-boundary tests passed,
including pending command/event/pool preservation, timing reset isolation,
optional dependency imports and an ImGui frame with a physics-only report.
The existing physics/character regression modules passed 116 tests; the CPU
smoke test (`python -m graaal`) also passed. No additional platform validation
was performed.

## Phase 6 workload measurements

Extended the [headless benchmark](../scenes/benchmarks/README.md#physics-synchronization)
with gravity-driven stacks, two-shape compounds and limp native humanoids while
retaining the isolated-box baseline. Each tick now includes event routing;
native timing means restart after warmup. Reports separate final ragdoll skinning
from physics, count actual contact activity and expose native body/bone counts
and dropped events so workload claims can be checked.

On Windows x64, CPython 3.14.10, NumPy 2.5.5 and pyjoltphysics 0.1.0a3, the
48-case matrix covered 100/1,000 bodies, 10/20 ragdolls, 1/2/4 threads and both
interpolation modes, using 60 warmup and 240 measured ticks. Every collision case
had contact records in all measured ticks, with no dropped events or rejected
commands. The native humanoid contains 11 physical bones. Local raw reports are
`scenes/benchmarks/results/physics_phase6_windows.json` and
`scenes/benchmarks/results/physics_phase6_ragdoll_recheck_windows.json` (ignored
benchmark artifacts); the benchmark guide contains the reproduction command.

At 1,000 bodies with interpolation, median fixed-tick time fell from 3.28 to
2.10 ms for boxes and 3.35 to 2.07 ms for compounds between one and four threads.
Rigid render-pose preparation cost about 0.27 ms; conversion alone cost about
0.001 ms. These measurements do not justify another snapshot-to-render path.
For 20 ragdolls, fixed-tick medians ranged from roughly 6.3 to 11.0 ms across
the matrix and a reverse-thread-order recheck, dominated by input preparation
(roughly 5.2–9.7 ms), with separate skinning costs. Variability prevents a stable
ragdoll thread-scaling claim. This prompted the held-target conversion/submission
measurement below; input cost exceeded native stepping in every case in this
historical matrix. These are local observations, not GPU
FPS or CI performance thresholds.

Validation: 46 benchmark, native headless and ragdoll tests passed. Workload
tests check contact activity, compound topology, warmup exclusion, two-thread
stepping, event draining, JSON serialization and the correct presentation path.

## Phase 6 ragdoll target preparation

Added the benchmark's `ragdoll_inputs` probe to separate per-owner/batched target
conversion, native submission, full adapter target updates and held preparation.
It runs after simulation, applying commands with `flush()` outside timed sections
without stepping or draining events. It submits rest targets, so it does not
measure animation evaluation or powered motor simulation.

Profiling identified repeated Python pose conversion and per-owner ECS reads as
the input bottleneck. The adapter now retains the native-parent conversion
already validated by `set_ragdoll_pose`, batches conversion by asset, and gathers
live owners' current entity frames together. Updates remain atomic across assets;
input order, repeated owners, detached caller buffers, animation-only bones,
generation checks and rebuild/removal cleanup are preserved. Native pose
submission remains per handle; these measurements do not justify new bindings.

On Windows x64, CPython 3.14.10, NumPy 2.5.5 and pyjoltphysics 0.1.0a3, matching
one-thread runs used 60 warmup and 240 measured ticks, 10/20 ragdolls and both
interpolation modes. At 20 ragdolls, median CPU times were:

| Measurement | Before (ms) | After (ms) |
| --- | --- | --- |
| Input preparation | 5.45–6.06 | 0.26–0.27 |
| Fixed tick | 6.63–7.40 | 1.46–1.56 |
| Full target update plus preparation probe | 9.61–9.90 | 0.66 |

Standalone per-owner conversion cost 3.39–3.50 ms versus 0.27–0.30 ms batched;
native submission cost roughly 0.15–0.16 ms in these cases.

A reverse-thread-order recheck covered 1/2/4 threads. At 20 ragdolls, preparation
medians ranged from 0.20 to 0.35 ms and full target update plus preparation from
0.58 to 0.99 ms. Every case had contacts in all 240 measured ticks, with no dropped
events or rejected commands. Variability still prevents a stable thread-scaling
claim. Raw local reports (ignored benchmark artifacts) are
`scenes/benchmarks/results/physics_ragdoll_inputs_probe_before_windows.json`,
`scenes/benchmarks/results/physics_ragdoll_inputs_probe_after_windows.json` and
`scenes/benchmarks/results/physics_ragdoll_inputs_recheck_windows.json`.

Validation: 185 focused physics, character, replay, locomotion, benchmark and
import-boundary checks passed, including mixed-asset/repeated-owner targets,
caller buffer reuse, current entity frames, invalid update atomicity, stale
owners, cache lifecycle and probe clock/event preservation. The import-boundary
module required a retry outside the sandbox because pytest's temporary directory
was inaccessible. `python -m graaal` and the 960-tick headless ragdoll example
passed. No GPU or additional platform validation was performed.

## Phase 6 body checkpoints and selective resets

Added same-adapter, in-memory `save_state`/`restore_state` for regular bodies and
joints, plus `reset_bodies` for selected regular owners. Checkpoints copy native
state, full-handle ownership, ECS targets and pending body inputs; restore flushes
without stepping and snaps presentation history. Topology/configuration guards,
native queue boundaries and generation checks prevent restoration onto rebuilt or
recycled owners. Selective resets preflight values before replacing only selected
owners' pending commands, preserving other owners' inputs and render history.

Native validation confirms that body/constraint state restores while the tick
counter stays monotonic. Character positions/held motion and ragdoll mode/strength
do not restore from native state bytes. Consequently the adapter rejects native
control resources rather than presenting pose-only restoration as full rollback.
Application clocks, command buffers, scheduled callbacks and gameplay side effects
remain caller-owned. The [checkpoint contract](../graaal/physics.md#checkpoints-and-selective-resets)
documents those limits and excludes future native commands/configuration changes.

Validation: 38 replay/reset checks passed, including one- and two-thread runs,
contact/constraint trajectories, sleep restoration, reusable queued inputs,
kinematic/static targets, generation/configuration guards and selective history
preservation. Another 113 existing physics and import-isolation tests passed;
`python -m graaal` and diff checks passed. Native exclusion tests establish the
character and ragdoll capture gaps rather than assuming those resources replay.

## Remaining work and acceptance criteria

- Full ragdoll/character replay and selective training resets require native
  control-state capture (held targets, modes/strengths, transitions, character
  timers/controls) as well as Python animation/gameplay clocks, queues and events.
  Body checkpoints and pose/velocity resets do not satisfy that training contract.
- The [active-ragdoll backend validation](active_ragdolls.md#backend-validation-status)
  now verifies native bulk torques, per-bone strengths and pose/velocity readback,
  and delivers standalone Python/NumPy neutral springs and progressive soft
  angular limits with a headless control benchmark. Anatomical tuning, expanded
  observations/actions, training resets and complete passive-motion validation
  remain pending. Record missing native capabilities without extending bindings.
  Revisit free-running simulation only with an explicit latency/synchronization
  contract.

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
replay and control/binding work remains explicit.

## Backend references

- [pyjoltphysics package](https://pypi.org/project/pyjoltphysics/) — integration pins `0.1.0a3`.
- [Backend API](https://github.com/mcfletch/pyjoltphysics/blob/main/docs/API.md).
- [Jolt architecture](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md).
