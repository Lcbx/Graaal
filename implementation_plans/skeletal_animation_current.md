# Skeletal animation status

Design: [pipeline plan](skeletal_animation_pipeline.md). Public contracts:
[animation guide](../graaal/skeletal_animation.md),
[materials](../graaal/materials.md#gpu-skinning),
[physics](../graaal/physics.md). This snapshot replaces incremental delivery
logs; validation evidence is scoped below.

| Milestone | Current scope | Remaining |
| --- | --- | --- |
| 1: foundation | Immutable skeletons/clips, batched CPU sampling/FK/skin matrices, glTF subset importer, ECS playback, CPU material bridge, shared GPU skinning and palette lifecycle | Broader asset support and measured allocation improvements |
| 2: transitions/root motion | Crossfades, named events, root extraction, clip-pair samples, separate advancement/finalization, bounded and native collision examples | Movement reconciliation policy remains application-owned |
| 3: GPU crowd evaluator | Packed clip baking, CPU reference, sampled graph eligibility, CPU/GPU graph routing and a compute pass writing shared skin palettes | Automatic tiering and crowd overrides |
| 4: graphs | Pose plans, masks/additive layers, 1D/2D selectors, editable definitions, clip-state and graph ECS adapters, batched playback with events and root motion | Weighted cycle timing |
| 5: sync/physics | Marker-synchronized ECS fades, fixed-reference 1D/2D blend synchronization, two-bone IK, axis alignment, stance locks, local twist limits, collision queries and ragdoll targets | Integration with full locomotion authoring and crowd overrides |
| 6: advanced | Not started | Warping, skeleton LOD, exact GPU evaluator, GPU IK and debugging |
| 7: tools | Headless event/graph authoring, Tk timeline, CPU-skinned preview, state/parameter/transition/blend-space forms, sync markers, manual/automatic clip and 1D/2D blend-state playback and explicit JSON project persistence | Forms for layers and masks, material-rendered preview |

## Integration choices to preserve

- Standalone resources/evaluators do not require ECS or graphics. Optional ECS
  adapters retain full generational owner handles and refresh selections.
- CPU quaternion math is NumPy-based. Runtime interpolation uses shortest-path
  nlerp; glTF resampling uses slerp. Clips retain both endpoints.
- Controllers emit samples, events and requested root motion. Applications
  decide accepted motion and physics order; pose finalization can run later.
  Crossfade root movement is a documented fixed-step approximation.
- Skin matrices use a shared 3×4 float32 palette. GPU material skinning lets
  owners share geometry; CPU vertex deformation needs independent meshes.
  PosePalette reconciles complete snapshots; callers coordinate draw removal,
  range reuse and GPU lifetime. Cached pose outputs must be consumed before reuse.
- PosePlan compiles dependencies into reusable buffers. Separate pre/post plans
  exchange copied local poses; queries and world/model conversion stay external.
- Editable definitions compile into existing state-machine, pose-plan and sync
  resources. Invalid drafts preserve the last compiled preview. Shared JSON
  projects persist graph drafts and event/sync tracks only on explicit save.
  Loads validate clip metadata and compile against imported markers before
  replacing a session; saves validate serialized content before replacing files.
  Source assets remain separate. This keeps both tools independent while letting
  marker edits round-trip through the timeline without losing a saved graph.
- Bound graph grids require distinct vertical weight slots. Grids used only by
  states may omit sample/weight bindings; the form preserves this choice.
  Synchronized playback
  re-anchors phase each tick to avoid long-fade float32 drift and lost crossings.
- Automatic graph preview adapts one row of the standalone runtime player to
  the editor's scalar controls. Manual preview uses the same state sampler.
  Fades preserve target time and speed at promotion, and parameter changes
  during a fade affect sampling without interrupting the fade.
- 1D/2D blend states select from applied parameters at a shared unwrapped seconds
  clock. Unsynchronized clips wrap independently; synchronized spaces map from
  an authored fixed reference, regardless of weights or the selected cell.
  All participants validate against compatible marker snapshots at compile time.
  Forms and JSON projects retain the optional reference. Fades blend state poses
  before authored layers through a compiled pose-input plan, preserving nested
  quaternion blending. Explicit transition mappings use clip or blend-reference
  clocks. Weighted cycle timing remains deferred. 2D states
  blend horizontal pairs before the vertical blend, independent of authored
  layer weights. Clip, 1D and 2D state bindings remain separate.
- Standalone batched graph playback owns a fixed set of independent state/fade
  clocks, with scalar or per-row time steps, selected-row resets and nonnegative
  speeds. It evaluates grouped state poses before application-owned layers and
  emits independent local TRS arrays. Clock snapshots are read-only; invalid
  updates preserve all rows. Optional state event tracks report crossings
  with row, state, tick offset and fade weight. State clocks drive events,
  including fixed references for synchronized blends. Optional root tracks
  generate character-space deltas over the traversed clip intervals. The ECS
  adapter retains full handles and routes events in query order. The authoring
  preview retains its single-row API.

## Examples and validation evidence

See the [scene catalog](../scenes/README.md) for playback, mixing, layers,
root motion, graphs, IK, physics locomotion and native character/ragdoll examples.
The native character scene supplies a synchronized walk/brisk blend, idle fades
and per-owner footstep events. Its controller handles sliding, steps and jumping.
The simpler shape-sweep locomotion example demonstrates blocking.

Regression coverage includes hierarchy/ownership validation, quaternion signs,
loop/reverse/event boundaries, transition promotion, root frames, long synchronized
fades, task dependencies, IK scales, entity recycling, palette ranges, glTF
reordering and GPU-versus-CPU material skinning at 1×/4× MSAA. Tool tests cover
stable marker IDs, invalid/stale drafts, compilation, preview deformation,
automatic fade promotion, pause/resume, seeking and synchronized preview timing.
Headless preview regressions compare poses and clocks with existing ECS playback.
1D/2D state regressions cover unequal clip loops, live parameters, clip/blend and
blend/blend fades, nested quaternion blending, pose-input layers, seeking, invalid
draft preservation and project reloads. Grid tests include one-row/column/cell
cases, bound/unbound forms and validation of paired slot bindings.
Blend-sync regressions cover batched pair/grid timing, reverse marker rates,
marker offsets and unequal intervals, stable reference clocks across parameter
changes and loops, reference-clock fades, snapshot ownership, form validation
and project round trips.
Project tests cover exact marker times, coincident ordering, graph/marker round
trips, source/baked duration differences, incompatible files, preserved graph
data in the timeline and failed-write preservation.
Blend-space synchronization validation (2026-11-21): focused sync, graph,
playback, graph UI, project and blend-space tests passed, followed by final
regressions for selected cells that exclude the reference clip. The full test
suite and `python -m graaal` also passed. Fixed-reference timing was selected
for stable clocks across parameter changes; weighted cycle timing remains
deferred. Prior bounded GPU scene results are historical; a separate GPU smoke
test was not run for this change. Use current tests rather than old test totals
as the coverage inventory.

Batched runtime graph validation: mixed-state tests assert batch clocks and
poses directly. Independent playback tests cover clip/1D/2D fades, marker
synchronization, unequal loops, live parameters and promotion. Focused runtime, graph, sync,
pose-plan and scene tests passed, including final empty-batch and parameterless
graph regressions. Selective resets, independent snapshots, per-row time steps
and speeds, overflow rejection and masked-layer skin matrices are covered.
The full suite, `python -m graaal` and
`python -m scenes.animation_graph_batch_example` passed. The batch size is fixed.

Graph event regressions cover independent track snapshots, stationary and
looping clocks, coincident ordering, tick offsets and fade weights for both
states, synchronized reference clocks, and crossings at and after promotion.
The full suite, CPU smoke and a headless event delivery check passed. The final
focused playback, graph, event, sync and ECS run also covered exact fade-end
ties with inverse timing rounded in either direction.
Wrap regressions check that cursor continuity prevents duplicate crossings
through promotion, and that corresponding sync markers keep the source first
at equal tick offsets. Adjacent events and reverse interval boundaries retain
their authored order.

`python -m scenes.benchmarks.benchmark_animation --entities 1000 10000 50000 --bones 12 --repeat 4`
measures controller, pose and combined CPU cost. The Greenman idle/move
crossfade probe used a real 12-bone rig and a Vulkan RTX 5060 Ti palette.
Median controller, pose and palette-submit CPU times were 0.49/4.66/0.66 ms
at 1k, 5.25/58.55/8.40 ms at 10k, and 26.69/311.55/46.54 ms at 50k. The
palette measurement includes validation, copies and queue submission. See the
[benchmark guide](../scenes/benchmarks/README.md) for timing conventions.

`BakedClipPool` used 22,080 bytes for the two clips at 30 Hz. Four sampled
0.25 s idle-to-move fades peaked at 1.4–1.9% of bind joint height in the CPU
reference. The GPU pass matches the CPU reference and can write beside CPU
palette ranges. The compute probe used two equally weighted samples per
instance. Over ten measured passes, median CPU staging and recording took
0.41/2.71/15.10 ms and GPU compute took 0.004/0.041/0.198 ms at 1k/10k/50k
instances. These GPU timestamps bracket the compute pass. Draw-inclusive
frame timing and broader asset validation remain open. Use the
[crowd roadmap](skeletal_animation_pipeline.md#gpu-pose-evaluation-for-crowds-and-distant-characters)
for tier acceptance criteria.

The graph eligibility validator compares the baked CPU reference with exact
base poses for clip, 1D and 2D states and authored fades. Tests cover the
four-sample limit, merging identical inputs, edge order and synchronized fade
clocks. With the river dummy's 30 Hz pool, 32 phase samples and playback rates
of 0.58×, 1.0× and 1.60×, the 2% height budget was 3.52 cm. The maximum sampled
errors for stand, walk and run were 0.05, 1.04 and 0.54 cm. The swim state
reached 38.44 cm and the largest fade error was 58.22 cm on swim to walk.

The river scene advances the ECS graph at fixed ticks and routes rendered poses
at draw time. Stand, walk and run usually use the crowd pass. The CPU evaluates
the swim state and fades. Both write disjoint owner ranges in one palette before
the material draw. Tests check clock and event parity, GPU palette values, and
the position error when switching evaluators at the start of a shoreline fade.
Mixed frames upload CPU matrices in one staging write and copy them into their
palette ranges before the crowd dispatch. GPU readback tests cover coalescing,
untouched ranges between copies and staging reuse on the next submission.

Before batching, exact pose and matrix evaluation at one representative tick
fell from 4.39 ms to 3.47 ms with hybrid routing, which took 0.16 ms. The 89
CPU owners spanned 24 separate palette upload ranges.

The upload comparison used 144 dummies at 640×480 offscreen on an NVIDIA GeForce
RTX 5060 Ti (Vulkan). Each trial began after an 850-tick preroll and five warm-up
frames. Each of the three trials per mode measured 12 frames from the fixed
update through GPU completion. Hybrid frames dispatched 51–59 owners on the GPU.
Both hybrid modes used the same poses and palette layout.

| Pose mode | CPU palette uploads per frame | Median pose preparation and recording across trials (ms) | Median full frame across trials (ms) |
| --- | --- | --- | --- |
| CPU only | 1 | 5.18–6.03 | 10.94–12.68 |
| Hybrid with direct uploads | 24–30 | 5.40–6.30 | 11.23–12.95 |
| Hybrid with staged copies | 1 | 5.44–5.59 | 11.21–11.69 |

Staged copies lowered median frame time relative to direct uploads in two of
three rounds. The frame-time ranges overlapped across all three modes.

Documentation audit: consolidated delivery history and validation counts into
capabilities, remaining work and maintenance contracts; API details remain in
the linked guides to avoid conflicting status copies.
