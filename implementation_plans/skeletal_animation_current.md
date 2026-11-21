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
| 3: GPU crowd evaluator | Not started; GPU vertex skinning already exists | Baked clip pool, compute evaluator, transition validator and tiering |
| 4: graphs | PosePlan, masks/additive layers, 1D/2D selectors, ordered AND-condition states, optional ECS state adapter, editable compiled definitions | Complete graph controller and blend-space state playback |
| 5: sync/physics | Marker-synchronized ECS fades, two-bone IK, axis alignment, pre/post-physics plans, collision queries and ragdoll targets | Blend-space synchronization, foot locking/twist constraints and crowd overrides |
| 6: advanced | Not started | Warping, skeleton LOD, exact GPU evaluator, GPU IK and debugging |
| 7: tools | Headless event/graph authoring, Tk timeline, CPU-skinned preview, graph JSON/transition/blend-space forms, sync markers, automatic single-clip state playback and explicit JSON project persistence | Remaining forms, blend-space state playback and material-rendered preview |

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
- Graph grids require distinct vertical weight slots. Synchronized playback
  re-anchors phase each tick to avoid long-fade float32 drift and lost crossings.
- Automatic graph preview reuses compiled transition rules and sync mappings in
  a headless single-instance clock; it stays independent of ECS. Fades promote
  the target without restarting its clip, preserve pause/resume and do not
  interrupt on parameter changes. Events and root motion remain outside preview.

## Examples and validation evidence

See the [scene catalog](../scenes/README.md) for playback, mixing, layers,
root motion, graphs, IK, physics locomotion and native character/ragdoll examples.
The native character scene now supplies sliding/steps/jumping through its
controller; the simpler shape-sweep locomotion example demonstrates blocking only.

Regression coverage includes hierarchy/ownership validation, quaternion signs,
loop/reverse/event boundaries, transition promotion, root frames, long synchronized
fades, task dependencies, IK scales, entity recycling, palette ranges, glTF
reordering and GPU-versus-CPU material skinning at 1×/4× MSAA. Tool tests cover
stable marker IDs, invalid/stale drafts, compilation, preview deformation,
automatic fade promotion, pause/resume, seeking and synchronized preview timing.
Headless preview regressions compare poses and clocks with existing ECS playback.
Project tests cover exact marker times, coincident ordering, graph/marker round
trips, source/baked duration differences, incompatible files, preserved graph
data in the timeline and failed-write preservation.
Authoring validation (2026-11-21): focused project, graph/playback and timeline UI
tests, the full test suite and `python -m graaal` passed. Prior bounded GPU scene results
are historical. Use the current tests rather than old test totals as the coverage
inventory.

`python -m scenes.benchmarks.benchmark_animation --entities 64 256 1024 --bones 12 60 --repeat 10`
measures controller, pose and combined CPU cost; see the
[benchmark guide](../scenes/benchmarks/README.md). Keep the GPU crowd evaluator
deferred until target workloads justify its extra backend.

Documentation audit: consolidated delivery history and validation counts into
capabilities, remaining work and maintenance contracts; API details remain in
the linked guides to avoid conflicting status copies.
