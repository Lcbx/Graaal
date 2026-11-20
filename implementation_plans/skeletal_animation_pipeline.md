# Animation system for Graaal — design and implementation plan

**Recommendation:** Adopt Esoterica's animation features and phased execution model, but implement them with NumPy arrays, batched systems and the existing ECS scheduler. Do not reproduce Esoterica's extensive C++ class hierarchy or its dynamically allocated per-frame task objects.

For large crowds and distant characters, add a second pose-evaluation backend: clips baked into a GPU-resident pool and mixed in a compute pass. The CPU keeps all control logic (state machines, events, root motion) and supplies overrides for IK and ragdolls. Both backends write to the same skin-matrix buffer, so the renderer does not need to know which one produced a pose (section 4).

**Assumptions**

- Native WGPU backends (Vulkan, DX12, Metal). Compute shaders and read-only storage buffers in the vertex stage are available. There is no WebGL2 fallback.
- The ECS API is as described in `ECS_API.md`. Code samples are illustrative and should be checked against it.

---

## 1. Good ideas to retain from Esoterica

### A. Separate animation into three responsibilities

| Layer                | Responsibility                                                           | Proposed implementation                                                                 |
| -------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| Animation controller | Select clips, evaluate state machines, calculate weights and root motion | ECS components + compiled graph definitions. Always on the CPU                          |
| Pose evaluator       | Sample clips, blend poses, run IK and procedural operations              | Two backends: NumPy (exact, full feature set) and GPU compute (baked clips, for crowds) |
| Skinning renderer    | Skin and draw meshes from a skin-matrix buffer                           | GPU skinning through WGPU                                                               |

This separation lets simple characters use a single clip, advanced characters use state machines, layered blending and procedural animation, and crowds use a cheaper evaluator, all without changing the renderer.

### B. Explicit animation phases

Esoterica has a particularly useful distinction between **pre-physics** and **post-physics** work.

| #   | Phase                | Work                                                                                         |
| --- | -------------------- | -------------------------------------------------------------------------------------------- |
| 1   | Graph evaluation     | Advance time, evaluate transitions, calculate blend weights, sync, events and root-motion delta |
| 2   | Root motion          | Apply or submit desired character displacement before physics                                |
| 3   | Pre-physics pose     | Sample and blend the poses that physics-driven animation requires                            |
| 4   | Physics              | Update character controllers, collisions, rigid bodies and ragdolls                          |
| 5   | Post-physics pose    | Apply ragdoll results, IK constraints and other simulation-dependent modifications           |
| 6   | Pose finalization    | Calculate global bone transforms, resolve attachments and produce skin matrices              |
| 7   | Render preparation   | Upload skin matrices and parameters, run the GPU pose pass, submit skinned meshes            |

The useful subtlety is that **not every pose operation needs to wait for physics**. Ordinary clip sampling and blending can execute before physics. Only operations that depend on simulation results need a post-physics stage.

Keep physics out of the animation evaluator itself. It should exchange root motion, constraints and poses through explicit interfaces.

### C. Features worth retaining

| Feature                                | Priority                          | Notes                                                                                   |
| -------------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------- |
| Clip playback, speed, looping, seeking | Essential                         | Simple animation player without a graph                                                 |
| Clip interpolation                     | Essential                         | Linear translation/scale, quaternion slerp or nlerp                                     |
| Root motion                            | Essential                         | Extract movement separately from bone poses                                             |
| Crossfading                            | Essential                         | Smooth transitions between clips                                                        |
| State machines                         | High                              | Gameplay-driven transitions                                                             |
| 1D/2D blend spaces                     | High                              | Walking/running and directional movement                                                |
| Animation events                       | High                              | Footsteps, attack windows, interaction triggers                                         |
| Sync tracks                            | High                              | Align footsteps or other animation markers when blending                                |
| Bone masks and additive layers         | High                              | Aim while running, upper-body attacks                                                   |
| Reusable/cached poses                  | High                              | Avoid repeated sampling of shared animations                                            |
| GPU crowd path                         | Medium (when crowd size requires) | Baked clips, GPU mixing, CPU overrides. See section 4                                   |
| Two-bone IK and foot placement         | Medium                            | Ground adaptation and hand targets                                                      |
| Orientation/target warping             | Medium                            | Adjust animations to a target location                                                  |
| Animation LOD                          | Medium                            | Tiered evaluation: exact on the CPU near the camera, baked on the GPU far away (section 4) |
| Ragdoll blending                       | Later                             | Requires a physics backend. The GPU path consumes it through override slots             |
| Secondary skeletons                    | Later                             | Independent attachments and additional rigs                                             |
| Animation authoring tools              | High                              | Configure graphs, inspect clip timelines and author event markers                        |

### D. Other good architectural ideas

**Compiled graphs:** Resolve graph structure and dependencies once, rather than interpreting an editable node graph every frame.

**Pooled pose buffers:** Reuse storage for sampling, blending and intermediate operations. Discard temporary results after their last use.

**Bone hierarchy and masks:** Store skeleton topology in a compact format and process only the bones necessary for an operation.

**Animation debugging:** Expose the active state, clip times, blend weights, root-motion deltas and pose tasks. Esoterica's recording and graph inspection tools are valuable models for this.

**Data-driven assets:** Keep skeletons, clips, masks, sync markers and graph definitions as reusable assets shared by character instances.

---

## 2. Adapting this to the NumPy ECS

The ECS API already provides most of the necessary infrastructure. I would **not modify the ECS itself** to implement animation.

### A. Proposed data model

Separate ECS-owned character state from animation-specific working memory.

**ECS components**

```python
from graaal.ECS import component
import numpy as np


@component
class SkeletonRef:
    id: np.uint32


@component
class ClipPlayer:
    clip_id: np.uint32
    time: np.float32
    speed: np.float32
    mode: np.uint8
    apply_root_motion: bool


@component
class AnimationGraph:
    graph_id: np.uint32
```

`ClipPlayer` provides simple animation playback. `AnimationGraph` is an alternative controller for advanced animation, introduced later.

The same character should not use both controllers unless an explicit precedence or composition policy is introduced.

Character movement, transform data, meshes and gameplay parameters stay in their existing components.

**Animation-owned arrays**

Skeletons and clips are shared, immutable resources. Runtime poses are managed by an external `AnimationRuntime`, rather than represented as ECS components.

For a batch of characters with the same skeleton (CPU path):

| Array          | Shape          | Purpose                 |
| -------------- | -------------- | ----------------------- |
| `rotation`     | `(N, B, 4)`    | Local bone quaternions  |
| `translation`  | `(N, B, 3)`    | Local bone translations |
| `scale`        | `(N, B, 3)`    | Local bone scales       |
| `model_matrix` | `(N, B, 4, 4)` | Final bone transforms   |
| `skin_matrix`  | `(N, B, 4, 4)` | GPU skinning transforms |

Here `N` is the number of animated characters and `B` the number of bones. Use `float32` and allocate these arrays in reusable buffers grouped by skeleton.

**Why not store the poses directly in ECS?**

Characters can have different skeletons and bone counts. Storing large, variable-size poses as Python objects in components would undermine the benefits of dense NumPy storage.

The ECS should own identity, parameters and control state; the animation runtime should own the pose buffers.

**Skeleton and clip conventions (fix these in milestone 1)**

These are cheap to decide now and expensive to change once clips and GPU buffers exist.

- **Bone order:** depth-first pre-order. A parent always precedes its children, and every subtree is a contiguous index range `[i, i + subtree_size)`. Forward kinematics, bone masks and override propagation become slice operations.
- **Uniform-rate clips:** resample clips to a fixed rate at import (30 Hz by default, higher for fast motion). Sampling becomes `frame = floor(t * fps)` plus a gather, which vectorizes across characters. There is no per-track keyframe search. The memory cost is small (section 4.2).
- **Transform representation:** local translation, rotation quaternion (xyzw) and scale, as `float32` on the CPU.
- **Quaternion sign continuity:** baked consecutive frames have a non-negative dot product. Blending still negates when needed.
- **Skin-matrix buffer:** one GPU storage buffer of 3x4 matrices indexed by `skin_base + bone`. This is the only interface between pose evaluators and the renderer.

### B. Integration with the existing scheduler

The ECS API supports `fixed_update`, `update` and `render_prep`, as well as explicit dependencies between systems. It does not have built-in pre/post-physics phases (see `ECS_API.md`).

I recommend implementing the animation phases as **ordered systems within `fixed_update`**:

```python
schedule = Scheduler(world)

schedule.add("anim_graph", anim.evaluate, phase="fixed_update")
schedule.add("anim_root", anim.apply_root_motion, after="anim_graph",
             phase="fixed_update")
schedule.add("anim_pre", anim.pre_physics, after="anim_root",
             phase="fixed_update")
schedule.add("physics", physics.step, after="anim_pre",
             phase="fixed_update")
schedule.add("anim_post", anim.post_physics, after="physics",
             phase="fixed_update")
schedule.add("anim_finalize", anim.finalize, after="anim_post",
             phase="fixed_update")

schedule.add("anim_upload", anim.prepare_render, phase="render_prep")
schedule.add("anim_gpu_pose", anim.dispatch_gpu_pose, after="anim_upload",
             phase="render_prep")
```

Illustrative integration. The `anim` and `physics` methods would implement the ECS callback signature `(world, commands, dt)`.

This assumes physics is executed within the same fixed-tick scheduler. If an external physics engine owns its stepping, the animation and physics phases must instead be coordinated around that external step.

The caller remains responsible for the fixed-timestep accumulator, as required by the current ECS API.

The GPU pose pass runs in `render_prep`, after every CPU override for the frame has been written. That makes it inherently a post-physics operation, with no readback or extra latency (section 4.5).

### C. Efficient ECS integration

The existing APIs are particularly suitable for this design:

| ECS facility              | Animation usage                                             |
| ------------------------- | ----------------------------------------------------------- |
| `world.query()`           | Cache matching entities and component-row mappings          |
| `selection.read(out=...)` | Gather controller parameters into reusable arrays           |
| `edit_dense()`            | Update homogeneous component fields efficiently             |
| `membership_version`      | Detect character/controller additions and removals          |
| `version(*fields)`        | Detect changes to control parameters or resource references |
| `CommandBuffer`           | Defer structural changes during simulation                  |
| `Scheduler`               | Order animation, physics and rendering                      |

For example:

```python
players = world.query(SkeletonRef, ClipPlayer)


def update(world, commands, dt):
    result = players.fetch()
    clip = result[ClipPlayer].selection
    clip.assign(time=clip.time + clip.speed * dt)
```

Actual playback must also handle looping, stopping and clip duration.

One important limitation: ECS component rows are not persistent IDs, and structural edits can invalidate selections. The animation runtime should map characters using their **full generational entity handles**, rebuilding affected mappings on membership changes. The same applies to GPU slot tables (skin-matrix ranges, override slots).

The ECS also does not automatically parallelize systems. Parallel animation evaluation would be a separate optimization.

---

## 3. Pose task architecture (CPU evaluator)

I would retain Esoterica's pose-task concept but use a simpler implementation.

A compiled graph produces a list of operations. Each operation references its input pose buffers and writes into a reusable output buffer. For example, locomotion with upper-body aiming:

```
 #  op            inputs   args                           out  phase
 0  SAMPLE        -        clip=walk,     t=0.42          P0   pre
 1  SAMPLE        -        clip=run,      t=0.42          P1   pre
 2  BLEND         P0, P1   w=speed_blend                   P2   pre
 3  SAMPLE        -        clip=aim_pose, t=0.00          P3   pre
 4  MASKED_BLEND  P2, P3   mask=upper_body, w=aim_weight   P4   pre
 5  IK_TWO_BONE   P4       chain=left_leg,  goal=foot_L    P4   post
 6  IK_TWO_BONE   P4       chain=right_leg, goal=foot_R    P4   post
```

Instead of Python objects for every task, store a compact execution plan using numeric opcodes and buffer indices:

```python
SAMPLE = 0
BLEND = 1
ADDITIVE = 2
MASKED_BLEND = 3
IK_TWO_BONE = 4
```

A compiled plan describes dependencies, buffer lifetimes and which operations require post-physics execution (operations 5 and 6 above). At runtime, graph evaluation updates only dynamic arguments such as clip times, weights and targets.

**Optimization priorities:**

- Batch characters with compatible skeletons and evaluation operations.
- Reuse scratch pose arrays.
- Avoid Python loops over individual bones for sampling and blending.
- Process forward kinematics by hierarchy depth, since parent-child dependencies prevent unrestricted vectorization.
- Cache shared computations within a character's evaluation graph.
- Keep the execution plan compact and avoid per-frame task-object allocations.

Initially, these operations should execute sequentially. The representation can support parallelism later without depending on it.

---

## 4. GPU pose evaluation for crowds and distant characters

The CPU evaluator costs roughly `N x B` NumPy work per frame plus Python dispatch overhead, so it will top out at some character count. A GPU path changes the scaling: per-frame Python work becomes one gather, one buffer upload and a few dispatches, regardless of `N`. The idea is to bake clips into a GPU-resident pool, mix them in a compute pass, and let the CPU modify the result through overrides.

### 4.1 Scope and tiers

| Tier         | Characters                                           | Evaluator            | Features                                                                    |
| ------------ | ---------------------------------------------------- | -------------------- | --------------------------------------------------------------------------- |
| Near / hero  | Player, close NPCs, anything needing exact poses     | CPU (sections 2 - 3) | Everything                                                                  |
| Far / crowd  | Distant or numerous characters                       | GPU compute          | Single clip, crossfade, blends of up to 4 clips, soft masks, CPU overrides  |

Tier is selected per character by an LOD system (distance, screen size, importance) with hysteresis:

```python
@component
class AnimLod:
    tier: np.uint8  # 0 = CPU evaluator, 1 = GPU evaluator
```

**Always on the CPU, for both tiers:** state machines, transitions, sync, animation events and root motion. The GPU only evaluates "which clips, at which times, with which weights". A far character therefore behaves identically to a near one; only the cost and precision of its pose differ.

### 4.2 Baked clip pool

At import, each clip is sampled at a uniform rate (default 30 Hz), evaluated through exact forward kinematics on the CPU, and stored as **model-space** transforms per bone per frame.

| Item                | Choice                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------ |
| Per bone-frame      | Quaternion (4) + translation (3) + uniform scale (1), as 8 half-floats = **16 bytes**      |
| Layout              | Frame-major: `base + frame * B + bone`, so one character's bones are contiguous            |
| Container           | One storage buffer per skeleton, plus a clip table (`base`, `frame_count`, `fps`, `loop`)  |
| Precision           | fp16: about 1 mm in position near 2 m, well under 0.1 degree in rotation                   |
| Size                | 60 bones x 30 Hz x 2 s is about 58 KB per clip. 100 clips is about 6 MB                    |
| Not stored          | Root-motion track (extracted on the CPU at bake time), non-uniform bone scale              |

WGSL can read fp16 without the `shader-f16` feature by storing pairs as `u32` and decoding with `unpack2x16float`.

**Buffer rather than texture.** A 2D texture atlas was the classic solution when vertex shaders could not read buffers. With WGPU a storage buffer is simpler: flat indexing, no 2D packing, no texture-dimension limit, and no sampler or filterable-format constraints (`rgba32float` is not filterable by default). Hardware filtering is the only real advantage of a texture, and it saves little here because blending between frames is a quaternion nlerp that needs a normalize anyway. "Atlas" here means one pooled allocation with a clip table. Check `max_storage_buffer_binding_size` (128 MiB by default) against the pool size; at the sizes above it is not a constraint.

**Why not vertex animation textures (baked vertex positions)?** They need no skeleton at runtime, but they are tied to one mesh, cost far more memory (per vertex rather than per bone), and cannot take overrides. Bone-based baking is mesh-independent and keeps the skinning shader unchanged.

### 4.3 Controller-to-GPU interface

For each GPU-tier character, the CPU controller emits up to `K = 4` samples. This is what the milestone 1 - 2 controllers already compute: one sample for a clip player, two for a crossfade, two to four for a 1D/2D blend space.

| Field           | Type                         | Notes                                              |
| --------------- | ---------------------------- | -------------------------------------------------- |
| `skin_base`     | `u32`                        | First skin-matrix slot of this character           |
| `sample_count`  | `u32`                        | 0 = pose written by the CPU path, skip             |
| `override_slot` | `i32`                        | -1 = no override                                   |
| padding         | `u32`                        |                                                    |
| `samples[4]`    | `(clip u32, time f32, weight f32)` | Weights normalized by the controller         |

That is 64 bytes per character. Records are gathered with `selection.read(out=...)` into one preallocated array and uploaded with a single `write_buffer`.

Use **render-time clip time** (`time + alpha * speed * dt`, where `alpha` is the fixed-tick interpolation factor). The GPU path then needs no pose interpolation between ticks.

### 4.4 Pose kernel and what is approximate

One dispatch per skeleton, one thread per `(character, bone)`:

```
# pseudocode, WGSL-style
for k in range(sample_count):
    f   = time[k] * fps(clip[k])          # wrap or clamp according to clip.loop
    f0  = floor(f);  f1 = f0 + 1;  a = f - f0
    s_k = blend(load(clip[k], f0, bone), load(clip[k], f1, bone), a)

q = normalize(sum(w[k] * q_k))             # q_k sign-aligned to q_0
p = sum(w[k] * p_k)
s = sum(w[k] * s_k)

if override_slot >= 0:
    (q, p) = blend((q, p), override[override_slot][bone], override_weight[bone])

skin[skin_base + bone] = to_3x4(q, p, s) * inv_bind[bone]
```

Mixing happens **per bone, not per vertex**. Vertex counts far exceed bone counts, and the vertex shader stays plain skinning.

**Why it is approximate.** Blending model-space transforms independently per bone does not preserve the hierarchy. A child's blended position minus its parent's blended position is a linear interpolation of two offsets of equal length, which is shorter than either.

| Source of error                                | Effect                                                                      | Magnitude                                                                     |
| ---------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Model-space blending of dissimilar poses       | Bones shorten, limbs shrink during the blend                                | Worst at 50% weight: length x cos(θ/2), where θ is the angle between the bone's direction in the two poses. 30° gives -3%, 60° gives -13%, 90° gives -29% |
| Interpolating model-space transforms per frame | Negligible at 30 Hz                                                         | Below visibility                                                              |
| fp16 storage                                   | Small positional and angular noise                                          | About 1 mm, under 0.1°                                                        |
| Single clip, no blending                       | Essentially exact                                                           | fp16 and bake-rate only                                                       |

The shortening is the same artifact as a vertex-space crossfade between two skinned meshes, which is why it is acceptable for short blends between similar poses and unacceptable for long blends between dissimilar ones.

**Not supported on this path:** additive layers (they need local-space deltas), hard bone masks (a pose discontinuity at the mask boundary), IK and procedural work. Soft per-bone weight masks are possible because bones are independent. Anything else means the CPU tier.

**Bake-time validator.** For every authored transition, compute the maximum bone-position error between this approximation and the exact local-space blend over the blend duration. Mark transitions below a threshold (start around 1 - 2% of character height and tune visually) as crowd-eligible. Characters whose active transition is ineligible stay on the CPU tier, or use a shorter fade.

**Cost.** For 10,000 characters x 60 bones x 2 samples, the pass issues about 2.4 M 16-byte loads (about 40 MB, much of it cache hits because characters share clips) and writes about 29 MB of skin matrices. That is small against GPU memory bandwidth, so the practical cost is likely the per-frame upload and dispatch overhead from Python rather than shader time. This is a back-of-envelope estimate and should be benchmarked.

**Bone independence is a feature.** Any single bone can be evaluated on its own, on the CPU or the GPU, with identical results. Section 4.5 relies on it.

### 4.5 CPU overrides: IK and ragdolls

**Rule: no GPU-to-CPU readback.** Readback stalls the pipeline or adds one to two frames of latency. Instead, the CPU computes whatever it needs itself and sends the result in the same frame.

**Mechanism: dense override slots.** A small pool of per-character slots, allocated only for characters with an active override. Each slot holds, per bone:

| Field                 | Type   | Notes                                       |
| --------------------- | ------ | ------------------------------------------- |
| rotation              | `vec4` | Model-space quaternion                      |
| translation + weight  | `vec4` | `xyz` = model-space position, `w` = weight  |

That is 32 bytes per bone, about 2 KB per slot for 60 bones. A weight of 0 leaves the animated bone untouched, so partial overrides need no sparse lists. Overrides are expressed in the character's model space (relative to its ECS transform). Scale comes from the animation.

The override is **not propagated to children**: the CPU supplies the whole affected subtree. With depth-first bone order that is a contiguous slice.

Illustrative API:

```python
ov = anim.acquire_override(entity)       # allocates a slot, sets override_slot
ov.rotation[bones]    = rotations        # (k, 4)
ov.translation[bones] = positions        # (k, 3)
ov.weight[bones]      = weights          # (k,)
...
anim.release_override(entity)            # weights go back to 0, slot is freed
```

**Ragdolls.** Physics bodies are converted to model-space transforms for every bone and written with a weight ramp (0 to 1 over roughly 0.1 - 0.3 s) so the character blends from animation into simulation. The root body's pose is written back to the entity transform, and the other bodies are expressed relative to it. Blending out of a ragdoll (get-up) needs an animation-driven recovery and belongs to a later milestone.

**IK.** IK needs the animated pose of the chain, but the CPU has no copy of the GPU result. It can reproduce what it needs:

1. Keep a CPU mirror of the baked pool (the same fp16 data).
2. Evaluate **only the chain bones** from the controller's samples, using the same per-bone blend as the kernel. Cost is `O(chain bones)` per IK character, not `O(B)`.
3. Raycast the ground, solve two-bone or foot IK on the CPU.
4. Write the chain and the end effector's subtree into the override slot with the IK weight.

The IK solver runs in the post-physics phase and the override is consumed by the compute pass in `render_prep`, in the same frame.

**Goals versus poses.** If many mid-distance characters need foot IK, move the two-bone solve into a small GPU pass fed with goals (target, pole, weight) from the CPU, so the CPU no longer evaluates chain bones. This is deferred until profiling shows the CPU cost matters.

**When to use the CPU tier instead.** Characters needing several IK chains, full-body procedural work, additive layers or hard masks should be promoted to the CPU tier rather than stretching the override mechanism.

### 4.6 Tier hand-off

Controller state is identical in both tiers, so a hand-off changes only which evaluator writes the skin matrices.

- Single-clip playback matches between tiers up to fp16 and bake-rate error, so switching outside a blend is invisible.
- During an active blend the two results differ by the approximation error of section 4.4. Defer tier switches while any blend weight is strictly between 0 and 1. Blends are short, so the delay is small.
- Use hysteresis on the LOD thresholds to prevent flip-flopping.

### 4.7 Exact GPU mode (escape hatch)

The limitation above belongs to **model-space mixing**, not to GPU mixing. If the error budget fails for important transitions, a second kernel mode can respect the chain:

- Bake **local-space** transforms.
- Run one workgroup per character, with one thread per bone.
- Blend local transforms per bone, then run forward kinematics level by level through shared memory with `workgroupBarrier`, then write skin matrices.

This is exact for crossfades and blend spaces, and it can support additive layers and hard masks. The cost is one barrier per hierarchy level, and it is limited to the default 256 invocations per workgroup (so at most 256 bones per character). It also lets overrides be applied in local space, so they propagate to children automatically. I would not build it first. Model-space mixing is simpler, and the validator above tells us whether we ever need this.

### 4.8 Risks and open questions

- **Two evaluators means two code paths.** The compute kernel needs a NumPy reference implementation of the same model-space blend, used in tests.
- **Attachments on crowd characters** (weapons, hats) should be rigidly skinned meshes reading the same skin buffer, not objects positioned from the CPU.
- **Culling.** Frustum and occlusion culling before the pose pass (with indirect dispatch) is an optimization for later. Initially, evaluate every GPU-tier character.
- **Crowd skeletons.** Distant characters will likely use a reduced bone set. This needs a per-LOD skeleton and skin-weight variant, each with its own pool.
- **Is it needed yet?** Benchmark the CPU path first (section 6). If the target character counts fit inside the NumPy budget, this section can wait.

---

## 5. Staged implementation plan

**1. Skeletal animation foundation — small**

Implement skeleton resources, clip resources, reference poses, sampling, local-to-model bone transforms and GPU skinning from a skin-matrix buffer. Fix the conventions in section 2.A.

Provide `SkeletonRef` and `ClipPlayer`. Support looping, speed, seeking and frame interpolation.

**Deliverable:** Multiple ECS entities playing skeletal animations, with GPU skinning and batched pose evaluation.

**2. Transitions and root motion — moderate**

Add two-clip crossfading, root-motion extraction, character displacement and animation events. Controllers emit `(clip, time, weight)` sample sets, which the CPU evaluator and the future GPU path both consume.

Separate the root-motion request from the actual physics-accepted displacement.

**Deliverable:** Characters transitioning between idle, walk and run with synchronized movement.

**3. GPU crowd path — moderate, independent track**

Add the baked clip pool, the compute kernel, `AnimLod` tiering and the bake-time transition validator. It depends only on milestones 1 - 2, so schedule it when crowd size requires it. Overrides come later, with physics.

**Deliverable:** Thousands of characters crossfading between clips with CPU-driven control, and seamless tier switching.

**4. Compiled animation graph — substantial**

Implement state machines, conditional transitions, parameter inputs, 1D/2D blend spaces, additive blending and bone masks. Introduce compact pose-task plans and reusable intermediate buffers.

**Deliverable:** A configurable locomotion controller with an independent upper-body animation layer.

**5. Synchronization and physics-aware animation — substantial**

Implement sync markers/tracks, two-bone IK, foot IK and physics-dependent pose processing. Fully exercise the pre/post-physics split. Add the override-slot mechanism for the GPU tier: CPU-solved IK first, then ragdoll blending once the physics backend exists.

**Deliverable:** A character that can walk over uneven terrain while preserving sensible foot placement and locomotion timing, and a crowd character that can drop into a ragdoll.

**6. Advanced features and optimization — optional**

Add orientation and target warping, secondary skeletons, skeleton LOD, GPU-side IK solve, the exact GPU kernel mode, culling, graph recording and visual debugging.

Benchmark NumPy implementations and accelerate only demonstrated bottlenecks.

**Deliverable:** A scalable, feature-rich character animation subsystem.

**7. Animation authoring and inspection tools — incremental**

Provide tools to configure animation graphs, visualize and scrub clip timelines,
and add, move, rename and remove animation events. Build the authoring data and
validation separately from the UI so scripts and applications can use them
without ECS or a graphics device. Continue to use the existing clip, event,
state-machine and pose-plan APIs for preview and playback.

- Define an editable, versioned representation for states, clip and blend-space
  inputs, transitions and conditions, pose tasks, masks and sync settings.
  Validate references, graph dependencies and unsupported combinations before
  compiling; errors should identify the editable item. Preserve reusable runtime
  resources and avoid inventing a second graph evaluator.
- Add graph configuration UI for parameters, states, transitions and conditions,
  blend spaces and layered poses. Provide a preview of the selected character,
  active state and transition weights, with save/load and clear validation errors.
- Add a timeline view showing clip duration, playback position, events and sync
  markers. Support play, pause, scrub and zoom, plus event insertion, selection,
  moving, renaming and deletion. Editing must preserve stable selection for
  coincident events and preserve the runtime event ordering and boundary rules.
- Save authored graph and marker data in a documented, portable format; round-trip
  and validate before replacing an existing asset. Preview edits without changing
  running instances until explicitly applied by the application. Define how sync
  marker constraints and clip-duration changes are surfaced in the editor.
- Test headless authoring, invalid inputs, event boundaries, serialization and
  compilation into existing runtime resources. Exercise UI interactions and a
  material-skinned preview when the graphics backend is available.

**Deliverable:** An author can configure a locomotion graph, preview and scrub its
clips, and add event markers that fire through the existing event runtime.

---

## 6. Key implementation decisions

| Decision                  | Recommendation                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------- |
| Graph representation      | Compiled flat graph, not runtime Python node objects                                   |
| Persistent state          | ECS components and compact per-instance graph state                                    |
| Clip storage (CPU)        | Immutable NumPy arrays at a uniform sample rate, compressed assets later               |
| Pose storage (CPU)        | Runtime-owned, pooled NumPy arrays                                                     |
| Sampling and blending     | NumPy vectorization, accelerate if necessary                                           |
| Task execution            | Compact operation list, no dynamic task allocation                                     |
| Timing                    | Fixed physics tick; optional render interpolation                                      |
| Physics integration       | Explicit root-motion and pose interfaces                                               |
| Skinning                  | GPU via WGPU, reading one skin-matrix buffer                                           |
| Character batching        | Group primarily by skeleton; subgroup compatible work                                  |
| Crowd and far characters  | GPU compute over baked model-space clips; control logic stays on the CPU               |
| GPU clip storage          | One fp16-packed storage buffer per skeleton, not a texture atlas                       |
| GPU mixing space          | Model space first (approximate, validated at bake time); local space with in-kernel FK as an escape hatch |
| CPU modifiers for GPU tier| Dense per-character override slots (ragdoll poses, CPU-solved IK); no readback         |
| LOD                       | Tier by distance with hysteresis; switch only outside active blends                    |
| Graph and timeline tools  | Incremental authoring model and validation first; graphical editing and preview build on the runtime |

### Testing priorities

Establish regression tests for clip looping and events, shortest-path quaternion interpolation, root-motion extraction, state transitions, skeleton hierarchy evaluation, and IK.

Also test component addition/removal during simulation, entity handle recycling, and stale ECS selections.

For the GPU path:

- Compare compute output against a NumPy reference of the same model-space blend, over random clips, times and weights, with an fp16 tolerance.
- Compare the approximate blend against the exact local-space blend for every authored transition and record the maximum error (this is the bake-time validator).
- Verify override blending: weight 0 is a no-op, weight 1 replaces the bone, and slots are freed correctly.
- Verify tier switches do not change the pose outside blends.

For performance, benchmark batches of characters with different skeleton sizes, both with and without graph blending. Compare allocation counts, animation CPU time and GPU skinning upload cost. Measure the CPU tier's practical character limit, since that decides when the GPU path becomes necessary, and measure per-frame Python cost of the GPU path at 1k, 10k and 50k characters.

---

## 7. Recommended scope for the first implementation

I would implement **milestones 1 and 2 before introducing animation graphs or a task DAG**.

This establishes the pieces that are hardest to retrofit later: skeleton representation, clip conventions, pose buffers, root-motion semantics and GPU skinning integration. The conventions in section 2.A (bone order, uniform-rate clips, the skin-matrix buffer) are fixed in milestone 1 even though the GPU crowd path comes later, because it depends on them.

The implementation should preserve three interfaces from the beginning:

`AnimationController → PoseEvaluator (CPU or GPU) → SkinningRenderer`

The controller could initially be a single `ClipPlayer`, then a crossfade controller, and eventually a compiled animation graph. The pose evaluator and renderer would remain largely unchanged, and a second evaluator backend can be added behind the same skin-matrix interface.

**The main architectural principle is to use Esoterica as a reference for behavior and processing stages, while letting NumPy, the existing ECS and the GPU dictate the data representation.**

That should retain the strongest ideas without introducing unnecessary abstraction or per-character Python overhead.
