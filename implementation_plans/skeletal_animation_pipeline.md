# Skeletal animation design and roadmap

Current delivery status: [status snapshot](skeletal_animation_current.md).
Implemented APIs: [animation guide](../graaal/skeletal_animation.md).
Esoterica informs behavior and phases; NumPy, WGPU and Graaal's ECS determine
storage and execution. The crowd evaluator below remains proposed.

## Architecture and conventions

`controller → pose evaluator → skinning renderer`

Controllers select clips, times, weights, events and requested root motion.
Evaluators sample/blend/modify poses; renderers consume a shared 3×4 skin-matrix
buffer. CPU and prospective GPU evaluators share that boundary. Keep physics
queries and movement acceptance outside evaluators.

- Skeletons use depth-first preorder: parents precede children and each subtree
  is contiguous. Clips are immutable uniform-rate local TRS arrays, normally
  resampled at 30 Hz with xyzw float32 rotations and sign-continuous frames.
- ECS owns identity/control state; runtime-owned arrays hold variable-sized
  poses. Group compatible work by skeleton, reuse buffers and vectorize over
  characters. Forward kinematics proceeds by hierarchy depth.
- Full generational handles identify characters and palette slots. Structural
  edits invalidate selections; rebuild mappings using membership/field versions.
- PosePlan compiles dependencies and buffer lifetimes once. Runtime inputs
  supply samples, weights and goals; avoid per-character task allocation.
- Explicit fixed-tick order is controller advancement, root-motion acceptance,
  pre-physics poses, physics, post-physics IK/ragdoll composition, final skin
  matrices, then render upload/dispatch. Not every pose operation waits for physics.
  The caller owns accumulation, dependencies and command-flush boundaries.

## Roadmap

| Milestone | Deliverable |
| --- | --- |
| 1: foundation | Immutable resources, batched sampling/FK, ECS players and shared GPU skinning |
| 2: transitions/movement | Crossfades, sample sets, events and root-motion request/acceptance |
| 3: crowds | Baked pool, compute evaluator, transition validator and hysteretic tiering; build when measurements justify it |
| 4: graphs | Compiled state/pose plans, parameters, 1D/2D blends, masks and additive layers |
| 5: sync/physics | Markers, phased IK, ragdoll composition and eventual GPU override slots |
| 6: advanced | Warping, secondary/reduced skeletons, exact GPU mode, GPU IK, culling and visual debugging |
| 7: authoring | Editable graph/marker data, validation, timeline/graph previews and portable persistence |

Authoring must reuse runtime resources, report item-specific validation errors,
preserve stable coincident-marker selection and apply drafts explicitly. Validate
round trips before replacing saved projects. The tools persist graph definitions
and markers through explicit JSON sidecar saves; glTF/GLB assets stay separate.
See the [tool guides](../tools/README.md) for available controls.

## GPU pose evaluation for crowds and distant characters

The packed clip pool and one-skeleton compute pass are available. See the
[crowd API](../graaal/skeletal_animation.md#crowd-pose-compute-pass) for the
supported interface and the [status snapshot](skeletal_animation_current.md)
for measurements and remaining work. Automatic tiering and overrides below
are design targets.

The CPU evaluator costs roughly `N x B` NumPy work per frame plus Python dispatch overhead, so it will top out at some character count. A GPU path batches Python work into a gather, an upload and a few dispatches; gathered/uploaded data still scales with `N`. The idea is to bake clips into a GPU-resident pool, mix them in a compute pass, and let the CPU modify the result through overrides.

### Scope and tiers

| Tier         | Characters                                           | Evaluator            | Features                                                                    |
| ------------ | ---------------------------------------------------- | -------------------- | --------------------------------------------------------------------------- |
| Near / hero  | Player, close NPCs, anything needing exact poses     | CPU evaluator | Everything                                                                  |
| Far / crowd  | Distant or numerous characters                       | GPU compute          | Single clip, crossfade, blends of up to 4 clips, soft masks, CPU overrides  |

Tier is selected per character by an LOD system (distance, screen size, importance) with hysteresis:

```python
@component
class AnimLod:
	tier: np.uint8  # 0 = CPU evaluator, 1 = GPU evaluator
```

**Always on the CPU, for both tiers:** state machines, transitions, sync, animation events and root motion. The GPU only evaluates "which clips, at which times, with which weights". A far character therefore behaves identically to a near one; only the cost and precision of its pose differ.

### Baked clip pool

At import, each clip is sampled at a uniform rate (default 30 Hz), evaluated through exact forward kinematics on the CPU, and stored as **model-space** transforms per bone per frame.

| Item                | Choice                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------ |
| Per bone-frame      | Quaternion (4) + translation (3) + uniform scale (1), as 8 half-floats = **16 bytes**      |
| Layout              | Frame-major: `base + frame * B + bone`, so one character's bones are contiguous            |
| Container           | One storage buffer per skeleton, plus a clip table (`base`, `frame_count`, `fps`, `loop`)  |
| Size                | 60 bones x 30 Hz x 2 s is about 58 KB per clip. 100 clips is about 6 MB                    |
| Not stored          | Root-motion track (extracted on the CPU at bake time), non-uniform bone scale              |

Use one storage buffer per skeleton with a clip-offset table. Pack half-floats
into u32 pairs and decode with `unpack2x16float`; check the device's storage
binding-size limit. Bone-based storage remains mesh-independent and supports
overrides. Measure quantization and sampling errors on target assets.

### Controller-to-GPU interface

For each GPU-tier character, the CPU controller emits up to `K = 4` samples. This is what the milestone 1 - 2 controllers already compute: one sample for a clip player, two for a crossfade, two to four for a 1D/2D blend space.

| Field           | Type                         | Notes                                              |
| --------------- | ---------------------------- | -------------------------------------------------- |
| `skin_base`     | `u32`                        | First skin-matrix slot of this character           |
| `sample_count`  | `u32`                        | 0 = pose written by the CPU path, skip             |
| `override_slot` | `i32`                        | -1 = no override                                   |
| padding         | `u32`                        |                                                    |
| `samples[4]`    | `(clip u32, time f32, weight f32)` | Weights normalized by the controller         |

That is 64 bytes per character. Records are gathered with `selection.read(out=...)` into one preallocated array and uploaded with a single `write_buffer`.

Use **render-time clip time** (`time + alpha * speed * dt`, where `alpha` is the fixed-tick interpolation factor). Validate loop and transition boundaries when extrapolating render-time samples.

### Pose kernel and approximation

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
| Interpolating baked model transforms | Sampling error | Validate against exact local-space evaluation on target clips |
| fp16 storage | Quantization | Depends on coordinate range and asset; measure at bake time |
| Single clip, no blending                       | Essentially exact                                                           | fp16 and bake-rate only                                                       |

The shortening is the same artifact as a vertex-space crossfade between two skinned meshes, which is why it is acceptable for short blends between similar poses and unacceptable for long blends between dissimilar ones.

**Not supported on this path:** additive layers (they need local-space deltas), hard bone masks (a pose discontinuity at the mask boundary), IK and procedural work. Soft per-bone weight masks are possible because bones are independent. Anything else means the CPU tier.

**Bake-time validator.** For every authored transition, compute the maximum bone-position error between this approximation and the exact local-space blend over the blend duration. Mark transitions below a threshold (start around 1 - 2% of character height and tune visually) as crowd-eligible. Characters whose active transition is ineligible stay on the CPU tier, or use a shorter fade.

For 10,000 characters × 60 bones × two samples, estimate 2.4 million
16-byte sample loads and 29 MB of skin-matrix writes per evaluation. Benchmark
shader time, controller gathering and uploads separately before assigning budgets.

### CPU overrides: IK and ragdolls

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

### Tier hand-off

Controller state is identical in both tiers, so a hand-off changes only which evaluator writes the skin matrices.

- Single-clip playback matches between tiers up to fp16 and bake-rate error, so validate switching outside a blend against the chosen error budget.
- During an active blend the evaluators differ by the model-space blending error. Defer tier switches while any blend weight is strictly between 0 and 1.
- Use hysteresis on the LOD thresholds to prevent flip-flopping.

### Exact GPU mode (escape hatch)

The limitation above belongs to **model-space mixing**, not to GPU mixing. If the error budget fails for important transitions, a second kernel mode can respect the chain:

- Bake **local-space** transforms.
- Run one workgroup per character, with one thread per bone.
- Blend local transforms per bone, then run forward kinematics level by level through shared memory with `workgroupBarrier`, then write skin matrices.

This is exact for crossfades and blend spaces, and it can support additive layers and hard masks. The cost is one barrier per hierarchy level, and it is limited to the default 256 invocations per workgroup (so at most 256 bones per character). It also lets overrides be applied in local space, so they propagate to children automatically. I would not build it first. Model-space mixing is simpler, and the validator above tells us whether we ever need this.

### Risks and open questions

- **Two evaluators means two code paths.** The compute kernel needs a NumPy reference implementation of the same model-space blend, used in tests.
- **Attachments on crowd characters** (weapons, hats) should be rigidly skinned meshes reading the same skin buffer, not objects positioned from the CPU.
- **Culling.** Frustum and occlusion culling before the pose pass (with indirect dispatch) is an optimization for later. Initially, evaluate every GPU-tier character.
- **Crowd skeletons.** Distant characters will likely use a reduced bone set. This needs a per-LOD skeleton and skin-weight variant, each with its own pool.
- Benchmark the CPU path first. If target character counts fit inside the NumPy budget, defer this backend.

## Validation and performance gates

Retain regressions for looping/events, quaternion signs, root frames, transitions,
hierarchy/masks, IK, entity recycling and selection expiry. Crowd tests must compare
the compute kernel with a NumPy reference, record exact-versus-approximate errors
for authored transitions, verify override weights/lifetimes, and check tier switches
outside blends. Attachments should read the shared skin palette rather than require
CPU readback.

Benchmark character/bone counts, single clips and graph blends, allocations, CPU
evaluation and palette uploads. Use target hardware to decide when a second
evaluator is necessary; future GPU scaling studies should include 1k/10k/50k
characters. Existing reproduction commands are in the
[benchmark guide](../scenes/benchmarks/README.md).

Documentation audit: consolidated repeated architecture/priority tables and
illustrative ECS scaffolding, retained crowd data/override contracts, and linked
delivery status separately so design proposals are not mistaken for current APIs.
