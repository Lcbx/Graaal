# Skeletal animation CPU foundation

`graaal.SkeletalAnimation` is independent of ECS and graphics. `Skeleton` takes a depth-first preorder forest of parent indices (`-1` for roots), local reference translation, xyzw rotation and scale, and affine inverse-bind matrices. Resources copy their inputs into read-only float32 arrays. Matrices act on column vectors; local transforms compose as translation, rotation, then scale.

`Clip` accepts equally spaced local TRS frames for every bone and a positive frame rate. Its duration is `(frames - 1) / fps`; looping wraps at that endpoint, so matching first and last frames are needed for a seamless loop. `sample_clip(clip, times, loop=True)` takes a one-dimensional array of seconds and returns batched translation, normalized quaternion rotation and scale. Set `loop=False` to clamp. The caller must match clips to skeletons with the same bone order and count.

`sample_crossfade(source, target, source_times, target_times, weights)` samples two clips and blends their local poses. Times are one-dimensional arrays of equal length, and weights are a scalar or one value per instance in `[0, 1]`. Each clip can loop or clamp independently with `source_loop` and `target_loop`. Translation and scale blend linearly; rotation uses shortest-path normalized linear interpolation. Both clips must have the same bone order and count. The caller controls transition timing and weights.

`model_matrices(skeleton, translation, rotation, scale)` evaluates batches of local poses, and `skin_matrices(skeleton, model)` returns contiguous `(instances, bones, 3, 4)` row-major float32 matrices representing `model @ inverse_bind`. `skin_vertices(positions, normals, joints, weights, matrices)` applies one pose to a mesh on the CPU, using the inverse transpose of each blended linear transform for normals. Joint weights must sum to one; singular blended transforms raise `ValueError`.

The sampling, model and skin functions accept optional `out` buffers. For
`sample_clip`, supply a tuple of translation, rotation and scale arrays;
the other functions take one output array each. Buffers must be writable,
C-contiguous float32 arrays with the exact corresponding output shapes; pose
output arrays must not overlap. These buffers avoid allocating the returned
arrays, although evaluation still creates intermediate arrays.

## Root-motion extraction

`RootMotion` extracts requested movement independently of ECS or physics:

```python
from graaal.RootMotion import RootMotion
from graaal.SkeletalAnimation import sample_clip

motion = RootMotion(skeleton, clip, root=0)
requests = motion.delta(previous_times, advanced_times, loop=True)  # (N, 4, 4)
character_requests = motion.character_delta(previous_times, advanced_times, loop=True)
in_place_pose = motion.in_place(*sample_clip(clip, advanced_times, loop=True))
```

Each float64 matrix describes the rigid transform from the root bone's starting
frame to its ending frame, using column vectors. For a character transform that
already represents the starting root frame, apply a request on the right:
`desired = current @ requests[i]`. If the character frame differs from the
root frame, convert between them before applying the delta. Physics may accept a
different displacement; the caller owns that decision and any reconciliation.

Times are finite one-dimensional arrays of equal length. Looping times can be
unwrapped in either direction; the extractor composes the clip's complete root
transform for every crossed cycle, including rotation. With `loop=False`, times
clamp to the clip endpoints. The selected bone must be a skeleton root and its
clip scale must remain unit scale.

`in_place` returns copied local pose arrays with root translation, rotation and
scale replaced by the skeleton's reference root; other bones are unchanged.
The root scale in both the supplied pose and the reference pose must be unit.
Use the returned pose with `model_matrices` and `skin_matrices` as usual.
`character_delta` conjugates the root-frame request by that same reference root,
so an in-place character can apply it as `desired = current @ character_requests[i]`.
The initial character transform must already align the reference root with the
starting animated root. The caller applies accepted movement and reconciles any
rejected displacement. The standalone API leaves crossfade motion blending to
the caller; the ECS adapter provides the policy described below.

For CPU material rendering, pass the deformed vertices to `MaterialRenderer.update_mesh_vertices`, or use `MaterialSkinningAdapter` to connect poses to independent material meshes. The material renderer can alternatively consume the GPU palette below, allowing independently animated owners to share a mesh. See the [material guide](materials.md#gpu-skinning) and [Greenman example](../scenes/animation_example.py).

## GPU skin matrices

`SkinMatrixBuffer(capacity)` owns one fixed-size GPU storage buffer of row-major
3x4 float32 matrices. With an initialized RenderContext, allocate one contiguous
range per persistent owner and write batches from the CPU evaluator:

```python
from graaal.SkinMatrixBuffer import SkinMatrixBuffer

palette = SkinMatrixBuffer(capacity=256)  # matrix slots, not characters
base = palette.acquire(entity, len(skeleton.parents))
palette.write(batch.entities, batch.matrices)
# A shader can read palette.buffer at base + joint_index.
palette.release(entity)
palette.close()
```

Allocate a range for every owner in a batch before writing it. The capacity is
fixed so bindings remain stable; allocation requires a contiguous free range and
can fail due to fragmentation. `range(owner)` returns its base and bone count.
Releasing coalesces free ranges but does not clear their contents; write a newly
acquired range before a shader reads it. The caller must use full generational
handles for ECS owners and ensure prior GPU work has finished using a range
before releasing or overwriting it. Validate indices and normalized weights
before dispatching shaders.

The reusable `graaal/shaders/skinning.shaderlib` defines `SkinMatrix` and
`skin_vertex`: pass four palette entries and their weights to obtain a position
and inverse-transpose normal. For a blended linear transform whose determinant
has magnitude at most `1e-8`, the GPU helper returns a zero normal; the CPU
helper raises for a singular matrix. GPU palette allocation and this shader
helper are also available for custom renderers. The material renderer uses this
layout in its vertex and transparent reconstruction shaders.

For automatic allocation and release from complete pose snapshots, use
`PosePalette`; see the [material guide](materials.md#gpu-skinning) for draw
ordering and lifetime requirements. The lower-level palette API remains usable
directly for custom pipelines.

## Skinned glTF import

`load_skinned_gltf(path, node=None, fps=30)` from `graaal.SkeletalGltf` loads
CPU data without an ECS or graphics device. It returns a `SkinnedAsset` with a
`skeleton`, `primitives`, named `clips`, and `joint_nodes` in skeleton order.
Each primitive has indexed vertices with `position`, `normal`, `uv`, `joints`
and `weights`, plus its original `material_id`. Assign materials separately;
this importer does not load material definitions or textures.

```python
from graaal.SkeletalGltf import load_skinned_gltf

asset = load_skinned_gltf("character.glb")
primitive = asset.primitives[0]
name, clip = asset.clips[0]
# Use asset.skeleton, primitive.vertices and primitive.indices with the
# animation and material APIs below.
```

The loader chooses the only skinned mesh node, or uses the explicit node index.
It reorders joints to depth-first preorder and remaps vertex joint indices and
inverse binds. LINEAR TRS tracks are resampled at at least the requested rate,
with the exact clip endpoint retained; quaternion tracks use shortest-path
slerp during resampling. Missing tracks use the node's reference transform.

The supported subset requires triangle lists, normals, four or fewer influences
in `JOINTS_0`/`WEIGHTS_0`, and TRS joint nodes. Mesh nodes and non-joint
ancestors must have identity transforms; non-joint nodes between joints,
animated ancestors, morph targets, additional influence sets, and STEP or
CUBICSPLINE tracks are rejected. Missing UVs become zero. Export or preprocess
assets outside this subset before loading them.

## ECS playback

`EcsAnimationAdapter` is optional; the CPU evaluator does not depend on ECS.
Register `SkeletonRef` and `ClipPlayer`, then supply mappings from their numeric
IDs to shared skeleton and clip resources:

```python
from graaal.ECS import ECS
from graaal.EcsAnimation import SkeletonRef, ClipPlayer, EcsAnimationAdapter

world = ECS()
world.register(SkeletonRef, ClipPlayer)
entity = world.create()[0]
world.add(entity, SkeletonRef(0), ClipPlayer(0, speed=1.0, loop=True))
animation = EcsAnimationAdapter(world, {0: skeleton}, {0: clip})

# Advance after gameplay changes; use batch.entities to associate poses with draws.
for batch in animation.update(dt):
	print(batch.entities, batch.matrices)  # (N,) full handles and (N, bones, 3, 4)
```

Playback advances in seconds. Negative speeds play backwards, `playing=False`
pauses, and editing `time` seeks. Looping wraps at clip duration; non-looping
clamps. Zero-duration clips stay at time zero. Each update normalizes player times
and returns batches grouped by skeleton and clip, with full generational entity
handles in the same order as the matrices. The adapter reuses pose and matrix
storage by skeleton and source clip, so consume or copy each batch's matrices
before the next call to `update` or `evaluate`, including calls that fail. Entity
arrays are copies; refetch after membership changes and do not identify an entity
by its component row. `clear_pose_buffers()` drops the adapter's cached storage;
future updates allocate it again. The
caller must map these handles to its renderables and ensure a clip uses the
specified skeleton's bone order. Unknown resource IDs or incompatible bone counts
raise before player times are changed.

Start a crossfade by assigning `transitioning=True`, `target_clip_id`,
`target_time`, `target_speed`, `target_loop`, `transition_time=0`, and a positive
`transition_duration` to the player. Both clips advance while playing; pausing
freezes their times and the fade. Fade weights progress linearly in wall-clock
seconds, independently of playback speeds. On completion the target clip and its
time, speed and loop setting replace the source, and `transitioning` becomes
false. An update that crosses the end advances the target for the whole update.
To restart or replace a fade, assign all target fields and reset transition time.
The batch's `clip_id` is the source ID at the start of that update, even if the
fade completes during it. Pose matrices already contain the blended result and
can be uploaded to the material renderer's skin palette as usual.

## Animation events

`graaal.AnimationEvents.EventTrack` stores named markers independently of ECS:

```python
from graaal.AnimationEvents import EventTrack

track = EventTrack(clip.duration, [0.2, 0.7], ["left_step", "right_step"])
crossings = track.between(0.6, 1.3, loop=True)
animation = EcsAnimationAdapter(world, {0: skeleton}, {0: clip}, event_tracks={0: track})
animation.update(dt)
for event in animation.events:
	print(event.entity, event.clip_id, event.name, event.role)
```

Tracks require a positive duration matching their clip and marker times in
`[0, duration)`. For a marker at the loop boundary use time zero. Traversal
excludes its start and includes its end in either direction, preserving marker
order at equal times. Looping emits every crossing, including multiple cycles;
non-looping clamps to the clip endpoints. A seek does not emit events by itself.
`EventOccurrence.time` is the unwrapped crossing time in the supplied interval.
`AnimationEvent.time` uses the player's normalized starting cycle, then remains
unwrapped for that update.

The adapter replaces `events` on every successful update; consume it after each
update, or after each scheduler evaluation. A paused player produces none. During
a crossfade, both source and target emit events regardless of blend weight. If
the fade ends within an update, the source emits only up to the fade's end and
the target emits for the full update. `role` identifies the clip as `source` or
`target`; after promotion it is `source` on the next update. Events are grouped
by pose batch and clip role, with playback order preserved for each entity and
role; there is no global chronological ordering across entities or roles.
For an executable ECS scheduler example, see
[animation_events_example.py](../scenes/animation_events_example.py).

For fixed-step scheduling, register `animation.evaluate` as a system; its latest
result is `animation.batches`. The application owns the fixed-step accumulator and
must order consumers after the animation system. For a complete material example,
see [animation_example.py](../scenes/animation_example.py).

For root-motion playback, provide a `RootMotion` for each `(skeleton_id, clip_id)`
pair and set `extract_root_motion=True` on the players that need it:

```python
motion = RootMotion(skeleton, clip)
animation = EcsAnimationAdapter(world, {0: skeleton}, {0: clip}, root_motions={(0, 0): motion})
world.get(ClipPlayer)[[entity]].assign(extract_root_motion=True)
pose_batches = animation.update(dt)
for batch in animation.root_motion_batches:
	for owner, request in zip(batch.entities, batch.matrices):
		pass  # Submit the character-frame request to movement or physics.
```

The corresponding pose batches already contain in-place roots and can be sent
to the material palette. Each request batch contains full entity handles and
character-frame float64 matrices. The adapter replaces its request batches on
every successful update; paused players receive identity requests. It does not
apply movement, and the application must account for physics-accepted movement.
The `RootMotion` must reference the same skeleton and clip objects as the
adapter. A crossfade requires a matching `RootMotion` for both clips, using the
same root bone. During the portion of a tick inside the fade, the adapter blends
the source and target character-frame movement deltas using the midpoint fade
weight: translations blend linearly and rotations use shortest-path normalized
quaternion interpolation. If the fade ends during the tick, the target's
remaining movement is composed after that blended delta. The pose stays in place
throughout, and the target is promoted normally.

This is a fixed-step approximation, not a timestep-independent integration of
motion curves. For rapidly changing or turning motion, use sufficiently short
fixed steps and validate the resulting movement; rotations spanning more than a
half turn in one step blend along the shorter arc. Applications needing another
policy can use the standalone extractor and handle the transition themselves.
