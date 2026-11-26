# Skeletal animation

This reference covers standalone evaluation, optional ECS playback, graph
definitions, events, IK and GPU palette output. See the
[scene catalog](../scenes/README.md) for runnable examples and the
[tool guides](../tools/README.md) for timeline and graph editing.

CPU animation modules live in `graaal.Animation`; for example,
`from graaal.Animation.SkeletalAnimation import Skeleton, Clip, sample_clip`.
ECS playback adapters live in `graaal.EcsAdapters`; GPU palette buffers live in
`graaal.RenderContext.SkinMatrixBuffer`.

`graaal.Animation.SkeletalAnimation` is independent of ECS and graphics. `Skeleton` takes a depth-first preorder forest of parent indices (`-1` for roots), local reference translation, xyzw rotation and scale, and affine inverse-bind matrices. Resources copy their inputs into read-only float32 arrays. Matrices act on column vectors; local transforms compose as translation, rotation, then scale.

`Clip` accepts equally spaced local TRS frames for every bone and a positive frame rate. Its duration is `(frames - 1) / fps`; looping wraps at that endpoint, so matching first and last frames are needed for a seamless loop. `sample_clip(clip, times, loop=True)` takes a one-dimensional array of seconds and returns batched translation, normalized quaternion rotation and scale. Set `loop=False` to clamp. The caller must match clips to skeletons with the same bone order and count.

Quaternion operations use xyzw arrays and are vectorized across instances and
bones through `graaal.Math.Quaternion`. Single-quaternion methods and
`multiply_batch`, `nlerp_batch`, `slerp_batch`, `matrix_batch`
and `from_matrices_batch` operate on NumPy arrays. Interpolation weights must
broadcast against the quaternion arrays, typically with a trailing singleton
dimension. Batch matrix methods use column-vector rotations. glTF track
resampling uses slerp before clips are uniformly baked,
while runtime interpolation uses nlerp.

`sample_crossfade(source, target, source_times, target_times, weights)` samples two clips and blends their local poses. Times are one-dimensional arrays of equal length, and weights are a scalar or one value per instance in `[0, 1]`. Each clip can loop or clamp independently with `source_loop` and `target_loop`. Translation and scale blend linearly; rotation uses shortest-path normalized linear interpolation. Both clips must have the same bone order and count. The caller controls transition timing and weights.

For an independent layer, `subtree_mask(skeleton, root)` selects a bone and its
depth-first descendants. `blend_poses(skeleton, base, overlay, weight, mask=mask)`
blends two batched local TRS pose tuples with a scalar or per-instance weight and
a per-bone mask of values in `[0, 1]`. The effective weight is their product;
translation and scale blend linearly, and rotation uses shortest-path normalized
quaternion interpolation. A missing mask selects every bone. Inputs are not
modified. Hierarchical transforms still propagate normally from each blended
local bone to its children. The result can be passed to `model_matrices` and
`skin_matrices` for the existing material palette.

`additive_pose(skeleton, base, overlay, weight, mask=mask)` applies the overlay
relative to the skeleton's local reference pose. Translation and scale use the
weighted difference from that reference; rotations compose the base with a
shortest-path, normalized interpolation from identity to
`inverse(reference_rotation) * overlay_rotation`. Scale is an additive offset,
not a multiplicative ratio, so zero reference scales are supported. The same
weight, mask, shape and ownership rules as `blend_poses` apply. The caller must
provide an overlay authored relative to this reference and manage root motion
separately.

`sample_clip_pairs(skeleton, clips, clip_ids, times, weights)` evaluates a batch
whose instances may each use a different pair of clips. `clips` maps integer IDs
to clips; the three input arrays have shape `(instances, 2)`. Weights must be
finite, lie in `[0, 1]`, and sum to one per instance. Times are in seconds and
clamp at clip endpoints; the controller handles wrapping before sampling. Both
clip IDs must exist and use the skeleton's bone order and count, even if one
weight is zero. The return value is the same local TRS pose tuple as
`sample_crossfade`, in the original instance order. For root-motion players,
apply the appropriate `RootMotion.in_place` after sampling. An optional `out`
tuple supplies writable, non-overlapping, C-contiguous float32 pose arrays.

`blend_space_1d(positions, clip_ids, values)` selects the neighboring clips and
linear weights for a batch of parameter values. Positions are finite, strictly
increasing axis coordinates, with one nonnegative integer clip ID per position.
Values outside the axis clamp to its endpoints; a single-position space always
selects that clip. The two returned arrays have shape `(instances, 2)` and can
be passed directly to `sample_clip_pairs` with separately managed clip times:

```python
from graaal.Animation.SkeletalAnimation import blend_space_1d, sample_clip_pairs

ids, weights = blend_space_1d([0, 1, 3], [idle_id, walk_id, run_id], speeds)
pose = sample_clip_pairs(skeleton, clips, ids, times, weights)
```

`times` contains the caller's wrapped or clamped time for each selected clip.
This selector does not advance playback, synchronize clips, blend root motion,
or dispatch events; applications must manage those separately. Clip IDs must
exist even for a zero-weight endpoint sample.

`blend_space_2d(x_positions, y_positions, clip_grid, values)` selects from a
rectangular grid, with `clip_grid[y, x]` and `(instances, 2)` input values.
Axes are strictly increasing and values clamp at their endpoints. It returns
IDs shaped `(instances, 2, 2)`, horizontal pair weights shaped `(instances, 2)`,
and vertical blend weights shaped `(instances,)`. A single-point axis is valid.
Use the two rows as sample slots in a `PosePlan`, then blend them vertically:

```python
from graaal.Animation.SkeletalAnimation import blend_space_2d

ids, horizontal, vertical = blend_space_2d(x_axis, y_axis, clip_grid, parameters)
samples = [(ids[:, row], times[:, row], horizontal) for row in range(2)]
pose = plan.evaluate(clips, samples, vertical[:, None])
```

Here `plan` has two sample tasks followed by a blend task using weight slot 0,
and `times` is `(instances, 2, 2)` in the same order as `ids`. Rotations are
blended horizontally and then vertically using the existing shortest-path
normalized interpolation; this order is part of the result. Playback, events,
and root motion remain the caller's responsibility.

## Compiled pose tasks

`PosePlan` compiles a fixed sequence of batched local-pose operations. Each
sample task evaluates one clip-pair sample slot; blend and additive tasks refer
to earlier task indices and use a column of per-instance layer weights. An
optional per-bone mask is copied when the plan is created.

```python
from graaal.Animation.PoseTasks import PosePlan
from graaal.Animation.SkeletalAnimation import model_matrices, skin_matrices, subtree_mask

plan = PosePlan(skeleton, [
	("sample", 0),
	("sample", 1),
	("blend", 0, 1, 0, subtree_mask(skeleton, upper_body_bone)),
])
samples = [(base_ids, base_times, base_weights), (upper_ids, upper_times, upper_weights)]
pose = plan.evaluate(clips, samples, layer_weights)  # layer_weights: (instances, 1)
matrices = skin_matrices(skeleton, model_matrices(skeleton, *pose))
```

Task forms are `("sample", sample_slot)` and
`("blend", base_task, overlay_task, weight_slot, mask)` or
`("additive", base_task, overlay_task, weight_slot, mask)`; `mask` can be
`None`. An `("in_place", base_task, motion_slot)` task anchors selected
instances' root to the reference pose before later operations. An
`("input", pose_slot)` task copies a supplied pose into the plan, and
`("two_bone_ik", base_task, chain, goal_slot, weight_slot)` applies IK.
`("align_axis", base_task, bone, goal_slot, weight_slot, local_axis)` aligns
a bone axis to a direction in model space.
Sample triples use the same arrays and validation as
`sample_clip_pairs`. Every layer weight must be finite and in `[0, 1]`.
Operation names can also be supplied as `PoseOp` string-enum members, such as
`PoseOp.SAMPLE` or `PoseOp.TWO_BONE_IK` from `graaal.Animation.PoseTasks`. The plan
validates and stores enum members when it compiles; plain strings remain valid.
The last task is the output. The plan reuses pose buffers based on task
dependencies; its returned arrays are views that must be consumed or copied
before its next evaluation, even if that evaluation fails. Different plans
own separate buffers. Blend operations still allocate intermediate arrays.
The plan does not advance playback, generate events or root motion, schedule
physics, or implement a state machine; applications supply the dynamic inputs.

For a pre/post-physics split, use two plans and pass the pre-physics output
into the post-physics plan after obtaining model-space physics goals:

```python
import numpy as np

pre = PosePlan(skeleton, [("sample", 0)])
post = PosePlan(skeleton, [("input", 0), ("two_bone_ik", 0, (hip, knee, ankle), 0, 0)])
base = pre.evaluate(clips, samples, np.empty((count, 0)))
# Run physics, then convert its targets and poles to the skeleton's model space.
pose = post.evaluate({}, [], ik_weights, poses=[base], goals=[(targets, poles)])
```

Each pose input is a local TRS tuple; inputs are copied before plan tasks run,
including when an input aliases a previous result of the same plan. Each goal
slot contains target and pole arrays shaped `(instances, 3)`. IK weights are
columns in the ordinary weights array. Axis-alignment goal slots contain a
one-element tuple with a direction array shaped `(instances, 3)`. See
`two_bone_pose` below for scale and
orientation limits. The application orders plans and physics and must keep the
pre-physics pose alive until the post-physics call; no scheduler or physics
backend is invoked by `PosePlan`.
The headless [foot_ik_example.py](../scenes/foot_ik_example.py) demonstrates
this order with root-motion requests and a synthetic terrain query.

For an in-place task, pass `root_motions=[(motion, instance_mask), ...]` to
`evaluate`. Each mask is a boolean array aligned with the sample instances;
`motion` must be a `RootMotion` using the plan's skeleton, or `None` when the
mask is all false. The operation uses `RootMotion.in_place`, including its root
scale requirements. When consuming an `EcsAnimationAdapter` sample batch,
`animation.in_place_motion(batch)` supplies a slot and checks that all selected
source and target clips have compatible root-motion resources. The adapter's
`root_motion_batches` remain movement requests for the application or physics
system; this pose task only anchors the visual pose. Apply the task before any
layer that should build on the in-place base.

## Editable graph definitions

`AnimationGraphDefinition` in `graaal.Animation.AnimationGraph` owns a copied, editable
dictionary in `definition.data`. `definition.compile(skeleton, named_clips)`
validates the draft and returns a `CompiledAnimationGraph` containing the existing
`AnimationStateMachine` and `PosePlan`. It needs neither ECS nor graphics. Edits
and failed compilation leave previously compiled resources unchanged; applying a
new graph to live instances remains application-owned. This API performs no file I/O.

```python
from graaal.Animation.AnimationGraph import AnimationGraphDefinition

definition = AnimationGraphDefinition({
	"version": 1,
	"states": {"idle": {"clip": "idle"}},
	"initial_state": "idle",
	"sample_slots": ["base"],
	"tasks": [{"name": "base", "op": "sample", "slot": "base"}],
})
graph = definition.compile(skeleton, {"idle": idle_clip})
```

Version 1 fields are:

| Field | Authoring value |
| --- | --- |
| `version` | Required integer `1`. |
| `parameters` | Optional ordered name-to-finite-default mapping. |
| `states` | Required nonempty ordered mapping of state names to `{"clip": clip_name}`, `{"blend_space": named_1d_space}` or `{"blend_space_2d": named_grid}`. Each state has exactly one source. |
| `initial_state` | Required state name. |
| `transitions` | Optional ordered list of `{source, target, conditions, duration}` mappings. Conditions are `{parameter, comparison, threshold}` mappings; all must match. Fade durations must fit positive float32 playback storage. |
| `sample_slots` | Required list of unique sample-slot names, populated by the caller. |
| `weight_slots` | Optional list of parameter names used as layer weights; their defaults must be in `[0, 1]`. |
| `root_motion_slots` | Optional list of names for caller-supplied in-place motion slots. |
| `masks` | Optional named masks, either per-bone weights in `[0, 1]` or `{"root": bone_index}` subtree definitions. |
| `tasks` | Required nonempty ordered task list; the final task is the output. |
| `sync_mappings` | Optional list of `{source, target}` clip-name pairs; supplied marker tracks compile into `SyncMapping` resources. |
| `blend_spaces` | Optional named 1D selectors: `{parameter, points: [{position, clip}, ...]}`. Positions must increase strictly. Optional `sync_reference` names a participating clip for marker synchronization. |
| `blend_spaces_2d` | Optional named rectangular grids: `{x_parameter, y_parameter, x_positions, y_positions, clips}`. `clips` has one row per Y point and one clip per X point. Optional `sample_slots` and `weight_slot` must be supplied together to bind the grid to pose-plan samples. Optional `sync_reference` names a participating clip. |

Each task has a unique `name` and a `PoseOp` or string `op`. `sample` names a
sample `slot`; `in_place` names an earlier `base` task and a root-motion `slot`.
`blend` and `additive` name earlier `base` and `overlay` tasks, a weight-slot
`weight`, and an optional named `mask` (`None` selects all bones). Forward and
cyclic task references are rejected. Errors identify the editable field/task.
Unknown fields and unsupported operations fail rather than being ignored.

Insertion order determines IDs and columns within a compiled result. Use
`graph.state_ids`, `clip_ids`, `parameter_columns`, `sample_slots` and
`root_motion_slots` to bind application data. `state_clips` contains only clip
states and feeds `EcsAnimationStateAdapter` for clip-only graphs.
`state_blend_spaces` maps blend-state IDs to names in `graph.blend_spaces`;
`state_blend_spaces_2d` does the same for `graph.blend_spaces_2d`. These maps
are separate, so 1D spaces and 2D grids can share names. The existing ECS state
adapter does not play blend states. `graph.clips` maps compiled clip IDs to the
original shared `Clip` objects. All supplied clips must match the skeleton's bone
count; the caller must also ensure matching bone order. Compiled metadata is read-only.

Initialize parameter rows with `np.tile(graph.parameter_defaults, (count, 1))`
and pass them to `graph.machine.select` or the existing state adapter. Fill the
sample and motion slots in their compiled order, then call
`graph.plan.evaluate(graph.clips, samples, parameters[:, graph.weight_columns], ...)`.
That output uses the existing model/skin matrix and material-palette path.
Compiled plans own reusable pose buffers with the usual `PosePlan` lifetime.
For externally composed state poses, `graph.input_plan` has the same authored
operations with sample tasks replaced by `PoseOp.INPUT`. Supply local TRS tuples
in sample-slot order through `poses=...`, and pass an empty sample list:
`graph.input_plan.evaluate(graph.clips, (), weights, poses=local_poses, ...)`.
This lets a controller compose each state's selected clips, crossfade the two
state poses, then apply the authored masks, in-place operations and additive
layers. Both plans own separate reusable buffers.
See [animation_graph_example.py](../scenes/animation_graph_example.py) for an ECS
locomotion controller and an independent masked upper-body layer compiled from
one definition.

### Batched runtime graph playback

`graaal.Animation.AnimationGraphPlayback.AnimationGraphPlayback` owns a fixed
batch of independent clocks for a compiled graph, without ECS or graphics:

```python
from graaal.Animation.AnimationGraphPlayback import AnimationGraphPlayback

player = AnimationGraphPlayback(graph, count)
parameters = np.tile(graph.parameter_defaults, (count, 1))
player.advance(dt, parameters)
local_pose = player.pose(parameters)
# For a graph with a single sample slot:
output = graph.input_plan.evaluate(graph.clips, (), parameters[:, graph.weight_columns], poses=[local_pose])
```

Clip, 1D and 2D states can coexist in one batch. Selection, sampling and fades
operate on groups of compatible states rather than looping over characters.
2D poses blend horizontally, then vertically, before the state fade. The
returned local TRS arrays own their storage and survive later player calls.
Supply additional layer poses in compiled sample-slot order, then apply masks,
additive layers and in-place operations through `input_plan`. Plans retain their
usual reusable-output lifetime; root-motion inputs remain application-owned.
See [animation_graph_batch_example.py](../scenes/animation_graph_batch_example.py)
for synchronized locomotion with a masked layer and final skin matrices.

`advance` accepts a nonnegative scalar `dt` or a `(count,)` array, allowing
individual rows to pause with zero time steps. A zero step can still select an
edge. Parameters must have shape `(count, len(graph.parameter_defaults))`;
omitting them uses the compiled defaults. `pose` uses the parameters supplied
to that call, so live blend changes need no clock reset.

`reset(states=None, times=0, *, speeds=1, rows=None)` seeks all or selected unique
integer row indices, cancels their fades and sets nonnegative playback speeds.
States are compiled integer IDs; omitted states use `graph.initial_state`.
Each value can be scalar or aligned with the selected rows. Clip clocks wrap;
blend clocks retain unwrapped seconds. Speed changes clip time, while fades
advance in `dt` seconds. All clips loop; reverse playback is not supported by
this controller.

`states`, `times`, `speeds`, `target_states`, `target_times`, `target_speeds`,
`elapsed` and `durations` are read-only array snapshots. A target state of `-1`
means no active fade. Failed advances/resets preserve every row. Each advance
selects at most one edge per row, never interrupts an active fade and promotes
without restarting target time. Explicit marker mappings use clip or blend
reference clocks; transition sync ends at promotion, retaining the endpoint
target speed. Fixed-reference synchronization inside a blend remains active.

Pass `root_motions={clip_id: RootMotion(rig, clip)}` for every clip used by a
state. All tracks must use the same skeleton root. Each advance publishes
read-only character-space deltas in `player.root_motion`. A blend uses the
selected clips' actual clock intervals and weights. A fade blends interval
deltas at its mean progress and composes any remaining target motion after
promotion. `pose(parameters, in_place=True)` holds that root at its reference
TRS. Use the raw pose with an authored in-place task when layer order matters.

To bind a compiled graph to ECS, register `GraphPlayer` from
`graaal.EcsAdapters.EcsAnimationGraph` and add `GraphPlayer(graph_id)` to each
owner. Create one `EcsAnimationGraphAdapter(world, graph_id, graph, ...)` per
graph, passing its optional root motion and event tracks. Supply parameters in
`adapter.entities()` order and call `adapter.update(dt, parameters)` to get
local TRS in that order. Surviving handles retain their state and clocks when
the query changes. `adapter.events` publishes full entity handles, while
`adapter.playback.root_motion` publishes requested movement aligned with
`adapter.owners`, the last update's handle snapshot.
Use `adapter.advance(dt, parameters)` to publish the same clocks, root motion
and events when evaluating poses separately. It returns the entity events.
The caller uses native controller or gameplay results for accepted motion.
See the [character example](../scenes/character_example.py).

To receive gameplay events, pass `event_tracks={state_id: track}` when creating
the player. Use `graph.state_ids` to look up compiled IDs. The player snapshots
each `EventTrack` independently of the tracks passed to `compile(sync_tracks=...)`.
For clip states, use the clip duration. For synchronized blends, use the
reference clip duration. For unsynchronized blends, supply a positive event cycle
duration in state-clock seconds. Each state reads its own track as parameters
change which clips contribute to the pose.

```python
moving = graph.state_ids["moving"]
player = AnimationGraphPlayback(graph, count, event_tracks={moving: moving_events})
for event in player.advance(dt, parameters):
	print(event.row, event.state, event.name, event.offset, event.role, event.weight)
```

`advance` returns an immutable tuple of `GraphEvent` records. `offset` is seconds
from the start of the tick and `weight` is the state's fade weight at that time.
Records are ordered by row and tick offset, with source first at equal offsets
and authored order at ties within a track. Traversal excludes the interval start
and includes its end. The player keeps the event cursor continuous across clip
wraps. During synchronized fades, matching markers share a tick timestamp.
Resets and zero steps produce no crossings. Both states emit during fades,
and the target retains its role for the rest of the tick
after promotion. The application decides how to handle overlapping events.

### Named blend selectors

Each `graph.blend_spaces[name]` stores a parameter column and an ordered clip
axis. Call `ids, weights = graph.blend_spaces[name].select(parameter_rows)` with
finite `(instances, parameter_columns)` rows. The existing `blend_space_1d`
selector clamps values at the endpoints and returns two clip IDs and weights per
instance. Supply those IDs and weights, plus caller-owned clip times, as a
sample-slot triple to `graph.plan.evaluate`. This selector owns neither playback
nor event traversal; layer masks still apply in the pose plan. See the graph
tool's sample bindings for a live preview example.

A `blend_spaces_2d` entry can bind two distinct sample slots and one existing
`weight_slot`, or omit both bindings for state evaluation. Its compiled selector
returns `ids, horizontal, vertical =
graph.blend_spaces_2d[name].select(parameter_rows)`, using the existing
`blend_space_2d`. Supply `ids[:, 0]` and `ids[:, 1]` as the two sample-slot ID
pairs, each with `horizontal` weights and caller-owned times. Set the named
weight-slot column to `vertical`, then blend the two sample tasks in the pose
plan. The row ordering follows increasing Y positions. X and Y parameters must
be distinct; axes must be finite and strictly increasing. The named weight-slot
parameter is a placeholder for standalone graph evaluation and must have a
default in `[0, 1]`; its preview value is replaced by the selector's vertical
weight when the grid is bound. This leaves the existing pose evaluator in charge
of the final local-pose blend. Each grid with sample bindings must use a distinct
weight slot; compilation rejects shared slots so simultaneously active grids
cannot overwrite one another's vertical blend weight.
For an unbound grid, compiled `sample_slots` is `()` and `weight_slot` is `None`;
selection still returns the same clip IDs and horizontal/vertical weights.
2D state playback blends the two horizontal pairs, then blends those row poses
vertically before crossfading complete state poses. State evaluation uses its
own weights and does not change authored layer-weight parameters, including when
the referenced grid also declares sample bindings.

Either kind of blend space can name `sync_reference` and receive
`sync_tracks` through `compile`. The reference must be a participating clip;
every distinct clip in the space needs a compatible marker track with matching
duration, including clips not currently selected. Invalid tracks identify the
space's `sync_reference` field and leave prior compiled resources unchanged.
The compiled selector's `sync` is a standalone `BlendSpaceSync`, or `None` when
no reference is authored. After selection, call
`times, rates = space.sync.map(ids, reference_times)` to obtain sample times.
Weights do not change the reference clock; reselecting clips or crossing grid
cells keeps the same phase source. See the synchronization API below.

For synchronized clip transitions, request `sync_mappings` in the definition
and supply `sync_tracks={clip_name: event_track, ...}` to `compile`. Each requested
pair needs distinct clips and marker tracks whose durations exactly match the
runtime clips. The existing `SyncMapping` validates marker count, unique names,
distinct times and cyclic ordering. Errors identify the pair and, for missing
tracks or mismatched durations, its source or target. Unrequested tracks do not
participate in compilation. Mappings are directional and keyed by compiled clip
IDs in `graph.sync_mappings`. Pass that same mapping to both
`EcsAnimationStateAdapter(..., sync_mappings=graph.sync_mappings)` and
`EcsAnimationAdapter(..., sync_mappings=graph.sync_mappings)`; synchronized players
must loop. For editable tracks, compile from `EventTimeline.snapshot()[1]` so
subsequent edits stay separate from the existing runtime mapping.

This authoring version supports single-clip and named 1D/2D blend states, blend-space
sample bindings, in-place root anchoring, masked blend/additive layers and
clip-pair sync settings. IK/phase authoring is not yet represented. The
[graph tool](../tools/animation_graph.md) provides a definition editor,
transition/condition forms, state diagram, manual layered-pose inspection and
automatic clip/1D/2D blend-state playback using this API. Unsynchronized blend
states loop their clips independently. Synchronized spaces interpret the state
clock in reference-clip seconds and map each selected clip's marker phase.
Explicit `sync_mappings` between state clock clips (a clip state or a blend's
reference) also synchronize editor state fades. ECS graph playback remains
application-owned. The standalone tools share
explicit [JSON project persistence](../tools/animation_project.md) for graph
definitions, events and sync markers. Runtime
blend-space, sync and IK APIs remain available for application-owned composition.

## Conditional state transitions

`AnimationStateMachine` selects transitions from per-instance numeric
parameters without owning playback. Each edge is `(source, target,
parameter_column, comparison, threshold, fade_duration)`. Comparisons are
`<`, `<=`, `>`, `>=`, `==` and `!=`; the first matching edge wins, so edge
order defines priority. State IDs are nonnegative integers and durations must
be positive.

For a transition that needs several conditions, use `(source, target,
conditions, fade_duration)`, where `conditions` is a nonempty list of
`(parameter_column, comparison, threshold)` tuples. Every condition must
match. For example, this edge requires both movement and ground contact:

```python
(0, 1, [(0, ">", 0.2), (1, "==", 1)], 0.3)
```

Each condition reads a column in the same per-instance parameter array. Use
separate edges for alternative conditions; edge order still defines priority.

```python
from graaal.Animation.AnimationStateMachine import AnimationStateMachine

machine = AnimationStateMachine([(0, 1, 0, ">", 0.2, 0.3), (1, 0, 0, "<=", 0.1, 0.3)])
selected, target_states, durations = machine.select(states, speeds[:, None], transitioning)
```

The returned arrays align with the input instances. Unselected rows retain
their current state and have zero duration; target IDs use `uint64`. Active transitions cannot be
interrupted. The application stores current and target states, maps states to
clips or blend spaces, starts the existing player crossfade for selected rows,
and promotes the target state when that crossfade completes. It also owns
playback time, events and root motion. This selector is independent of ECS and
does not itself update a `ClipPlayer` or `PosePlan`.

For single-clip states, `EcsAnimationStateAdapter` connects this selector to
`ClipPlayer`. Register `AnimationState` with the ECS and attach it alongside
`ClipPlayer` and `SkeletonRef`; initialize the state and player clip to match
the supplied state-to-clip mapping.

```python
from graaal.EcsAdapters.EcsAnimationStateMachine import AnimationState, EcsAnimationStateAdapter

control = EcsAnimationStateAdapter(world, machine, {0: idle_id, 1: walk_id})
entities = control.query.fetch().entities
parameters = speeds_for_entities(entities)[:, None]
control.update(parameters)  # before animation.update(dt)
animation.update(dt)
```

Parameter rows must follow the current query's entity order, with no structural
ECS changes between fetching that order and `update`. Without a sync mapping,
the adapter initializes target time to zero and inherits the source player's
speed and loop setting.
The existing playback adapter advances the fade and handles events and root
motion. On the next control update after completion, the target state is
promoted; a new transition may begin immediately. Paused players retain their
pending state. A transition must remain owned by this adapter until completion;
external changes to its source or target clip are rejected. The application
must supply valid clips and any event or root-motion resources to playback.
Control and playback updates are separate operations; a playback error does not
undo a transition already started by control. Blend-space states still need
an application-specific controller.

For a complete headless composition of this controller, `PosePlan`, a bone
mask and skin matrices, see [animation_graph_example.py](../scenes/animation_graph_example.py).
The synthetic example does not enable root extraction, but its pose plan includes
the in-place task and obtains its slot from each controller sample batch.

`model_matrices(skeleton, translation, rotation, scale)` evaluates batches of local poses, and `skin_matrices(skeleton, model)` returns contiguous `(instances, bones, 3, 4)` row-major float32 matrices representing `model @ inverse_bind`. `skin_vertices(positions, normals, joints, weights, matrices)` applies one pose to a mesh on the CPU, using the inverse transpose of each blended linear transform for normals. Joint weights must sum to one; singular blended transforms raise `ValueError` when transforming normals. Pass `normals=None` to skip normal transforms and return `(positions, None)`; this supports collapsed geometry in position-only previews.

The sampling, model and skin functions accept optional `out` buffers. For
`sample_clip`, supply a tuple of translation, rotation and scale arrays;
the other functions take one output array each. Buffers must be writable,
C-contiguous float32 arrays with the exact corresponding output shapes; pose
output arrays must not overlap. These buffers avoid allocating the returned
arrays, although evaluation still creates intermediate arrays.

## Root-motion extraction

`RootMotion` extracts requested movement independently of ECS or physics:

```python
from graaal.Animation.RootMotion import RootMotion
from graaal.Animation.SkeletalAnimation import sample_clip

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

For CPU material rendering, pass the deformed vertices to `MaterialRenderer.update_mesh_vertices`, or use `MaterialSkinningAdapter` to connect poses to independent material meshes. The material renderer can alternatively consume the GPU palette below, allowing independently animated owners to share a mesh. See the [material guide](materials.md#gpu-skinning) and [Greenman example](../scenes/simple_animation_example.py).

## GPU skin matrices

`SkinMatrixBuffer(capacity)` owns a fixed-size GPU storage buffer of row-major
3x4 float32 matrices. With an initialized RenderContext, allocate one contiguous
range per persistent owner and write batches from the CPU evaluator:

```python
from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer

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

For a batch whose owners occupy scattered ranges, use
`palette.record_write(commands, owners, matrices)`. It uploads the poses
together and records one copy per contiguous palette range. Record the copies
before any pass that needs those matrices. The palette owns a reusable default
staging buffer. To record several batches before submission, pass a distinct
`GpuBuffer` with `COPY_SRC | COPY_DST` usage as `staging=` for each additional
batch. Submit before reusing a staging buffer or closing it or the palette.

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

### Crowd pose compute pass

`BakedClipPool(rig, clips, fps=30)` bakes positive uniform-scale clips into
packed model-space transforms. Clips are indexed in sequence order, and their
endpoints are retained. Use in-place clip poses in this pool if the application
already applies extracted root movement. `pool.crossfade_position_error(ids,
times, weights)` reports per-sample maximum bone-position error in rig units
against an exact local-space pair. Sample the transition phases and poses your
application uses before choosing its crowd eligibility threshold.

`validate_crowd_graph` from `graaal.Animation.CrowdGraphValidation` extends that
comparison to the base poses of a compiled graph. Build the pool with clips in
compiled ID order and pass a maximum bone-position error in rig units.

```python
from graaal.Animation.BakedClipPool import BakedClipPool
from graaal.Animation.CrowdGraphValidation import validate_crowd_graph

pool = BakedClipPool.from_graph(graph)
report = validate_crowd_graph(graph, pool, error_budget=0.02 * character_height,
	phase_samples=32, playback_rates=(0.75, 1.0, 1.25))
```

The report lists states by name and transitions in authored order. Each result
includes the sample count, maximum error over evaluated samples, the count of
rows requiring more than four samples, and an `eligible` flag. Rows combine
identical clip IDs and times before checking the limit. Eligibility requires
every sampled row to fit the limit and the measured maximum to meet the budget.

The sweep samples authored axis knots, blend interiors, defaults, start phases
and fade progress. Unsynchronized spaces sample each participating cycle and
additional relative phases for unequal durations. Transition sampling preserves
marker synchronization and varies the requested playback rates. It isolates
each edge from guards and priority so the validator can vary parameters during
the fade. The result covers the sampled base poses. Applications account for
authored layers, root anchoring and IK separately.

#### Record a graph's base poses

`GraphPosePass` evaluates current `AnimationGraphPlayback` poses into a
caller-owned `SkinMatrixBuffer`. Allocate the owner ranges for the rig first and
pass the same palette to `MaterialRenderer`. Create the pass with a validation
report to use crowd compute for eligible rows and exact local-space sampling
for the rest. Omitting the report selects exact CPU poses for the whole batch.

```python
from graaal.RenderPasses.CrowdPose import GraphPosePass

poses = GraphPosePass(palette, max_instances, eligibility=report)
# After advancing playback, record before any draws that read the palette.
poses.record(commands, owners, playback, parameters)
```

With ECS, call `EcsAnimationGraphAdapter.advance` at the simulation tick and
pass `adapter.owners` and `adapter.playback` to `record`. Owners, parameters
and the optional boolean `gpu_mask` use playback row order. A distance or
importance policy can supply the mask to restrict which validated rows can use
the GPU. Record again when either the pose inputs or the mask change. The
application owns playback advancement, events and movement acceptance.

CPU poses are staged and copied into the palette in command order, including
when the entire batch uses the CPU. Different graph passes can write disjoint
owner ranges of the same palette in one submission. Each pass owns its staging
buffers. Submit before recording through the same pass again or calling
`poses.close()`. Coordinate owner allocation and release with draw membership
and recorded GPU work.

#### Frustum-based pose selection

`frustum_pose_mask` tests conservative animated spheres against the draw view
on the CPU. Supply world-space centers in playback row order, using the same
presentation transforms as the draws. Pass one world-space radius for the
batch or an array of radii in the same row order. Spheres that touch the frustum
stay selected.

```python
from graaal.Animation.CrowdVisibility import frustum_pose_mask

view_projection = camera.view() @ camera.projection(width / height)
visible = frustum_pose_mask(view_projection, render_centers, render_radii)
poses.record(commands, owners, playback, parameters, pose_mask=visible)
```

`pose_mask` selects rows to evaluate and write through both pose evaluators.
It composes with `gpu_mask`, which controls evaluator preference among those
rows. Keep advancing all playback rows and record a current pose before a
newly visible owner is drawn, including on frames without a simulation tick.
If one evaluation supplies poses for multiple views, use the union of their
masks.

#### Distance-based evaluator selection

`DistancePoseLod` selects a GPU preference using world-space camera distance.
Owners enter the GPU tier at or beyond `far` and return to the CPU at or within
`near`. Between the thresholds they keep their previous preference. A new
owner starts with a CPU preference and can enter the GPU tier on its first
update. Pass the full live batch each time. Owner IDs can span the full `uint64`
range, including generational ECS handles. History follows owners across row
reordering and is discarded when they leave the batch. To clear all history,
pass `owners=[]` and a position array of shape `(0, 3)`.

```python
from graaal.Animation.AnimationLod import DistancePoseLod

lod = DistancePoseLod(near=18, far=22)
# On each render update, positions and parameters follow playback row order.
mask = lod.update(owners, positions, camera.position, hold=playback.blending(parameters))
poses.record(commands, owners, playback, parameters, gpu_mask=mask)
```

`AnimationGraphPlayback.blending()` marks active fades and interior blend-space
weights. Passing it as `hold` defers distance-driven switches while those poses
are blended. The router still evaluates ineligible poses on the CPU. Update
the policy when the camera or owner positions move, including render frames
without a simulation tick. A changed mask can require recording a pose again
at the same playback time.

#### Automatic skeleton LOD

`reduce_crowd_skeleton` generates a compact skeleton, clips and remapped
primitives from an imported skin. It greedily merges leaf influences into
their parents within a conservative vertex displacement budget over the
supplied clip frames. Roots and ancestors of surviving joints are retained.
The result includes a source-to-compact joint map. If the budget allows no
merge, the function returns `None`.

```python
from graaal.Animation.CrowdGraphValidation import validate_reduced_crowd_graph
from graaal.Animation.CrowdSkeletonLod import reduce_crowd_skeleton

clips = tuple(graph.clips[i] for i in range(len(graph.clips)))
variant = reduce_crowd_skeleton(graph.plan.skeleton, clips, primitives, error_budget=0.035)
report = None if variant is None else validate_reduced_crowd_graph(graph, variant, error_budget=0.035)
poses = GraphPosePass(palette, max_instances, eligibility=report)
```

Validation measures the combined reduction, bake and model-space blending
error on skinned vertex positions against the exact source graph. It samples
states and transitions on the same phase, blend and rate grid as the
full-rig validator. Use a budget in mesh units appropriate to the viewing
distance. The report describes the sampled base poses. Normals and
application-owned pose changes require separate validation.

Use the original mesh and palette ranges sized for the source rig with `GraphPosePass`.
The crowd pass evaluates only the compact joints and expands their matrices
to source slots through the joint map. Exact CPU poses can then use those
same slots when distance or graph eligibility changes. The compact
primitives can also be drawn with compact palette ranges using a low-level
`CrowdPosePass` built from the reduced pool.

#### Compose custom pose paths

`CrowdGraphRouter(report)` from `graaal.Animation.CrowdGraphSamples` exposes the
same partition without a GPU. Call `cpu_rows, batch = router.split(playback,
parameters, gpu_mask=mask)` to get exact rows and packed clip samples. The mask
is optional. Validation and packing determine whether a masked-in row can use
the compute pass. Use `playback.remap(cpu_rows).pose(...)` for exact local poses,
then apply any root anchoring, authored layers or IK through the existing pose
APIs before writing skin matrices. The exact and crowd writers own disjoint
ranges for the frame.

`CrowdPosePass(pool, palette, max_instances)` from
`graaal.RenderPasses.CrowdPose` records one batch before the material draws.
Allocate the owner ranges in a `SkinMatrixBuffer` first. Call
`crowd.record(commands, owners, ids, times, weights)` with one to four samples
per owner, clamped clip times, and normalized weights. The compute pass writes
directly into those palette ranges. Each batch uses one rig and shares the
palette with disjoint CPU-evaluated ranges. Pass `palette_joint_map` when
each output palette slot should copy the matrix of a baked joint. The map
length must match each allocated range. The router's report supplies the
map as `report.joint_map` for reduced pools. Submit before calling `record`
again on the same pass, and release ranges after their use by recorded work.
The palette's CPU mirror does not contain compute results. Call `crowd.close()`
to release the pass's buffers.

The [river crossing](../scenes/river_crossing_example.py) uses `GraphPosePass` to
share a palette between the CPU and compute writers. Model-space blending trades
accuracy for throughput, particularly for dissimilar poses. See the
[animation status](../implementation_plans/skeletal_animation_current.md) for
performance and pose errors on that workload.

## Skinned glTF import

`load_skinned_gltf(path, node=None, fps=30)` from `graaal.Animation.SkeletalGltf` loads
CPU data without an ECS or graphics device. It returns a `SkinnedAsset` with a
`skeleton`, `primitives`, named `clips`, and `joint_nodes` in skeleton order.
Each primitive has indexed vertices with `position`, `normal`, `uv`, `joints`
and `weights`, plus its original `material_id`. Assign materials separately;
this importer does not load material definitions or textures.

```python
from graaal.Animation.SkeletalGltf import load_skinned_gltf

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
from graaal.EcsAdapters.EcsAnimation import SkeletonRef, ClipPlayer, EcsAnimationAdapter

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

Each successful update also replaces `animation.sample_batches` with the clip
samples used for that pose. Each batch has a `skeleton_id`, full `entities`,
`clip_ids` and wrapped or clamped `times` in seconds of shape `(N, 2)`, `weights` of shape `(N, 2)`,
and an `in_place` flag per entity. Non-transitioning players use weights `(1, 0)`
and repeat their clip and time in both slots. During a fade the second slot is
the target; a fade completed in this update has weights `(0, 1)`. Times have
already been wrapped or clamped, so sample them with `loop=False` to preserve a
clamped endpoint. `in_place` identifies players whose selected root is replaced
by the reference root after sampling. Sample arrays are independent snapshots;
the adapter retains the previous list if an update fails. These samples can be
used to prepare another pose evaluator without deriving controller state from
skin matrices.

```python
from graaal.Animation.SkeletalAnimation import sample_clip_pairs, model_matrices, skin_matrices

samples = animation.sample_batches[0]
pose = sample_clip_pairs(skeleton, clips, samples.clip_ids, samples.times, samples.weights)
# Apply RootMotion.in_place to flagged rows before evaluating matrices, if needed.
matrices = skin_matrices(skeleton, model_matrices(skeleton, *pose))
```

## Two-bone IK geometry

`solve_two_bone` solves batched two-segment chains without ECS, rendering or a
physics backend. All positions and the pole point must share a coordinate space:

```python
from graaal.Animation.InverseKinematics import solve_two_bone

new_joint, new_end = solve_two_bone(roots, joints, ends, targets, poles)
```

Each input is a finite `(instances, 3)` array. Both bone lengths must be
positive. The solver preserves lengths, clamps unreachable targets to the
reachable distance, and uses each pole point to choose a bend direction. If a
pole is collinear, it retains the existing bend when possible and otherwise
chooses a deterministic perpendicular. Results are float64 position arrays;
this function alone does not modify a pose.

`two_bone_pose` applies the solve to a batched local pose and returns copied
translation, rotation and scale arrays that can feed `model_matrices` and
`skin_matrices`:

```python
from graaal.Animation.InverseKinematics import two_bone_pose

pose = two_bone_pose(skeleton, pose, (hip, knee, ankle), targets, poles, weight=weights)
```

The bones must be a directly connected chain, and targets and poles are in
skeleton model space. Ancestors of the hip, the hip and the knee must have
positive uniform local scale; other scales are unrestricted. The root and joint
rotations are blended with the original rotations using a scalar or per-instance
weight in `[0, 1]`. A partial weight does not necessarily place the end at the
target. Descendants follow the modified rotations, including the end bone;
the function does not separately constrain end-effector orientation. The
caller owns physics queries, converting world goals to model space, and
post-physics scheduling.

To orient a foot after solving its position, `align_bone_axis(skeleton, pose,
bone, local_axis, directions, weight=...)` rotates the selected bone so its
local axis points along each supplied model-space direction, such as a terrain
normal. Its position stays fixed and descendants follow the rotation. The bone
and all ancestors require positive uniform local scale. A weight of zero keeps
the original local rotation; partial weights interpolate rotations. This aligns
one axis. Use `limit_twist(rotations, reference, axis, minimum, maximum)` to
clamp signed local twist in radians while preserving swing. Use
`foot_plant(targets, normals, stance, locked, anchors, anchor_normals)` to keep
world-space position and normal goals through a caller-defined supported stance.
Retain the returned anchors and stance per foot. Clear locks on teleports and
convert the goals into model space before solving. The headless foot IK example
combines the IK and axis-alignment pose tasks and shows planting with `--plant`.

## Animation events

`EventTimeline` provides an editable, headless model for event markers. It
starts from an `EventTrack` and returns stable IDs for selecting and editing
individual markers, including markers at the same time:

```python
from graaal.Animation.AnimationEvents import EventTimeline, EventTrack

timeline = EventTimeline(EventTrack(clip.duration, [], []))
event_id = timeline.add(0.2, "footstep")
timeline.update(event_id, time=0.3, name="left_step")
ids, track = timeline.snapshot()
timeline.remove(event_id)
```

Snapshots return IDs in the same time order as the independent runtime track;
ties retain insertion order. Removed IDs are not reused within the timeline.
Edits validate against the same `[0, duration)` and nonempty-name rules as
`EventTrack`, and a failed edit leaves the timeline unchanged. A snapshot does
not update an existing player's track automatically: the application decides
when to replace its event resource. Graphical editing, clip preview and asset
persistence are not provided by this model.

For phase matching between looping clips, `SyncMapping` reuses `EventTrack`
markers. Each track must contain the same unique names in the same cyclic
order, at distinct times, with at least two markers. The first marker in time
may differ between tracks. Track durations must match their respective clips.

```python
from graaal.Animation.AnimationEvents import EventTrack
from graaal.Animation.AnimationSync import SyncMapping

walk = EventTrack(1.0, [0.1, 0.6], ["left", "right"])
run = EventTrack(0.8, [0.05, 0.35], ["left", "right"])
sync = SyncMapping(walk, run)
run_times, rate = sync.map(walk_times)
```

`map` accepts finite, possibly unwrapped source times and returns wrapped target
times and per-interval rate ratios. The target time preserves progress between
corresponding markers; multiplying source speed by the ratio gives the target
speed within that interval. Pass `reverse=True` for reverse playback so the
rate at an exact marker comes from the preceding interval. Reevaluate at marker
boundaries: a single rate over a step crossing markers does not maintain phase
when interval ratios differ. Mapping does not advance players, fire events,
or synchronize root-motion requests automatically. Applications own those
decisions and can use the mapped times with `sample_clip_pairs`.

For a step that crosses markers or loops, `sync.advance(source_start,
source_end, target_start)` returns unwrapped target end times. Start with
`target_start = sync.map(source_start)[0]` or an equivalent aligned time;
subsequent calls can use the previous result. The returned interval can be
passed to `EventTrack.between` and `RootMotion.character_delta`, preserving
crossings in either direction. All arrays must be finite, equal-length 1D
arrays. Loop counts with magnitude `2**52` or greater are rejected to avoid
losing cycle precision. The method calculates timing only; the caller applies
it to playback and handles events and movement requests.

`BlendSpaceSync` composes those mappings around a fixed reference clock:

```python
from graaal.Animation.AnimationSync import BlendSpaceSync

group = BlendSpaceSync(walk_id, {walk_id: walk, run_id: run})
times, rates = group.map(selected_clip_ids, reference_times)
```

Tracks use nonnegative integer clip IDs. Construction snapshots the tracks and
validates all participants, including the reference, with `SyncMapping`.
Selected IDs are integer arrays shaped `(instances, ...)`, such as a 1D blend's
`(instances, 2)` or a grid's `(instances, 2, 2)`; reference times are finite seconds
shaped `(instances,)`, including negative or unwrapped times. Outputs match the
IDs' shape and own their storage. `reverse=True` selects preceding-interval rates
at exact markers. Unknown IDs fail before sampling. `group.reference` identifies
the reference clip, and read-only `group.mappings` exposes the individual phase
mappings for application-owned interval traversal. Multiply returned rates by
reference playback speed when deriving clip speeds.

The reference keeps advancing independently of blend weights, even when its
weight is zero or it is outside the selected cell. Parameter changes therefore
do not switch clock leaders or change cycle speed. Weighted cycle timing is not
implemented. Events, root-motion traversal and movement acceptance remain
application-owned; mapped pose times alone do not preserve crossing intervals.

To synchronize ECS crossfades, pass the same clip-pair mapping to both the
state controller and playback adapter:

```python
sync_mappings = {(walk_id, run_id): sync}
control = EcsAnimationStateAdapter(world, machine, state_clips, sync_mappings=sync_mappings)
animation = EcsAnimationAdapter(world, skeletons, clips, sync_mappings=sync_mappings)
```

Each configured `(source_clip_id, target_clip_id)` pair synchronizes all its
crossfades. Both clips must loop, their durations must match the tracks, and
their target time must be aligned at the start. The state adapter initializes
that time automatically; callers starting fades directly must set it using
`sync.map`. Playback checks alignment, advances the target through marker
intervals, and uses the resulting intervals for event and root-motion handling.
When a fade ends, the target continues at the mapped speed at the fade endpoint;
there is no ongoing synchronization after promotion. Supply mappings in both
directions if transitions can run in both directions. An unconfigured pair
keeps ordinary independent playback.

`graaal.Animation.AnimationEvents.EventTrack` stores named markers independently of ECS:

```python
from graaal.Animation.AnimationEvents import EventTrack

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
see [simple_animation_example.py](../scenes/simple_animation_example.py).

To separate control and pose evaluation, call
`animation.update(dt, evaluate_poses=False)` to advance playback, events and
root-motion requests without evaluating CPU poses. It returns an empty list and
replaces `sample_batches`; it does not change `animation.batches`. Later,
`animation.evaluate_samples()` evaluates the
latest samples without advancing playback; pass an earlier sample-batch list to
evaluate that snapshot instead. It returns pose batches and does not change
`animation.batches`. The samples must retain the adapter's grouping by skeleton
and source clip, with at most one batch per pair; referenced resources and
root-motion definitions must remain available. Consume or copy returned matrix
views before the next pose evaluation or normal update, including a failed one.

For scheduled phases, use `animation.advance` and `animation.finalize` instead of
`animation.evaluate`. `advance` produces controller outputs and clears the old
`batches`; `finalize` evaluates those samples and stores the new `batches`.
Order any root-motion consumer and physics between them, and order rendering
after finalization. Samples retain full entity handles from the advance step;
applications must coordinate structural changes so removed entities are not
rendered from stale snapshots. A separate post-physics `PosePlan` can consume
the copied pre-physics pose for IK or alignment; see `foot_ik_example.py`.

Run `python -m scenes.root_motion_example` for a fixed-step idle/walk/run
example that submits root-motion requests to a bounded movement system and
reports the accepted displacement. It uses synthetic clips and a simple boundary.
Run `python -m scenes.physics_locomotion_example --headless` to accept the same requests
against a native shape sweep in one ECS scheduler. The character's kinematic
transform receives only the accepted displacement, while the animation pose
remains in place. This demonstrates blocking, not sliding or step climbing.
For the same flow with GPU-skinned materials, run
`python main.py scenes/locomotion_example.py`. Greenman's walk clip is augmented
with synthetic root translation; the boundary is an example movement constraint.

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
