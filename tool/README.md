# Animation timeline tool

From the repository root, install the asset dependencies and launch the tool:

```powershell
python -m pip install -e ".[assets]"
python -m tool.animation_timeline scenes/resources/greenman.glb
```

The file argument is optional. Use **Open glTF / GLB** to choose another file.
The tool also needs Python's Tkinter support; it does not require a GPU.

Select an animation to inspect each node/property channel and its keyframes.
The timeline automatically fits the animation duration when selecting a clip
and follows panel resizing until zoom is adjusted manually. **Fit timeline**
restores automatic fitting. Click or drag on the timeline to scrub; click a
keyframe to see its value and interpolation in the status bar. Use Play/Pause,
the zoom slider and the scrollbars to navigate. Choose **Events** or **Sync markers**
beside the marker controls. Add a marker by entering its name and time, or click
**Use playhead** to fill in the current time. Select an event in the list or
click its timeline marker to inspect it. Change its name or time and choose
**Update selected**, or drag the marker to move it. Both marker rows remain
visible; clicking a marker switches to its editor. Selection stays attached
to the event when times are reordered; use the list to choose between events
at the same time. Dragging clamps times to `[0, duration)`. Use **Remove
selected** to remove the selected event.

Selecting or dragging a marker pauses playback and moves the playhead to it.
During playback, **Preview events** shows the last crossed markers using the
runtime `EventTrack.between` rules, including loop crossings and stable ordering
at coincident times. It shows up to eight occurrences from the last update
that crossed events. Scrubbing and editing do not fire events. Playback takes
fresh snapshots of the in-memory events; it does not change a running engine
instance or write to the asset.

The right viewport follows the playhead and shows the animated bones and a
flat-shaded skinned mesh where supported by `load_skinned_gltf`. Its camera
targets the center of the bone AABB sampled across the selected animation;
distance accounts for those bounds and the viewport aspect ratio. Drag the
divider to resize the panels. Mesh and bone visibility can be toggled separately.
Supported rigs use the imported runtime `Clip` with `sample_clip` and
`model_matrices`, including runtime nlerp between baked frames. Source keys
remain unchanged in the timeline. Mesh deformation uses `skin_matrices` and
position-only `skin_vertices`; projection uses `graaal.Math.Camera` matrices.
Unsupported skinned mesh features fall back to a skeleton preview with a reason
shown in the viewport. The bone preview evaluates LINEAR, STEP and CUBICSPLINE
TRS channels; morph weights are displayed as tracks but are not deformed in the
viewport. glTF materials and textures are not shown in this CPU preview.

Events and sync markers are kept separately for each animation in memory. They are discarded
when the tool closes or a different file is opened; no asset is written. Event
times must be at least zero and strictly less than the animation duration,
matching `EventTrack`. Zero-duration animations can be inspected but cannot
receive markers. For imported rigs, sync-marker duration uses the baked runtime
clip's endpoint. glTF sparse accessors are not supported by the current reader.

For a supported skinned asset, open **Graph configuration**. The separate window
has a JSON definition editor, a scrollable state/transition diagram, preview
parameters and sample-slot bindings. Its starting draft has one state per clip;
setting `advance` above `0.5` selects a transition to the next clip in the cycle.
Choose a state in the list or diagram to inspect it in the existing viewport.

Edit states, transition conditions, masks and layered tasks in the draft, then
choose **Compile draft**. Validation uses
[`AnimationGraphDefinition`](../graaal/skeletal_animation.md#editable-graph-definitions)
and identifies invalid fields/tasks. Successful compilation resets the preview
to the initial state, parameter defaults, zero fade and state-driven sample
bindings. Invalid drafts preserve the previous compiled preview and controls.
No graph or glTF file is saved.

Choose **Edit transitions** to configure rules in a form using the current valid
JSON draft. Select a rule to edit its source, target and fade duration, or choose
**New rule**. Add, update or remove conditions in the form, then choose **Add
rule** or **Update selected rule** to stage those fields. All conditions must
match. **Higher priority** and **Lower priority** change rule order; the first
matching rule wins. **Remove selected** removes a staged rule.

**Write transitions to draft** validates and replaces the JSON text as one undoable
edit. It preserves the other graph fields and leaves the compiled preview alone;
choose **Compile draft** to apply it. Closing the form discards changes that have
not been written. If the JSON text changes while the form is open, writing is
blocked: close and reopen the form to use the latest draft. Parameters and states
are chosen from the draft; add new ones in JSON before opening the form.

Edit preview parameter values and choose **Apply parameters / evaluate conditions**
to inspect the first matching transition and its configured duration. The
**Manual transition weight** slider blends its source and target poses; it does
not run an automatic state controller or promote the target state. Sample slots
can use that state/transition pair or a specific clip, allowing independent
masked override and additive layers defined in the draft. Clips loop at the
timeline's current time in seconds; configured sync mappings align the target
phase to the source markers. Root-motion extraction is disabled, so in-place
tasks receive an empty instance mask. Event feedback
still belongs to the selected timeline clip, rather than a graph event controller.

To preview a 1D blend space, add a named `blend_spaces` entry to the JSON draft,
then compile and choose `<blend: name>` in a sample binding. For example:

```json
"blend_spaces": {
  "locomotion": {
    "parameter": "speed",
    "points": [{"position": 0, "clip": "1: 1_idle"}, {"position": 1, "clip": "2: 2_move"}]
  }
}
```

Declare `speed` in `parameters`. Apply preview parameters to move along the
axis; values outside the points use the nearest clip. The binding supplies the
selected pair to the existing pose plan at timeline time. Each slot can choose
its own blend space, so the same graph can layer a masked clip over locomotion.
For a 2D grid, define `blend_spaces_2d` with separate X/Y parameters, strictly
increasing `x_positions` and `y_positions`, and a rectangular `clips` list in
Y-row order. Give it `sample_slots` (two distinct existing slots) and a
`weight_slot` (an existing layer weight parameter). The pose plan should sample
both slots and blend their tasks using that weight. For example:

```json
"blend_spaces_2d": {
  "motion": {
    "x_parameter": "speed", "y_parameter": "turn",
    "x_positions": [0, 1], "y_positions": [-1, 1],
    "clips": [["1: 1_idle", "2: 2_move"], ["1: 1_idle", "2: 2_move"]],
    "sample_slots": ["lower", "upper"], "weight_slot": "grid_mix"
  }
}
```

Choose `<grid: motion / lower>` in either paired sample binding; the other
binding is set automatically. The grid selects both horizontal clip pairs and
overrides `grid_mix` with the vertical blend weight for the viewport pose. Choosing
a different binding on either slot disconnects the pair. The sample times use
the timeline clock; clip phase, events and root motion remain application-owned.
Give each named 2D grid its own weight slot so multiple grids can be previewed
at once; shared slots fail compilation.

Graph poses use the existing `PosePlan`, model matrices and CPU-skinned viewport.
Camera framing uses the union of sampled clip bone bounds. Selecting a clip in
the timeline turns graph preview off; enable **Preview graph in viewport** to
restore it. Reopening the graph window retains its draft. Closing it returns to
clip inspection; loading another asset discards the graph draft and closes it.
See `python -m scenes.animation_graph_example` for ECS integration of the same
definition format. Forms for other graph items and blend spaces, and
persistence remain future work.

To synchronize two clips, author at least two sync markers on each clip using
the same unique names in the same cyclic order. Their times must be distinct
and in `[0, clip duration)`. Then add a requested pair to the graph draft, using
the clip names shown in its sample bindings, for example with Greenman:

```json
"sync_mappings": [{"source": "1: 1_idle", "target": "2: 2_move"}]
```

Choose **Compile draft** to validate and apply fresh marker snapshots. The
transition description shows **marker sync** when its pair has a mapping. Later
marker edits stay separate from the compiled preview until recompilation; a
failed compile preserves the previous mapping. Missing markers, duplicate names
or times, incompatible cyclic ordering and clip-duration mismatches identify the
invalid pair. Sync markers never fire as gameplay events. Mappings are
directional; author the reverse pair separately if needed. No marker data is saved.
