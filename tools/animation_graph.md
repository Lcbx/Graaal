# Animation graph tool

From the repository root, install asset dependencies and launch the editor:

```powershell
python -m pip install -e ".[assets]"
python -m tools.animation_graph scenes/resources/greenman.glb
```

The file argument is optional; without it, choose a file in the opening dialog.
A supported skinned asset is required. The window has a JSON definition editor,
a scrollable state/transition diagram, preview
parameters and sample-slot bindings. Its starting draft has one state per clip;
setting `advance` above `0.5` selects a transition to the next clip in the cycle.
Choose a state in the list or diagram to inspect it in the viewport. The graph
clock can be scrubbed or played independently of the timeline tool.

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
graph clock time in seconds; configured sync mappings align the target
phase to the source markers. Root-motion extraction is disabled, so in-place
tasks receive an empty instance mask. This graph preview does not dispatch events.

Choose **Edit 1D blends** to add, rename, update or remove named 1D blend
spaces. Pick a draft parameter, then add points with numeric positions and
imported clips. Points are sorted by position; repeated positions fail
validation. **Write 1D blends to draft** changes only the in-memory JSON as one
undoable edit. Choose **Compile draft** to update the preview. Concurrent JSON
edits block writing until the form is reopened.

You can also add a named `blend_spaces` entry to the JSON draft directly. After
compiling, choose `<blend: name>` in a sample binding. For example:

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
selected pair to the existing pose plan at graph clock time. Each slot can choose
its own blend space, so the same graph can layer a masked clip over locomotion.
Choose **Edit 2D grids** to configure named rectangular grids in a form. Select
separate X/Y parameters, enter comma-separated increasing axis positions and
choose **Apply axes**. Existing cell assignments at unchanged coordinates are
kept; new cells initially use the first imported clip. Select each cell and
assign its clip, then choose the two sample slots and a weight slot. **Write 2D
grids to draft** validates the full graph and changes the in-memory JSON as one
undoable edit. Choose **Compile draft** to update the preview. Concurrent JSON
edits block writing until the form is reopened.

You can also define `blend_spaces_2d` in JSON with separate X/Y parameters, strictly
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
the graph clock; clip phase, events and root motion remain application-owned.
Give each named 2D grid its own weight slot so multiple grids can be previewed
at once; shared slots fail compilation.

Graph poses use the existing `PosePlan`, model matrices and CPU-skinned viewport.
Camera framing uses the union of sampled clip bone bounds. Selecting a clip in
the graph tool turns graph preview off; enable **Preview graph in viewport** to
restore it. Closing the window discards the draft.
See `python -m scenes.animation_graph_example` for ECS integration of the same
definition format. Forms for other graph items and persistence remain future work.

Choose **Edit sync markers** to add, update or remove markers for each imported
clip. Markers remain in memory and must have unique names and times within each
clip. Add matching names in the same cyclic order to both clips, then request a
directional pair in the graph draft:

```json
"sync_mappings": [{"source": "1: 1_idle", "target": "2: 2_move"}]
```

Choose **Compile draft** to snapshot and validate the marker tracks. Later
marker edits do not change the compiled preview until recompilation. Invalid
markers or incompatible tracks leave the last compiled preview in place. The
timeline tool has its own temporary marker session; no marker data is saved.

