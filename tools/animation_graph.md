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
and identifies invalid fields/tasks. Successful compilation pauses and resets the preview
to the initial state, parameter defaults, zero clock/fade and state-driven sample
bindings. Invalid drafts preserve the previous compiled preview and controls.
Compilation does not save files. Use **Save project** to persist the current draft
and all event/sync markers in a JSON sidecar. **Load project** validates and applies
a project for the open asset. See [authoring projects](animation_project.md) for
the shared format and replacement rules.

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
**Manual transition weight** slider blends its source and target poses without
promoting the target state.

Enable **Automatic state playback**, apply parameters and choose **Play** to run
the compiled clip and 1D/2D blend states using one row of the runtime graph
player. Manual preview uses the same state sampler. The first matching rule starts a fade using its
configured duration. Its target starts at time zero, or at the corresponding
marker phase when a sync mapping exists. Both clips advance during the fade;
completion promotes the target with its current time and speed. Applied parameter
changes do not interrupt an active fade. Conditions are checked again on the next
playback tick after promotion, so a condition that remains true can cycle states.
Even a tick longer than a fade starts at most one transition.

**Pause** holds the clip clocks and fade progress; **Play** resumes them.
Scrubbing in automatic mode pauses, cancels the fade and seeks the current state
at unit speed. Selecting a state restarts it at time zero. Switching modes cancels
the fade at the displayed source time and restores unit speed. The manual weight
slider is disabled in automatic mode. All preview clips loop, including targets
after promotion; marker synchronization ends when the fade completes, retaining
the target speed at that endpoint. Static clips can still transition.

To use a named blend space as a state, replace that state's `clip` entry in JSON:

```json
"states": {
  "idle": {"clip": "1: 1_idle"},
  "moving": {"blend_space": "locomotion"},
  "turning": {"blend_space_2d": "motion"}
}
```

Define `locomotion` in `blend_spaces` and `motion` in `blend_spaces_2d` as shown
below, then compile. Each state must name exactly one source. State/transition
bindings evaluate its blend from the applied parameters in both manual and
automatic modes. A 2D state blends its two horizontal clip pairs, then blends
the row poses vertically. During a fade, the complete state poses are crossfaded
before authored masks and additive layers.
Parameter changes update active blend weights without restarting clocks or fades.

A blend state's clock measures elapsed seconds and remains unwrapped; each
selected clip loops at its own duration. This preserves clip clocks when the
selected pair changes or clips have unequal durations. Static clips sample time
zero. Without synchronization, the clip dropdown shows the state's first 1D point
or first grid cell as the representative asset. With synchronization it shows the
reference clip. These state definitions also round-trip through JSON project saves.

To synchronize clips within a blend, add `"sync_reference": "clip name"` to its
`blend_spaces` or `blend_spaces_2d` entry, or choose **Sync reference (optional)**
in its form. The reference must occur in that space. Every distinct participating
clip needs compatible sync markers, even if its current weight is zero. Choose
an empty reference to restore independent clip looping. Writing a form changes
the draft; compile it to apply the reference and marker snapshot.

The displayed state clock measures reference-clip seconds. Selected clips map
that clock to corresponding marker phases, including marker offsets and unequal
intervals. The reference keeps leading when parameters change or select a cell
that does not contain it; weights do not change cycle speed. Direct blend/grid
sample bindings interpret the displayed clock in their own reference seconds.
No events or root motion are dispatched by this preview.

State fades also use explicit `sync_mappings` between clock clips: a clip state's
clip or a synchronized blend state's reference. For example, a mapping from
`walk` to `run` aligns a walk-referenced blend to a run-referenced blend. Without
a configured pair, automatic targets start at time zero and advance independently.
Synchronization between state clocks ends at promotion, retaining target speed;
each blend's internal marker mapping remains active. Applied parameter changes
do not restart either clock or an active fade. Weighted cycle timing remains
future work.

Sample slots can use the state/transition pair or a specific clip, allowing
independent masked override and additive layers defined in the draft. Other
bindings use the displayed source-state clock, which follows the target clock
after promotion. In manual mode both state poses use the graph clock, with sync
mappings aligning target phase between state clock clips. Root-motion extraction is disabled, so in-place
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
assign its clip. For direct sample bindings, enable **Bind sample slots** and
choose two sample slots and a weight slot. Disable it for a grid used only by
states; no extra sample or weight slots are needed. **Write 2D
grids to draft** validates the full graph and changes the in-memory JSON as one
undoable edit. Choose **Compile draft** to update the preview. Concurrent JSON
edits block writing until the form is reopened.

You can also define `blend_spaces_2d` in JSON with separate X/Y parameters, strictly
increasing `x_positions` and `y_positions`, and a rectangular `clips` list in
Y-row order. For direct sample bindings, give it `sample_slots` (two distinct
existing slots) and a `weight_slot` (an existing layer weight parameter). The pose plan should sample
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
the graph clock, interpreted through markers when a sync reference is authored;
events and root motion remain application-owned.
Give each named 2D grid its own weight slot so multiple grids can be previewed
at once; shared slots fail compilation. Omit both `sample_slots` and
`weight_slot` for a grid used only by a `blend_space_2d` state. Such grids do not
appear in the sample-binding dropdowns; choose the state with a state/transition
binding to preview them. A grid can serve both uses when bindings are declared;
state evaluation keeps its internal vertical weight separate from authored
layer weights. One-row, one-column and single-cell grids are supported.

Graph poses use the existing `PosePlan`, model matrices and CPU-skinned viewport.
Camera framing uses the union of sampled clip bone bounds. Selecting a clip in
the graph tool turns graph preview off; enable **Preview graph in viewport** to
restore it. Closing the window discards unsaved changes.
See `python -m scenes.animation_graph_example` for ECS integration of the same
definition format. Forms for other graph items remain future work.

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
timeline tool has its own marker session; exchange markers through explicitly
saved [authoring projects](animation_project.md).

