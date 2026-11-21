# Animation timeline tool

From the repository root, install the asset dependencies and launch the tool:

```powershell
python -m pip install -e ".[assets]"
python -m tools.animation_timeline scenes/resources/greenman.glb
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

Events and sync markers are kept separately for each animation in memory. Use
**Save project** to persist them for all animations in a JSON sidecar, and
**Load project** to restore them for the open asset. Projects can be exchanged with
the graph editor; a graph in a loaded project is preserved when saving marker edits.
See [authoring projects](animation_project.md) for the format and validation rules.
Unsaved edits are discarded when the tool closes or a different source file is
opened; glTF/GLB assets are not modified. Event
times must be at least zero and strictly less than the animation duration,
matching `EventTrack`. Zero-duration animations can be inspected but cannot
receive markers. For imported rigs, sync-marker duration uses the baked runtime
clip's endpoint. glTF sparse accessors are not supported by the current reader.
