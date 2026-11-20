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
the zoom slider and the scrollbars to navigate. Add an event by entering its name and time, or click
**Use playhead** to fill in the current time. Select an event in the list to
move the playhead to it, and use **Remove selected** to remove it.

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

Events are kept separately for each animation in memory. They are discarded
when the tool closes or a different file is opened; no asset is written. Event
times must be at least zero and strictly less than the animation duration,
matching `EventTrack`. Zero-duration animations can be inspected but cannot
receive events. glTF sparse accessors are not supported by the current reader.
