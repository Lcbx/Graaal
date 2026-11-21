# Animation authoring projects

The [graph](animation_graph.md) and [timeline](animation_timeline.md) tools share
**Save project** and **Load project** controls. Open the source glTF/GLB first,
then load its project. Projects are UTF-8 JSON sidecar files, conventionally
`character.animation.json`. Source assets, animation channels, materials and
textures remain separate; the project stores no absolute asset paths.

Save captures events and sync markers for every animation. The graph editor also
saves the current JSON draft, including valid edits not yet applied to its preview.
Write staged form changes to the draft before saving. Saving validates the
serialized graph and markers without changing the compiled preview. Invalid graph
drafts or incompatible referenced sync tracks cannot replace a saved project.
Writes replace the destination only after validation and a complete temporary
file write; failed saves preserve the previous file.

Load checks animation count, names, order, source durations and baked sync
durations against the open asset. These checks compare clip metadata, not asset
file identity. Any graph compiles against that asset's skeleton and the imported
sync tracks before the session changes. Invalid files preserve the draft,
playback and marker selections. Successful loads replace all marker tracks.
Marker IDs are local to a session; coincident event order survives loading.

The graph editor applies the loaded graph, pauses, resets its preview and closes
open editing forms. Loading a marker-only project keeps the current graph draft
and recompiles it against the imported markers. The timeline clears marker
selection, dragging and event-preview history, and pauses at the current playhead.
When it loads a project containing a graph, it preserves that graph on subsequent
saves and validates its sync mappings against the edited markers.

Projects contain authored data. Preview clocks, transition progress, applied
parameter overrides and sample-binding selections are not stored. Both tools
save only on request; save before closing or opening another source asset.

## JSON format

Version 1 requires `version` and an `animations` list in source animation order.
Every animation has these fields, including empty lists for tracks with no markers:

```json
{
  "version": 1,
  "animations": [
    {
      "name": "walk",
      "duration": 1.0,
      "sync_duration": 1.0,
      "events": [{"time": 0.25, "name": "footstep"}],
      "sync_markers": [{"time": 0.0, "name": "left"}, {"time": 0.5, "name": "right"}]
    }
  ]
}
```

`duration` is the source timeline endpoint. `sync_duration` is the imported runtime
clip endpoint, or the source duration when no runtime clip is available. Durations
must match exactly; resampling settings must therefore agree between sessions.
Marker times are seconds in `[0, duration)` or `[0, sync_duration)`, respectively.
Zero-duration tracks require empty lists. Event names may repeat and events may
coincide. Sync mappings additionally require compatible unique marker names and
distinct times, as enforced by the runtime compiler.

Optional `graph` contains the existing
[`AnimationGraphDefinition` format](../graaal/skeletal_animation.md#editable-graph-definitions).
Its clip references use `index: name`, matching the graph editor's labels.
Marker-only files support assets outside the runtime skinned-import subset;
graph-containing projects require a supported skinned asset. Unknown fields,
duplicate JSON keys, non-finite numbers and unsupported versions fail validation.

## Headless use

From the repository root, tools can persist the same format without Tk or ECS:

```python
from tools.animation_asset import load_animations
from tools.animation_project import read_project, write_project

animations = load_animations("character.glb")
write_project("character.animation.json", animations)  # Marker-only project.
definition, graph, timelines = read_project("character.animation.json", animations)
```

Pass a graph definition dictionary as the optional third argument to
`write_project`. `read_project` returns an optional definition, optional compiled
graph, and `(events, sync_markers)` timeline pairs in source order. It does not
modify `animations`; callers explicitly install those timelines after validation.
Each zero-duration track returns `None`. See
[project tests](../tests/test_animation_project.py) for round trips, compatibility
checks and failed-write preservation, and the graph/timeline UI tests for session
replacement behavior.
