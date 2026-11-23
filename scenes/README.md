# Interactive scenes

Run the examples from the repository root after installing the optional dependencies:

```powershell
pip install -e ".[render,assets,ui,examples,test]"
python main.py scenes/draw_batches_example.py
python main.py scenes/direct_draw_example.py
python main.py scenes/prototype.py
python main.py scenes/transparency_example.py
python main.py scenes/material_example.py
python main.py scenes/postprocessing_example.py
python main.py scenes/msaa_example.py
python main.py scenes/simple_animation_example.py
python main.py scenes/mixing_animation_example.py
python main.py scenes/layered_animation_example.py
python main.py scenes/locomotion_example.py
```

Add `--frames N` to stop a RenderContext scene after N displayed frames, for
example `python main.py scenes/material_example.py --frames 120`. Closing the
window can end a run earlier. The modules export `run_scene()` and can be
imported without opening a GPU device. `main.py` invokes that entry point and
cleans up the context afterward.

| Scene | What to try |
| --- | --- |
| `draw_batches_example.py` | ECS movement and entity changes, LOD, indirect draws and Hi-Z culling; Space switches camera, scroll zooms. |
| `direct_draw_example.py` | Direct drawing with instance changes and shadows; scroll zooms. |
| `prototype.py` | ECS-backed actor/health panels in ImGui. Combat resolution is a placeholder. |
| `transparency_example.py` | OPAQUE/MASK/BLEND glTF geometry, banded transparency and diagnostics; see below. F opens settings; P opens the performance profiler. |
| `material_example.py` | Shared material dictionaries, per-instance overrides, rim lighting and animated stripes. Space orbits; M toggles MSAA; D changes a default material; F opens settings; P opens the performance profiler. |
| `postprocessing_example.py` | Drive forward or backward with W/S or Up/Down, look with A/D or Left/Right, and adjust maximum blur, acceleration boost and vignette exponent in the panel. Space toggles blur. |
| `msaa_example.py` | Minimal full-window rotating triangle; 1 selects single sampling and 4 selects 4x MSAA. |
| `simple_animation_example.py` | Three ECS players share the textured dummy mesh with independent GPU skinning. Keys 1–3 select rest pose, Walking and Running for all. Space pauses. |
| `mixing_animation_example.py` | Compare a manually weighted clip blend with an ECS crossfade. Choose clips, adjust playback and fade settings in ImGui; Space pauses, F opens rendering settings and P opens the profiler. |
| `layered_animation_example.py` | Compare a base clip with a masked override or additive layer. Adjust the clips, bone mask, layer weight and playback speeds in ImGui; Space pauses, F opens rendering settings and P opens the profiler. |
| `locomotion_example.py` | The dummy crossfades between rest, walk and run. A separate root track moves it through the looping city. Compare requested and accepted displacement in ImGui. |

The four rendered animation scenes use [dummy_rigged.glb](resources/dummy_rigged.glb)
and baked clips from [Walking.fbx](resources/Walking.fbx) and
[Running.fbx](resources/Running.fbx). If you edit the source assets, rebuild the
clip bundle with Blender from the repository root:

```powershell
blender --background --factory-startup --python-exit-code 1 --python tools/bake_dummy_animations.py
```

These files are runnable examples rather than a stable scene API. For reusable
components, see the [engine guide](../graaal/README.md), [ECS reference](../graaal/ECS_API.md),
[material guide](../graaal/materials.md), and
[transparency guide](../graaal/stochastic_transparency.md).

## Headless animation and physics

Run the examples as `python -m scenes.<name>`. Add `--headless` to the physics
viewers when you want simulation results in the terminal.

| Module | Demonstrates | Extra |
| --- | --- | --- |
| `animation_events_example` | Forward/reverse playback and scheduled footstep-event consumption | Core |
| `root_motion_example` | Idle/walk/run crossfades with requested and boundary-accepted movement | Core |
| `animation_graph_example` | Compiled conditional states and a masked upper-body layer for two characters | Core |
| `animation_graph_batch_example` | Standalone batched graph playback, synchronized 1D states and a masked upper-body layer | Core |
| `foot_ik_example` | Pre/post-physics pose plans with analytic terrain, two-bone IK, normal alignment and optional `--plant` | Core |
| `physics_headless_example` | Standalone native self-test, batched boxes, snapshots and events; no ECS | `physics` |
| `physics_locomotion_example` | Root-motion acceptance against a native shape sweep; blocking only | `physics` |
| `physics_queries_example` | Batched collision rays driving foot IK, owner lookup and sweeps | `physics` |
| `physics_joints_example` | Hinge/slider limits, motor reversal and a breakable attachment | `physics` |
| `passive_ragdoll_example` | Python-authored 13-body humanoid with feet/ankles, passive neutral/limit forces and bounded zero-gravity/drop/stair/impact checks; no ECS or renderer | `physics` |
| `active_ragdoll_example` | Headless powered pose tracking, support observations and root-assisted standing balance. `--no-balance` runs tracking alone | `physics` |

Install the native backend with `python -m pip install -e ".[physics]"`.
`physics_headless_example` defaults to 120 ticks at 120 Hz; use `--ticks`,
`--tick-hz` and `--threads` to vary the run. The
[physics guide](../graaal/physics.md) covers units, read-only snapshots,
UID identity, commands and supported collider combinations.

The falling-box, joint, locomotion and query demos open a playback window by
default. Install the viewer dependencies with
`python -m pip install -e ".[physics,render]"`, then choose a demo.

```sh
python -m scenes.physics_headless_example
python -m scenes.physics_joints_example
python -m scenes.physics_locomotion_example
python -m scenes.physics_queries_example
```

Playback runs at the recorded sample rate. Press Space to pause or resume and R
to restart. The queries viewer shows the IK leg in blue, the comparison leg in
orange, the ray in yellow and the swept sphere at impact in green. Add
`--offscreen` to render two samples without opening a window.

Run the passive humanoid checks with:

```powershell
python -m scenes.passive_ragdoll_example --headless --scenario all --frames 960 --threads 1 --output passive_humanoid.json
python -m scenes.passive_ragdoll_example --headless --scenario zero_gravity --frames 960 --no-passive
```

`--scenario` also accepts `drop`, `stairs` and `impact`. `--height` and `--mass`
scale the example asset from 1.8 m/70 kg; its gains scale with mass times length
squared. Simulations use 120 Hz lockstep and zero active motor strength.
`--no-passive` disables the Python neutral spring and soft-limit torques.
At the built-in size, the stair case starts just above the highest step
with a 1.2 m/s velocity toward descending steps.
The impact case queues an off-centre chest impulse at `min(24, frames // 3)`
and records its immediate linear/angular velocity changes without advancing time.
The impulse uses the root when a tuned rig has no bone named `chest`. The
reported impact point offset is relative to the struck bone's joint origin.

Use `--tuning PATH` in any viewing mode or headless to load a
[ragdoll tuner project](../tools/ragdoll_tuner.md) or native
`RagdollSettings.as_dict()` JSON. Tuner projects supply their saved neutral pose
and passive gains. Native settings start with the rest pose and zero passive
gains. The loaded rig keeps its saved dimensions and masses, so `--height` and
`--mass` apply only to the built-in humanoid.

```sh
python -m scenes.passive_ragdoll_example --tuning rig.tuning.json --scenario stairs
python -m scenes.passive_ragdoll_example --tuning rig.tuning.json --headless --scenario all
```

The optional JSON report owns final positions/orientations after shutdown and
includes pose error, anchor separation, contact-record counts, peak velocities,
late maximum velocities, raw contact depth/impulse peaks and final **linear**
kinetic energy. Peak velocities include the starting state and the immediate
impulse response. Late maxima use post-step samples from the last 120 ticks
(or the whole shorter run). Contact counts include begin, persist and end
records. These bounded mechanical checks cover the authored scenarios and their
default parameters.

Open the viewer with `python -m scenes.passive_ragdoll_example`.
The viewer needs the `physics` and `render` extras and a working WGPU backend.
It starts with the drop. Use `--scenario stairs` to start on the stairs, or
switch in the window with **1** for zero gravity, **2** for drop, **3** for stairs
and **4** for impact. Install the `ui` extra to also use the **Scenario** dropdown.
The viewer records each scenario the first time it is selected and loops it at
120 simulated ticks per second. In a window, `--output` writes the last viewed
scenario's report when you close it.
Press Space to pause or resume and R to restart. Blue marks the left limbs and
orange marks the right. Capsules, spheres, boxes and cylinders follow the collider poses.
To play a scenario with the skinned dummy, install the `assets` extra and get a
copy of `dummy_rigged.glb` from the scene author. The viewer uses this rig's
specific joint layout.
Pass the path explicitly, or place the file at `scenes/resources/dummy_rigged.glb`
and pass `--model` by itself.

```sh
python -m scenes.passive_ragdoll_example --scenario zero_gravity --model "path/to/dummy_rigged.glb"
```

Model mode draws the dummy and terrain. The mesh follows the simulated pelvis
position and joint rotations while keeping the dummy's bone lengths. Tuning
files used with `--model` must keep the built-in humanoid's bone names, parents,
and order.
Add `--no-passive` to inspect the same scenario with only native constraints
and damping, or add `--offscreen` for a bounded render check.

These physics scenes also support windows through `python main.py scenes/<name>.py`:

| Scene | Behavior and controls | Bounded headless run |
| --- | --- | --- |
| `physics_example` | Falling cubes, obstacles, moving platform and a sensor tinted by box occupancy; render interpolation between 120 Hz ticks | `python -m scenes.physics_example --headless --frames 240` |
| `ragdoll_example` | Eight-second animated/powered/limp/recovery cycle; procedural skinned boxes show the native humanoid | `python -m scenes.ragdoll_example --headless --frames 960` |
| `character_example` | Animated native controller, collision foot IK, triggers and ragdoll hand-off; WASD moves, Space jumps, R falls, G requests recovery, T resumes the scripted hurdle-jump cycle | `python -m scenes.character_example --headless --frames 960` |

Rendered modes need the rendering/scene dependencies as well as `physics`.
Use `python -m scenes.<name> --offscreen --frames N` for a bounded GPU check
(for example, 2 frames for bodies or 6 for skinning). Gameplay uses the latest
physics tick; presentation interpolates without changing that state. Recovery
requires standing clearance. Native physics and active-balancing plans are
described in the [physics guide](../graaal/physics.md).

The physics viewers share [window, offscreen and draw helpers](physics_viewing_utils.py).
The live examples call those helpers from their own loops. The replay examples
use the playback helper to draw sampled poses. The [passive ragdoll example](passive_ragdoll_example.py)
owns its scenario selector and dummy model playback. It shares
[collider geometry and pose calculations](physics_geometry.py) with the
[ragdoll tuner](../tools/ragdoll_tuner.md).

## Rendering and transparency settings

In `transparency_example.py`, the ImGui settings panel is visible at startup.
`material_example.py` opens it with **F**. Both use
`RenderingSettings` in [`graaal/UI/rendering_settings.py`](../graaal/UI/rendering_settings.py); its `handle_key` callback
handles keyboard editing of visible controls. The rendering resolution slider
scales scene, AO, and AA targets from 25% to 100% of the framebuffer; a final
sampled pass via [`ResolutionPresenter`](../graaal/RenderPasses/render_resolution.py) fills the unchanged
window, then ImGui draws at full size. The top section adjusts the camera's
vertical FoV (20–120 degrees), selects MSAA, LXAA, FAAA, DDAA2, or no
antialiasing, and controls SSAO; transparency controls are below it. **A** toggles
SSAO and **O** switches between its compose and
separate-pass implementations. `example_overlay.py` connects **M**, **A**, **O**,
**P**, and **F** in both scenes. **P** toggles the profiler and starts monitored
submissions; GPU timings appear when the device supports timestamp queries.
Both examples start with `TransparencyPass` defaults.

**Banded + dual Kawase** captures the nearest transparent
layer exactly and uses a complementary banded pair for the remaining tail.
**Down/up pairs** controls the blur (0–8, default 1). **Proxy residual**
subtracts a local albedo estimate before blurring to reduce color halos.
It is **off by default**. **Shade one pixel
per 2 x 2 block** is an optional lower-cost, noisier approximation. Tail
estimation defaults to 1x even when the scene uses MSAA; exact nearest,
transmittance and opaque/MASK geometry still use the selected native sample count.
There is no tail-MSAA control in this UI (see `set_banded_options()` for the API).

**Grazing fade** controls BLEND opacity toward silhouettes. The low-level pass
defaults to enabled from 83 to 86 degrees; materials can provide their own
grazing parameters. **Screen noise cell size** changes the shared tail noise.

Diagnostic views are **final**, **transmittance**, **winner**, **reconstruction**,
**nearest**, **nearest_mask**, and **tail_transmittance**. The **winner** view
shows signed residuals when the proxy is active.

| Input | Action in `transparency_example.py` |
| --- | --- |
| Space / mouse scroll | Toggle orbit / zoom |
| R | Reverse transparent draw order |
| D | Cycle diagnostic views |
| [ / ] | Decrease/increase screen noise-cell size |
| G | Toggle grazing fade |
| M | Toggle 4x MSAA (starts off) |
| P | Show/hide the performance profiler |
| 1 / 3 / 5 / 9 | Select 0 / 1 / 2 / 3 Kawase pairs |
| F | Show/hide the settings panel |
| Left / Right | Change blur pairs while the panel is visible |
| Home | Reset blur to one pair |

No TAA or temporal history is used. MSAA improves opaque/MASK and nearest-layer
edges, but the 1x tail can miss subpixel details. For Python controls, use
`set_kawase_passes()`, `set_banded_options()`, `set_noise_scale()` and
`set_debug_view()` as documented in the [transparency guide](../graaal/stochastic_transparency.md).

The transparency scene loads `resources/small_scene.glb`, with
`Simple_foliage.001` treated as MASK. It supports static glTF node transforms,
base color, roughness/metallic and emission. Normal maps, skinning, animation,
material extensions and sampler overrides are not covered by this example.
The [blue-noise data and license](../graaal/data/blue_noise/README.md) are bundled.

## Scene tests and benchmarks

For untracked in-place edits to ECS object fields, call `touch()` so dependent
readers can observe them. Run the import, scene-state and cache tests with:

```powershell
python -m pytest tests/test_scenes.py tests/test_scene_caches.py tests/test_library_boundaries.py -q -p no:cacheprovider
```

For subsystem regressions, use [ECS tests](../graaal/ECS_API.md),
[material tests](../graaal/materials.md#implementation-and-verification), and
[transparency tests](../graaal/stochastic_transparency.md#implementation-and-verification).
An installed-package CPU smoke test is `python -m graaal`; an offscreen GPU
smoke test is `python -m graaal --offscreen --frames 1` (requires a working
WGPU backend). Import tests do not verify the visual output of a scene.

See [benchmarks](benchmarks/README.md) for CLI usage of diagnostic scripts.
