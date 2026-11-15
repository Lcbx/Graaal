# Interactive scenes

Run the examples from the repository root after installing the optional dependencies:

```powershell
pip install -e ".[render,assets,ui,examples,test]"
python main.py scenes/draw_batches_example.py
python main.py scenes/direct_draw_example.py
python main.py scenes/prototype.py
python main.py scenes/transparency_example.py
python main.py scenes/material_example.py
python main.py scenes/msaa_example.py
```

Add `--frames N` to stop a RenderContext scene after N displayed frames, for
example `python main.py scenes/material_example.py --frames 120`. Closing the
window can end a run earlier. The six modules export `run_scene()` and can be
imported without opening a GPU device. `main.py` invokes that entry point and
cleans up the context afterward.

| Scene | What to try |
| --- | --- |
| `draw_batches_example.py` | ECS movement and entity changes, LOD, indirect draws and Hi-Z culling; Space switches camera, scroll zooms. |
| `direct_draw_example.py` | Direct drawing with instance changes and shadows; scroll zooms. |
| `prototype.py` | ECS-backed actor/health panels in ImGui. Combat resolution is a placeholder. |
| `transparency_example.py` | OPAQUE/MASK/BLEND glTF geometry, banded transparency and diagnostics; see below. |
| `material_example.py` | Shared material dictionaries, per-instance overrides, rim lighting and animated stripes. Space orbits; M toggles MSAA; D changes a default material; F opens settings. |
| `msaa_example.py` | Minimal full-window rotating triangle; 1 selects single sampling and 4 selects 4x MSAA. |

These files are runnable examples rather than a stable scene API. For reusable
components, see the [engine guide](../README.md), [ECS reference](../graaal/ECS_API.md),
[material guide](../graaal/materials.md), and
[transparency guide](../graaal/stochastic_transparency.md).

## Rendering and transparency settings

In `transparency_example.py`, the ImGui settings panel is visible at startup.
`material_example.py` opens it with **F**. Both use
`RenderingSettings` in `rendering_settings.py`; its `handle_key` callback
handles keyboard editing of visible controls. The top section selects MSAA,
LXAA, FAAA, DDAA2, or no antialiasing, and controls SSAO; transparency controls are
below it. **A** toggles SSAO and **O** switches between its compose and
separate-pass implementations.
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

See [benchmarks](benchmarks/README.md) for CLI usage of diagnostic scripts.
