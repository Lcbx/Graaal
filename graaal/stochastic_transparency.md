# Banded transparency

[TransparencyPass.py](TransparencyPass.py) renders exact nearest transparency,
two complementary banded tail winners, and dual Kawase tail blur.
The caller supplies geometry, materials, camera values, targets and commands.
[MaterialRenderer](materials.md) records these stages for dictionary materials;
the [interactive scene](../scenes/transparency_example.py) shows direct use.

## Implementation and verification

- **Entry points:** [TransparencyPass.py](TransparencyPass.py) owns the pass
  resources and recording methods; [transparency_utils.py](transparency_utils.py)
  wires the transparent stages; [TransparencyData.py](TransparencyData.py)
  supplies input layouts.
- **Invariants:** Resize/update camera inputs before recording, preserve the
  [recording order](#recording), and keep geometry metadata and deferred inputs
  in the same packed instance order. Caller-owned targets must satisfy the
  [geometry and target contracts](#geometry-and-targets). This pipeline uses
  no temporal accumulation.
- **Example:** [transparency_example.py](../scenes/transparency_example.py).
- **Tests:** [pass recording](../tests/test_transparency_pass.py),
  [nearest winner](../tests/test_transparency_exact_nearest.py),
  [MSAA](../tests/test_transparency_msaa.py), and
  [banded tail](../tests/test_banded_tail.py).

```sh
python -m pytest tests/test_transparency_pass.py tests/test_transparency_exact_nearest.py tests/test_transparency_msaa.py tests/test_banded_tail.py -q
```

Rendering tests require a working WGPU backend.

## Useful APIs

Import from `graaal.TransparencyPass`. Initialize RenderContext before creating
GPU resources.

| API | Usage |
| --- | --- |
| `TransparencyPass(materials=..., base_textures=..., vertices=...)` | Create the pass with GPU material, texture-array and deferred-vertex resources. Optional `target_format` must match the output attachment. |
| `resize(size, targets=None)` | Allocate or adopt targets before updating camera data or recording. `targets` is keyword-only. Nonpositive sizes retain existing resources. |
| `update_frame(view=..., projection=..., camera_position=..., near=..., far=..., frame=None, lighting=None)` | Upload camera values and return `view @ projection` for culling. All arguments are keyword-only. |
| `update_camera(camera, frame=None, lighting=None)` | Camera-object adapter to `update_frame()`. Optional arguments are keyword-only. |
| `create_geometry_passes(metadata, transparent=..., instances=...)` | Create DrawBatches passes for caller-owned metadata and transforms. Use `transparent=False` for OPAQUE/MASK, `True` for BLEND. |
| `opaque_pass(commands, clear=..., load=False)` | Render opaque/MASK geometry. `load=True` preserves caller-owned color and depth. |
| `nearest_pass(commands)`, `shade_nearest(commands, geometry)` | Capture and shade the nearest nonzero-opacity transparent fragment at each native sample. |
| `transmittance_pass(commands)` | Accumulate tail transmission and clear band allocation resources. Record once per frame, before allocation. |
| `tail_statistics_pass(commands)` | Accumulate single-sample tail statistics when `mixed_tail` is true. |
| `banded_pass(commands)`, `shade_winner(commands, geometry)` | Allocate both complementary winners, then shade both in one dispatch. |
| `compose(commands, output)` | Blur the tail and compose into a caller-provided color attachment. Does not submit or present. |
| `set_opaque_inputs(color, depth)` | Adopt sampleable opaque textures. Pass both as `None` to restore internally allocated textures. |
| `set_sample_count(1 or 4)` | Select native MSAA and rebuild geometry/targets while retaining live settings. |
| `set_kawase_passes(count)` | Set 0-8 down/up pairs; default 1. Zero displays the unblurred tail. |
| `set_banded_options(proxy=None, subset=None, tail_msaa=None)` | Change boolean options; `None` preserves the current value. All default to off. |
| `set_noise_scale(pixels)` | Set the positive finite 1x screen-noise cell size; default 1. |
| `set_grazing_fade(GrazingFade(...))` | Set angular BLEND opacity and MASK coverage fade; default enabled from 83 to 86 degrees. |
| `set_debug_view(name)` | Select one of the views below. |
| `set_ambient_occlusion(bool)`, `set_ambient_occlusion_settings(settings)` | Configure AO in composition using `AmbientOcclusionSettings`. |

```python
renderer = MaterialRenderer(items, kawase_passes=1)
renderer.set_banded_options(proxy=True, subset=False)
renderer.transparency.set_kawase_passes(3)
renderer.transparency.set_debug_view("reconstruction")
```

`proxy=True` subtracts an opacity-weighted texture-albedo estimate before
blurring and restores its local contribution afterward. `subset=True` shades
one fixed pixel per 2x2 block and uses compact payloads; a wider blur can help
with the extra noise. Both options default off.

`sample_count=4` samples opaque/MASK geometry, transmission and the nearest
layer at four hardware positions. Tail estimation remains at 1x unless
`tail_msaa=True`; `tail_sample_count` reports its sample count, and `mixed_tail`
reports whether native and tail counts differ. Noise cells follow the tail
sample density: native tail MSAA halves the base cell width.
No temporal accumulation is used.

## Recording

Call `resize()` and `update_frame()` or `update_camera()` first. After opaque
rendering, record transparent geometry using its shared `GeometryPasses`:

```python
with transparency.nearest_pass(commands) as render:
	batches.draw(render, visibility, "main", pass_override=geometry.nearest)
transparency.shade_nearest(commands, geometry)
with transparency.transmittance_pass(commands) as render:
	batches.draw(render, visibility, "transmittance")
if transparency.mixed_tail:
	with transparency.tail_statistics_pass(commands) as render:
		batches.draw(render, visibility, "tail_statistics")
with transparency.banded_pass(commands) as render:
	batches.draw(render, visibility, "bands")
transparency.shade_winner(commands, geometry)
transparency.compose(commands, output)
```

Alternatively, `graaal.transparency_utils.record_transparency(commands,
transparency, geometry, draw)` records the transparent stages with a
`draw(render, pass_name)` callback. The callback uses `geometry.nearest` for
`"nearest"` and the named geometry pass otherwise. Composition and submission
remain with the caller.

## Geometry and targets

`GeometryPasses.render_shader` registers with DrawBatches; `.nearest` is a
pass override for nearest capture. Supply the same geometry object to both
shading methods. Metadata, vertices and materials share one address space
across all transparent draws. Raster vertices use `VERTEX_DTYPE`, deferred
vertices use `SHADING_VERTEX_DTYPE`, and transforms use `MESH_INSTANCE_DTYPE`.
Metadata is a nonempty vec4u table in packed instance order: triangle offset,
material index, stable surface ID and a reserved slot used by the material adapter.

Material hooks supplied through `material_source` or `material_library`
support custom shading, opacity and matched raster/deferred vertex deformation.
`create_geometry_passes()` accepts `geometry_resources`, `shading_resources`,
`geometry_bindings` and `shading_bindings` for additional inputs. Additional
geometry bind groups start at 4; additional shading groups start at 3.

`TransparencyTargets` adopts caller-owned textures matching the requested
width, height, formats, usage and sample layout:

| Target | Format and layout |
| --- | --- |
| `main_color`, `transmittance`, `nearest_alpha` | `rgba16float`, native sample count, sampled/render attachment |
| `depth`, `nearest_depth` | `depth32float`, native sample count, sampled/render attachment |
| `nearest_payload` | `rgba32uint` at 1x; `rgba16uint` at 4x, sampled/render attachment |
| `nearest_extra` | Required at 4x: `rgba16uint`, 4x sampled/render attachment |
| `nearest_color` | `rgba16float`, single-sample sampled/storage, one layer per native sample |
| `winner_color`, `reconstruction_color` | `rgba16float`, single-sample sampled/storage, one layer per tail sample |

Opaque inputs must match the target dimensions and native sample count.
`set_sample_count()` rejects a change incompatible with adopted opaque inputs.
In mixed mode `reconstruction_color.rgb` holds tail mean color; native opacity
is applied during composition. With matched counts it holds the premultiplied
tail contribution.

## Diagnostics and opacity

| View | Display |
| --- | --- |
| `final` | Composed scene |
| `transmittance` | Total scalar transmission including the nearest layer |
| `winner` | Raw tail color; signed residual displayed around neutral grey when proxy is enabled |
| `reconstruction` | Transparent contribution including the exact nearest layer |
| `nearest` | Nearest color multiplied by RGB opacity |
| `nearest_mask` | Nonzero-opacity nearest coverage |
| `tail_transmittance` | Scalar transmission excluding the nearest fragment |

`GrazingFade(enabled=True, start_angle=83, end_angle=86)` fades BLEND opacity
and MASK coverage toward silhouettes; angles must satisfy
`0 <= start_angle < end_angle <= 90`.
MASK applies the alpha cutoff, then dithers surviving samples by the grazing
fade factor before writing opaque depth. Its screen-space thresholds are stable
across the depth and color passes and independent of the blue-noise texture and
tail blur. MASK UVs and depth agree at native MSAA sample positions.

Banded tail color approximates an opacity-weighted mix rather than sorted
layer composition. Atomic allocation order can vary between frames. The 1x
tail can miss subpixel features even with native MSAA. Tail counters must fit
the device storage-buffer limit, and summed weights must remain below 65536
full-opacity tail fragments per sample. Exact nearest capture accepts every
nonzero opacity.
