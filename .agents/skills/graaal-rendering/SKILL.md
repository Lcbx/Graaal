---
name: graaal-rendering
description: Use when editing Graaal wgpu resources, WGSL or shader templates, NumPy/GPU layouts, draw batching and culling, materials, render passes, postprocessing, or rendering UI integration.
---

# Rendering changes

Read the relevant [engine contracts](../../../graaal/README.md#architecture-and-maintenance-contracts). Use [materials](../../../graaal/materials.md), [transparency](../../../graaal/stochastic_transparency.md) or [antialiasing](../../../graaal/antialiasing.md) only for the paths involved. Existing [direct drawing](../../../scenes/direct_draw_example.py), [material](../../../scenes/material_example.py) and [transparency](../../../scenes/transparency_example.py) scenes are integration examples.

## Ownership, ordering and lifetimes

- Passes record into caller-owned commands; applications submit and present. Keep graphics policy out of ECS extraction adapters. Texture-creation helpers that upload and submit are documented exceptions; inspect their contracts rather than generalizing them to pass recording.
- `DrawBatches.sync()` can change `instance_order`. Reorder every per-instance attribute and metadata table together. Visibility belongs to one view and must be prepared again after layout invalidation. Frustum/Hi-Z culling records GPU work; it is not a CPU visibility query.
- GPU resources belong to a RenderContext device/generation and cannot survive its replacement or cleanup. Caller targets must match dimensions, formats, sample counts and usage. Resize and update camera inputs before recording; preserve the transparency guide's stage order.
- Keep generic material schema/packing separate from default lighting/alpha policy. Nested dictionary edits need `touch_materials()`; ECS field writes already advance versions. Preserve distinct stable `surface_id` values across repacking.
- Frame data for multiple views recorded before one submission needs separate buffers or aligned ranges. Updating buffer contents and replacing a buffer handle have different bind-group/cache consequences.

## Shaders and data layouts

Use existing reflected layouts, `ShaderInterface`, `UniformInfo` offsets/strides and canonical NumPy dtypes rather than guessing a packed struct matches WGSL. Inspect both CPU packing and shader consumers when changing the ABI. Preserve padded matrix layouts and explicit caller-provided binding layouts.

For built-in material passes, read [bind groups and lifetime](../../../graaal/materials.md#bind-group-convention-and-lifetime): group 0 is shared Frame/Lighting, group 1 is pass indexing or textures, group 2 holds material resources, and group 3 is raster pass depth/noise. This convention is specific to those passes; low-level custom layouts remain usable. DrawBatches defaults to group 1 for instances/visible indices and exposes overrides.

Read the [shader utilities](../../../graaal/README.md#resources-and-pipelines-through-rendercontext) before changing shorthand/Jinja processing. Reuse granular shader includes/macros; material-family helpers use `PREFIX` for uniqueness. Built-in shaders belong in `graaal/shaders/` and installed package data; `scenes/shaders/` is for example-specific assets.

For MSAA changes, inspect both declared texture types and load helpers, including layered textures, resolve targets and 1x/4x variants. Do not assume the banded tail uses the scene's native sample count or that this renderer has temporal history. Shared CPU settings belong in `RenderPasses/pass_settings.py`; UI displays the tool's settings.

The [blue-noise contract](../../../graaal/data/blue_noise/README.md) defines threshold normalization, bundled data/license and device-scoped texture caching. Preserve those rules when touching noise loading.

## Verify the affected path

Choose focused tests rather than running every rendering group for every change:

| Changed area | Tests under `tests/` |
| --- | --- |
| Resource/device lifetime, commands | `test_render_context.py`, `test_resource_lifetimes.py`, `test_procedural_api.py` |
| Shader layouts, source and reflection | `test_shader.py`, `test_shader_source.py`, `test_shader_pruning.py`, `test_array_contracts.py` |
| Grouping, visibility and culling | `test_draw_batches.py`, `test_material_culling.py` |
| Material packing/extraction/recording | `test_materials.py`, `test_ecs_materials.py`, `test_material_recording.py`, `test_material_pooling.py` |
| Transparency | `test_transparency_pass.py`, `test_transparency_exact_nearest.py`, `test_transparency_msaa.py`, `test_banded_tail.py` |
| Postprocessing/presentation/settings | `test_postprocessing_pass.py`, `test_antialiasing.py`, `test_render_resolution.py`, `test_settings_validation.py` |
| UI and diagnostics | `test_imgui.py`, `test_diagnostics.py`, `test_diagnostics_visualizer.py` |

Run selected files with `python -m pytest tests/<file>.py -q -p no:cacheprovider`. Check the guides and neighboring tests for feature-specific coverage such as MASK, skinning, blue noise or GPU banded-tail behavior. GPU tests and `python -m graaal --offscreen --frames 1` require optional dependencies and a working backend; report skips or unavailable validation. Imports alone do not verify rendering.
