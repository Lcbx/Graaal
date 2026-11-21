---
name: graaal-animation
description: Use when editing Graaal skeletal sampling, pose tasks, animation graphs, events or sync, root motion, IK, skinning integration, or the timeline/graph authoring tools and JSON projects.
---

# Animation and authoring changes

Use the [animation guide](../../../graaal/skeletal_animation.md) for the affected API and `tools/module_inspection.py` for implementation navigation. CPU resources and evaluators live in `graaal/Animation/`, ECS playback in `graaal/EcsAdapters/`, GPU palettes in `graaal/RenderContext/SkinMatrixBuffer.py`, and the independent material bridge in `graaal/Materials/MaterialSkinning.py`.

## Runtime contracts

- Skeletons are depth-first preorder forests with contiguous subtrees. Immutable resources copy inputs to read-only float32 arrays; rotations use xyzw and matrices act on column vectors with local TRS composition. Clips must match skeleton bone order/count.
- Clips retain both endpoints; duration is `(frames - 1) / fps`. Runtime interpolation uses shortest-path nlerp, while glTF resampling uses slerp. Preserve looping, reverse crossings, coincident-event order and synchronized phase behavior when changing clocks.
- Controllers choose clips/times/weights and emit events and requested root motion. Applications decide accepted movement and physics order. Keep physics queries and world/model conversions outside pose evaluators.
- Batch by compatible skeleton and vectorize over characters/bones. Reusable `PosePlan` outputs are views that must be consumed or copied before the next evaluation, including a failing evaluation. Separate pre/post-physics plans exchange local poses; they do not own scheduling.
- Adapters retain full generational owner handles and refetch after structural changes. GPU skinning lets owners share geometry; CPU deformation needs independent meshes. Shared palettes use 3x4 float32 skin matrices. Read [GPU palette ownership](../../../graaal/skeletal_animation.md#gpu-skin-matrices) before changing reconciliation, range reuse or draw removal.

For animation/physics integration, preserve controller advancement, movement acceptance, pre-physics pose work, physics, post-physics IK/ragdoll composition, final skin matrices and upload order as applicable. Read the [physics guide](../../../graaal/physics.md) for native authority and fixed ticks.

The [status snapshot](../../../implementation_plans/skeletal_animation_current.md) distinguishes delivered features from the [roadmap](../../../implementation_plans/skeletal_animation_pipeline.md). GPU vertex skinning exists; a GPU crowd pose evaluator remains proposed. Do not treat roadmap code or budgets as supported APIs.

## Authoring tools

Read the relevant [timeline](../../../tools/animation_timeline.md), [graph](../../../tools/animation_graph.md) or [project](../../../tools/animation_project.md) guide when changing tools. Both editors share loading, CPU preview and JSON projects while remaining independently usable without ECS or GPU initialization.

Keep graph drafts separate from compiled previews. Invalid compilation or loading preserves the prior usable session. Validate graph definitions and imported marker tracks before replacing state. Apply changes explicitly; preserve stable marker selection and coincident-event order.

Projects store authored graph/markers in explicit JSON sidecar saves, separate from source glTF/GLB assets. Validate clip count/names/order and source/baked durations, and compile against imported tracks before installing a loaded project. Validate serialization and complete the temporary write before replacing a destination. Timeline marker edits must preserve a loaded graph on later saves. Preview clocks and transient controls are not persisted.

## Verify the affected behavior

| Changed area | Tests under `tests/` |
| --- | --- |
| Sampling, transforms and ownership | `test_skeletal_animation.py`, `test_animation_buffers.py`, `test_math_quaternion.py` |
| Layers, samples and pose tasks | `test_animation_layers.py`, `test_animation_sample_batches.py`, `test_animation_blend_space.py`, `test_pose_tasks.py` |
| Controllers, events and sync | `test_animation_state_machine.py`, `test_animation_events.py`, `test_animation_sync.py`, `test_ecs_animation.py`, `test_ecs_animation_sync.py` |
| Graph definitions and ECS rules | `test_animation_graph.py`, `test_ecs_animation_state_machine.py` |
| Root motion and IK | `test_root_motion.py`, `test_ecs_root_motion.py`, `test_inverse_kinematics.py` |
| Import and skinning | `test_skeletal_gltf.py`, `test_pose_palette.py`, `test_skin_matrix_buffer.py`, `test_material_skinning.py`, `test_material_gpu_skinning.py` |
| Authoring sessions/projects | `test_animation_project.py`, `test_animation_graph_ui.py`, `test_animation_graph_playback.py`, `test_animation_timeline_ui.py`, `test_animation_timeline_tool.py` |

Run relevant files with `python -m pytest tests/<file>.py -q -p no:cacheprovider`; include affected scene tests for integration changes. Use the [scene catalog](../../../scenes/README.md#headless-animation-and-physics) for bounded headless examples. GPU skinning requires rendering dependencies/backend; tool imports and CPU preview should remain headless-testable.
