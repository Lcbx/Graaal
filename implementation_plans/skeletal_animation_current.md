# Skeletal animation implementation status

Source plan: [skeletal_animation_pipeline.md](skeletal_animation_pipeline.md).

## Milestone 1: foundation — in progress

- Started the standalone CPU evaluator in `graaal/SkeletalAnimation.py`: immutable copied skeleton and uniformly sampled clip arrays; validated depth-first preorder; xyzw float32 local transforms; normalized, sign-continuous rotations; batched clip sampling and depth-batched local-to-model transforms; row-major 3x4 skin matrices.
- Clips include both endpoints at `(frame_count - 1) / fps`; loop playback wraps at that duration, so authored seamless loops must match at the endpoints. Sampling uses normalized linear quaternion interpolation and supports negative times, clamping, and single-frame clips.
- Added focused CPU regression coverage for hierarchy validation, interpolation, looping, resource ownership, and transforms.
- Validation: `tests/test_skeletal_animation.py` passes with NumPy and pytest installed in a temporary directory. Reusable GPU and ECS integration are not yet available to validate.
- `SkeletalAnimation.skin_vertices` provides CPU mesh deformation with inverse-transpose normals. `MaterialRenderer.update_mesh_vertices` updates raster and deferred vertices together, so ordinary material shaders and transparency work with animated geometry. Each independently animated pose currently needs its own registered mesh and conservative bounds; vertex uploads scale with mesh size.
- `scenes/animation_example.py` loads Greenman's rig and clips with an asset-specific loader and renders its CPU-skinned mesh through the material system. A reusable GPU skinning renderer remains outstanding.
- Validation: 51 focused animation and material tests pass, including transparent rendering after a vertex update; the updated Greenman scene passed a three-frame GPU smoke test.
- Remaining: reusable asset import/resampling and pose buffers, ECS player components and lifecycle-safe batching, and a reusable GPU skin buffer and skinning renderer. Milestone 1 is not complete until the planned ECS and GPU deliverable works.

## Later milestones

Milestones 2–6 have not started. Crossfades, root motion, events, graphs, physics integration and the crowd GPU evaluator remain future work.
