---
name: graaal-physics
description: Use when changing Graaal optional pyjoltphysics integration, ECS bodies/colliders, native queries/events/joints, characters, ragdolls, fixed scheduling, interpolation, checkpoints or native resource ownership.
---

# Physics integration changes

Read the relevant [physics guide](../../../graaal/physics.md) sections and inspect affected modules with `tools/module_inspection.py`. The [integration status](../../../implementation_plans/physics_integration_plan.md) records delivered scope and remaining backend gaps; [learned active-ragdoll control](../../../implementation_plans/active_ragdolls.md) is a separate proposal.

## Dependency and authority boundaries

Standalone applications use `pyjoltphysics.World` directly. ECS integration belongs in `EcsAdapters`; skeleton conversion belongs in standalone `PhysicsRagdoll`. Components contain numeric declarations/asset keys, while the adapter owns native handles/resources. Constructing an adapter loads the optional backend once; component imports remain free of native and graphics dependencies.

Use the repository's pinned `physics` extra and supported backend contracts. Do not infer native capabilities from plans. Keep `physics.world` accessible for advanced native use rather than hiding it behind a replacement API.

Dynamic bodies own their native pose/velocity; ECS stores a mirror and ordinary mirror writes do not teleport. Kinematic transforms supply targets. Static moves require explicit teleport. Native characters own accepted motion. Render interpolation must not write into gameplay state or controller targets.

## Ticks, identities and lifetimes

- The caller owns fixed-step accumulation/count and render alpha. One adapter `step` is one native tick with `fixed_dt == 1 / settings.tick_hz`. Preserve `reconcile -> prepare_inputs -> step -> pull -> dispatch_events`, inserting animation/IK phases as documented. Scheduler dependencies alone do not flush queued ECS edits.
- Full uint64 generational ECS handles and native uint32 UIDs are distinct. Resolve rows for each snapshot; native compaction and ECS slot reuse must not change ownership. Body, joint, character and ragdoll-bone event namespaces need their documented mappings. Entity zero is valid; use the documented world/unresolved sentinel.
- Native snapshots are read-only borrowed views. Release promptly and copy retained data. Dispatch events once per fixed tick after pull, preserving retired ownership mapping long enough for routing and suppressing delivery to deleted owners.
- Interpolation retains owned pose copies. Pull on every tick even for sleeping bodies. Render transforms preserve requested handle order and duplicates; UID replacement and teleport reset the relevant history. See [interpolation](../../../graaal/physics.md#render-interpolation) for edge cases.
- Shape/topology changes rebuild resources while preserving appropriate native state. Shared body/ragdoll/character capacity must be preflighted before removals. At capacity, the pinned backend can need removals plus `flush()` before additions; this does not advance time. Queries must not race reconciliation. Preserve scheduler-thread ownership and idempotent shutdown.

Read guide restrictions before changing compounds, sensor combinations, nonuniform scale, ragdoll mapping, query filters or checkpoint/reset behavior. A body/joint replay checkpoint is not a complete character/controller/application-state checkpoint.

## Verify the affected path

Install the native extra only when needed: `python -m pip install -e ".[physics]"`.

| Changed area | Tests under `tests/` |
| --- | --- |
| Native world and rigid-body adapter | `test_physics_headless.py`, `test_ecs_physics.py` |
| Events, queries, filters and joints | `test_ecs_physics_interactions.py` |
| Ragdolls and characters | `test_physics_ragdoll.py`, `test_ecs_characters.py`, `test_ragdoll_example.py`, `test_character_example.py` |
| Checkpoints and diagnostics | `test_physics_replay.py`, `test_physics_diagnostics.py`, `test_benchmark_physics.py` |
| Movement/animation integration | `test_physics_locomotion_example.py`, `test_foot_ik_example.py`, `test_ecs_material_transforms.py` |

Run selected files with `python -m pytest tests/<file>.py -q -p no:cacheprovider`. Include `test_library_boundaries.py` for dependency/import changes. A bounded integration run is `python -m scenes.physics_example --headless --frames 240`. See the [scene catalog](../../../scenes/README.md#headless-animation-and-physics) for native ragdoll/character runs. Report native-backend skips separately from passing CPU imports; headless native tests do not require a GPU.
