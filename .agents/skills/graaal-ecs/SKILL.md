---
name: graaal-ecs
description: Use when changing Graaal ECS component storage, NumPy selections and queries, mutation/version tracking, deferred commands, scheduling, or adapters that depend on those contracts.
---

# ECS changes

Use the [ECS API guide](../../../graaal/ECS_API.md) for the contract being changed: entity lifecycle and selections for row validity, structural queries for invalidation, version tokens for cache notification, and commands/scheduling for visibility and recovery. Inspect `graaal/ECS/core.py`, `ecs_schema.py`, `ecs_types.py`, `ecs_commands.py` or `ecs_schedule.py` as appropriate, using `tools/module_inspection.py`.

## Preserve storage and mutation semantics

- Keep full generational entity handles distinct from component rows. Selections expire on structural changes to their store; refetch after structural flushes. Independent stores need joins or gathers to align rows.
- Numeric selection reads copy. `selection.x[:] = ...` changes a temporary; use `assign()`, bracket assignment or the documented augmented assignment. Object reads retain references and in-place changes need `touch()` for dependent caches.
- Use `read(name, out=array)` for reusable gathers with exact dtype and shape. `edit_dense()` borrows numeric columns only within its context: views cannot escape, structural changes are forbidden while borrowed, and exit touches fields even after exceptions without rolling back writes.
- Preserve preflight validation before value/counter changes. Successful nonempty writes advance affected field tokens; field, membership and layout versions have different meanings. Compare version tokens only for the same store and ordered fields.
- Keep stale-handle behavior, object-reference cleanup, multicomponent ordering and duplicate-row NumPy assignment semantics intact. Read the relevant guide sections before changing these paths.

## Preserve command and schedule boundaries

Deferred commands execute FIFO with generation checks. Targets are copied at enqueue time; component payloads remain referenced. A failed flush consumes the failing command, retains earlier edits and leaves the tail pending; it is not a whole-buffer transaction. Deferred tokens belong to their originating buffer.

Systems are synchronous and sequential. Dependencies order callbacks but do not flush commands. Use explicit `flush_after=True` where a later system needs queued edits in the same phase; refetch selections afterward. The caller owns fixed-step accumulation and counts. Preserve graph validation before execution and documented failure recovery.

## Verify the affected behavior

Start with the relevant files from:

```sh
python -m pytest tests/test_ecs_main.py tests/test_ecs_queries.py tests/test_ecs_schedule.py -q -p no:cacheprovider
```

Storage/allocation changes also have focused coverage in [array contracts](../../../tests/test_array_contracts.py), [ECS optimizations](../../../tests/test_ecs_optimizations.py), [follow-up regressions](../../../tests/test_ecs_followup.py) and [tags](../../../tests/test_ecs_tags.py). Run adapter tests when changed row lifetimes, versions or command visibility affect extraction. Use `world.check_invariants()` for explicit storage audits; it is not per-frame work and requires assertions enabled.
