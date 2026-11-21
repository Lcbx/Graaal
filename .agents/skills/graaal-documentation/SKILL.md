---
name: graaal-documentation
description: Use when writing, correcting or reorganizing Graaal API guides, README files, scene/tool documentation, or implementation plan and status documents, including documentation updates after verified code changes.
---

# Documentation maintenance

Start with the [repository overview](../../../README.md) or [engine guide](../../../graaal/README.md) to locate the relevant subsystem guide. Use the [scene catalog](../../../scenes/README.md), [tool guides](../../../tools/README.md) and [benchmark guide](../../../scenes/benchmarks/README.md) for their respective usage contracts. Read only references relevant to the task.

## Content and sources

Document public API usage, architectural boundaries, integration contracts and invariants that maintainers must preserve. Prefer short examples and links to canonical scenes/tests over repeating implementations. Do not put implementation diaries or descriptions of completed work in API guides.

Keep API details in their canonical guides; skills should carry decision rules and links rather than copied API inventories. Check current code and tests before changing claims about behavior. Prefer [module inspection](../../../tools/README.md#module-inspection) for discovering Python APIs; it imports targets and lists declarations, not call sites or proof of runtime behavior. Use focused text searches, AST inspection or direct file reads when module inspection cannot answer the question.

For a known API, narrow inspection with a quoted `--filter 'Class.method*'` glob or `--regex` pattern and add `--source-ranges` to open its implementation directly. Filters are case-sensitive and mutually exclusive; use qualified names to avoid ambiguous matches. Source locations are absolute paths with one-based inclusive line ranges, including decorators; re-exports point to their defining files. Missing locations mean source is unavailable, not that the definition is absent. Keep runtime import/dependency requirements in mind even with filtered output.

Consult an existing `implementation_plans/` document only when its feature is in scope. Keep an existing task plan/status document current with what changed, how and why, and record validation evidence there. Distinguish delivered contracts from proposals and historical measurements: the backlog, GPU crowd evaluator, grass scattering, illumination probes, SDF shadows and learned active-ragdoll plans do not establish implemented APIs. Historical passing tests or timings do not establish current validation or performance thresholds.

## Updates after code changes

For directory/import refactoring, adapt documentation only after the code changes and verification required by the refactoring workflow. Describe the resulting public paths and behavior, and update affected examples, links and navigation. Documentation-only corrections do not require a code refactor or an engine test run unless verifying the corrected claim needs one.

## Check the result

Check relative links and section anchors from each edited file's location. Verify referenced files and commands still exist and examples match current signatures, imports and dependency requirements. Keep CPU-only, optional native and GPU requirements explicit. Report unavailable validation without claiming it passed.
