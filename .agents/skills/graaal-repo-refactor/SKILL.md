---
name: graaal-repo-refactor
description: Use when locating Graaal modules and APIs, assessing downstream impact, moving or renaming files and packages, or changing imports and public exports. Documentation-only maintenance uses graaal-documentation.
---

# Repository navigation and refactoring

Run commands from the repository root. Prefer the [module inspection tool](../../../tools/module_inspection.py) for discovering Python APIs. Use focused text searches, AST inspection or direct file reads when module inspection cannot answer the question. Read its [usage guide](../../../tools/README.md#module-inspection) for filters and runtime behavior.

```sh
python tools/module_inspection.py graaal/ECS/core.py --only classes methods properties --no-comments --no-docstrings
python tools/module_inspection.py graaal/ECS/core.py --filter 'ECS.query*' --source-ranges --no-comments --no-docstrings
python tools/module_inspection.py graaal/ECS --regex '^ECS\.(query|register)$' --source-ranges --no-comments --no-docstrings
python tools/module_inspection.py graaal/ECS/core.py --only imports --no-docstrings
python tools/module_inspection.py graaal/ECS -r --no-comments --no-docstrings
```

Start with the relevant module or package, then read the implementations needed for the change. Use `--include-imports` when both signatures and imports matter. Avoid dumping the entire engine when a subsystem is sufficient.

When a definition name is known, narrow the outline with `--filter` (case-sensitive glob) or `--regex` (case-sensitive regex search); these options are mutually exclusive. Quote patterns. Filters match bare, class-qualified and module-qualified names, including exported aliases. Prefer qualified patterns when a bare name is ambiguous. A matching class includes its members subject to `--only`; a matching member retains its containing class. Import listings are independent of definition filters.

Use `--source-ranges` when the next step is reading or editing a definition. Open the emitted `# source: /absolute/path/module.py:START-END` directly instead of searching again for the name. Ranges are one-based and inclusive, include decorators, and point re-exports/aliases to their implementation files; methods and property accessors have individual locations. Locations remain visible with `--no-comments` and are omitted when source is unavailable. Read surrounding code only as needed for context.

Inspection imports targets and executes their top-level code. Dependencies must be available; import failures identify the affected module. Do not inspect executable entry points with import-time side effects just to discover their contents. Read those files directly. The outline omits most private names and is neither a dependency graph nor a call-site index. Inspect downstream modules and tests before changing an interface; use a focused Python AST scan when actual import/reference sites are needed. Use directory listings and direct reads for Markdown, shaders and other non-Python files.

Name filters reduce output, not the modules imported. Source ranges locate definitions, not downstream uses; retain the separate caller/impact checks before changing interfaces.

## Find the relevant contracts

Read only the guide sections relevant to the task. Paths in this table are relative to the repository root; links resolve from this skill.

| Area | Implementations | Canonical guide |
| --- | --- | --- |
| ECS storage, queries, commands, scheduling | `graaal/ECS/` | [ECS API](../../../graaal/ECS_API.md) |
| CPU animation, graphs, pose tasks, sampling | `graaal/Animation/` | [Skeletal animation](../../../graaal/skeletal_animation.md) |
| ECS extraction and shared transforms | `graaal/EcsAdapters/` | [Engine boundaries](../../../graaal/README.md#architecture-and-maintenance-contracts) |
| Native physics, characters, ragdolls | `graaal/EcsAdapters/`, `graaal/PhysicsRagdoll.py` | [Physics](../../../graaal/physics.md) |
| GPU resources, commands, meshes, shader utilities | `graaal/RenderContext/` | [Resources and pipelines](../../../graaal/README.md#resources-and-pipelines-through-rendercontext) |
| Draw grouping and culling | `graaal/DrawBatches/`, `graaal/EcsAdapters/EcsDrawing.py` | [Draws and visibility](../../../graaal/README.md#shared-draws-and-independent-visibility) |
| Materials and ECS integration | `graaal/Materials/`, `graaal/EcsAdapters/EcsMaterials.py` | [Materials](../../../graaal/materials.md) |
| Transparency, HZB, postprocessing, presentation | `graaal/RenderPasses/` | [Transparency](../../../graaal/stochastic_transparency.md), [antialiasing](../../../graaal/antialiasing.md) |
| CPU/render diagnostics and GPU profiling | `graaal/Diagnostics/` | [Diagnostics](../../../graaal/README.md#examples-and-diagnostics) |
| ImGui controls and diagnostic displays | `graaal/UI/` | [Scene controls](../../../scenes/README.md#rendering-and-transparency-settings) |
| Timeline, graph editor, shared project files | `tools/` | [Tool guides](../../../tools/README.md#animation-tools) |
| Runnable examples and benchmarks | `scenes/`, `scenes/benchmarks/` | [Scenes](../../../scenes/README.md), [benchmarks](../../../scenes/benchmarks/README.md) |

## Type annotations

Python modules under `graaal/` must have type annotations verified by mypy. When adding or modifying these modules, annotate function and method parameters and return values, and annotate variables where inference is insufficient.

Run `python -m mypy` from the repository root using the existing `[tool.mypy]` configuration in `pyproject.toml`, which targets `graaal/` and disallows untyped definitions. Resolve typing errors introduced by the change and report any remaining errors or unavailable validation explicitly.

## Change repository structure

Before a move or interface change, identify the affected files, public exports, downstream imports and tests. For changes crossing packages, briefly state the intended dependency direction and interface changes. Keep the plan proportional to the task.

Make directory and code changes first, including imports, exports, package data and entry points that depend on the old paths. Do not create compatibility scripts, forwarding modules or import shims to preserve old import locations. Update callers to the intended paths.

Then verify that the code works: run affected subsystem tests, the CPU smoke test, and library-boundary tests when imports or dependencies change. Resolve failures before adapting documentation. If required validation is unavailable, report the gap and do not describe the restructuring as verified.

Finally adapt guides, examples in documentation, links and navigation to the verified structure, following the [documentation skill](../graaal-documentation/SKILL.md).

## Verification scope

Use the affected guide's regression tests first. Package moves and optional-dependency changes also need:

```sh
python -m pytest tests/test_library_boundaries.py tests/test_scenes.py tests/test_scene_caches.py -q -p no:cacheprovider
python -m graaal
```

For changes to the inspection tool, run `python -m pytest tests/test_module_inspection.py -q -p no:cacheprovider`. Use AGENTS.md for installation, full-suite and GPU smoke commands. Import tests check import behavior, not rendered output. Report skipped GPU/native tests and unavailable backends explicitly.

