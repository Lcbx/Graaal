# AGENTS.md

## Project
Python game engine built on wgpu and numpy. It has an ECS core and a set of rendering tools that each work on their own.
See [README.md](README.md) for an overview and [graaal/README.md](graaal/README.md)
for APIs and architecture.

## Navigation and skills
Prefer [tools/module_inspection.py](tools/module_inspection.py) for discovering
Python APIs. Use focused text searches, AST inspection or direct file reads when
module inspection cannot answer the question. Use the relevant repo-local skill for its workflow and
guide/test map; read only references relevant to the task:

- [Repository refactoring](.agents/skills/graaal-repo-refactor/SKILL.md): navigation, downstream impact, directory structure, imports and exports.
- [Documentation](.agents/skills/graaal-documentation/SKILL.md): API guides, examples, links and plan/status documents.
- [ECS](.agents/skills/graaal-ecs/SKILL.md): storage, queries, mutations, commands and scheduling.
- [Rendering](.agents/skills/graaal-rendering/SKILL.md): GPU resources, shaders/layouts, draws, materials, passes and UI.
- [Animation](.agents/skills/graaal-animation/SKILL.md): sampling, graphs, IK, skinning and authoring tools.
- [Physics](.agents/skills/graaal-physics/SKILL.md): native integration, fixed ticks, identity and interpolation.
- [Performance](.agents/skills/graaal-performance/SKILL.md): array access, allocations, benchmarks and profiling.

## Architecture
ECS, DrawBatches, Materials, RenderContext and Diagnostics expose their main APIs
from `__init__.py`; edit implementations in `core.py` and sibling modules.
Animation, EcsAdapters, RenderPasses and UI expose individual modules.
Keep package imports free of optional backends until an API requiring them is requested.

## Ground rules
- Keep tools independent of the ECS; shared infrastructure dependencies are allowed. ECS integration belongs in adapters under `graaal/EcsAdapters/`.
- Do what was asked. Suggestions are welcome, but wait for approval before acting on them.
- Explain incorrect assumptions and ask before proceeding. Ask about missing information rather than making assumptions, even when they seem minor.
- Don't add abstraction layers the task doesn't need. Low-level APIs must stay usable.

## Code style
- Indent Python code with tabs, never spaces. Respect required indentation in other formats.
- Keep existing comments and variable names unless the task requires changing them.
- add concise comments when they help readability. do not enumerate steps (ex ```# 1. first step```) 
- Prefer composition over inheritance.
- Prefer procedural code over OOP.
- Prefer concise, readable functional expressions such as comprehensions and short lambdas. Use `map` when it improves clarity.
- Keep call arguments on one line. Break them only if an argument is a complex expression or there are 6+ parameters.
- Use vectorized numpy operations instead of Python loops over array data.

## Validation
- Install from the repository root: `python -m pip install -e ".[render,assets,ui,examples,test]"`.
- CPU smoke test: `python -m graaal`.
- Full suite: `python -m pytest tests -q -p no:cacheprovider`.
- Optional GPU smoke test: `python -m graaal --offscreen --frames 1`.
- Run tests for changed subsystems first. Rendering tests and GPU smoke tests
  require optional dependencies and a working WGPU backend; report skipped or
  unavailable validation rather than claiming it passed.

## Responses
- Be concise. Use simple formatting short paragraphs, minimal headers, bullets only when they help.
- For larger or non-trivial tasks, end with a short summary of the choices made.
