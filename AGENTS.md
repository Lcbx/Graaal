# AGENTS.md

## Project
Python game engine built on wgpu and numpy. It has an ECS core and a set of rendering tools that each work on their own.
See [README.md](README.md) for an overview and [graaal/README.md](graaal/README.md)
for APIs and architecture.

## Where to look
- ECS storage, queries, commands, and scheduling: `graaal/ECS.py`, `graaal/ecs_*.py`;
  reference: [graaal/ECS_API.md](graaal/ECS_API.md).
- GPU resources, command recording, and shaders: `graaal/RenderContext.py`,
  `graaal/ShaderSyntax.py`, `graaal/ShaderShorthand.py`.
- Draw grouping/culling: `graaal/DrawBatches.py`, `graaal/draw_*.py`,
  `graaal/EcsDrawing.py`.
- Materials and ECS integration: `graaal/Materials.py`, `graaal/MaterialRenderer.py`,
  `graaal/EcsMaterials.py`; reference: [graaal/materials.md](graaal/materials.md).
- Transparency: `graaal/TransparencyPass.py`;
  reference: [graaal/stochastic_transparency.md](graaal/stochastic_transparency.md).
- Runnable examples: [scenes/README.md](scenes/README.md).
- Tests: `tests/test_*.py` (use the subsystem guides to find relevant tests).

Consult specialized guides when working on their subsystems; do not read every
reference file for unrelated changes.

## Ground rules
- Each tool must work without the ECS or any other tool. ECS integration goes in an adapter, not in the tool.
- Do what was asked. Suggestions are welcome, but wait for approval before acting on them.
- If the request rests on a wrong assumption or leaves out critical information, point it out and stop. Don't guess.
- Don't add abstraction layers the task doesn't need. Low-level APIs must stay usable.

## Code style
- Indent with tabs, never spaces.
- Keep existing comments and variable names unless the task requires changing them.
- add concise comments when they help readability. do not enumerate steps (ex ```# 1. first step```) 
- Prefer composition over inheritance.
- Prefer procedural code over OOP.
- Use functional style for small expressions (comprehensions, short lambdas, `map`).
- Keep call arguments on one line. Break them only if an argument is a complex expression or there are 6+ parameters.
- Use vectorized numpy operations instead of Python loops over array data.

## Docs
- Document public API usage, architectural boundaries, integration contracts,
  and invariants that a maintainer must preserve. Prefer short examples and links
  to canonical scenes/tests over repeating implementation details.
- Don't write implementation diaries or describe work done in API guides.
- If a task has a plan/status doc, record what was done, how, and why in it,
  and keep it current as work progresses.

## Validation
- Install from the repository root: `python -m pip install -e ".[render,assets,ui,examples,test]"`.
- CPU smoke test: `python -m graaal`.
- Focused ECS tests: `python -m pytest tests/test_ecs_main.py -q`.
- Full suite: `python -m pytest tests -q -p no:cacheprovider`.
- Optional GPU smoke test: `python -m graaal --offscreen --frames 1`.
- Run tests for changed subsystems first. Rendering tests and GPU smoke tests
  require optional dependencies and a working WGPU backend; report skipped or
  unavailable validation rather than claiming it passed.

## Responses
- Be concise. Use simple formatting short paragraphs, minimal headers, bullets only when they help.
- For larger or non-trivial tasks, end with a short summary of the choices made.