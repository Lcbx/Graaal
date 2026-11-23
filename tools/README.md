# Repository tools

## Module inspection

`module_inspection.py` prints a public API outline with signatures, annotations,
docstrings and comments immediately preceding declarations. It accepts a Python
file, package directory or directory of modules. Package imports retain their
qualified names so relative imports work.

```sh
python tools/module_inspection.py graaal/ECS/core.py
python tools/module_inspection.py graaal/ECS/core.py --only classes methods properties --no-comments --no-docstrings
python tools/module_inspection.py graaal/ECS/core.py --include-imports
python tools/module_inspection.py graaal/ECS/core.py --only imports --no-docstrings
python tools/module_inspection.py graaal/ECS --source-ranges --no-comments --no-docstrings
python tools/module_inspection.py graaal/ECS --filter '*Query*' --source-ranges
python tools/module_inspection.py graaal/ECS --regex '^ECS\.(query|register)$' --no-docstrings
python tools/module_inspection.py graaal -r -o api.txt
```

`--only` accepts one or more of `classes`, `functions`, `methods`, `properties`,
`attributes` and `imports`. Classes remain as containers for selected members; selecting
only `classes` omits their methods and attributes. Properties include getters,
setters and deleters. Class members are limited to those defined on the class,
including `__init__`; other private names are omitted. Module `__all__`, when
present, controls exports and includes re-exported classes and functions.

`--include-imports` adds import statements to the outline; `--only imports`
selects just imports. In Python, use `dump_module(module, include_imports=True)`
or `dump_module(module, only={"imports"})`. Imports are collected from available
source in source order, preserving aliases and relative import syntax, with
multiline statements normalized to one line. This includes conditional and local
imports regardless of `__all__`; it does not detect dynamic imports such as
`importlib.import_module()`. Import listing is omitted by default.

`--source-ranges` adds `# source: /absolute/path/module.py:START-END` before
classes, functions, methods and each property accessor. Line numbers are
one-based and inclusive, covering decorators through the end of the definition.
Re-exports and aliases point to the definition's file; wrapped functions follow
`__wrapped__`. Locations remain visible with `--no-comments` and are omitted
when source is unavailable, such as for built-ins. In Python, use
`dump_module(module, source_ranges=True)`.

`--filter PATTERN` matches definition names using case-sensitive shell wildcards
(`*`, `?`, `[abc]`); a plain string matches an exact name, and `*text*` matches
names containing `text`. `--regex PATTERN` uses case-sensitive regular expression
search instead; use `^` and `$` for exact matches. These options are mutually
exclusive. Quote patterns to prevent shell expansion.

Both filters check bare names (`query`), class-qualified names (`ECS.query`),
and module-qualified names (`graaal.ECS.core.ECS.query`), including exported
aliases and attributes. A matching class includes its members, subject to
`--only`; a matching member keeps its containing classes. Private names remain
excluded unless exported by module `__all__`. Comments, docstrings and source
ranges stay with the retained definitions; module headings and docstrings remain.
Import listings are independent of definition filters. In Python, use
`dump_module(module, name_filter="*Query*")` or
`dump_module(module, regex=r"^ECS\.query$")`.

`-r` includes subdirectories when given a directory or `__init__.py`.
`--no-comments` and `--no-docstrings` work independently. The same filters are
available through `tools.module_inspection.dump_module()`:

```python
from graaal import ECS
from tools.module_inspection import dump_module

print(dump_module(ECS, only={"classes", "methods"}, include_comments=False, include_docstrings=False))
```

This is runtime inspection: target modules are imported, their top-level code
runs, and their dependencies must be available. Import failures stop the dump
and identify the affected module. The output is an outline, not an executable
stub or a list of references and call sites. Regression tests are in
[`tests/test_module_inspection.py`](../tests/test_module_inspection.py).

## Animation tools

- [Animation timeline](animation_timeline.md): inspect tracks and author events and sync markers.
- [Animation graph](animation_graph.md): configure and preview a graph for a supported skinned asset.
- [Authoring projects](animation_project.md): explicitly save/load graph definitions and markers in shared JSON sidecar files.

Both tools use the shared asset loader and CPU preview; neither requires the other to run.

## Physics tools

- [Ragdoll tuner](ragdoll_tuner.md): edit per-bone colliders, mass, hard and soft limits, and neutral springs in a live physics preview.
