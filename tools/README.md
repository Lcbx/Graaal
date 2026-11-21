# Repository tools

## Module inspection

`module_inspection.py` prints a public API outline with signatures, annotations,
docstrings and comments immediately preceding declarations. It accepts a Python
file, package directory or directory of modules. Package imports retain their
qualified names so relative imports work.

```sh
python tools/module_inspection.py graaal/ECS.py
python tools/module_inspection.py graaal/ECS.py --only classes methods properties --no-comments --no-docstrings
python tools/module_inspection.py graaal/ECS.py --include-imports
python tools/module_inspection.py graaal/ECS.py --only imports --no-docstrings
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
