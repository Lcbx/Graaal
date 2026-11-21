"""API outlines preserve signatures and load packages by their qualified names."""

import ast
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools.module_inspection import dump_module, main


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "module_inspection.py"
SOURCE = '''"""Module documentation."""
from pathlib import Path
from math import sin

# Constant comment.
label: str = "hello"
pending: int
Alias = list[int]
_private = 1

# Function comment.
async def run(value: int = 1, /, *, enabled: bool = True) -> str:
	"""Function documentation."""
	raise AssertionError("Function bodies must not run")

class Parent:
	def inherited(self): pass

# Class comment.
class Item(Parent):
	"""Class documentation."""
	# Field comment.
	field: int
	text = "value"
	def __init__(self, value: int = 0): pass
	def _private_method(self): pass
	# Method comment.
	def method(self, value: int) -> int:
		"""Method documentation."""
		raise AssertionError("Method bodies must not run")
	@staticmethod
	def create(value: int): pass
	@classmethod
	def factory(cls): pass
	# Getter comment.
	@property
	def value(self) -> int:
		"""Property documentation."""
		raise AssertionError("Properties must not run")
	# Setter comment.
	@value.setter
	def value(self, value: int): pass
	# Deleter comment.
	@value.deleter
	def value(self): pass
	class Nested:
		def nested_method(self): pass

class Empty: pass

class _Internal: pass
PublicAlias = _Internal
item = Item()
if True:
	# Conditional comment.
	def conditional(): pass
'''


@pytest.fixture
def module(tmp_path, monkeypatch):
	path = tmp_path / "inspection_fixture.py"
	path.write_text(SOURCE, encoding="utf-8")
	module = ModuleType("inspection_fixture")
	module.__file__ = str(path)
	monkeypatch.setitem(sys.modules, module.__name__, module)
	exec(compile(SOURCE, str(path), "exec"), module.__dict__)
	return module


def test_public_declarations_and_signatures(module):
	text = dump_module(module)
	assert 'label: str = \'hello\'' in text
	assert "pending: int" in text
	assert "Alias = list[int]" in text
	assert "async def run(value: int = 1, /, *, enabled: bool = True) -> str:" in text
	assert "class Item(Parent):" in text
	assert "field: int" in text
	assert "@staticmethod\n\tdef create(value: int):" in text
	assert "@classmethod\n\tdef factory(cls):" in text
	assert "def __init__(self, value: int = 0):" in text
	assert "class PublicAlias:" in text
	assert "class item:" not in text
	assert "item: inspection_fixture.Item" in text
	assert "class Path" not in text and "def sin" not in text
	assert "_private" not in text and "class _Internal" not in text
	tree = ast.parse(text)
	item = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Item")
	assert "inherited" not in {getattr(node, "name", None) for node in item.body}


@pytest.mark.parametrize("comments,docstrings", [(True, True), (False, True), (True, False), (False, False)])
def test_comments_and_docstrings_are_independent(module, comments, docstrings):
	text = dump_module(module, include_comments=comments, include_docstrings=docstrings)
	for marker in ("Constant", "Function", "Class", "Field", "Method", "Getter", "Setter", "Deleter", "Conditional"):
		assert (f"# {marker} comment." in text) == comments
	for marker in ("Module", "Function", "Class", "Method", "Property"):
		assert (f"{marker} documentation." in text) == docstrings
	ast.parse(text)


@pytest.mark.parametrize("kind", ["classes", "functions", "methods", "properties", "attributes"])
def test_member_kind_filters_keep_only_selected_declarations(module, kind):
	text = dump_module(module, include_docstrings=False, only={kind})
	tree = ast.parse(text)
	functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
	classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
	methods = [node for cls in classes for node in cls.body if isinstance(node, ast.FunctionDef)]
	properties = [node for node in methods if node.decorator_list and ast.unparse(node.decorator_list[0]) in {"property", "value.setter", "value.deleter"}]
	attributes = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
	assert bool(functions) == (kind == "functions")
	assert bool(properties) == (kind == "properties")
	assert bool([node for node in methods if node not in properties]) == (kind == "methods")
	assert bool(attributes) == (kind == "attributes")
	assert bool(classes) == (kind != "functions")
	assert ("class Empty:" in text) == (kind == "classes")
	assert ("class Nested:" in text) == (kind in {"classes", "methods"})


def test_properties_include_all_accessors_and_matching_comments(module):
	text = dump_module(module, only={"properties"})
	assert "# Getter comment.\n\t@property" in text
	assert "# Setter comment.\n\t@value.setter" in text
	assert "# Deleter comment.\n\t@value.deleter" in text
	assert "class Empty:" not in text


def test_explicit_exports_include_reexports_and_private_names(module):
	module.__all__ = ["Path", "sin", "_private", "pending", "PublicAlias"]
	text = dump_module(module, only={"classes", "functions", "attributes"}, include_docstrings=False)
	assert "class Path(" in text
	assert "def sin(" in text
	assert "_private = 1" in text
	assert "pending: int" in text
	assert "class PublicAlias:" in text
	assert "class Item" not in text and "def run" not in text and "label:" not in text


def test_recursive_class_aliases_and_unrepresentable_instances(module):
	module.Item.Self = module.Item
	module.Item.Nested.Owner = module.Item
	def bad_repr(self):
		raise RuntimeError("No representation")
	module.Item.__repr__ = bad_repr
	text = dump_module(module)
	assert "class Item(" in text
	assert "item: inspection_fixture.Item" in text
	assert "class Self" not in text and "class Owner" not in text


def test_source_unavailable_and_docstring_escaping():
	module = ModuleType("dynamic_api", 'A path C:\\tools and triple quotes """ ending in "')
	module.count = 3
	text = dump_module(module)
	assert ast.get_docstring(ast.parse(text)) == module.__doc__
	assert "count = 3" in text


def test_invalid_and_empty_filters(module):
	with pytest.raises(ValueError, match="Unknown member kinds: invalid"):
		dump_module(module, only={"invalid"})
	assert not ast.parse(dump_module(module, only=set(), include_docstrings=False)).body


@pytest.mark.parametrize("pattern,names", [
	("run", {"run"}),
	("meth*", {"Item", "method"}),
	("*method", {"Item", "method", "Nested", "nested_method"}),
	("Item.Nested", {"Item", "Nested", "nested_method"}),
	("inspection_fixture.Item.method", {"Item", "method"}),
	("label", {"label"}),
	("pending", {"pending"}),
	("Item.field", {"Item", "field"}),
	("Item.text", {"Item", "text"}),
	("PublicAlias", {"PublicAlias"}),
	("item", {"item"}),
	("RUN", set()),
	("missing*", set()),
])
def test_name_globs_filter_declarations_and_keep_containing_classes(module, pattern, names):
	text = dump_module(module, name_filter=pattern, include_docstrings=False)
	actual = set()
	for node in ast.walk(ast.parse(text)):
		if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
			actual.add(node.name)
		elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
			actual.add(node.id)
	assert actual == names


@pytest.mark.parametrize("options", [{"name_filter": "*"}, {"regex": ""}])
def test_match_all_preserves_default_output(module, options):
	assert dump_module(module, **options) == dump_module(module)


@pytest.mark.parametrize("pattern,names", [
	(r"^run$", {"run"}),
	(r"method$", {"Item", "method", "Nested", "nested_method"}),
	(r"^Item\.method$", {"Item", "method"}),
	(r"^inspection_fixture\.Item\.method$", {"Item", "method"}),
	(r"^(run|conditional)$", {"run", "conditional"}),
])
def test_regex_search_matches_bare_and_qualified_names(module, pattern, names):
	text = dump_module(module, regex=pattern, include_docstrings=False)
	actual = {node.name for node in ast.walk(ast.parse(text)) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
	assert actual == names


def test_matching_class_includes_members_and_respects_kind_selection(module):
	text = dump_module(module, name_filter="Item", only={"classes", "methods"}, include_docstrings=False)
	assert "class Item(" in text and "class Nested:" in text
	assert "def method(" in text and "def nested_method(" in text
	assert "@staticmethod" in text and "@classmethod" in text
	assert "def __init__(" in text
	assert "@property" not in text and "field:" not in text
	assert "class Parent:" not in text and "class Empty:" not in text
	assert "class PublicAlias:" not in text
	assert "def run(" not in text


def test_matching_property_includes_accessors_and_filters_metadata(module):
	text = dump_module(module, name_filter="Item.value", source_ranges=True)
	assert "class Item(" in text
	assert all(f"@{decorator}" in text for decorator in ("property", "value.setter", "value.deleter"))
	assert text.count("# source:") == 4
	assert "# Getter comment." in text and "# Setter comment." in text and "# Deleter comment." in text
	assert "# Function comment." not in text and "# Method comment." not in text
	assert "def method(" not in text and "class Nested:" not in text
	assert not ast.parse(dump_module(module, name_filter="Item.value", only={"methods"}, include_docstrings=False)).body


def test_definition_filters_do_not_filter_imports(module):
	text = dump_module(module, name_filter="missing*", include_imports=True, source_ranges=True, include_docstrings=False)
	assert [ast.unparse(node) for node in ast.parse(text).body] == ["from pathlib import Path", "from math import sin"]
	assert "# source:" not in text and "comment." not in text


def test_name_filters_respect_visibility_and_exported_aliases(module):
	assert not ast.parse(dump_module(module, name_filter="_private", include_docstrings=False)).body
	module.__all__ = ["_private", "PublicAlias"]
	assert "_private = 1" in dump_module(module, name_filter="_private")
	assert "class PublicAlias:" in dump_module(module, name_filter="PublicAlias")
	assert "class PublicAlias:" not in dump_module(module, name_filter="_Internal")


def test_invalid_name_filters_are_rejected(module):
	with pytest.raises(ValueError, match="Invalid regex"):
		dump_module(module, regex="[")
	with pytest.raises(ValueError, match="Use either name_filter or regex"):
		dump_module(module, name_filter="*", regex=".*")


def test_imports_are_opt_in_and_independent_of_member_filters(module):
	assert "from pathlib import Path" not in dump_module(module)
	text = dump_module(module, include_imports=True, only={"classes"}, include_docstrings=False, include_comments=False)
	assert "from pathlib import Path\nfrom math import sin" in text
	assert "class Item(" in text
	assert "def run(" not in text
	assert "documentation." not in text and "comment." not in text
	module.__all__ = []
	text = dump_module(module, only={"imports"}, include_docstrings=False)
	assert [ast.unparse(node) for node in ast.parse(text).body] == ["from pathlib import Path", "from math import sin"]


def test_import_listing_preserves_aliases_relative_imports_and_source_order(tmp_path):
	path = tmp_path / "imports.py"
	path.write_text('''import os as operating_system, sys
from . import sibling as local
from ..other import (
	First,
	Second as Alias,
)
if TYPE_CHECKING:
	from optional_dependency import Thing
def deferred():
	import local_dependency as dependency
try:
	from .fallback import *
except ImportError:
	pass
module = __import__("dynamic_dependency")
''', encoding="utf-8")
	module = ModuleType("imports_fixture")
	module.__file__ = str(path)
	text = dump_module(module, only={"imports"}, include_docstrings=False)
	assert [ast.unparse(node) for node in ast.parse(text).body] == [
		"import os as operating_system, sys",
		"from . import sibling as local",
		"from ..other import First, Second as Alias",
		"from optional_dependency import Thing",
		"import local_dependency as dependency",
		"from .fallback import *",
	]
	assert "dynamic_dependency" not in text


def test_import_listing_without_available_source():
	module = ModuleType("dynamic_api")
	module.dependency = sys
	assert not ast.parse(dump_module(module, only={"imports"}, include_docstrings=False)).body


@pytest.mark.parametrize("comments", [True, False])
def test_source_ranges_cover_classes_callables_decorators_and_accessors(module, comments):
	assert "# source:" not in dump_module(module)
	text = dump_module(module, source_ranges=True, include_comments=comments, include_docstrings=False)
	filename = Path(module.__file__).resolve().as_posix()
	lines = SOURCE.splitlines()
	cases = [
		("", "class Item(Parent):", "class Item(Parent):", "\t\tdef nested_method(self): pass"),
		("", "async def run", "async def run(value: int = 1, /, *, enabled: bool = True) -> str:", '\traise AssertionError("Function bodies must not run")'),
		("\t", "def method", "\tdef method(self, value: int) -> int:", '\t\traise AssertionError("Method bodies must not run")'),
		("\t", "@staticmethod\n\tdef create", "\t@staticmethod", "\tdef create(value: int): pass"),
		("\t", "@classmethod\n\tdef factory", "\t@classmethod", "\tdef factory(cls): pass"),
		("\t", "@property\n\tdef value", "\t@property", '\t\traise AssertionError("Properties must not run")'),
		("\t", "@value.setter\n\tdef value", "\t@value.setter", "\tdef value(self, value: int): pass"),
		("\t", "@value.deleter\n\tdef value", "\t@value.deleter", "\tdef value(self): pass"),
		("\t", "class Nested:", "\tclass Nested:", "\t\tdef nested_method(self): pass"),
		("\t\t", "def nested_method", "\t\tdef nested_method(self): pass", "\t\tdef nested_method(self): pass"),
		("", "class PublicAlias:", "class _Internal: pass", "class _Internal: pass"),
	]
	for indent, declaration, first_line, last_line in cases:
		first = lines.index(first_line) + 1
		last = lines.index(last_line, first - 1) + 1
		assert f"{indent}# source: {filename}:{first}-{last}\n{indent}{declaration}" in text
	assert ("# Getter comment." in text) == comments
	ast.parse(text)


def test_source_ranges_are_omitted_when_source_is_unavailable(module):
	module.__all__ = ["sin"]
	text = dump_module(module, source_ranges=True, include_docstrings=False)
	assert "def sin(" in text and "# source:" not in text
	dynamic = ModuleType("dynamic_definitions")
	exec("def function(): pass", dynamic.__dict__)
	text = dump_module(dynamic, source_ranges=True)
	assert "def function():" in text and "# source:" not in text


@pytest.fixture
def package(tmp_path):
	package = tmp_path / "sample_package"
	(package / "nested").mkdir(parents=True)
	(package / "__init__.py").write_text('"""Package documentation."""\nfrom .api import Public\n__all__ = ["Public"]\n', encoding="utf-8")
	(package / "api.py").write_text('class Public:\n\tdef method(self): pass\n', encoding="utf-8")
	(package / "nested" / "__init__.py").write_text("", encoding="utf-8")
	(package / "nested" / "api.py").write_text('from ..api import Public\nclass NestedPublic(Public): pass\n', encoding="utf-8")
	return package


def cli(*args, cwd):
	return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=cwd, capture_output=True, text=True, encoding="utf-8")


@pytest.mark.parametrize("target,recursive", [("directory", False), ("directory", True), ("init", False), ("init", True), ("file", False)])
def test_cli_loads_qualified_package_names_from_another_directory(package, tmp_path, target, recursive):
	path = {"directory": package, "init": package / "__init__.py", "file": package / "api.py"}[target]
	args = [path, "--no-comments", "--no-docstrings", "--only", "classes", "methods"]
	if recursive:
		args.append("-r")
	result = cli(*args, cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "class Public:" in result.stdout
	assert ("sample_package module" in result.stdout) == (target != "file")
	assert ("sample_package.api module" in result.stdout) == (target != "init" or recursive)
	assert ("sample_package.nested.api module" in result.stdout) == recursive
	assert "Package documentation." not in result.stdout


def test_cli_loose_module_directory_recursive_and_output(tmp_path):
	directory = tmp_path / "loose"
	(directory / "nested").mkdir(parents=True)
	(directory / "api.py").write_text('def first(): pass\n', encoding="utf-8")
	(directory / "nested" / "api.py").write_text('def second(): pass\n', encoding="utf-8")
	output = tmp_path / "outline.txt"
	result = cli(directory, "-r", "-o", output, cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert result.stdout == ""
	text = output.read_text(encoding="utf-8")
	assert "api module" in text and "nested.api module" in text
	assert "def first():" in text and "def second():" in text


@pytest.mark.parametrize("imports_only", [False, True])
def test_cli_import_listing_and_recursive_relative_imports(package, tmp_path, imports_only):
	args = [package, "-r", "--no-comments", "--no-docstrings"]
	args.extend(["--only", "imports"] if imports_only else ["--include-imports"])
	result = cli(*args, cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "from .api import Public" in result.stdout
	assert "from ..api import Public" in result.stdout
	assert ("class Public:" in result.stdout) == (not imports_only)


def test_cli_source_ranges_use_definition_file_for_reexports(package, tmp_path):
	result = cli(package / "__init__.py", "--source-ranges", "--only", "classes", "methods", "--no-comments", "--no-docstrings", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	filename = (package / "api.py").resolve().as_posix()
	assert f"# source: {filename}:1-2\nclass Public:" in result.stdout
	assert f"\t# source: {filename}:2-2\n\tdef method(self):" in result.stdout
	assert f"# source: {(package / '__init__.py').resolve().as_posix()}" not in result.stdout


@pytest.mark.parametrize("flag,pattern", [("--filter", "Public.met*"), ("--regex", r"^Public\.method$")])
def test_cli_definition_filters_work_recursively_with_ranges(package, tmp_path, flag, pattern):
	result = cli(package, "-r", flag, pattern, "--source-ranges", "--only", "methods", "--no-comments", "--no-docstrings", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert "class Public:" in result.stdout and "def method(self):" in result.stdout
	assert "class NestedPublic" not in result.stdout
	assert f"# source: {(package / 'api.py').resolve().as_posix()}:2-2" in result.stdout


@pytest.mark.parametrize("args,message", [(["--regex", "["], "Invalid regex"), (["--filter", "*", "--regex", ".*"], "not allowed with argument")])
def test_cli_rejects_invalid_filters_before_importing(tmp_path, args, message):
	path = tmp_path / "should_not_import.py"
	path.write_text("raise AssertionError('Must not import')\n", encoding="utf-8")
	result = cli(path, *args, cwd=tmp_path)
	assert result.returncode == 2
	assert message in result.stderr
	assert "Must not import" not in result.stderr and "Traceback" not in result.stderr


def test_source_ranges_unwrap_decorated_functions_across_modules(package, tmp_path):
	(package / "wrappers.py").write_text('''from functools import wraps
def traced(function):
	@wraps(function)
	def wrapper(*args, **kwargs):
		return function(*args, **kwargs)
	return wrapper
''', encoding="utf-8")
	path = package / "decorated.py"
	path.write_text('''from .wrappers import traced
@traced
def operation(
	value: int,
):
	return value

# Trailing comments are outside the range.
''', encoding="utf-8")
	result = cli(path, "--source-ranges", "--only", "functions", "--no-docstrings", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	assert f"# source: {path.resolve().as_posix()}:2-6\ndef operation" in result.stdout
	assert "wrappers.py:" not in result.stdout


def test_source_ranges_preserve_multiline_strings_in_nested_definitions(package, tmp_path):
	path = package / "text.py"
	path.write_text('''def identity(cls): return cls
@identity
class Text:
	@staticmethod
	def render():
		"""Example:
no indentation
"""
		return 1
''', encoding="utf-8")
	result = cli(path, "--source-ranges", "--only", "classes", "methods", "--no-docstrings", cwd=tmp_path)
	assert result.returncode == 0, result.stderr
	filename = path.resolve().as_posix()
	assert f"# source: {filename}:2-9\nclass Text:" in result.stdout
	assert f"\t# source: {filename}:4-9\n\t@staticmethod\n\tdef render():" in result.stdout


@pytest.mark.parametrize("case,message", [("missing", "Path does not exist"), ("extension", "Expected a Python file"), ("import", "Cannot import"), ("output", "Cannot write")])
def test_cli_errors_are_actionable(tmp_path, case, message):
	path = tmp_path / ("sample.txt" if case == "extension" else "sample.py")
	if case != "missing":
		path.write_text('import definitely_missing_inspection_dependency\n' if case == "import" else "value = 1\n", encoding="utf-8")
	args = [path]
	if case == "output":
		args.extend(["-o", tmp_path / "missing" / "outline.txt"])
	result = cli(*args, cwd=tmp_path)
	assert result.returncode == 2
	assert message in result.stderr
	assert "Traceback" not in result.stderr


def test_main_restores_import_path_after_failure(tmp_path):
	path = tmp_path / "broken_inspection_module.py"
	path.write_text("raise RuntimeError('Broken import')\n", encoding="utf-8")
	before = sys.path[:]
	with pytest.raises(SystemExit):
		main([str(path)])
	assert sys.path == before


def test_main_rejects_cached_module_from_a_different_path(tmp_path, monkeypatch, capsys):
	path = tmp_path / "cached_inspection_module.py"
	path.write_text("", encoding="utf-8")
	module = ModuleType(path.stem)
	module.__file__ = str(tmp_path / "other" / path.name)
	monkeypatch.setitem(sys.modules, path.stem, module)
	before = sys.path[:]
	with pytest.raises(SystemExit):
		main([str(path)])
	assert "imported from another path" in capsys.readouterr().err
	assert sys.path == before
