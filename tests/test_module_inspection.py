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
