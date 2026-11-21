#!/usr/bin/env python3
"""Print a filtered outline of an imported module or package's public API."""

import argparse
import sys
import ast
import importlib
import inspect
import io
import re
import tokenize
from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import Path
from types import ModuleType, UnionType
from typing import get_origin


MEMBER_KINDS = frozenset({"classes", "functions", "methods", "properties", "attributes", "imports"})
DEFAULT_MEMBER_KINDS = MEMBER_KINDS - {"imports"}


def _name_matcher(module_name: str, name_filter: str | None, regex: str | None) -> Callable[[tuple[str, ...]], bool] | None:
	if name_filter is not None and regex is not None:
		raise ValueError("Use either name_filter or regex, not both")
	if regex is not None:
		try:
			match = re.compile(regex).search
		except re.error as error:
			raise ValueError(f"Invalid regex: {error}") from error
	elif name_filter is not None:
		match = lambda name: fnmatchcase(name, name_filter)
	else:
		return None
	return lambda path: any(match(name) for name in (path[-1], ".".join(path), ".".join((module_name, *path))))


def _doc(obj: object) -> str | None:
	doc = getattr(obj, "__doc__", None)

	if not isinstance(doc, str):
		return None

	return inspect.cleandoc(doc) or None


def _annotation(value: object) -> str:
	if isinstance(value, str):
		return value

	return inspect.formatannotation(value)


def _signature(obj: object) -> str:
	try:
		return str(inspect.signature(obj, eval_str=False))
	except (TypeError, ValueError):
		return "(...)"


def _is_type(value: object) -> bool:
	if isinstance(value, (type, UnionType)):
		return True
	if get_origin(value) is not None or hasattr(value, "__value__"):
		return True
	module = getattr(type(value), "__module__", "")
	return module in ("typing", "types", "typing_extensions")


def _annotation_str(value: object) -> str:
	if hasattr(value, "__value__"):
		return _annotation(getattr(value, "__value__"))
	return _annotation(value)


def _value(value: object) -> str | None:
	try:
		text = repr(value)
	except Exception:
		return None
	return text if len(text) < 80 and not text.startswith("<") else None


def _doc_lines(doc: str, indent: str) -> list[str]:
	doc = doc.replace("\\", "\\\\").replace('"""', '\\"""')
	lines = doc.splitlines()

	if len(lines) == 1 and not lines[0].endswith('"'):
		return [f'{indent}"""{lines[0]}"""']

	return [
		f'{indent}"""',
		*(f"{indent}{line}" for line in lines),
		f'{indent}"""',
	]


def _node_start(node: ast.AST) -> int:
	lines = [node.lineno]

	if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
		lines.extend(decorator.lineno for decorator in node.decorator_list)

	return min(lines)


def _source_info(module: ModuleType) -> tuple[dict[tuple[str, ...], int], dict[int, str]]:
	try:
		source = inspect.getsource(module)
	except (OSError, TypeError):
		return {}, {}

	nodes: dict[tuple[str, ...], int] = {}

	def walk(body: list[ast.AST], prefix: tuple[str, ...] = ()) -> None:
		for node in body:
			if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
				path = prefix + (node.name,)
				if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
					for decorator in node.decorator_list:
						if isinstance(decorator, ast.Attribute) and decorator.attr in {"setter", "deleter"}:
							path += (decorator.attr,)
				nodes[path] = _node_start(node)

				if isinstance(node, ast.ClassDef):
					walk(node.body, path)

			elif isinstance(node, (ast.Assign, ast.AnnAssign)):
				targets = node.targets if isinstance(node, ast.Assign) else [node.target]

				for target in targets:
					if isinstance(target, ast.Name):
						nodes[prefix + (target.id,)] = node.lineno
			else:
				walk(list(ast.iter_child_nodes(node)), prefix)

	tree = ast.parse(source)
	walk(tree.body)

	lines = source.splitlines()
	comments_by_line: dict[int, str] = {}

	for token in tokenize.generate_tokens(io.StringIO(source).readline):
		if token.type != tokenize.COMMENT:
			continue

		line, column = token.start
		if not lines[line - 1][:column].strip():
			comments_by_line[line] = token.string

	comments: dict[int, str] = {}

	for line in nodes.values():
		current = line - 1
		found: list[str] = []

		while current in comments_by_line:
			found.append(comments_by_line[current])
			current -= 1

		if found:
			comments[line] = "\n".join(reversed(found))

	return nodes, comments


def _source_imports(module: ModuleType) -> list[str]:
	try:
		source = inspect.getsource(module)
	except (OSError, TypeError):
		return []
	nodes = (node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom)))
	return [ast.unparse(node) for node in sorted(nodes, key=lambda node: node.lineno)]


def _emit_comments(
	out: list[str],
	path: tuple[str, ...],
	nodes: dict[tuple[str, ...], int],
	comments: dict[int, str],
	indent: str,
) -> None:
	line = nodes.get(path)

	if line not in comments:
		return

	for comment in comments[line].splitlines():
		out.append(indent + comment)


def _emit_doc(out: list[str], obj: object, indent: str) -> None:
	doc = _doc(obj)

	if doc is not None:
		out.extend(_doc_lines(doc, indent))


def _emit_source_range(out: list[str], obj: object, indent: str) -> None:
	try:
		obj = inspect.unwrap(obj)
		filename = inspect.getsourcefile(obj)
		if filename is None:
			return
		lines, start = inspect.getsourcelines(obj)
		if lines[0][:1].isspace():
			# Preserve indentation inside multiline strings in nested definitions.
			tree = ast.parse("if True:\n" + "".join(lines))
			body = tree.body[0].body
			start -= 1
		else:
			body = ast.parse("".join(lines)).body
	except (OSError, TypeError, ValueError, SyntaxError):
		return
	for node in body:
		if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
			first = start + _node_start(node) - 1
			last = start + node.end_lineno - 1
			out.append(f"{indent}# source: {Path(filename).resolve().as_posix()}:{first}-{last}")
			return


def _emit_attribute(
	out: list[str],
	name: str,
	value: object,
	annotation: object | None,
	indent: str = '',
) -> None:
	if _is_type(value):
		val_str = _annotation_str(value)
		if annotation is not None:
			text = f"{name}: {_annotation(annotation)} = {val_str}"
		else:
			text = f"{name} = {val_str}"
	else:
		rendered = _value(value)
		if annotation is not None:
			text = f"{name}: {_annotation(annotation)}"
			if rendered is not None:
				text += f" = {rendered}"
		elif rendered is not None:
			text = f"{name} = {rendered}"
		else:
			text = f"{name}: {_annotation(type(value))}"

	out.append(indent + text)


def _emit_function(
	out: list[str],
	name: str,
	obj: object,
	path: tuple[str, ...],
	nodes: dict[tuple[str, ...], int],
	comments: dict[int, str],
	indent: str = "",
	decorator: str | None = None,
	*,
	include_docstrings: bool = True,
	source_ranges: bool = False,
) -> None:
	_emit_comments(out, path, nodes, comments, indent)
	if source_ranges:
		_emit_source_range(out, obj, indent)

	if decorator is not None:
		out.append(f"{indent}@{decorator}")

	prefix = "async def" if inspect.iscoroutinefunction(obj) else "def"
	out.append(f"{indent}{prefix} {name}{_signature(obj)}:")
	if include_docstrings:
		_emit_doc(out, obj, indent + "\t")
	out.append(indent + "\t...")


def _emit_class(
	out: list[str],
	cls: type,
	path: tuple[str, ...],
	nodes: dict[tuple[str, ...], int],
	comments: dict[int, str],
	indent: str = "",
	name: str | None = None,
	ast_path: tuple[str, ...] | None = None,
	*,
	only: frozenset[str] = DEFAULT_MEMBER_KINDS,
	include_docstrings: bool = True,
	ancestors: tuple[type, ...] = (),
	source_ranges: bool = False,
	matches: Callable[[tuple[str, ...]], bool] | None = None,
) -> None:
	if cls in ancestors:
		return
	ancestors += (cls,)
	class_matches = matches is None or matches(path)
	child_matches = None if class_matches else matches
	ast_path = ast_path or path
	class_start = len(out)
	_emit_comments(out, ast_path, nodes, comments, indent)
	if source_ranges:
		_emit_source_range(out, cls, indent)

	class_name = name or cls.__name__
	bases = [base.__qualname__ for base in cls.__bases__ if base is not object]
	suffix = f"({', '.join(bases)})" if bases else ""

	out.append(f"{indent}class {class_name}{suffix}:")
	body_start = len(out)
	child_indent = indent + "\t"

	if include_docstrings:
		_emit_doc(out, cls, child_indent)
	members_start = len(out)

	annotations = inspect.get_annotations(cls, eval_str=False)
	emitted_attributes: set[str] = set()

	for attr_name, annotation in annotations.items():
		if attr_name.startswith("_") or "attributes" not in only:
			continue
		if child_matches is not None and not child_matches(path + (attr_name,)):
			continue

		child_ast_path = ast_path + (attr_name,)
		_emit_comments(out, child_ast_path, nodes, comments, child_indent)

		if attr_name in cls.__dict__:
			_emit_attribute(
				out, attr_name, cls.__dict__[attr_name], annotation, child_indent
			)
		else:
			out.append(f"{child_indent}{attr_name}: {_annotation(annotation)}")

		emitted_attributes.add(attr_name)

	for attr_name, raw in cls.__dict__.items():
		if attr_name.startswith("_") and attr_name != "__init__":
			continue

		child_path = path + (attr_name,)
		child_ast_path = ast_path + (attr_name,)

		if not inspect.isclass(raw) and child_matches is not None and not child_matches(child_path):
			continue

		if isinstance(raw, staticmethod) and "methods" in only:
			_emit_function(
				out, attr_name, raw.__func__, child_ast_path, nodes, comments,
				child_indent, "staticmethod",
				include_docstrings=include_docstrings, source_ranges=source_ranges,
			)

		elif isinstance(raw, classmethod) and "methods" in only:
			_emit_function(
				out, attr_name, raw.__func__, child_ast_path, nodes, comments,
				child_indent, "classmethod",
				include_docstrings=include_docstrings, source_ranges=source_ranges,
			)

		elif isinstance(raw, property):
			if "properties" not in only:
				continue

			for accessor, decorator in ((raw.fget, "property"), (raw.fset, f"{attr_name}.setter"), (raw.fdel, f"{attr_name}.deleter")):
				if accessor is None:
					continue
				accessor_path = child_ast_path if decorator == "property" else child_ast_path + (decorator.rsplit(".", 1)[1],)
				_emit_function(
					out, attr_name, accessor, accessor_path, nodes, comments,
					child_indent, decorator, include_docstrings=include_docstrings, source_ranges=source_ranges,
				)

		elif inspect.isroutine(raw):
			if "methods" in only:
				_emit_function(
					out, attr_name, raw, child_ast_path, nodes, comments, child_indent,
					include_docstrings=include_docstrings, source_ranges=source_ranges,
				)

		elif inspect.isclass(raw):
			_emit_class(
				out, raw, child_path, nodes, comments, child_indent, name=attr_name,
				ast_path=child_ast_path, only=only, include_docstrings=include_docstrings, ancestors=ancestors,
				source_ranges=source_ranges, matches=child_matches,
			)

		elif not isinstance(raw, (staticmethod, classmethod)) and "attributes" in only and attr_name not in emitted_attributes and not attr_name.startswith("_"):
			_emit_comments(out, child_ast_path, nodes, comments, child_indent)
			_emit_attribute(out, attr_name, raw, None, child_indent)

	if ("classes" not in only or not class_matches) and len(out) == members_start:
		del out[class_start:]
	elif len(out) == body_start:
		out.append(child_indent + "...")


def dump_module(
	module: ModuleType,
	*,
	include_comments: bool = True,
	include_docstrings: bool = True,
	only: frozenset[str] | None = None,
	include_imports: bool = False,
	source_ranges: bool = False,
	name_filter: str | None = None,
	regex: str | None = None,
) -> str:
	"""Return public declarations; classes contain only selected member kinds.

	Explicit ``__all__`` includes re-exports; otherwise imported classes and
	functions and private names are omitted. Only directly defined class members
	are shown, including ``__init__``. Classes remain as containers when selecting
	methods, properties or class attributes. Optional imports are collected from
	source, including conditional and local imports, independently of exports.
	``source_ranges`` adds source locations for classes and callables when available.
	Name filters match exported names, class-qualified names or module-qualified
	names. Matching classes include their selected members; matching members keep
	their containing classes. Imports are independent of definition name filters.
	"""
	matches = _name_matcher(module.__name__, name_filter, regex)
	only = DEFAULT_MEMBER_KINDS if only is None else frozenset(only)
	if include_imports:
		only |= {"imports"}
	if only - MEMBER_KINDS:
		raise ValueError(f"Unknown member kinds: {', '.join(sorted(only - MEMBER_KINDS))}")
	nodes, comments = _source_info(module) if include_comments else ({}, {})

	module_name = module.__name__
	title = f" {module_name} module "
	lpad = (80 - len(title)) //2
	rpad = (80 - len(title) - lpad)
	out = [
		'#' * 80,
		f"{'#' * lpad}{title}{'#' * rpad}",
		""
	]

	if include_docstrings:
		_emit_doc(out, module, "")

	if out[-1]:
		out.append("")

	if "imports" in only:
		imports = _source_imports(module)
		if imports:
			out.extend(imports)
			out.append("")

	annotations = getattr(module, "__annotations__", {})
	exports = getattr(module, "__all__", None)

	def public(name: str) -> bool:
		return name in exports if exports is not None else not name.startswith("_")

	emitted: set[str] = set()

	for name, value in module.__dict__.items():
		if not public(name) or isinstance(value, ModuleType):
			continue

		path = (name,)
		member_start = len(out)
		if not inspect.isclass(value) and matches is not None and not matches(path):
			continue

		if inspect.isclass(value):
			if exports is None and value.__module__ != module_name:
				continue
			_emit_class(
				out, value, path, nodes, comments, name=name, ast_path=tuple(value.__qualname__.split(".")),
				only=only, include_docstrings=include_docstrings, source_ranges=source_ranges, matches=matches,
			)

		elif inspect.isroutine(value):
			if "functions" not in only or (exports is None and getattr(value, "__module__", None) != module_name):
				continue
			_emit_function(out, name, value, (value.__name__,), nodes, comments, include_docstrings=include_docstrings, source_ranges=source_ranges)

		elif "attributes" in only:
			_emit_comments(out, path, nodes, comments, "")
			_emit_attribute(out, name, value, annotations.get(name))

		if len(out) > member_start:
			out.append("")
			emitted.add(name)

	for name, annotation in annotations.items():
		if "attributes" in only and public(name) and name not in emitted and name not in module.__dict__:
			if matches is not None and not matches((name,)):
				continue
			_emit_comments(out, (name,), nodes, comments, "")
			out.append(f"{name}: {_annotation(annotation)}")
			out.append("")

	return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description="Dump a Python file or directory's public API. Target modules are imported and their top-level code runs.")
	parser.add_argument("package", type=Path, help="Python file, package directory or directory of modules")
	parser.add_argument("-o", "--output", type=Path)
	parser.add_argument("-r", "--recursive", action="store_true", help="Include subdirectories (or sibling modules when targeting __init__.py)")
	parser.add_argument("--no-comments", action="store_true", help="Omit comments preceding declarations")
	parser.add_argument("--no-docstrings", action="store_true", help="Omit module, class and callable docstrings")
	parser.add_argument("--include-imports", action="store_true", help="Include source import statements, including conditional and local imports")
	parser.add_argument("--source-ranges", action="store_true", help="Show definition file paths and inclusive line ranges, including decorators")
	parser.add_argument("--only", nargs="+", choices=sorted(MEMBER_KINDS), help="Include only these member kinds; classes are retained as containers")
	filters = parser.add_mutually_exclusive_group()
	filters.add_argument("--filter", dest="name_filter", metavar="PATTERN", help="Case-sensitive name glob, e.g. '*Query*' or 'ECS.query*'")
	filters.add_argument("--regex", metavar="PATTERN", help="Case-sensitive regex search over definition names")
	args = parser.parse_args(argv)
	try:
		_name_matcher("", args.name_filter, args.regex)
	except ValueError as error:
		parser.error(str(error))

	package = args.package.resolve()
	if not package.exists():
		parser.error(f"Path does not exist: {package}")
	if not package.is_dir() and package.suffix != ".py":
		parser.error(f"Expected a Python file or directory: {package}")

	directory = package if package.is_dir() else package.parent
	parent_directory = directory
	while (parent_directory / "__init__.py").is_file():
		parent_directory = parent_directory.parent

	if package.is_file() and not (args.recursive and package.name == "__init__.py"):
		paths = [package]
	else:
		glob_func = directory.rglob if args.recursive else directory.glob
		paths = sorted(path for path in glob_func("*.py") if "__pycache__" not in path.relative_to(directory).parts)

	modules = []
	previous_path = sys.path[:]
	sys.path.insert(0, str(parent_directory))
	try:
		for path in paths:
			parts = path.relative_to(parent_directory).with_suffix("").parts
			module_name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
			try:
				module = importlib.import_module(module_name)
			except Exception as error:
				parser.error(f"Cannot import {module_name} ({path}): {type(error).__name__}: {error}")
			module_file = getattr(module, "__file__", None)
			if module_file is None or Path(module_file).resolve() != path.resolve():
				parser.error(f"Module {module_name} was imported from another path: {module_file}")
			modules.append(module)
		text = "\n\n".join(
			dump_module(module, include_comments=not args.no_comments, include_docstrings=not args.no_docstrings, only=args.only, include_imports=args.include_imports, source_ranges=args.source_ranges, name_filter=args.name_filter, regex=args.regex)
			for module in modules
		)
	finally:
		sys.path[:] = previous_path

	if args.output:
		try:
			args.output.write_text(text, encoding="utf-8")
		except OSError as error:
			parser.error(f"Cannot write {args.output}: {error}")
	else:
		print(text, end="")


if __name__ == "__main__":
	main()
