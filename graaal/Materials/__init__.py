"""Public Materials API. Implementation and optional backends load on first access."""
from importlib import import_module as _import_module


__all__ = (
	'KINDS',
	'MaterialBinding',
	'MaterialLibrary',
	'MaterialShader',
	'Parameter',
	'ParameterLayout',
	'SHADERS',
)


def __getattr__(name):
	if name not in __all__:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	value = getattr(_import_module('.core', __name__), name)
	globals()[name] = value
	return value
