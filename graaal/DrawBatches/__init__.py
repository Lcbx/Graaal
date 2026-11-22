"""Public DrawBatches API. Implementation and optional backends load on first access."""
from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from .core import (
		DrawBatches,
		FrustumCuller,
		HiZCuller,
		INDIRECT_DTYPE,
		RenderShader,
		ShaderPass,
		Visibility,
		standard_render_shader,
	)


__all__ = (
	'DrawBatches',
	'FrustumCuller',
	'HiZCuller',
	'INDIRECT_DTYPE',
	'RenderShader',
	'ShaderPass',
	'Visibility',
	'standard_render_shader',
)


def __getattr__(name: str) -> Any:
	if name not in __all__:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	value = getattr(_import_module('.core', __name__), name)
	globals()[name] = value
	return value
