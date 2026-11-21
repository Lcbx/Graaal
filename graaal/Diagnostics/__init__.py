"""CPU and rendering diagnostics. GPU tools load only when requested."""
from importlib import import_module as _import_module


__all__ = (
	'WatchTimer',
	'setup_gc_monitor',
	'setup_memory_monitor',
	'inspect_physics',
	'RenderMonitor',
	'get_draw_counts',
	'get_shader_sources',
	'inspect_render_memory',
	'GpuProfiler',
)


def __getattr__(name):
	if name not in __all__:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	module = '.gpu_profiler' if name == 'GpuProfiler' else '.RenderDiagnostics' if name in ('RenderMonitor', 'get_draw_counts', 'get_shader_sources', 'inspect_render_memory') else '.core'
	value = getattr(_import_module(module, __name__), name)
	globals()[name] = value
	return value
