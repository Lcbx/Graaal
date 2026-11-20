"""Rendering diagnostics; importing this module requires the rendering extra."""

import ctypes
import json
import logging
import os
import tracemalloc
from ctypes import wintypes
from time import perf_counter as get_time

from .DrawBatches import DrawBatches
from .MaterialRenderer import MaterialRenderer
from .GpuProfiler import GpuProfiler
from .RenderContext import GpuBuffer, RenderContext, Texture, TextureView

try:
	import psutil  # type: ignore[import-untyped]
except ModuleNotFoundError as error:
	if error.name != "psutil": raise
	psutil = None


def get_draw_counts(renderer, passes=None) -> tuple[int, int]:
	"""Return (drawable batches, submitted draw calls) without GPU visibility readback.

	DrawBatches defaults to the "main" pass; pass ("prepass", "main") to count
	both. MaterialRenderer defaults to all draws in record(), including fullscreen
	composition, for a nonzero output size. Compute dispatches are excluded.
	"""
	if isinstance(renderer, MaterialRenderer):
		draws = renderer.draws
	elif isinstance(renderer, DrawBatches):
		draws = renderer
	else:
		raise TypeError("Expected DrawBatches or MaterialRenderer")
	batches = [batch for batch in draws.draw_batches if batch.mesh_id is not None]
	if passes is not None or isinstance(renderer, DrawBatches):
		passes = ("main",) if passes is None else (passes,) if isinstance(passes, str) else tuple(passes)
		return len(batches), sum(name in draws.shaders[batch.shader_id].passes for batch in batches for name in passes)

	oit = renderer.transparency
	opaque = sum(batch.shader_id == 0 for batch in batches)
	transparent = sum(batch.shader_id == 1 for batch in batches)
	draw_calls = opaque + transparent * (3 + int(oit.mixed_tail)) + 1
	return len(batches), draw_calls


def get_shader_sources(renderer) -> list[str]:
	"""Return unique expanded WGSL sources of this renderer's registered shaders.

	Includes its culling shader and, for MaterialRenderer, shading/composition
	shaders. External passes such as a separately owned HZB are excluded.
	"""
	if isinstance(renderer, MaterialRenderer):
		draws = renderer.draws
	elif isinstance(renderer, DrawBatches):
		draws = renderer
	else:
		raise TypeError("Expected DrawBatches or MaterialRenderer")
	shaders = [draws._cull_shader]
	for shader in draws.shaders.values():
		shaders.extend(spec.pipeline.shader for spec in shader.passes.values())
	if isinstance(renderer, MaterialRenderer):
		oit = renderer.transparency
		shaders.extend((oit.shading_shader, oit.nearest_shading_shader, oit.compose_shader))
		shaders.extend(getattr(oit, name, None) for name in ("band_clear_shader", "kawase_shader"))
	return list(dict.fromkeys(shader.info.source for shader in shaders if shader is not None))


_TEXTURE_BYTES = {"r8": 1, "rg8": 2, "rgba8": 4, "bgra8": 4, "r16": 2, "rg16": 4,
	"rgba16": 8, "r32": 4, "rg32": 8, "rgba32": 16, "depth16unorm": 2,
	"depth24plus": 4, "depth24plus-stencil8": 4, "depth32float": 4,
	"depth32float-stencil8": 8}


def inspect_render_memory() -> dict:
	"""Inspect live engine buffers/textures and process memory without GPU readback.

	GPU bytes are logical texture payload plus current buffer capacity, not driver
	allocation, in-flight replaced handles, or resources created directly through wgpu.
	"""
	resources = []
	for resource in tuple(RenderContext._tracked_resources):
		if getattr(resource, "_closed", False) or resource._device_generation != RenderContext.device_generation:
			continue
		if isinstance(resource, GpuBuffer):
			resources.append(dict(kind="buffer", name=_resource_name(resource),
				gpu_bytes=resource.handle.size, cpu_mirror_bytes=resource.content.nbytes if resource.content is not None else 0,
				usage=int(resource.usage)))
		elif isinstance(resource, Texture):
			format_name = str(resource.format).rsplit(".", 1)[-1]
			bytes_per_texel = next((size for prefix, size in _TEXTURE_BYTES.items() if format_name.startswith(prefix)), None)
			width, height, layers = resource.size
			texels = sum(max(1, width >> level) * max(1, height >> level) *
				(max(1, layers >> level) if resource.dimension == "3d" else layers) for level in range(resource.mip_level_count))
			resources.append(dict(kind="texture", name=_resource_name(resource),
				gpu_bytes=texels * resource.sample_count * bytes_per_texel if bytes_per_texel is not None else None,
				cpu_mirror_bytes=0, format=format_name, size=resource.size, mip_levels=resource.mip_level_count,
				samples=resource.sample_count, usage=int(resource.usage)))
	resources.sort(key=lambda item: (item["kind"], item["name"]))
	if psutil is not None:
		cpu_rss = psutil.Process(os.getpid()).memory_info().rss
	else:
		cpu_rss = None
		if os.name == "nt":
			class ProcessMemoryCounters(ctypes.Structure):
				_fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD)] + [
					(name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "peak_paged_pool",
						"paged_pool", "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")]
			info = ProcessMemoryCounters()
			info.cb = ctypes.sizeof(info)
			ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
			ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD)
			if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(info), info.cb):
				cpu_rss = info.working_set
		elif os.path.exists("/proc/self/statm"):
			with open("/proc/self/statm", encoding="ascii") as statm:
				cpu_rss = int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
	return dict(cpu_rss_bytes=cpu_rss,
		python_traced_bytes=tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else None,
		cpu_buffer_mirrors_bytes=sum(item["cpu_mirror_bytes"] for item in resources),
		gpu_tracked_bytes=sum(item["gpu_bytes"] or 0 for item in resources),
		gpu_unestimated_count=sum(item["gpu_bytes"] is None for item in resources), resources=resources)


class RenderMonitor:
	"""Opt-in command, pass and resource log for one submission at a time.

	Use commands()/submit() in place of RenderContext.commands()/submit(). GPU
	pass times require a device created with the timestamp-query feature; without
	it, GPU times are None. CPU timings measure Python recording and queue submit.
	"""
	def __init__(self, renderer=None, *, log=False, max_passes=64):
		self.renderer = renderer
		self.log = log
		self.max_passes = max_passes
		self._finished = []
		self.last_report = None

	def commands(self, label=None):
		commands = RenderContext.commands(label)
		profiler = GpuProfiler(self.max_passes) if "timestamp-query" in RenderContext.device.features else None
		return _MonitoredCommands(self, profiler.wrap(commands) if profiler else commands, profiler, label)

	def submit(self, *command_buffers):
		if len(command_buffers) != len(self._finished) or any(buffer is not entry[0] for buffer, entry in zip(command_buffers, self._finished)):
			raise ValueError("Submit each monitored command buffer once, in finished order")
		start = get_time()
		RenderContext.submit(*command_buffers)
		submit_ms = (get_time() - start) * 1000
		commands = []
		try:
			for _, entry in self._finished:
				profiler = entry.pop("_profiler")
				if profiler is not None:
					try:
						profiler.read()
						entry["gpu_span_ms"] = profiler.last_frame_ms if entry["passes"] else None
						for scope, elapsed in zip(entry["passes"], profiler.last_samples):
							scope["gpu_ms"] = elapsed
					finally:
						profiler.close()
				commands.append(entry)
		finally:
			self._finished.clear()
		batches, draws = get_draw_counts(self.renderer) if self.renderer is not None else (None, None)
		encoded_draws = sum(call["operation"].startswith("draw") for entry in commands for scope in entry["passes"] for call in scope["calls"])
		encoded_dispatches = sum(call["operation"].startswith("dispatch") for entry in commands for scope in entry["passes"] for call in scope["calls"])
		self.last_report = dict(commands=commands, cpu_submit_ms=submit_ms, draw_batches=batches,
			predicted_draw_calls=draws, encoded_draw_calls=encoded_draws, encoded_dispatches=encoded_dispatches,
			memory=inspect_render_memory(),
			gpu_timing_available="timestamp-query" in RenderContext.device.features)
		if self.log:
			logging.getLogger(__name__).info("render submission %s", json.dumps(self.last_report))
		return self.last_report


def _resource_name(resource):
	if isinstance(resource, TextureView): resource = resource.texture
	if isinstance(resource, GpuBuffer): return f"{resource.label or 'buffer'}@{id(resource):x}"
	if isinstance(resource, Texture): return f"{resource.label or 'texture'}@{id(resource):x}"
	return None


class _MonitoredCommands:
	def __init__(self, monitor, commands, profiler, label):
		self.monitor, self.commands, self.profiler = monitor, commands, profiler
		self.entry = dict(label=label, encode_ms=None, gpu_span_ms=None, passes=[], calls=[])
		self.start = get_time()

	def __getattr__(self, name):
		return getattr(self.commands, name)

	def render_pass(self, **kwargs):
		return _MonitoredPass(self, self.commands.render_pass(**kwargs), "render", kwargs.get("label"))

	def compute_pass(self, **kwargs):
		return _MonitoredPass(self, self.commands.compute_pass(**kwargs), "compute", kwargs.get("label"))

	def clear_buffer(self, buffer, **kwargs):
		start = get_time()
		self.commands.clear_buffer(buffer, **kwargs)
		self.entry["calls"].append(dict(operation="clear_buffer", buffer=_resource_name(buffer),
			cpu_encode_ms=(get_time() - start) * 1000, gpu_ms=None, **kwargs))

	def copy_buffer(self, source, destination, **kwargs):
		start = get_time()
		self.commands.copy_buffer(source, destination, **kwargs)
		self.entry["calls"].append(dict(operation="copy_buffer", source=_resource_name(source), destination=_resource_name(destination),
			cpu_encode_ms=(get_time() - start) * 1000, gpu_ms=None, **kwargs))

	def finish(self, label=None):
		result = self.commands.finish(label=label)
		self.entry["encode_ms"] = (get_time() - self.start) * 1000
		self.entry["_profiler"] = self.profiler
		self.monitor._finished.append((result, self.entry))
		return result


class _MonitoredPass:
	_CALLS = {"draw", "draw_indexed", "draw_indexed_indirect", "draw_mesh", "dispatch", "dispatch_indirect"}

	def __init__(self, commands, scope, kind, label):
		self.commands, self.scope, self.kind, self.label = commands, scope, kind, label

	def __enter__(self):
		self.start = get_time()
		self.scope.__enter__()
		self.entry = dict(kind=self.kind, label=self.label, cpu_encode_ms=None, gpu_ms=None,
			calls=[], resources=[])
		self.commands.entry["passes"].append(self.entry)
		for attachment in getattr(self.scope, "color", ()):
			self._add_view(attachment.view)
			if attachment.resolve_target is not None: self._add_view(attachment.resolve_target)
		if depth := getattr(self.scope, "depth", None): self._add_view(depth.view)
		return self

	def _add_view(self, view):
		texture_handle = getattr(view, "texture", None)
		for resource in tuple(RenderContext._tracked_resources):
			if isinstance(resource, Texture) and (resource.handle is texture_handle or
				(resource._view is not None and resource._view.handle is view)):
				self._add_resource(resource)
				break

	def _add_resource(self, resource):
		name = _resource_name(resource)
		if name is not None and name not in self.entry["resources"]: self.entry["resources"].append(name)

	def __getattr__(self, name):
		attribute = getattr(self.scope, name)
		if name not in self._CALLS and name not in ("set_pipeline", "set_bind_group", "set_bind_groups", "set_vertex_buffer", "set_index_buffer"):
			return attribute
		def record(*args, **kwargs):
			start = get_time()
			result = attribute(*args, **kwargs)
			if name in self._CALLS:
				self.entry["calls"].append(dict(operation=name, arguments=[str(arg) for arg in args] +
					[f"{key}={value}" for key, value in kwargs.items()],
					cpu_encode_ms=(get_time() - start) * 1000, gpu_ms=None))
			else:
				self.entry["calls"].append(dict(operation=name, cpu_encode_ms=(get_time() - start) * 1000, gpu_ms=None))
			if name in ("set_bind_group", "set_bind_groups"):
				for binding in (*args, *((kwargs["bindings"],) if "bindings" in kwargs else ())):
					for resource in binding.resources.values(): self._add_resource(resource)
			elif name in ("set_vertex_buffer", "set_index_buffer"):
				index = 1 if name == "set_vertex_buffer" else 0
				self._add_resource(args[index] if len(args) > index else kwargs["buffer"])
			elif name == "draw_indexed_indirect" or name == "dispatch_indirect":
				self._add_resource(args[0] if args else kwargs["indirect_buffer" if name == "draw_indexed_indirect" else "buffer"])
			elif name == "draw_mesh":
				mesh = args[0] if args else kwargs["mesh"]
				self._add_resource(mesh.vertex_buffer)
				self._add_resource(mesh.index_buffer)
				if kwargs.get("instances") is not None: self._add_resource(kwargs["instances"])
			return result
		return record

	def __exit__(self, *args):
		try:
			return self.scope.__exit__(*args)
		finally:
			self.entry["cpu_encode_ms"] = (get_time() - self.start) * 1000
