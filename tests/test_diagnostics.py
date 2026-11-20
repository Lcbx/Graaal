"""CPU-only render diagnostics tests."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from graaal.Diagnostics import RenderMonitor, get_draw_counts, get_shader_sources, inspect_render_memory
from graaal import RenderDiagnostics


@pytest.fixture
def renderer_types(monkeypatch):
	class DrawBatches: pass
	class MaterialRenderer: pass
	monkeypatch.setattr(RenderDiagnostics, "DrawBatches", DrawBatches)
	monkeypatch.setattr(RenderDiagnostics, "MaterialRenderer", MaterialRenderer)
	return DrawBatches, MaterialRenderer


def shader(source):
	return NS(info=NS(source=source))


def pass_(source):
	return NS(pipeline=NS(shader=shader(source)))


def draws(renderer_types):
	DrawBatches, _ = renderer_types
	renderer = DrawBatches()
	renderer.draw_batches = [NS(mesh_id=10, shader_id=0), NS(mesh_id=11, shader_id=0),
		NS(mesh_id=None, shader_id=0), NS(mesh_id=12, shader_id=1)]
	renderer.shaders = {
		0: NS(passes={"main": pass_("opaque"), "prepass": pass_("opaque")}),
		1: NS(passes={"main": pass_("transparent"), "transmittance": pass_("transparent"),
			"nearest": pass_("transparent"), "bands": pass_("transparent")}),
	}
	renderer._cull_shader = shader("cull")
	return renderer


def material_renderer(renderer_types, **options):
	_, MaterialRenderer = renderer_types
	settings = dict(mixed_tail=True, shading_shader=shader("shade"), nearest_shading_shader=shader("nearest"),
		compose_shader=shader("compose"), band_clear_shader=shader("clear"), kawase_shader=shader("kawase"))
	settings.update(options)
	renderer = MaterialRenderer()
	renderer.draws = draws(renderer_types)
	renderer.draws.shaders[0].passes = {"opaque": pass_("opaque")}
	renderer.transparency = NS(**settings)
	return renderer


def test_draw_batches_counts(renderer_types):
	renderer = draws(renderer_types)
	assert get_draw_counts(renderer) == (3, 3)
	assert get_draw_counts(renderer, passes=("prepass", "main")) == (3, 5)
	assert get_draw_counts(renderer, passes="bands") == (3, 1)
	assert get_draw_counts(renderer, passes=()) == (3, 0)


def test_material_renderer_counts(renderer_types):
	renderer = material_renderer(renderer_types)
	# Two opaque draws, nearest/transmission/statistics/bands for one transparent
	# batch, and one composition draw. Kawase uses compute dispatches.
	assert get_draw_counts(renderer) == (3, 7)
	assert get_draw_counts(renderer, passes="opaque") == (3, 2)
	assert get_draw_counts(renderer, passes="transmittance") == (3, 1)
	assert get_draw_counts(material_renderer(renderer_types, mixed_tail=False)) == (3, 6)


def test_sources_deduplicated_and_active(renderer_types):
	assert get_shader_sources(draws(renderer_types)) == ["cull", "opaque", "transparent"]
	assert get_shader_sources(material_renderer(renderer_types)) == ["cull", "opaque", "transparent", "shade", "nearest", "compose", "clear", "kawase"]
	assert get_shader_sources(material_renderer(renderer_types, kawase_shader=None))[-1] == "clear"


def test_unsupported_input(renderer_types):
	with pytest.raises(TypeError, match="DrawBatches or MaterialRenderer"):
		get_draw_counts(object())
	with pytest.raises(TypeError, match="DrawBatches or MaterialRenderer"):
		get_shader_sources(object())
	# Attribute-compatible objects are not accepted unless they have the right type.
	with pytest.raises(TypeError, match="DrawBatches or MaterialRenderer"):
		get_draw_counts(NS(draw_batches=[], shaders={}))


def test_render_monitor_records_submitted_work_without_gpu_timestamps(monkeypatch, renderer_types):
	from graaal.RenderContext import RenderContext

	class Scope:
		def __enter__(self): return self
		def __exit__(self, *args): pass
		def draw(self, count): pass
		def dispatch(self, count): pass

	class Commands:
		def render_pass(self, **kwargs): return Scope()
		def compute_pass(self, **kwargs): return Scope()
		def finish(self, label=None): return object()

	monkeypatch.setattr(RenderContext, "device", NS(features=()), raising=False)
	monkeypatch.setattr(RenderContext, "commands", lambda label=None: Commands())
	monkeypatch.setattr(RenderContext, "submit", Mock())
	monitor = RenderMonitor(renderer=draws(renderer_types), log=False)
	cmd = monitor.commands("frame")
	with cmd.render_pass(label="opaque") as render:
		render.draw(3)
	with cmd.compute_pass(label="cull") as compute:
		compute.dispatch(4)
	buffer = cmd.finish()
	with pytest.raises(ValueError, match="Submit each"):
		monitor.submit(object())
	report = monitor.submit(buffer)
	assert RenderContext.submit.call_args.args == (buffer,)
	assert [entry["label"] for entry in report["commands"][0]["passes"]] == ["opaque", "cull"]
	assert [entry["calls"][0]["operation"] for entry in report["commands"][0]["passes"]] == ["draw", "dispatch"]
	assert report["encoded_draw_calls"] == report["encoded_dispatches"] == 1
	assert report["draw_batches"] == report["predicted_draw_calls"] == 3
	assert report["commands"][0]["passes"][0]["gpu_ms"] is None
	assert report["commands"][0]["encode_ms"] >= 0
	assert report["cpu_submit_ms"] >= 0
	assert not report["gpu_timing_available"]
	assert monitor.last_report is report
	assert isinstance(inspect_render_memory()["resources"], list)


def test_render_monitor_keeps_gpu_samples_for_repeated_pass_labels(monkeypatch):
	from graaal.RenderContext import RenderContext

	class Profiler:
		def __init__(self, max_passes):
			self.last_frame_ms = 3.0
			self.last_samples = [1.0, 2.0]
			self.closed = False
		def wrap(self, commands): return commands
		def read(self): pass
		def close(self): self.closed = True

	class Scope:
		def __enter__(self): return self
		def __exit__(self, *args): pass

	class Commands:
		def render_pass(self, **kwargs): return Scope()
		def finish(self, label=None): return object()

	monkeypatch.setattr(RenderDiagnostics, "GpuProfiler", Profiler)
	monkeypatch.setattr(RenderContext, "device", NS(features=("timestamp-query",)), raising=False)
	monkeypatch.setattr(RenderContext, "commands", lambda label=None: Commands())
	monkeypatch.setattr(RenderContext, "submit", Mock())
	monitor = RenderMonitor()
	cmd = monitor.commands("frame")
	with cmd.render_pass(label="same"): pass
	with cmd.render_pass(label="same"): pass
	report = monitor.submit(cmd.finish())
	assert [entry["gpu_ms"] for entry in report["commands"][0]["passes"]] == [1.0, 2.0]
	assert report["commands"][0]["gpu_span_ms"] == 3.0


def test_render_memory_counts_buffer_capacity_and_texture_mips(monkeypatch):
	import numpy as np
	from graaal.RenderDiagnostics import _resource_name
	from graaal.RenderContext import BufferUsage, GpuBuffer, RenderContext, Texture, TextureUsage

	device = NS(create_buffer=lambda **kwargs: NS(size=kwargs["size"]), create_texture=lambda **kwargs: NS())
	monkeypatch.setattr(RenderContext, "device", device, raising=False)
	buffer = GpuBuffer(np.zeros(4, dtype=np.uint32), BufferUsage.COPY_DST, upload=False, label="tracked_bytes")
	texture = Texture((4, 4), format="rgba8unorm", usage=TextureUsage.TEXTURE_BINDING,
		mip_level_count=3, label="tracked_color")
	resources = inspect_render_memory()["resources"]
	buffer_row = next(row for row in resources if row["name"] == f"tracked_bytes@{id(buffer):x}")
	texture_row = next(row for row in resources if row["name"] == f"tracked_color@{id(texture):x}")
	assert buffer_row["gpu_bytes"] == buffer_row["cpu_mirror_bytes"] == 16
	assert texture_row["gpu_bytes"] == (16 + 4 + 1) * 4
	assert buffer_row["name"] == _resource_name(buffer)
	assert texture_row["name"] == _resource_name(texture)
	with pytest.raises(ValueError) as invalid:
		GpuBuffer(None, BufferUsage.COPY_DST, dtype=None, count=1)
	assert invalid.value is not None
	assert inspect_render_memory()["resources"]
	buffer.close()
	texture.close()
	assert all(row["name"] not in (buffer_row["name"], texture_row["name"]) for row in inspect_render_memory()["resources"])
