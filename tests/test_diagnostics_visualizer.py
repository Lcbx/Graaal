"""Profiler widget state and ImGui frame smoke tests."""
import pytest

from graaal.UI.diagnostics_visualizer import DiagnosticsVisualizer


def report(gpu=None, cpu=2.0):
	return dict(commands=[dict(label="frame", encode_ms=cpu, gpu_span_ms=gpu, calls=[], passes=[
		dict(kind="render", label="main", gpu_ms=gpu, cpu_encode_ms=cpu, calls=[
			dict(operation="draw", arguments=["3"], cpu_encode_ms=0.1)], resources=["color@1"])])],
		cpu_submit_ms=0.2, draw_batches=1, encoded_draw_calls=1, encoded_dispatches=0,
		gpu_timing_available=gpu is not None,
		memory=dict(cpu_rss_bytes=1024, cpu_buffer_mirrors_bytes=256, gpu_tracked_bytes=512,
			gpu_unestimated_count=0, resources=[dict(kind="texture", name="color@1", gpu_bytes=512,
				format="rgba8unorm", size=(8, 8, 1))]))


def test_history_is_bounded_and_pause_holds_report():
	widget = DiagnosticsVisualizer(history_size=2)
	widget.update(report(cpu=1))
	widget.update(report(cpu=2))
	widget.update(report(cpu=3))
	assert list(widget.history) == [2, 3]
	widget.paused = True
	widget.update(report(cpu=4))
	assert widget.report["commands"][0]["encode_ms"] == 3
	assert list(widget.history) == [2, 3]
	widget.paused = False
	widget.update(report(gpu=1.5))
	assert list(widget.history) == [1.5]
	with pytest.raises(ValueError, match="history_size"):
		DiagnosticsVisualizer(history_size=0)


def test_widget_draws_report_in_imgui_frame():
	from imgui_bundle import imgui

	previous = imgui.get_current_context()
	context = imgui.create_context()
	try:
		io = imgui.get_io()
		io.set_ini_filename("")
		io.display_size = (900, 700)
		io.delta_time = 1 / 60
		io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
		widget = DiagnosticsVisualizer()
		widget.update(report(gpu=1.5))
		imgui.new_frame()
		widget.draw()
		imgui.render()
		assert imgui.get_draw_data().cmd_lists_count > 0
	finally:
		imgui.destroy_context(context)
		if previous is not None: imgui.set_current_context(previous)


def test_monitor_reports_real_gpu_pass_and_attached_texture():
	from graaal.Diagnostics import RenderMonitor
	from graaal.RenderContext import RenderContext, RenderPipeline, Shader, Texture, TextureUsage

	RenderContext.setup_graphics_backend(True, optional_features=("timestamp-query",))
	shader = Shader(source="""
		@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
			var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
			return vec4f(positions[index], 0.0, 1.0);
		}
		@fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
	""")
	output = Texture((8, 8), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT, label="profile_target")
	monitor = RenderMonitor()
	commands = monitor.commands("frame")
	with commands.render_pass(color=output.color_attachment(clear=(0, 0, 0, 1)), label="triangle") as render:
		render.set_pipeline(RenderPipeline(shader, depth_test=None, depth_write=False))
		render.draw(3)
	report = monitor.submit(commands.finish())
	assert report["encoded_draw_calls"] == 1
	assert report["commands"][0]["passes"][0]["label"] == "triangle"
	assert f"profile_target@{id(output):x}" in report["commands"][0]["passes"][0]["resources"]
	if report["gpu_timing_available"]:
		assert report["commands"][0]["passes"][0]["gpu_ms"] is not None
	else:
		assert report["commands"][0]["passes"][0]["gpu_ms"] is None


def test_monitored_keyword_calls_and_custom_resolve_view():
	import numpy as np
	from graaal.Diagnostics import RenderMonitor
	from graaal.RenderContext import BufferUsage, GpuBuffer, Mesh, RenderContext, RenderPipeline, Shader, Texture, TextureUsage

	RenderContext.setup_graphics_backend(True)
	shader = Shader(source="""
		@vertex fn vertex(@location(0) position: vec2f) -> @builtin(position) vec4f { return vec4f(position, 0.0, 1.0); }
		@fragment fn fragment() -> @location(0) vec4f { return vec4f(1.0); }
	""")
	vertices = np.zeros(3, dtype=[("position", "f4", 2)])
	mesh = Mesh(vertices, np.arange(3, dtype=np.uint32))
	buffer = GpuBuffer(vertices, BufferUsage.VERTEX, label="keyword_buffer")
	target = Texture((8, 8), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT, mip_level_count=2, label="resolve")
	multisampled = Texture((4, 4), format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT, sample_count=4)
	view = target.view(base_mip_level=1, mip_level_count=1)
	monitor = RenderMonitor()
	commands = monitor.commands()
	with commands.render_pass(color=multisampled.color_attachment(clear=(0, 0, 0, 1), resolve_target=view)) as render:
		render.set_pipeline(RenderPipeline(shader, depth_test=None, depth_write=False, sample_count=4))
		render.set_vertex_buffer(slot=0, buffer=buffer)
		render.draw_mesh(mesh=mesh)
	report = monitor.submit(commands.finish())
	resources = report["commands"][0]["passes"][0]["resources"]
	assert f"resolve@{id(target):x}" in resources
	assert f"keyword_buffer@{id(buffer):x}" in resources
	assert any(name.endswith(f"@{id(mesh.index_buffer):x}") for name in resources)
