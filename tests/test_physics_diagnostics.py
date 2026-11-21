"""Headless native statistics collection and profiler presentation contracts."""

import json
from types import SimpleNamespace as NS

import pytest

from graaal.Diagnostics import inspect_physics
from graaal.diagnostics_visualizer import DiagnosticsVisualizer


def test_inspection_preserves_pending_work_events_and_snapshot_pool():
	pj = pytest.importorskip("pyjoltphysics")
	with pj.World(pj.WorldSettings(threads=1)) as world:
		world.add_bodies(pj.BodyBatch.boxes([[0, 2, 0]]))
		pending = inspect_physics(world)
		assert pending["tick"] == 0
		assert pending["tick_hz"] == 120
		assert pending["lockstep"] is True
		assert pending["stats"]["queued_bodies"] == 1
		assert pending["stats"]["bodies"] == 0
		assert pending["stats"]["pool"] == 0
		world.run_ticks(2)
		before = world.stats()
		report = inspect_physics(world)
		assert report["tick"] == 2
		assert report["stats"]["bodies"] == 1
		assert report["stats"] == before == world.stats()
		assert {"tick_ms", "step_ms", "publish_ms", "events_ms", "events_dropped", "pool", "queries", "query_wait_ms", "query_wait_max_ms"} <= report["stats"].keys()
		json.dumps(report)
		world.reset_stats()
		assert inspect_physics(world)["stats"]["timed_ticks"] == 0
		assert report["stats"]["timed_ticks"] == 2
		assert pending["stats"]["bodies"] == 0


def physics_report(tick, mean):
	return dict(physics=dict(tick=tick, tick_hz=120, lockstep=True,
		stats=dict(tick_ms=mean, step_ms=mean / 2, bodies=2, events_dropped=3, pool=4, query_wait_max_ms=0.25)))


def test_physics_history_is_bounded_paused_and_separate_from_render_times():
	widget = DiagnosticsVisualizer(history_size=2)
	widget.update(physics_report(1, 1.0))
	widget.update(physics_report(2, 2.0))
	combined = physics_report(3, 3.0)
	combined.update(commands=[dict(encode_ms=8.0, gpu_span_ms=5.0)], gpu_timing_available=True)
	widget.update(combined)
	assert list(widget.physics_history) == [2.0, 3.0]
	assert list(widget.history) == [5.0]
	widget.paused = True
	widget.update(physics_report(4, 4.0))
	assert widget.report is combined
	assert list(widget.physics_history) == [2.0, 3.0]
	widget.paused = False
	widget.update(dict(commands=[dict(encode_ms=7.0)]))
	assert list(widget.physics_history) == [2.0, 3.0]
	assert list(widget.history) == [7.0]


def test_physics_panel_displays_native_units_and_all_counters():
	widget = DiagnosticsVisualizer()
	widget.update(physics_report(42, 1.5))
	lines, plots = [], []
	ui = NS(collapsing_header=lambda name: True, text=lines.append, text_disabled=lines.append,
		plot_lines=lambda *args, **kwargs: plots.append((args, kwargs)))
	widget._draw_physics(ui)
	assert "Tick: 42 at 120 Hz   Lockstep: True" in lines
	assert "tick_ms: 1.50 ms" in lines
	assert "events_dropped: 3" in lines
	assert "pool: 4" in lines
	assert "query_wait_max_ms: 0.25 ms" in lines
	assert len(plots) == 1
	widget.update({})
	widget._draw_physics(ui)
	assert len(plots) == 1


def test_physics_only_report_draws_in_imgui_frame():
	imgui = pytest.importorskip("imgui_bundle").imgui
	previous = imgui.get_current_context()
	context = imgui.create_context()
	try:
		io = imgui.get_io()
		io.set_ini_filename("")
		io.display_size = (900, 700)
		io.delta_time = 1 / 60
		io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
		widget = DiagnosticsVisualizer()
		widget.update(physics_report(42, 1.5))
		imgui.new_frame()
		widget.draw()
		imgui.render()
		assert imgui.get_draw_data().cmd_lists_count > 0
	finally:
		imgui.destroy_context(context)
		if previous is not None: imgui.set_current_context(previous)
