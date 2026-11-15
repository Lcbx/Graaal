"""Keyboard settings and range validation without creating a GPU device."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import glfw

from graaal.TransparencyPass import DEBUG_VIEWS
from graaal.RenderContext import RenderContext
from scenes.rendering_settings import FilterSettings
from scenes import transparency_example as scene
from scenes.example_overlay import ExampleOverlay


def make_renderer(**overrides):
	values = dict(debug_view="final", banded_proxy=False, kawase_passes=1)
	values.update(overrides)
	renderer = SimpleNamespace(**values)
	renderer.set_kawase_passes = lambda value: setattr(renderer, "kawase_passes", value)
	renderer.set_debug_view = lambda value: setattr(renderer, "debug_view", value)
	return renderer


def test_antialiasing_modes_disable_msaa_for_postprocess():
	renderer = make_renderer(sample_count=4)
	renderer.set_sample_count = lambda count: setattr(renderer, "sample_count", count)
	aa = SimpleNamespace(mode="off")
	panel = FilterSettings(renderer, aa_pass=aa)
	panel.set_aa_mode("lxaa")
	assert renderer.sample_count == 1 and aa.mode == "lxaa"
	panel.set_aa_mode("faaa")
	assert renderer.sample_count == 1 and aa.mode == "faaa"
	panel.set_aa_mode("ddaa2")
	assert renderer.sample_count == 1 and aa.mode == "ddaa2"
	panel.set_aa_mode("msaa")
	assert renderer.sample_count == 4 and aa.mode == "msaa"
	panel.set_aa_mode("off")
	assert renderer.sample_count == 1 and aa.mode == "off"



def test_transparency_msaa_shortcut_updates_antialiasing_mode(monkeypatch):
	renderer = make_renderer(sample_count=1)
	renderer.set_sample_count = lambda count: setattr(renderer, "sample_count", count)
	aa = SimpleNamespace(mode="off")
	monkeypatch.setattr(scene, "transparency", renderer, raising=False)
	monkeypatch.setattr(scene, "msaa_enabled", False)
	panel = FilterSettings(renderer, on_msaa_changed=scene.set_msaa_enabled, aa_pass=aa)
	monkeypatch.setattr(scene, "settings", panel, raising=False)
	monkeypatch.setattr(scene, "overlay", scene.ExampleOverlay(panel), raising=False)
	panel.set_aa_mode("ddaa2")
	scene.on_char(ord("m"))
	assert renderer.sample_count == 4 and panel.aa_mode == aa.mode == "msaa"
	scene.on_char(ord("m"))
	assert renderer.sample_count == 1 and panel.aa_mode == aa.mode == "off"


def test_example_overlay_shortcuts_share_settings_and_profiler():
	renderer = make_renderer(sample_count=1)
	renderer.set_sample_count = lambda count: setattr(renderer, "sample_count", count)
	settings = FilterSettings(renderer, visible=False)
	method = ["compose"]
	overlay = ExampleOverlay(settings, get_ao_method=lambda: method[0], set_ao_method=lambda value: method.__setitem__(0, value))
	assert overlay.handle_char(ord("m")) and renderer.sample_count == 4
	assert overlay.handle_char(ord("a")) and settings.ssao_enabled
	assert overlay.handle_char(ord("o")) and method[0] == "pass"
	assert overlay.handle_char(ord("p")) and overlay.profiler.visible
	assert overlay.handle_key(glfw.KEY_F, 0, glfw.PRESS, 0) and settings.visible
	assert not overlay.handle_char(ord("z"))


def test_example_overlay_collects_reports_only_while_profiler_is_visible(monkeypatch):
	settings = FilterSettings(make_renderer(sample_count=1), visible=False)
	overlay = ExampleOverlay(settings)
	plain = SimpleNamespace(finish=Mock(return_value="plain"))
	monitored = SimpleNamespace(finish=Mock(return_value="monitored"))
	monkeypatch.setattr(RenderContext, "commands", Mock(return_value=plain))
	monkeypatch.setattr(RenderContext, "submit", Mock())
	overlay.monitor = SimpleNamespace(commands=Mock(return_value=monitored), submit=Mock(return_value={"commands": []}))
	overlay.profiler.update = Mock()
	assert overlay.commands("scene") is plain
	overlay.submit(plain)
	RenderContext.submit.assert_called_once_with("plain")
	overlay.monitor.submit.assert_not_called()
	overlay.handle_char(ord("p"))
	assert overlay.commands("scene") is monitored
	overlay.submit(monitored)
	overlay.monitor.submit.assert_called_once_with("monitored")
	overlay.profiler.update.assert_called_once_with({"commands": []})



def test_keyboard_edits_clamp_and_reset_blur():
	renderer = make_renderer()
	settings = FilterSettings(renderer)
	def key(code, action=glfw.PRESS): return settings.handle_key(code, 0, action, 0)
	for _ in range(12): key(glfw.KEY_RIGHT)
	assert renderer.kawase_passes == 8
	for _ in range(12): key(glfw.KEY_LEFT)
	assert renderer.kawase_passes == 0
	key(glfw.KEY_RIGHT, glfw.RELEASE)
	assert renderer.kawase_passes == 0
	key(glfw.KEY_HOME)
	assert renderer.kawase_passes == 1
	key(glfw.KEY_F)
	assert not settings.visible and not key(glfw.KEY_RIGHT)
	key(glfw.KEY_F, glfw.REPEAT)
	assert not settings.visible
	key(glfw.KEY_F)
	assert settings.visible
	assert not key(glfw.KEY_F5)


def test_shortcuts_change_screen_noise_and_cycle_supported_diagnostics(monkeypatch):
	renderer = make_renderer(noise_scale=1.0)
	renderer.set_noise_scale = lambda value: setattr(renderer, "noise_scale", value)
	monkeypatch.setattr(scene, "transparency", renderer, raising=False)
	visited = set()
	for _ in DEBUG_VIEWS:
		scene.on_char(ord("d"))
		visited.add(renderer.debug_view)
	assert visited == set(DEBUG_VIEWS)
	for _ in range(30): scene.on_char(ord("["))
	assert renderer.noise_scale == 0.25
	for _ in range(60): scene.on_char(ord("]"))
	assert renderer.noise_scale == 4
	for digit, pairs in zip("1359", range(4)):
		scene.on_char(ord(digit))
		assert renderer.kawase_passes == pairs
	assert FilterSettings.debug_view_label(renderer, "winner") == "winner"
	renderer.banded_proxy = True
	assert FilterSettings.debug_view_label(renderer, "winner") == "signed residual"
