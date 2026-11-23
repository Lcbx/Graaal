"""Playback timing and recorded data for the optional physics viewers."""

import argparse
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scenes import physics_viewing_utils


def test_playback_cli_modes_are_mutually_exclusive():
	parser = argparse.ArgumentParser()
	physics_viewing_utils.add_viewing_options(parser)
	assert vars(parser.parse_args([])) == dict(view=False, headless=False, offscreen=False)
	assert parser.parse_args(["--headless"]).headless
	with pytest.raises(SystemExit):
		parser.parse_args(["--headless", "--offscreen"])


def test_playback_uses_sample_rate_and_supports_pause_and_restart(monkeypatch):
	keys = SimpleNamespace(PRESS=1, KEY_SPACE=2, KEY_R=3)
	frames = iter(range(10))

	def window_loop():
		frame = next(frames, None)
		if frame in (3, 6, 9):
			key = keys.KEY_R if frame == 9 else keys.KEY_SPACE
			context.on_key(key, 0, keys.PRESS, 0)
		return frame is not None

	context = SimpleNamespace(window_loop=window_loop, frame_time=0.1)
	context.subscribe_event = lambda channel, callback: setattr(context, "on_key", callback)
	monkeypatch.setitem(sys.modules, "graaal.RenderContext", SimpleNamespace(RenderContext=context))
	monkeypatch.setitem(sys.modules, "glfw", keys)
	monkeypatch.setattr(physics_viewing_utils, "draw_frame", lambda *args: None)
	displayed = []
	renderer = SimpleNamespace(sync=lambda: None)
	physics_viewing_utils.play_frames(renderer, None, None, displayed.append, label="test", sample_count=3, sample_hz=4)
	assert displayed == [0, 0, 0, 1, 1, 1, 1, 1, 2, 0]


def test_switch_restarts_playback_at_sample_zero(monkeypatch):
	keys = SimpleNamespace(PRESS=1, KEY_SPACE=2, KEY_R=3)
	frames = iter((True,) * 5 + (False,))
	context = SimpleNamespace(window_loop=lambda: next(frames), frame_time=0.1, subscribe_event=lambda *args: None)
	monkeypatch.setitem(sys.modules, "graaal.RenderContext", SimpleNamespace(RenderContext=context))
	monkeypatch.setitem(sys.modules, "glfw", keys)
	monkeypatch.setattr(physics_viewing_utils, "draw_frame", lambda *args: None)
	displayed = []
	renderer = SimpleNamespace(sync=lambda: None)
	physics_viewing_utils.play_frames(renderer, None, None, displayed.append, label="test", sample_count=8, sample_hz=10,
		before_frame=lambda: len(displayed) == 3)
	assert displayed == [0, 1, 2, 0, 1]


def test_native_box_capture_matches_the_headless_result():
	pytest.importorskip("pyjoltphysics")
	from scenes.physics_headless_example import run_scene

	samples = []
	options = dict(ticks=12, tick_hz=60)
	result = run_scene(**options, on_pose=lambda position, rotation: samples.append((position, rotation)))
	headless = run_scene(**options)
	assert len(samples) == result["tick"] + 1
	np.testing.assert_array_equal(samples[0][0][:, 1], [2, 3, 4])
	np.testing.assert_allclose(samples[-1][0], result["position"])
	np.testing.assert_allclose(result["position"], headless["position"])


def test_locomotion_capture_tracks_accepted_movement():
	pytest.importorskip("pyjoltphysics")
	from scenes.physics_locomotion_example import run_scene

	samples = []
	run_scene(on_pose=lambda position, rotation: samples.append(position))
	positions = np.array(samples)
	assert len(positions) == 13
	np.testing.assert_allclose(positions[:, 1, 0], 2)
	assert positions[0, 0, 0] == 0
	np.testing.assert_allclose(positions[-2:, 0, 0], 1.495, atol=1e-6)


def test_query_capture_aligns_world_space_legs_to_collision_terrain():
	pytest.importorskip("pyjoltphysics")
	from scenes.physics_queries_example import run_scene, terrain_geometry

	samples = []
	run_scene(on_sample=lambda *sample: samples.append(sample))
	assert len(samples) == 8
	joints, ray_hit, sweep_center = samples[-1]
	terrain, _ = terrain_geometry()
	x, y = terrain[::2, :2].T
	np.testing.assert_allclose(joints[0, -1, 1], np.interp(joints[0, -1, 0], x, y), atol=1e-6)
	assert joints[1, -1, 1] == 0
	np.testing.assert_allclose(ray_hit, [1, np.interp(1, x, y), 0], atol=1e-6)
	assert sweep_center[1] > ray_hit[1]


@pytest.mark.parametrize("end", ((3, 1, 4), (1, 1, 1)))
def test_segment_transform_preserves_world_space_endpoints(end):
	from scenes.physics_geometry import segment_transform

	start, end = np.array([1, 2, 1]), np.array(end)
	center, rotation, scale = segment_transform(start, end, 0.09)
	half_segment = rotation.matrix33[:, 1] * scale[1] / 2
	np.testing.assert_allclose(center - half_segment, start)
	np.testing.assert_allclose(center + half_segment, end)


def test_collider_poses_apply_local_offsets_and_rotations_in_bone_space():
	from graaal.Math import Quaternion
	from scenes.physics_geometry import collider_poses

	positions = np.array([[4, 5, 6], [10, 20, 30]])
	orientations = np.array([Quaternion.from_z_rotation(np.pi / 2), Quaternion()])
	offsets = np.array([[2, 0, 0], [1, 2, 3]])
	shape_rotations = np.array([Quaternion.from_x_rotation(np.pi / 2), Quaternion()])
	centers, rotations = collider_poses(positions, orientations, offsets, shape_rotations)
	np.testing.assert_allclose(centers, [[4, 7, 6], [11, 22, 33]])
	np.testing.assert_allclose(Quaternion.matrix_batch(rotations) @ [1, 2, 3], [[3, 1, 2], [1, 2, 3]])
