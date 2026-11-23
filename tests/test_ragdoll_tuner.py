import json

import numpy as np
import pytest

from graaal.Math import Quaternion
from tools.ragdoll_preview import ColliderViewport, RagdollPreview, SCENARIOS
from tools.ragdoll_preview_raster import PreviewRaster
from tools.ragdoll_project import default_project, open_project, read_project, write_project
from tools.ragdoll_tuner import RagdollTuner, _bone_fields, _edited_project


@pytest.fixture
def project():
	pytest.importorskip("pyjoltphysics")
	return default_project()


def test_tuned_project_preserves_native_settings_and_passive_controls(project, tmp_path):
	initial = _bone_fields(project, 1)
	fields = initial | {"shape": "Cylinder", "dimensions": "0.14 0.32", "mass": "18.5", "shape_offset": "0 0.13 0.05", "shape_rotation": "0 90 40", "twist": "-20 25",
		"swing": "30 45", "bend": "10", "twist_axis": "0 1 0", "plane_axis": "1 0 0", "neutral": "15 5 -20", "margin": "4", "spring_stiffness": "2", "limit_max_torque": "1.7"}
	tuned = _edited_project(project, 1, fields, initial)
	path = tmp_path / "tuned.json"
	write_project(path, tuned)
	saved = read_project(path)
	assert saved.settings.as_dict() == tuned.settings.as_dict()
	for saved_gains, tuned_gains in ((saved.spring, tuned.spring), (saved.soft_limit, tuned.soft_limit)):
		for key in saved_gains: np.testing.assert_array_equal(saved_gains[key], tuned_gains[key])
	bone = saved.settings.bones[1]
	assert bone.shape.kind.name == "CYLINDER" and bone.shape.params == (0.14, 0.32, 0, 0)
	assert bone.mass == 18.5 and project.settings.bones[1].mass == 22
	assert saved.settings.as_dict()["bones"][2:] == project.settings.as_dict()["bones"][2:]
	np.testing.assert_allclose(np.rad2deg(bone.twist), [-20, 25])
	np.testing.assert_allclose(np.rad2deg(saved.soft_limit["margin"][1]), 4)
	np.testing.assert_allclose(saved.neutral[1], np.asarray(Quaternion.from_eulers(np.deg2rad([15, 5, -20]))))
	displayed = _bone_fields(saved, 1)
	eulers = np.deg2rad(np.fromstring(displayed["shape_rotation"], sep=" "))
	np.testing.assert_allclose(Quaternion.from_eulers(eulers).matrix33, Quaternion(bone.shape_rotation).matrix33, atol=1e-10)


def test_plain_native_settings_get_rest_neutral_pose_and_zero_passive_gains(project, tmp_path):
	path = tmp_path / "native.json"
	path.write_text(json.dumps(project.settings.as_dict()), encoding="utf-8")
	loaded, native_import = open_project(path)
	assert native_import
	assert loaded.settings.as_dict() == project.settings.as_dict()
	np.testing.assert_array_equal(loaded.neutral, loaded.settings.rest_rotations)
	for gain in ("stiffness", "damping", "max_torque"):
		assert not loaded.spring[gain].any()
		assert not loaded.soft_limit[gain].any()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_native_preview_steps_with_tuned_mass_and_rebuilds_at_tick_zero(project, scenario):
	initial = _bone_fields(project, 1)
	tuned = _edited_project(project, 1, initial | {"mass": "20"}, initial)
	first = RagdollPreview(tuned, scenario)
	try:
		tick_zero = first.state.position.copy()
		velocity = first.state.linear_velocity[1].copy()
		first.push(1, (2, 0, 0))
		np.testing.assert_allclose(first.state.linear_velocity[1] - velocity, [0.1, 0, 0], atol=1e-6)
		first.step(48)
		assert first.world.tick == 48
		assert np.isfinite(first.state.position).all()
		assert not np.array_equal(first.state.position, tick_zero)
	finally:
		first.close()
	restarted = RagdollPreview(tuned, scenario)
	try:
		assert restarted.world.tick == 0
		np.testing.assert_array_equal(restarted.state.position, tick_zero)
	finally:
		restarted.close()


@pytest.mark.parametrize("tuned", (False, True))
def test_stair_preview_spawns_lowest_collider_over_highest_tread(project, tuned):
	import pyjoltphysics as pj

	foot = project.settings.bone_index("foot_r")
	if tuned:
		initial = _bone_fields(project, foot)
		fields = initial | {"dimensions": "0.18 0.17 0.32 0", "shape_rotation": "10 15 23", "shape_offset": "0.01 -0.07 0.09"}
		project = _edited_project(project, foot, fields, initial)
	preview = RagdollPreview(project, "Stairs")
	try:
		positions, sizes = preview.terrain
		top = positions[-1, 1] + sizes[-1, 1] / 2
		assert preview.world.tick == 0
		assert preview.bounds()[0][1] > top
		bone = project.settings.bones[foot]
		turn = preview.state.orientation[foot]
		center = preview.state.position[foot] + Quaternion.matrix_batch(turn) @ bone.shape_offset
		np.testing.assert_allclose(center[[0, 2]], positions[-1, [0, 2]], atol=1e-6)
		terrain_only = pj.QueryFilter(exclude=tuple(map(int, preview.uids)))
		rotation = Quaternion.multiply_batch(turn, bone.shape_rotation)
		assert not len(preview.world.overlap(bone.shape, center, rotation, terrain_only))
	finally:
		preview.close()


def test_stair_view_keeps_feet_visible_above_treads(project):
	preview = RagdollPreview(project, "Stairs")
	try:
		view = ColliderViewport(preview)
		assert view.yaw < 0 < view.pitch
		full_stairs = view.stairs
		feet = [project.settings.bone_index(name) for name in ("foot_l", "foot_r")]
		for yaw in (view.yaw, 0.45):
			view.yaw = yaw
			view.stairs = full_stairs[:0]
			view.render(preview, 0, 650, 570)
			unobscured = np.isin(view.raster.owners, feet)
			assert unobscured.any()
			view.stairs = full_stairs
			view.render(preview, 0, 650, 570)
			np.testing.assert_array_equal(np.isin(view.raster.owners, feet), unobscured)
	finally:
		preview.close()


def test_overlapping_triangles_resolve_depth_per_pixel_in_either_draw_order():
	raster = PreviewRaster()
	raster.clear(40, 40, near=0.1)
	screen = np.array([[6, 6], [30, 6], [6, 30]])
	xy = (screen - raster.center) / [raster.focal, -raster.focal]
	depth = np.array([[1, 4, 4], [2.05, 2.05, 2.05]])
	faces = np.concatenate((xy * depth[..., None], depth[..., None]), axis=-1)
	colors = np.array([[170, 30, 15], [25, 75, 180]])
	owners = np.array([1, 2])
	raster.triangles(faces, colors, owners)
	image = raster.rgb.copy()
	picked = raster.owners.copy()
	assert picked[8, 8] == 1 and picked[12, 20] == 2
	np.testing.assert_array_equal(image[8, 8], colors[0])
	np.testing.assert_array_equal(image[12, 20], colors[1])
	raster.clear(40, 40, near=0.1)
	raster.triangles(faces[::-1], colors[::-1], owners[::-1])
	np.testing.assert_array_equal(raster.rgb, image)
	np.testing.assert_array_equal(raster.owners, picked)


def test_near_clipping_keeps_visible_part_of_triangle():
	raster = PreviewRaster()
	raster.clear(40, 40, near=0.1)
	faces = np.array([[[-0.02, 0.03, 0.01], [0.16, 0.11, 0.4], [0.02, -0.12, 0.4]]])
	raster.triangles(faces, np.array([[170, 30, 15]]), np.array([1]))
	assert (raster.owners == 1).any()
	assert raster.depth.max() <= 10


def test_editor_applies_to_live_preview_and_saves_separate_project(project, tmp_path, monkeypatch):
	tk = pytest.importorskip("tkinter")
	try: root = tk.Tk()
	except tk.TclError as error: pytest.skip(f"Tk display unavailable: {error}")
	errors = []
	root.report_callback_exception = lambda *error: errors.append(error)
	native = tmp_path / "native.json"
	original = json.dumps(project.settings.as_dict())
	native.write_text(original, encoding="utf-8")
	app = None
	try:
		app = RagdollTuner(root, project, native, native_import=True)
		width, height = root.minsize()
		root.geometry(f"{width}x{height}")
		root.update()
		assert app.canvas.winfo_width() > 350 and (app.viewport.raster.owners == 0).any()
		assert app.preview_image.width() == app.canvas.winfo_width()
		assert app.tabs.tab(1, "state") == "disabled"
		canvas = app.canvas
		pixels = np.argwhere(app.viewport.raster.owners == 1)
		assert len(pixels)
		y, x = map(int, pixels[len(pixels) // 2])
		assert app.preview_image.get(x, y) == tuple(map(int, app.viewport.raster.rgb[y, x]))
		yaw, pitch = app.viewport.yaw, app.viewport.pitch
		canvas.event_generate("<ButtonPress-1>", x=x, y=y)
		canvas.event_generate("<B1-Motion>", x=x + 20, y=y + 10)
		canvas.event_generate("<B1-Motion>", x=x, y=y)
		canvas.event_generate("<ButtonRelease-1>", x=x, y=y)
		assert app.bone == 0
		np.testing.assert_allclose((app.viewport.yaw, app.viewport.pitch), (yaw, pitch))
		canvas.event_generate("<ButtonPress-1>", x=x, y=y)
		canvas.event_generate("<B1-Motion>", x=x + 1, y=y)
		canvas.event_generate("<ButtonRelease-1>", x=x, y=y)
		assert app.bone == 1
		np.testing.assert_allclose((app.viewport.yaw, app.viewport.pitch), (yaw, pitch))
		canvas.event_generate("<ButtonPress-1>", x=x, y=y)
		canvas.event_generate("<B1-Motion>", x=x + 20, y=y + 10)
		canvas.event_generate("<ButtonRelease-1>", x=x + 20, y=y + 10)
		np.testing.assert_allclose((app.viewport.yaw, app.viewport.pitch), (yaw - 0.2, pitch + 0.1))
		app.scenario.set("Stairs")
		app.reset()
		assert app.preview.scenario == "stairs" and app.viewport.yaw < 0
		app.viewport.yaw, app.viewport.pitch, app.viewport.zoom = 0.25, 0.5, 0.8
		app.reset()
		assert (app.viewport.yaw, app.viewport.pitch, app.viewport.zoom) == (0.25, 0.5, 0.8)
		app.tree.selection_set("1")
		root.update()
		assert app.bone == 1 and app.tabs.tab(1, "state") == "normal"
		app.tabs.select(2)
		app.step()
		preview = app.preview
		app.fields["mass"].set("-5")
		app.apply()
		assert app.preview is preview and preview.world.tick == 1
		assert "mass is positive" in app.status.get()
		app.tree.selection_set("2")
		root.update()
		assert app.bone == 1 and app.fields["mass"].get() == "-5"
		app.fields["mass"].set("19")
		app.fields["shape"].set("Sphere")
		app.change_shape()
		app.fields["dimensions"].set("0.14")
		app.fields["spring_stiffness"].set("3")
		app.apply()
		assert app.preview is not preview and app.preview.world.tick == 0
		assert (app.viewport.yaw, app.viewport.pitch, app.viewport.zoom) == (0.25, 0.5, 0.8)
		assert app.project.settings.bones[1].mass == 19
		assert app.project.settings.bones[1].shape.kind.name == "SPHERE"
		assert app.preview.project.spring["stiffness"][1] == 3
		assert app.tree.set("1", "mass") == "19"
		monkeypatch.setattr(app.filedialog, "asksaveasfilename", lambda **_: str(native))
		app.save()
		assert native.read_text(encoding="utf-8") == original
		project_path = tmp_path / "tuned.json"
		monkeypatch.setattr(app.filedialog, "asksaveasfilename", lambda **_: str(project_path))
		app.save()
		assert native.read_text(encoding="utf-8") == original
		assert read_project(project_path).settings.bones[1].mass == 19
		assert not errors
	finally:
		if app is not None: app.dispose()
		root.destroy()
