"""Authored feet/ankles and bounded passive falls use the existing native API."""

import json
import subprocess
import sys

import numpy as np
import pytest

from graaal.Animation.SkeletalAnimation import Skeleton, model_matrices
from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import RagdollAsset
from scenes.passive_ragdoll_example import main, make_humanoid, run_scene, terrain_boxes


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


def test_passive_scene_import_needs_no_ecs_rendering_or_native_backend():
	result = subprocess.run([sys.executable, "-c", '''
import sys
sys.modules['pyjoltphysics'] = None
sys.modules['wgpu'] = None
sys.modules['graaal.ECS'] = None
from scenes.passive_ragdoll_example import make_humanoid, run_scene
assert 'graaal.ECS.core' not in sys.modules
try:
	make_humanoid()
except ModuleNotFoundError as error:
	assert error.name == 'pyjoltphysics'
else:
	raise AssertionError('Construction must require the physics extra')
'''], capture_output=True, text=True, timeout=30)
	assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("height,mass", ((1.8, 70), (0.9, 35), (2.16, 84)))
def test_authored_humanoid_topology_feet_mass_and_skeleton_compatibility(pj, height, mass):
	settings, neutral, spring, limit = make_humanoid(height=height, mass=mass)
	assert len(settings) == 13
	assert sum(bone.mass for bone in settings.bones) == pytest.approx(mass)
	assert settings.self_collision and settings.root_force == settings.root_torque == 0
	assert all(bone.strength == bone.friction == 0 for bone in settings.bones)
	rig = Skeleton(settings.parents, settings.rest_translations, settings.rest_rotations, np.ones((13, 3)), np.tile(np.eye(4), (13, 1, 1)))
	asset = RagdollAsset(rig, settings)
	assert len(asset.bones) == 13
	model = model_matrices(rig, rig.translation[None], rig.rotation[None], rig.scale[None])[0]
	for name, parent in (("foot_l", "shin_l"), ("foot_r", "shin_r")):
		i = settings.bone_index(name)
		bone = settings.bones[i]
		assert bone.parent == settings.bone_index(parent)
		assert bone.shape.kind == pj.ShapeKind.BOX
		np.testing.assert_allclose(bone.shape.params[:3], np.array([0.14, 0.10, 0.26]) * (height / 1.8))
		assert bone.shape_offset[1] + bone.shape.params[1] / 2 == pytest.approx(0)
		assert model[i, 1, 3] + bone.shape_offset[1] - bone.shape.params[1] / 2 == pytest.approx(0, abs=1e-7)
		head = settings.bone_index("head")
	assert model[head, 1, 3] + settings.bones[head].shape_offset[1] + settings.bones[head].shape.params[0] == pytest.approx(height, abs=5e-7)
	assert neutral.shape == (13, 4)
	assert np.any(neutral[1:, :3] != 0)
	for controls in (spring, limit):
		for key in ("stiffness", "damping", "max_torque"):
			assert controls[key].shape == (13,)
			assert controls[key][0] == 0
			assert np.all(controls[key][1:] > 0)


def test_rendered_limb_capsules_match_authored_joint_extents(pj):
	from scenes.passive_ragdoll_viewer import rounded_vertices

	settings = make_humanoid()[0]
	template = np.empty(0, dtype=[("position", "f4", 3), ("normal", "f4", 3), ("uv", "f4", 2)])
	for parent, child in (("upper_arm", "lower_arm"), ("thigh", "shin"), ("shin", "foot")):
		for side in ("l", "r"):
			bone = settings.bones[settings.bone_index(f"{parent}_{side}")]
			tip = settings.bones[settings.bone_index(f"{child}_{side}")].translation
			assert bone.shape.kind == pj.ShapeKind.CAPSULE
			vertices = rounded_vertices(template, *bone.shape.params[:2])["position"]
			poles = vertices[[np.argmin(vertices[:, 1]), np.argmax(vertices[:, 1])]]
			poles = poles @ Quaternion(bone.shape_rotation).matrix33.T + bone.shape_offset
			start = (0, 0.085, 0) if parent == "thigh" else (0, 0, 0)
			np.testing.assert_allclose(poles, [start, tip], atol=1e-7)


def test_playback_samples_include_initial_and_final_poses(pj):
	poses = []
	result = run_scene("drop", frames=6, on_pose=lambda position, orientation: poses.append((position.copy(), orientation.copy())))
	assert len(poses) == result["ticks"] + 1
	assert poses[0][0][0, 1] > poses[-1][0][0, 1]
	np.testing.assert_array_equal(poses[-1][0], result["position"])
	np.testing.assert_array_equal(poses[-1][1], result["orientation"])


def test_missing_dummy_model_reports_how_to_supply_it(monkeypatch, tmp_path, capsys):
	from scenes import passive_ragdoll_viewer

	path = tmp_path / "dummy_rigged.glb"
	monkeypatch.setattr(passive_ragdoll_viewer, "DEFAULT_DUMMY_MODEL", path)
	monkeypatch.setattr(sys, "argv", ["passive_ragdoll_example", "--offscreen", "--model"])
	with pytest.raises(SystemExit) as error:
		main()
	assert error.value.code == 2
	message = capsys.readouterr().err
	assert str(path) in message and "--model PATH" in message


@pytest.mark.parametrize("args, expected", (([], "drop"), (["--view", "--scenario", "stairs"], "stairs")))
def test_passive_cli_defaults_to_window(monkeypatch, args, expected):
	from scenes import passive_ragdoll_viewer

	def view(scenario, *, offscreen, **options):
		assert scenario == expected and not offscreen
		return dict(scenario=scenario, ticks=1, initial_joint_error_rad=0, final_joint_error_rad=0, contact_records=0, max_anchor_error_m=0)

	monkeypatch.setattr(passive_ragdoll_viewer, "run_viewer", view)
	monkeypatch.setattr(sys, "argv", ["passive_ragdoll_example", *args])
	main()


def test_passive_viewer_switches_terrain_and_reuses_recordings(pj, monkeypatch):
	from contextlib import nullcontext
	from types import SimpleNamespace

	pytest.importorskip("wgpu")
	glfw = pytest.importorskip("glfw")
	from graaal.Materials import MaterialRenderer as materials
	from graaal.RenderContext import RenderContext
	from graaal.UI import ImGui
	from scenes import passive_ragdoll_viewer as viewer

	titles, keys, recorded = [], [], []
	monkeypatch.setattr(glfw, "set_window_title", lambda window, title: titles.append(title))
	monkeypatch.setattr(RenderContext, "subscribe_event", lambda channel, handler: keys.append(handler))
	monkeypatch.setattr(viewer, "viewing_target", lambda *args, **kwargs: nullcontext())
	monkeypatch.setattr(materials, "MaterialRenderer", lambda instances, **kwargs: SimpleNamespace(instances=instances))
	imgui = SimpleNamespace(Cond_=SimpleNamespace(first_use_ever=0),
		WindowFlags_=SimpleNamespace(always_auto_resize=1, no_saved_settings=2, no_collapse=4),
		set_next_window_pos=lambda *args: None, set_next_item_width=lambda *args: None,
		begin=lambda *args, **kwargs: (True, None), end=lambda: None, combo=lambda *args: (True, 0))
	monkeypatch.setattr(ImGui, "ImGui", lambda: nullcontext(SimpleNamespace(imgui=imgui, render=lambda draw: draw())))

	def record(name, **options):
		recorded.append(name)
		return run_scene(name, **options)

	def play(renderer, camera, output, update, *, before_frame, draw_overlay, **options):
		drop_target = camera.target.copy()
		assert len(renderer.instances) == 14
		keys[0](glfw.KEY_3, 0, glfw.PRESS, 0)
		assert before_frame()
		update(0)
		assert len(renderer.instances) == 13 + len(terrain_boxes("stairs", 1)[0])
		assert not np.allclose(camera.target, drop_target)
		assert " | Stairs | " in titles[-1]
		draw_overlay()
		assert before_frame()
		assert len(renderer.instances) == 13
		assert " | Zero gravity | " in titles[-1]
		keys[0](glfw.KEY_2, 0, glfw.PRESS, 0)
		assert before_frame()
		np.testing.assert_allclose(camera.target, drop_target)

	monkeypatch.setattr(viewer, "run_scene", record)
	monkeypatch.setattr(viewer, "play_frames", play)
	assert viewer.run_viewer("drop", frames=2)["scenario"] == "drop"
	assert recorded == ["drop", "stairs", "zero_gravity"]


@pytest.mark.parametrize("height", (0.9, 1.8, 2.16))
def test_stair_spawn_clears_top_step(pj, height):
	poses = []
	settings = make_humanoid(height=height)[0]
	result = run_scene("stairs", height=height, frames=1, on_pose=lambda p, q: poses.append((p.copy(), q.copy())))
	positions, orientations = poses[0]
	ground_positions, ground_sizes = terrain_boxes("stairs", height / 1.8)
	top = np.max(ground_positions[:, 1] + ground_sizes[:, 1] / 2)
	for name in ("foot_l", "foot_r"):
		i = settings.bone_index(name)
		bone = settings.bones[i]
		rotation = Quaternion.matrix_batch(orientations[i])
		center = positions[i] + rotation @ bone.shape_offset
		half_height = np.abs(rotation[1]) @ np.asarray(bone.shape.params[:3]) / 2
		assert center[1] - half_height > top
	assert result["peak_contact_depth_m"] == 0
	if height == 1.8: assert result["stair_contact_records"] == 0


def test_peak_speeds_include_immediate_impact_response(pj):
	result = run_scene("impact", frames=1)
	impact = result["impact"]
	assert impact["tick"] == 0
	assert result["max_linear_speed_m_s"] >= np.linalg.norm(impact["delta_linear_velocity"]) - 1e-6
	assert result["max_angular_speed_rad_s"] >= np.linalg.norm(impact["delta_angular_velocity"]) - 1e-6


def test_authored_gain_scaling_and_independent_factory_outputs(pj):
	a, neutral_a, spring_a, limit_a = make_humanoid()
	b, neutral_b, spring_b, limit_b = make_humanoid(height=0.9, mass=35)
	np.testing.assert_allclose(neutral_a, neutral_b)
	for controls_a, controls_b in ((spring_a, spring_b), (limit_a, limit_b)):
		for name in ("stiffness", "damping", "max_torque"):
			np.testing.assert_allclose(controls_b[name], controls_a[name] * 0.125)
			assert not np.shares_memory(controls_a[name], controls_b[name])
	b.bones[1].mass = 1000
	assert a.bones[1].mass == 22


@pytest.mark.parametrize("height,mass", ((0, 70), (-1, 70), (np.nan, 70), (1.8, 0), (1.8, -1), (1.8, np.inf)))
def test_invalid_authored_scale_and_mass_rejected(pj, height, mass):
	with pytest.raises(ValueError, match="finite and positive"): make_humanoid(height=height, mass=mass)


@pytest.mark.parametrize("threads", (1, 2))
def test_passive_zero_gravity_relaxes_without_active_motors(pj, threads):
	baseline = run_scene("zero_gravity", frames=960, threads=threads, passive=False)
	result = run_scene("zero_gravity", frames=960, threads=threads)
	assert baseline["final_joint_error_rad"] == pytest.approx(baseline["initial_joint_error_rad"], abs=1e-7)
	assert result["final_joint_error_rad"] < result["initial_joint_error_rad"] * 0.3
	assert result["contact_records"] == 0
	assert result["max_anchor_error_m"] < 0.001
	assert result["mode"] == pj.RagdollMode.LIMP and result["active_strength"] == 0
	assert result["stats"]["bodies"] == result["bodies"] == 13
	assert result["stats"]["rejected"] == result["stats"]["events_dropped"] == 0
	assert result["position"].flags.owndata and result["orientation"].flags.owndata
	settings = make_humanoid()[0]
	position = result["position"]
	local = (position - position[0]) @ Quaternion.matrix_batch(result["orientation"][0])
	for side, sign in (("l", 1), ("r", -1)):
		shoulder, elbow, hip, knee, ankle = (local[settings.bone_index(f"{name}_{side}")] for name in ("upper_arm", "lower_arm", "thigh", "shin", "foot"))
		assert (elbow[0] - shoulder[0]) * sign > 0.04
		assert elbow[2] > shoulder[2] + 0.025
		assert (knee[0] - hip[0]) * sign > 0.035
		assert knee[2] > max(hip[2], ankle[2]) + 0.08


@pytest.mark.parametrize("scenario", ("drop", "stairs", "impact"))
@pytest.mark.parametrize("threads", (1, 2))
def test_passive_contact_scenarios_are_bounded_and_settle(pj, scenario, threads):
	result = run_scene(scenario, frames=960, threads=threads)
	assert result["ticks"] == result["stats"]["timed_ticks"] == 960
	assert result["mode"] == pj.RagdollMode.LIMP and result["active_strength"] == 0
	assert result["stats"]["bodies"] == (20 if scenario == "stairs" else 14)
	assert result["stats"]["rejected"] == result["stats"]["events_dropped"] == 0
	assert result["floor_contact_records"] > 0 and result["foot_contact_records"] > 0
	if scenario == "stairs": assert result["stair_contact_records"] > 0 and result["self_contact_records"] > 0
	assert result["max_anchor_error_m"] < 0.06
	assert result["peak_contact_depth_m"] < 0.08
	assert result["max_linear_speed_m_s"] < 20
	assert result["late_max_linear_speed_m_s"] < 0.3
	assert result["late_max_angular_speed_rad_s"] < 2
	assert result["final_linear_kinetic_energy_j"] < 0.1
	assert np.isfinite(result["position"]).all() and np.isfinite(result["orientation"]).all()
	if scenario == "impact":
		impact = result["impact"]
		assert impact["tick"] == 24
		np.testing.assert_allclose(impact["delta_linear_velocity"], impact["impulse_ns"] / 22, atol=1e-6)
		assert np.linalg.norm(impact["delta_angular_velocity"]) > 1
	else: assert result["impact"] is None
	json.dumps(result, default=lambda value: value.tolist(), allow_nan=False)


@pytest.mark.parametrize("scenario,frames,threads", (("unknown", 1, 1), ("drop", 0, 1), ("drop", 1, 0)))
def test_invalid_scene_parameters_rejected(pj, scenario, frames, threads):
	with pytest.raises(ValueError): run_scene(scenario, frames=frames, threads=threads)
