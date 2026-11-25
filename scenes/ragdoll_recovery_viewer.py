"""Replay seeded ragdoll standing and push-recovery trials in 3D."""

import argparse
from functools import cache, lru_cache
import json
from pathlib import Path

import numpy as np

from scenes.benchmarks.benchmark_ragdoll_push_recovery import PUSH_SECONDS, PUSH_TICK, RECOVERY_SECONDS, TICK_HZ, make_trial_controller, run_quiet_trial, run_trial
from scenes.physics_geometry import collider_bounds, collider_poses, cube_vertices, segment_transform
from scenes.physics_viewing_utils import Playback, add_viewing_options, has_blue_pixels, play_frames, ragdoll_collider_instances, viewing_target
from tools.ragdoll_project import default_project, read_project


CONTROLLER_LABELS = {"ankle_balance": "Ankle balance", "scripted_step": "Scripted backstep", "ppo_residual": "Residual PPO",
	"ppo_stepping": "Stepping PPO", "ppo_whole_body": "Whole-body PPO", "ppo_recovery": "Recovery PPO", "ppo_direct": "Direct PPO"}
SUITE_REPORTS = {"hard": "Hard backward pushes", "light": "Light pushes", "original": "Original push grid",
	"margin": "Strong pushes", "quiet-replay": "Quiet standing"}
RESULT_FILTERS = ("All results", "Passed", "Failed", "With steps")
PASS_OUTCOMES = ("recovered", "settled")
PASS_COLOR = (0.52, 0.92, 0.69, 1)
FAIL_COLOR = (1.0, 0.66, 0.51, 1)


def read_report(path):
	report = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(report, dict) or not report.get("cells") or report.get("tick_hz") != TICK_HZ:
		raise ValueError(f"{path} must be a full benchmark report with individual trials")
	return report


def suite_reports(prefix):
	return [(path, label) for suffix, label in SUITE_REPORTS.items()
		if (path := prefix.with_name(f"{prefix.name}-{suffix}.json")).is_file()]


def report_assets(path, report, controller):
	metadata = report.get("checkpoints", {}).get(controller, {})
	checkpoint = metadata.get("path") or (report.get("ppo_checkpoint") if controller == "ppo_residual" else None)

	def resolve(filename):
		if filename is None:
			return None
		asset = Path(filename.replace("\\", "/"))
		return next((candidate.resolve() for candidate in (asset, path.parent / asset, path.parent / asset.name) if candidate.is_file()), asset)

	return resolve(checkpoint), resolve(metadata.get("tuning"))


def record_trial(task, policy, cell, trial, controller):
	"""Record the benchmark's 120 Hz samples as owned collider and COM arrays."""
	positions, orientations, centers, speeds = [], [], [], []

	def capture(task):
		state = task.state
		positions.append(state.position[0].copy())
		orientations.append(state.orientation[0].copy())
		centers.append(state.center_of_mass[0].copy())
		speeds.append(np.hypot(*state.center_of_mass_velocity[0, [0, 2]]))

	try:
		if cell["bone"] is None:
			result = run_quiet_trial(task, policy, seed=trial["seed"], on_sample=capture)
		else:
			result = run_trial(task, policy, seed=trial["seed"], bone=cell["bone"], impulse=cell["impulse_ns"],
				scripted_step=controller == "scripted_step", on_sample=capture)
	finally:
		task.close()
	end = round(result["elapsed_seconds"] * TICK_HZ)
	if len(positions) != end + 1:
		raise RuntimeError("replay samples do not match the trial duration")
	bones = task.settings.bones
	positions, orientations = np.asarray(positions), np.asarray(orientations)
	poses = collider_poses(positions, orientations, [bone.shape_offset for bone in bones], [bone.shape_rotation for bone in bones])
	lower, upper = collider_bounds(bones, positions, orientations)
	bounds = lower.min(axis=(0, 1)), upper.max(axis=(0, 1))
	push = None
	if cell["bone"] is not None and not result["outcome"].startswith("pre_push"):
		start = poses[0][PUSH_TICK, task.settings.bone_index(cell["bone"])]
		push = start, np.asarray(cell["impulse_ns"]) / float(cell["magnitude_ns"])
	return result, poses, np.asarray(centers), np.asarray(speeds), bounds, push


def outcome_text(outcome):
	return {"pre_push_fall": "Fell before push", "post_push_fall": "Fell after push", "pre_push_unsettled": "Unsettled before push",
		"fall": "Fell", "timeout": "Timed out"}.get(outcome, outcome.replace("_", " ").capitalize())


def case_key(cell):
	return cell["bone"], cell.get("direction"), cell.get("magnitude_ns")


def case_label(cell):
	bone, direction, magnitude = case_key(cell)
	return "Quiet standing" if bone is None else f"{bone.title()} / {direction} / {magnitude:g} N s"


def filtered_trials(trials, result_filter):
	if result_filter == "With steps":
		return [(i, trial) for i, trial in enumerate(trials) if trial.get("step_count", 0)]
	if result_filter == "All results":
		return list(enumerate(trials))
	passed = result_filter == "Passed"
	return [(i, trial) for i, trial in enumerate(trials) if (trial["outcome"] in PASS_OUTCOMES) == passed]


def paired_selection(reports, current, report_index, *, case_index=None, name=None, preferred=None, result_filter="All results"):
	"""Find the same test and seed in the new report or controller when present."""
	old_report, old_case, old_name, old_start = current
	old_cell = reports[old_report]["cells"][old_case]
	seed = old_cell["controllers"][old_name]["episodes"][old_start]["seed"]
	cells = reports[report_index]["cells"]
	if case_index is None:
		case_index = next((i for i, cell in enumerate(cells) if case_key(cell) == case_key(old_cell)), 0)
	controllers = cells[case_index]["controllers"]
	name = next((candidate for candidate in (name, old_name, preferred) if candidate in controllers), next(iter(controllers)))
	trials = controllers[name]["episodes"]
	matches = filtered_trials(trials, result_filter) or list(enumerate(trials))
	start = next((i for i, trial in matches if trial["seed"] == seed), matches[0][0])
	return report_index, case_index, name, start


def trial_label(trial, *, show_seed=True):
	prefix = f"{trial['seed']}   " if show_seed else ""
	text = prefix + outcome_text(trial["outcome"])
	if (recovery := trial.get("recovery_seconds")) is not None:
		text += f"   {recovery:.2f} s"
	if steps := trial.get("step_count", 0):
		text += f"   {steps} {'step' if steps == 1 else 'steps'}"
	return text


def view_reports(paths, reports, *, controller=None, labels=None, headless=False, offscreen=False, frames=None):
	controller = controller or next(iter(reports[0]["cells"][0]["controllers"]))
	first_report = next((index for index, report in enumerate(reports) if controller in report["cells"][0]["controllers"]), None)
	if first_report is None:
		raise ValueError(f"the reports have no {controller} controller")
	selection = (first_report, 0, controller, 0)
	labels = labels if labels is not None else [path.stem for path in paths]

	@cache
	def assets(report_index, name):
		return report_assets(paths[report_index], reports[report_index], name)

	@cache
	def load_controller(name, checkpoint, tuning, seed, spin, tilt, seconds):
		task, policy, _ = make_trial_controller(name, seed=seed, checkpoint=checkpoint, tuning=tuning,
			max_start_spin=spin, max_start_tilt=tilt, episode_seconds=seconds)
		return task, policy

	def trial_controller(report_index, name):
		report = reports[report_index]
		return load_controller(name, *assets(report_index, name), report["first_seed"], report["max_start_spin_rad_s"],
			report["max_start_tilt_rad"], report.get("episode_seconds", PUSH_SECONDS + RECOVERY_SECONDS))

	@lru_cache(maxsize=24)
	def recording(report_index, cell_index, name, start_index):
		cell = reports[report_index]["cells"][cell_index]
		trial = cell["controllers"][name]["episodes"][start_index]
		task, policy = trial_controller(report_index, name)
		return record_trial(task, policy, cell, trial, name)

	current = selection
	clip = recording(*current)
	if headless:
		return clip[0]

	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh, MaterialRenderer
	from graaal.Math import Camera

	playback = Playback()
	yaw, pitch, zoom = 42.0, 22.0, 1.0
	view_offset = (0.0, 0.0)
	error_message = ""
	with viewing_target("Ragdoll tests", window_size=(1280, 800), offscreen=offscreen) as output:
		shader = DefaultMaterialShader()
		cube = cube_vertices()
		box = MaterialMesh(cube, shader)
		floor = DrawInstance(box, material={"base_color": (0.22, 0.25, 0.29, 1)}, position=(0, -0.5, 0), scale=(8, 1, 8))
		grid = [DrawInstance(box, material={"base_color": (0.34, 0.38, 0.43, 1)}, position=point, scale=size)
			for offset in range(-3, 4) for point, size in (((offset, 0.003, 0), (0.012, 0.004, 8)), ((0, 0.003, offset), (8, 0.004, 0.012)))]
		com = DrawInstance(box, material={"base_color": (0.20, 0.95, 0.63, 1)}, scale=(0.07, 0.012, 0.07))
		arrow = [DrawInstance(box, material={"base_color": (1, 0.80, 0.12, 1)}) for _ in range(3)]
		static = [floor, *grid, com]
		initial_tuning = assets(current[0], current[2])[1]
		all_tunings = {initial_tuning} if offscreen else {assets(index, name)[1] for index, report in enumerate(reports)
			for name in report["cells"][0]["controllers"]}
		parts = {}
		for tuning in all_tunings:
			if tuning == initial_tuning:
				settings = trial_controller(current[0], current[2])[0].settings
			else:
				if tuning is not None and not tuning.is_file():
					continue
				settings = (default_project() if tuning is None else read_project(tuning)).settings
			parts[tuning] = ragdoll_collider_instances(settings.bones, cube, box, shader)
		meshes = [box, *(part.mesh for group in parts.values() for part in group)]
		renderer = MaterialRenderer(static, meshes=meshes)
		camera = Camera((3, 2, 3), (0, 0.8, 0), (0, 1, 0), 43, near=0.03, far=50)

		def switch_trial():
			nonlocal clip, current, selection, error_message
			if current == selection:
				return False
			compare = current[:2] == selection[:2]
			previous_seed = clip[0]["seed"]
			try:
				clip = recording(*selection)
			except (ImportError, OSError, RuntimeError, ValueError) as error:
				error_message = str(error)
				selection = current
				return False
			current = selection
			error_message = ""
			if compare and clip[0]["seed"] == previous_seed:
				playback.cursor = min(playback.cursor, sample_count() - 1)
				return False
			return True

		def update(tick):
			_, (centers, rotations), mass_centers, _, (lower, upper), push = clip
			group = parts[assets(current[0], current[2])[1]]
			for part, position, rotation in zip(group, centers[tick], rotations[tick], strict=True):
				part.position, part.rotation = position, rotation
			com.position = (mass_centers[tick, 0], 0.013, mass_centers[tick, 2])
			renderer.instances[:] = [*static, *group]
			if push is not None and PUSH_TICK <= tick < PUSH_TICK + TICK_HZ // 3:
				start, direction = push
				tip = start + direction * 0.85
				base = tip - direction * 0.20
				side = np.cross(direction, (0, 1, 0)) * 0.13
				for segment, a, b in zip(arrow, (start, base + side, base - side), (tip, tip, tip), strict=True):
					segment.position, segment.rotation, segment.scale = segment_transform(a, b, 0.035)
				renderer.instances.extend(arrow)
			azimuth, elevation = np.deg2rad((yaw, pitch))
			direction = np.array((np.sin(azimuth) * np.cos(elevation), np.sin(elevation), np.cos(azimuth) * np.cos(elevation)))
			camera.target[:] = (lower + upper) * 0.5
			# Center the rig in the part of the viewport visible between the panels.
			right = np.array((np.cos(azimuth), 0, -np.sin(azimuth)))
			distance = max(2.2, float((upper - lower).max()) * 1.9) / zoom
			shift = view_offset[0] * right + view_offset[1] * np.cross(direction, right)
			camera.target[:] -= shift * distance * np.tan(np.deg2rad(camera.fovy_deg * 0.5))
			camera.position[:] = camera.target + direction * distance

		def sample_count():
			return len(clip[2])

		if offscreen:
			last = sample_count() - 1
			midpoint = min(PUSH_TICK + 6, last) if clip[5] is not None else last // 2
			samples = tuple(sorted({0, midpoint, last}))
			play_frames(renderer, camera, output, update, label="Ragdoll replay", sample_count=sample_count, sample_hz=TICK_HZ,
				offscreen_samples=samples, check_frame=has_blue_pixels)
			return clip[0]

		from graaal.UI.ImGui import ImGui

		with ImGui() as gui:
			imgui = gui.imgui

			result_filter = 0
			scroll_to_selection = True
			flags = imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize

			def draw_browser(sidebar_width, sidebar_height):
				nonlocal selection, result_filter, scroll_to_selection
				report_index, cell_index, name, start_index = current
				report = reports[report_index]
				cells = report["cells"]
				cell = cells[cell_index]
				controllers = cell["controllers"]
				trials = controllers[name]["episodes"]
				filter_label = RESULT_FILTERS[result_filter]
				if len(paths) > 1:
					imgui.text("Scenario group")
					imgui.set_next_item_width(-1)
					changed, index = imgui.combo("##Scenario group", report_index, labels)
					if changed:
						selection = paired_selection(reports, current, index, preferred=controller)
				else:
					imgui.text_wrapped(labels[report_index])
				names = list(controllers)
				imgui.set_next_item_width(min(170, sidebar_width * 0.55))
				changed, index = imgui.combo("Controller", names.index(name), [CONTROLLER_LABELS.get(item, item) for item in names])
				if changed:
					selection = paired_selection(reports, current, report_index, case_index=cell_index, name=names[index])
				imgui.set_next_item_width(min(170, sidebar_width * 0.55))
				changed, result_filter = imgui.combo("Results", result_filter, RESULT_FILTERS)
				if changed:
					filter_label = RESULT_FILTERS[result_filter]
					scroll_to_selection = True

				visible_cases = []
				for i, item in enumerate(cells):
					block = item["controllers"].get(name)
					if block is not None and (matches := len(filtered_trials(block["episodes"], filter_label))):
						visible_cases.append((i, item, block, matches))
				visible_starts = filtered_trials(trials, filter_label)
				all_results = result_filter == 0
				imgui.separator_text(f"Cases ({len(visible_cases)}/{len(cells)})")
				case_height = max(64, min(sidebar_height * 0.30, (len(visible_cases) + 1) * imgui.get_text_line_height_with_spacing()))
				try:
					if imgui.begin_child("Cases", (0, case_height), imgui.ChildFlags_.borders):
						if not visible_cases:
							imgui.text_wrapped("No cases match. Try another controller or result filter.")
						for i, item, block, matches in visible_cases:
							passed = block["settled"] if item["bone"] is None else block["recovered"]
							count = f"{passed}/{block['trials']} passed"
							if filter_label == "With steps":
								count = f"{matches}/{block['trials']} with steps"
							elif filter_label == "Failed":
								count = f"{matches}/{block['trials']} failed"
							label = f"{case_label(item)}   {count}##case{i}"
							if imgui.selectable(label, i == cell_index)[0]:
								selection = paired_selection(reports, current, report_index, case_index=i, result_filter=filter_label)
							if scroll_to_selection and i == cell_index:
								imgui.set_scroll_here_y(0.5)
				finally:
					imgui.end_child()

				imgui.separator_text(f"Seeds ({len(visible_starts)}/{len(trials)})")
				start_ids = [i for i, _ in visible_starts]
				position = start_ids.index(start_index) if start_index in start_ids else -1
				for label, target in (("Previous", position - 1), ("Next", position + 1)):
					if label == "Next":
						imgui.same_line()
					imgui.begin_disabled(not 0 <= target < len(start_ids))
					try:
						if imgui.button(f"{label} seed") and 0 <= target < len(start_ids):
							selection = (report_index, cell_index, name, start_ids[target])
					finally:
						imgui.end_disabled()
				if not all_results and start_index not in start_ids:
					imgui.text_disabled("Current replay is outside the filter.")
				if error_message and selection == current:
					imgui.text_wrapped(error_message)
				try:
					if imgui.begin_child("Seeds", (0, 0), imgui.ChildFlags_.borders):
						if not visible_starts:
							imgui.text_wrapped("No seeds match in this case.")
						for index, trial in visible_starts:
							color = PASS_COLOR if trial["outcome"] in PASS_OUTCOMES else FAIL_COLOR
							imgui.push_style_color(imgui.Col_.text, color)
							try:
								if imgui.selectable(f"{trial_label(trial)}##start{index}", index == start_index)[0]:
									selection = (report_index, cell_index, name, index)
							finally:
								imgui.pop_style_color()
							if scroll_to_selection and index == start_index:
								imgui.set_scroll_here_y(0.5)
				finally:
					imgui.end_child()
				scroll_to_selection = selection != current

			def draw_transport(width):
				nonlocal yaw, pitch, zoom
				report_index, cell_index, name, start_index = current
				cell = reports[report_index]["cells"][cell_index]
				saved = cell["controllers"][name]["episodes"][start_index]
				result = clip[0]
				last = sample_count() - 1
				tick = min(int(playback.cursor), last)
				imgui.text_wrapped(f"{case_label(cell)}   |   {CONTROLLER_LABELS.get(name, name)}   |   Seed {result['seed']}")
				color = PASS_COLOR if result["outcome"] in PASS_OUTCOMES else FAIL_COLOR
				status = trial_label(result, show_seed=False)
				if cell["bone"] is None:
					status += f"   |   Drift {result['peak_drift_m'] * 100:.1f} cm"
				imgui.text_colored(color, status)
				if selection != current:
					imgui.same_line()
					imgui.text_disabled("Loading replay...")
				if any(saved.get(key) != result.get(key) for key in ("outcome", "elapsed_seconds", "recovery_seconds", "step_count")):
					imgui.text_wrapped(f"Saved result differs: {trial_label(saved)}")
				if imgui.button("Play" if playback.paused else "Pause"):
					playback.paused = not playback.paused
				imgui.same_line()
				if imgui.button("Restart"):
					playback.cursor = 0
				if clip[5] is not None:
					imgui.same_line()
					if imgui.button("Push"):
						playback.cursor, playback.paused = PUSH_TICK, True
				if result.get("landings"):
					imgui.same_line()
					if imgui.button("First landing"):
						touchdown = result["landings"][0]["seconds_after_push"]
						playback.cursor, playback.paused = PUSH_TICK + round(touchdown * TICK_HZ), True
				imgui.same_line()
				if imgui.button("View"):
					imgui.open_popup("View controls")
				if imgui.begin_popup("View controls"):
					try:
						imgui.set_next_item_width(134)
						_, playback.speed = imgui.slider_float("Speed", playback.speed, 0.10, 2.0, "%.2fx")
						imgui.set_next_item_width(134)
						_, yaw = imgui.slider_float("Orbit", yaw, -180.0, 180.0, "%.0f deg")
						imgui.set_next_item_width(134)
						_, pitch = imgui.slider_float("Elevation", pitch, 5.0, 80.0, "%.0f deg")
						imgui.set_next_item_width(134)
						_, zoom = imgui.slider_float("Zoom", zoom, 0.5, 2.5, "%.2fx")
					finally:
						imgui.end_popup()
				if width >= 520:
					imgui.same_line()
					imgui.text_disabled(f"COM {clip[3][tick]:.2f} m/s   |   {playback.speed:.2f}x")
				imgui.set_next_item_width(-1)
				changed, seconds = imgui.slider_float("##Timeline", tick / TICK_HZ, 0, last / TICK_HZ, f"%.2f / {last / TICK_HZ:.2f} s")
				if changed:
					playback.cursor, playback.paused = round(seconds * TICK_HZ), True

			def draw_controls():
				nonlocal view_offset
				width, height = imgui.get_io().display_size
				margin = 12.0
				sidebar_width = min(390, width * 0.43)
				transport_height = 138
				view_offset = (sidebar_width / max(height, 1), transport_height / max(height, 1))
				sidebar_height = max(1, height - 2 * margin - (transport_height + margin if width < 850 else 0))
				imgui.set_next_window_pos((margin, margin))
				imgui.set_next_window_size((sidebar_width, sidebar_height))
				expanded, _ = imgui.begin("Trial browser", flags=flags)
				try:
					if expanded:
						draw_browser(sidebar_width, sidebar_height)
				finally:
					imgui.end()

				left = margin if width < 850 else sidebar_width + 2 * margin
				transport_width = max(1, width - left - margin)
				imgui.set_next_window_pos((left, max(margin, height - transport_height - margin)))
				imgui.set_next_window_size((transport_width, transport_height))
				expanded, _ = imgui.begin("Playback", flags=flags | imgui.WindowFlags_.no_title_bar)
				try:
					if expanded:
						draw_transport(transport_width)
				finally:
					imgui.end()

			play_frames(renderer, camera, output, update, label="Ragdoll replay", sample_count=sample_count, sample_hz=TICK_HZ,
				before_frame=switch_trial, draw_overlay=lambda: gui.render(draw_controls), playback=playback, frame_limit=frames)
	return clip[0]


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("reports", type=Path, nargs="*", help="full benchmark JSON reports")
	parser.add_argument("--suite", type=Path, help="load standing and push reports sharing this filename prefix")
	parser.add_argument("--controller", choices=CONTROLLER_LABELS, help="controller selected when the viewer opens")
	parser.add_argument("--frames", type=int, help="close the window after this many rendered frames")
	add_viewing_options(parser)
	args = parser.parse_args()
	if args.frames is not None and args.frames < 1:
		parser.error("--frames must be positive")
	try:
		paths = args.reports
		labels = [path.stem for path in paths]
		if args.suite is not None:
			matches = suite_reports(args.suite)
			if not matches:
				raise ValueError(f"no reports found for suite {args.suite}")
			suite_paths = {path.resolve() for path, _ in matches}
			matches.extend((path, label) for path, label in zip(paths, labels, strict=True) if path.resolve() not in suite_paths)
			paths, labels = zip(*matches)
		if not paths:
			raise ValueError("pass report paths or --suite")
		reports = [read_report(path) for path in paths]
		result = view_reports(paths, reports, controller=args.controller, labels=labels, headless=args.headless, offscreen=args.offscreen, frames=args.frames)
	except (ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
		parser.error(str(error))
	if args.headless or args.offscreen:
		print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
	main()
