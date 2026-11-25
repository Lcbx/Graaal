"""Evaluate joint-only ragdoll standing or recovery from a paired grid of pushes."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from graaal.Math import Quaternion
from scenes.physics_geometry import collider_poses
from tools.ragdoll_project import default_project, read_project
from tools.ragdoll_recovery_step import ScriptedBackstep
from tools.ragdoll_standing import CONTROL_HZ, DEFAULT_START_TILT, DIRECT_REWARD_KEYS, HARD_BACKWARD_PUSHES, PUSH_BONES, PUSH_DIRECTIONS, PUSH_SECONDS, PUSH_TICK, RECOVERY_STEP_DISTANCE, SETTLE_SECONDS, TICK_HZ, StandingTask, _support_margin, evaluate_episode


RECOVERY_SECONDS = 5
QUIET_SECONDS = 10
HOLD_TICKS = SETTLE_SECONDS * TICK_HZ
STEP_DISTANCE = RECOVERY_STEP_DISTANCE
MAX_START_SPIN = 0.1
HARD_BACKWARD_CELLS = {(bone, magnitude, "-Z") for bone, magnitude in HARD_BACKWARD_PUSHES}
LANDING_WINDOW_TICKS = TICK_HZ // 5
DIRECT_DIAGNOSTICS = ("target_saturation_fraction", "max_foot_clearance_m", *DIRECT_REWARD_KEYS)


def run_trial(task, policy, *, seed, bone, impulse, scripted_step=False, on_sample=None):
	"""Return a trial outcome. on_sample(task) sees the initial pose and each physics tick."""
	observation, _ = task.reset(seed=seed)

	def capture():
		on_sample(task)

	if on_sample is not None:
		capture()
	result = dict(seed=seed, recovery_seconds=None, step_count=0, landings=[])
	while task.world.tick < PUSH_TICK:
		observation, _, fallen, completed, info = task.step(policy(observation), after_tick=capture if on_sample is not None else None)
		if fallen or completed:
			if task.direct:
				result["direct_diagnostics"] = {key: info[key] for key in DIRECT_DIAGNOSTICS}
			return result | dict(outcome="pre_push_fall" if fallen else "pre_push_unsettled", elapsed_seconds=task.world.tick / TICK_HZ)
	if task.settled_ticks < HOLD_TICKS:
		if task.direct:
			result["direct_diagnostics"] = {key: info[key] for key in DIRECT_DIAGNOSTICS}
		return result | dict(outcome="pre_push_unsettled", elapsed_seconds=PUSH_SECONDS)
	task.apply_push(task.settings.bone_index(bone), impulse)
	backstep = None
	step_options = {}
	if scripted_step and tuple(impulse[:2]) == (0, 0) and -impulse[2] >= {"chest": 8, "pelvis": 12}.get(bone, np.inf):
		backstep = ScriptedBackstep(task)
		step_options = dict(edit_motors=backstep.edit_motors, joint_feedback=backstep.feedback)
		result["scripted_step"] = True
	track_load = backstep is not None or task.recovery is not None or task.direct
	if track_load:
		import pyjoltphysics as pj

	tracked_landings = [None] * len(task.feet)
	planted = task.state.contacts[0, task.feet]
	last_contact = task.state.position[0, task.feet][:, [0, 2]]
	seen_contact = planted.copy()
	result.update(peak_com_speed_m_s=0.0, arrest_seconds=None)

	def after_tick():
		nonlocal planted
		if result["recovery_seconds"] is not None:
			return
		if on_sample is not None:
			capture()
		tick = task.world.tick
		com_speed = float(np.hypot(*task.state.center_of_mass_velocity[0, [0, 2]]))
		result["peak_com_speed_m_s"] = max(result["peak_com_speed_m_s"], com_speed)
		if com_speed < 0.05 and result["peak_com_speed_m_s"] >= 0.05 and result["arrest_seconds"] is None:
			result["arrest_seconds"] = (tick - PUSH_TICK) / TICK_HZ
		contact = task.state.contacts[0, task.feet]
		position = task.state.position[0, task.feet][:, [0, 2]]
		landed = contact & ~planted & seen_contact
		moved = np.linalg.norm(position - last_contact, axis=1) >= STEP_DISTANCE
		relocated = np.flatnonzero(landed & moved)
		margin = None
		if len(relocated) and task.support_area:
			feet = task.feet[contact]
			centers, turns = collider_poses(task.state.position[0, feet], task.state.orientation[0, feet], task.foot_offsets[contact], task.foot_rotations[contact])
			soles = centers[:, None] + task.sole_corners[contact] @ Quaternion.matrix_batch(turns).transpose(0, 2, 1)
			margin = _support_margin(soles.reshape(-1, 3)[:, [0, 2]], task.state.center_of_mass[0, [0, 2]])
		for foot in relocated:
			landing = dict(foot=task.settings.bones[task.feet[foot]].name, seconds_after_push=(tick - PUSH_TICK) / TICK_HZ,
				takeoff_xz_m=last_contact[foot].tolist(), landing_xz_m=position[foot].tolist(),
				com_speed_at_touchdown_m_s=com_speed, com_speed_after_0_2s_m_s=None, support_margin_at_touchdown_m=margin)
			result["landings"].append(landing)
			if track_load:
				landing.update(contact_ticks=0, peak_estimated_floor_impulse_ns=0.0)
				tracked_landings[foot] = landing
		if track_load:
			for foot, landing in enumerate(tracked_landings):
				if not contact[foot]:
					tracked_landings[foot] = None
				elif landing is not None:
					events = task.events
					uid = task.uids[0, task.feet[foot]]
					pair = ((events["a"] == uid) & np.isin(events["b"], task.floor)) | ((events["b"] == uid) & np.isin(events["a"], task.floor))
					active = pair & (events["kind"] == pj.EventKind.CONTACT) & np.isin(events["phase"], (pj.EventPhase.BEGIN, pj.EventPhase.PERSIST))
					impulse_estimate = np.max(events["impulse"][active], initial=0)
					landing["contact_ticks"] += 1
					landing["peak_estimated_floor_impulse_ns"] = max(landing["peak_estimated_floor_impulse_ns"], float(impulse_estimate))
		result["step_count"] = len(result["landings"])
		for landing in result["landings"]:
			if tick - PUSH_TICK - round(landing["seconds_after_push"] * TICK_HZ) == LANDING_WINDOW_TICKS:
				landing["com_speed_after_0_2s_m_s"] = com_speed
		last_contact[contact] = position[contact]
		seen_contact[:] |= contact
		planted = contact
		if task.settled_ticks >= HOLD_TICKS:
			result["recovery_seconds"] = (task.world.tick - PUSH_TICK) / TICK_HZ

	while True:
		action = policy(observation) if backstep is None else backstep.action()
		observation, _, fallen, completed, info = task.step(action, after_tick=after_tick, **step_options)
		recovery = result["recovery_seconds"]
		if task.direct and (recovery is not None or fallen or completed):
			result["direct_diagnostics"] = {key: info[key] for key in DIRECT_DIAGNOSTICS}
		if recovery is not None:
			return result | dict(outcome="recovered", elapsed_seconds=PUSH_SECONDS + recovery)
		if fallen or completed:
			return result | dict(outcome="post_push_fall" if fallen else "timeout", elapsed_seconds=task.world.tick / TICK_HZ)


def summarize(trials):
	recoveries = [trial["recovery_seconds"] for trial in trials if trial["outcome"] == "recovered"]
	return dict(trials=len(trials), recovered=len(recoveries), success_rate=len(recoveries) / len(trials),
		mean_recovery_seconds=float(np.mean(recoveries)) if recoveries else None,
		mean_step_count=float(np.mean([trial["step_count"] for trial in trials])),
		recovered_with_step=sum(trial["step_count"] > 0 and trial["outcome"] == "recovered" for trial in trials))


def run_quiet_trial(task, policy, *, seed, on_sample=None):
	metrics = evaluate_episode(task, policy, seed=seed, on_sample=on_sample)
	outcome = "settled" if metrics["settled_success"] else "unsettled" if metrics["success"] else "fall"
	return dict(seed=seed, outcome=outcome, **metrics)


def make_trial_controller(name, *, seed, checkpoint=None, tuning=None, max_start_spin=MAX_START_SPIN, max_start_tilt=DEFAULT_START_TILT,
		episode_seconds=PUSH_SECONDS + RECOVERY_SECONDS):
	"""Build the same controller for measured and visualized trials."""
	project = read_project(tuning) if tuning is not None else default_project()
	options = dict(episode_steps=episode_seconds * CONTROL_HZ, tuned_stance=True, vary_start=True,
		initial_angular_speed=max_start_spin, max_start_tilt=max_start_tilt, support_area=True)
	if name == "ankle_balance":
		task = StandingTask(project, ankle_balance=True, **options)
		fixed = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), len(project.settings) - 1)
		return task, lambda _: fixed, None
	if name == "scripted_step":
		task = StandingTask(project, whole_body=True, **options)
		zero = np.zeros(task.action_shape, dtype=np.float32)
		return task, lambda _: zero, None
	mode = name.removeprefix("ppo_")
	if mode not in ("residual", "stepping", "whole_body", "recovery", "direct") or checkpoint is None:
		raise ValueError(f"missing controller or checkpoint for {name}")

	import torch
	from stable_baselines3 import PPO

	torch.set_num_threads(1)
	model = PPO.load(checkpoint, device="cpu")
	task = StandingTask(project, **{mode: True}, **options)
	try:
		observation, _ = task.reset(seed=seed)
	finally:
		task.close()
	if model.action_space.shape != task.action_shape or model.observation_space.shape != observation.shape:
		raise ValueError(f"{checkpoint} must match the {name} action space and support-area observations")
	metadata = dict(path=str(checkpoint), training_steps=model.num_timesteps)
	if mode == "direct":
		metadata.update(tuning=str(tuning) if tuning is not None else None,
			motor_frequency_hz=task.settings.motor_frequency, motor_damping=task.settings.motor_damping)
	return task, lambda obs: model.predict(obs, deterministic=True)[0], metadata


def benchmark(*, seed, episodes, impulses, checkpoint=None, stepping_checkpoint=None, whole_body_checkpoint=None, recovery_checkpoint=None, scripted_step=False, direct_checkpoint=None, hard_backward_only=False, direct_tuning=None, quiet=False):
	if quiet and (hard_backward_only or scripted_step):
		raise ValueError("--quiet cannot be combined with --hard-backward-only or --scripted-step")
	episode_seconds = QUIET_SECONDS if quiet else PUSH_SECONDS + RECOVERY_SECONDS
	setup = dict(seed=seed, episode_seconds=episode_seconds)
	controllers = {}
	checkpoints = {}
	learned_paths = {"residual": checkpoint, "stepping": stepping_checkpoint, "whole_body": whole_body_checkpoint,
		"recovery": recovery_checkpoint, "direct": direct_checkpoint}
	try:
		baseline, policy, _ = make_trial_controller("ankle_balance", **setup)
		controllers["ankle_balance"] = baseline, policy
		if scripted_step:
			scripted, policy, _ = make_trial_controller("scripted_step", **setup)
			controllers["scripted_step"] = scripted, policy
		for mode, path in learned_paths.items():
			if path is None:
				continue
			name = f"ppo_{mode}"
			learned, policy, metadata = make_trial_controller(name, checkpoint=path, tuning=direct_tuning if mode == "direct" else None, **setup)
			controllers[name] = learned, policy
			checkpoints[name] = metadata
		metadata = dict(rig="built-in humanoid", stance="tuned", tick_hz=TICK_HZ, control_hz=CONTROL_HZ, first_seed=seed,
			episodes_per_cell=episodes, max_start_tilt_rad=DEFAULT_START_TILT, max_start_spin_rad_s=MAX_START_SPIN, checkpoints=checkpoints)
		if quiet:
			results = {}
			summary = {}
			for name, (task, policy) in controllers.items():
				trials = [run_quiet_trial(task, policy, seed=start) for start in range(seed, seed + episodes)]
				settled = sum(trial["settled_success"] for trial in trials)
				summary[name] = dict(trials=episodes, settled=settled, success_rate=settled / episodes)
				results[name] = summary[name] | dict(episodes=trials)
			return metadata | dict(scenario="quiet_standing", episode_seconds=episode_seconds, hold_seconds=SETTLE_SECONDS,
				summary=summary, cells=[dict(bone=None, controllers=results)])
		cells = []
		grid = sorted(HARD_BACKWARD_CELLS) if hard_backward_only else product(PUSH_BONES, impulses, PUSH_DIRECTIONS)
		for bone, magnitude, direction in grid:
			vector = (magnitude * np.asarray(PUSH_DIRECTIONS[direction])).tolist()
			results = {}
			for name, (task, policy) in controllers.items():
				trials = [run_trial(task, policy, seed=start, bone=bone, impulse=vector, scripted_step=name == "scripted_step")
					for start in range(seed, seed + episodes)]
				results[name] = summarize(trials) | dict(episodes=trials)
			cells.append(dict(bone=bone, magnitude_ns=magnitude, direction=direction, impulse_ns=vector, controllers=results))
		hard_cells = [cell for cell in cells if (cell["bone"], cell["magnitude_ns"], cell["direction"]) in HARD_BACKWARD_CELLS]
		return metadata | dict(
			push_seconds=PUSH_SECONDS, recovery_window_seconds=RECOVERY_SECONDS, hold_seconds=SETTLE_SECONDS, step_distance_m=STEP_DISTANCE,
			ppo_checkpoint=checkpoints.get("ppo_residual", {}).get("path"), ppo_training_steps=checkpoints.get("ppo_residual", {}).get("training_steps"),
			summary={name: summarize([trial for cell in cells for trial in cell["controllers"][name]["episodes"]]) for name in controllers},
			hard_backward_summary={name: summarize([trial for cell in hard_cells for trial in cell["controllers"][name]["episodes"]])
				for name in controllers} if hard_cells else {}, cells=cells)
	finally:
		for task, _ in controllers.values():
			task.close()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--checkpoint", type=Path, help="include a residual PPO checkpoint trained with --support-area")
	parser.add_argument("--stepping-checkpoint", type=Path, help="include a hip and knee PPO checkpoint trained with --support-area")
	parser.add_argument("--whole-body-checkpoint", type=Path, help="include a whole-body PPO checkpoint trained with --support-area")
	parser.add_argument("--recovery-checkpoint", type=Path, help="include a PPO checkpoint trained with --action-space recovery")
	parser.add_argument("--direct-checkpoint", type=Path, help="include a PPO checkpoint trained with --action-space direct")
	parser.add_argument("--direct-tuning", type=Path, help="ragdoll project used to train the direct checkpoint")
	parser.add_argument("--quiet", action="store_true", help="evaluate ten-second quiet standing and save replayable trials")
	parser.add_argument("--hard-backward-only", action="store_true", help="run only the three historically failing backward push cells")
	parser.add_argument("--scripted-step", action="store_true", help="include a timed backward recovery step on strong -Z pushes")
	parser.add_argument("--episodes", type=int, default=10, help="number of paired starts per grid cell")
	parser.add_argument("--seed", type=int, default=55000, help="first evaluation seed")
	parser.add_argument("--impulses", type=float, nargs="+", default=(0.5, 1.0, 2.0), help="impulse magnitudes in N s")
	parser.add_argument("--report", type=Path, help="write the complete report to a JSON file")
	args = parser.parse_args()
	if args.episodes < 1 or args.seed < 0 or args.seed + args.episodes > 2**32:
		parser.error("episodes must be positive and evaluation seeds must fit uint32")
	if not all(np.isfinite(value) and 0 < value <= np.finfo(np.float32).max for value in args.impulses):
		parser.error("impulse magnitudes must be positive and finite in float32")
	if args.direct_tuning is not None and args.direct_checkpoint is None:
		parser.error("--direct-tuning requires --direct-checkpoint")
	try:
		report = benchmark(seed=args.seed, episodes=args.episodes, impulses=args.impulses, checkpoint=args.checkpoint,
			stepping_checkpoint=args.stepping_checkpoint, whole_body_checkpoint=args.whole_body_checkpoint, recovery_checkpoint=args.recovery_checkpoint,
			direct_checkpoint=args.direct_checkpoint, scripted_step=args.scripted_step, hard_backward_only=args.hard_backward_only,
			direct_tuning=args.direct_tuning, quiet=args.quiet)
		if args.report is not None:
			args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
	except (ImportError, OSError, ValueError) as error:
		parser.error(str(error))
	print(json.dumps(report if args.report is None else report["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
	main()
