"""Evaluate joint-only ragdoll recovery from a paired grid of horizontal pushes."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from tools.ragdoll_project import default_project
from tools.ragdoll_recovery_step import ScriptedBackstep
from tools.ragdoll_standing import CONTROL_HZ, DEFAULT_START_TILT, PUSH_BONES, PUSH_DIRECTIONS, PUSH_SECONDS, PUSH_TICK, SETTLE_SECONDS, TICK_HZ, StandingTask


RECOVERY_SECONDS = 5
HOLD_TICKS = SETTLE_SECONDS * TICK_HZ
STEP_DISTANCE = 0.02
MAX_START_SPIN = 0.1


def run_trial(task, policy, *, seed, bone, impulse, scripted_step=False):
	"""Return a trial outcome, counting foot relocations after the push."""
	observation, _ = task.reset(seed=seed)
	result = dict(seed=seed, recovery_seconds=None, step_count=0, landings=[])
	while task.world.tick < PUSH_TICK:
		observation, _, fallen, completed, _ = task.step(policy(observation))
		if fallen or completed:
			return result | dict(outcome="pre_push_fall" if fallen else "pre_push_unsettled", elapsed_seconds=task.world.tick / TICK_HZ)
	if task.settled_ticks < HOLD_TICKS:
		return result | dict(outcome="pre_push_unsettled", elapsed_seconds=PUSH_SECONDS)
	task.world.add_impulses(task.uids[:, task.settings.bone_index(bone)], [impulse])
	task.settled_ticks = 0
	backstep = None
	step_options = {}
	if scripted_step and tuple(impulse[:2]) == (0, 0) and -impulse[2] >= {"chest": 8, "pelvis": 12}.get(bone, np.inf):
		import pyjoltphysics as pj

		backstep = ScriptedBackstep(task)
		step_options = dict(edit_motors=backstep.edit_motors, joint_feedback=backstep.feedback)
		result["scripted_step"] = True
		tracked_landings = [None] * len(task.feet)
	planted = task.state.contacts[0, task.feet]
	last_contact = task.state.position[0, task.feet][:, [0, 2]]
	seen_contact = planted.copy()

	def after_tick():
		nonlocal planted
		if result["recovery_seconds"] is not None:
			return
		contact = task.state.contacts[0, task.feet]
		position = task.state.position[0, task.feet][:, [0, 2]]
		landed = contact & ~planted & seen_contact
		moved = np.linalg.norm(position - last_contact, axis=1) >= STEP_DISTANCE
		for foot in np.flatnonzero(landed & moved):
			landing = dict(foot=task.settings.bones[task.feet[foot]].name, seconds_after_push=(task.world.tick - PUSH_TICK) / TICK_HZ,
				takeoff_xz_m=last_contact[foot].tolist(), landing_xz_m=position[foot].tolist())
			result["landings"].append(landing)
			if backstep is not None:
				landing.update(contact_ticks=0, peak_estimated_floor_impulse_ns=0.0)
				tracked_landings[foot] = landing
		if backstep is not None:
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
		last_contact[contact] = position[contact]
		seen_contact[:] |= contact
		planted = contact
		if task.settled_ticks >= HOLD_TICKS:
			result["recovery_seconds"] = (task.world.tick - PUSH_TICK) / TICK_HZ

	while True:
		action = policy(observation) if backstep is None else backstep.action()
		observation, _, fallen, completed, _ = task.step(action, after_tick=after_tick, **step_options)
		recovery = result["recovery_seconds"]
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


def benchmark(*, seed, episodes, impulses, checkpoint=None, stepping_checkpoint=None, whole_body_checkpoint=None, scripted_step=False):
	project = default_project()
	options = dict(episode_steps=(PUSH_SECONDS + RECOVERY_SECONDS) * CONTROL_HZ, tuned_stance=True, vary_start=True,
		initial_angular_speed=MAX_START_SPIN, max_start_tilt=DEFAULT_START_TILT, support_area=True)
	baseline = StandingTask(project, ankle_balance=True, **options)
	fixed = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), len(project.settings) - 1)
	controllers = {"ankle_balance": (baseline, lambda _: fixed)}
	if scripted_step:
		scripted = StandingTask(project, whole_body=True, **options)
		zero = np.zeros(scripted.action_shape, dtype=np.float32)
		controllers["scripted_step"] = (scripted, lambda _: zero)
	checkpoints = {}
	learned_paths = {"residual": checkpoint, "stepping": stepping_checkpoint, "whole_body": whole_body_checkpoint}
	try:
		if any(learned_paths.values()):
			import torch
			from stable_baselines3 import PPO

			torch.set_num_threads(1)
			for mode, path in learned_paths.items():
				if path is None:
					continue
				name = f"ppo_{mode}"
				model = PPO.load(path, device="cpu")
				learned = StandingTask(project, **{mode: True}, **options)
				controllers[name] = (learned, lambda obs, policy=model: policy.predict(obs, deterministic=True)[0])
				observation, _ = learned.reset(seed=seed)
				learned.close()
				if model.action_space.shape != learned.action_shape or model.observation_space.shape != observation.shape:
					raise ValueError(f"{path} must match the {name} action space and support-area observations")
				checkpoints[name] = dict(path=str(path), training_steps=model.num_timesteps)
		cells = []
		for bone, magnitude, (direction, axis) in product(PUSH_BONES, impulses, PUSH_DIRECTIONS.items()):
			vector = (magnitude * np.asarray(axis)).tolist()
			results = {}
			for name, (task, policy) in controllers.items():
				trials = [run_trial(task, policy, seed=start, bone=bone, impulse=vector, scripted_step=name == "scripted_step")
					for start in range(seed, seed + episodes)]
				results[name] = summarize(trials) | dict(episodes=trials)
			cells.append(dict(bone=bone, magnitude_ns=magnitude, direction=direction, impulse_ns=vector, controllers=results))
		return dict(rig="built-in humanoid", stance="tuned", tick_hz=TICK_HZ, control_hz=CONTROL_HZ, first_seed=seed,
			episodes_per_cell=episodes, max_start_tilt_rad=DEFAULT_START_TILT, max_start_spin_rad_s=MAX_START_SPIN,
			push_seconds=PUSH_SECONDS, recovery_window_seconds=RECOVERY_SECONDS, hold_seconds=SETTLE_SECONDS, step_distance_m=STEP_DISTANCE,
			ppo_checkpoint=checkpoints.get("ppo_residual", {}).get("path"), ppo_training_steps=checkpoints.get("ppo_residual", {}).get("training_steps"), checkpoints=checkpoints,
			summary={name: summarize([trial for cell in cells for trial in cell["controllers"][name]["episodes"]]) for name in controllers}, cells=cells)
	finally:
		for task, _ in controllers.values():
			task.close()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--checkpoint", type=Path, help="include a residual PPO checkpoint trained with --support-area")
	parser.add_argument("--stepping-checkpoint", type=Path, help="include a hip and knee PPO checkpoint trained with --support-area")
	parser.add_argument("--whole-body-checkpoint", type=Path, help="include a whole-body PPO checkpoint trained with --support-area")
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
	try:
		report = benchmark(seed=args.seed, episodes=args.episodes, impulses=args.impulses, checkpoint=args.checkpoint,
			stepping_checkpoint=args.stepping_checkpoint, whole_body_checkpoint=args.whole_body_checkpoint, scripted_step=args.scripted_step)
		if args.report is not None:
			args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
	except (ImportError, OSError, ValueError) as error:
		parser.error(str(error))
	print(json.dumps(report if args.report is None else report["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
	main()
