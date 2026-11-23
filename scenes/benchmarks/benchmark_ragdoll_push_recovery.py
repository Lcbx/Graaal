"""Evaluate joint-only ragdoll recovery from a paired grid of horizontal pushes."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from tools.ragdoll_project import default_project
from tools.ragdoll_standing import CONTROL_HZ, DEFAULT_START_TILT, SETTLE_SECONDS, TICK_HZ, StandingTask


PUSH_SECONDS = 2
RECOVERY_SECONDS = 5
PUSH_TICK = PUSH_SECONDS * TICK_HZ
HOLD_TICKS = SETTLE_SECONDS * TICK_HZ
STEP_DISTANCE = 0.02
MAX_START_SPIN = 0.1
PUSH_BONES = ("pelvis", "chest")
DIRECTIONS = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def run_trial(task, policy, *, seed, bone, impulse):
	"""Return a trial outcome, counting foot relocations after the push."""
	observation, _ = task.reset(seed=seed)
	result = dict(seed=seed, recovery_seconds=None, step_count=0)
	while task.world.tick < PUSH_TICK:
		observation, _, fallen, completed, _ = task.step(policy(observation))
		if fallen or completed:
			return result | dict(outcome="pre_push_fall" if fallen else "pre_push_unsettled", elapsed_seconds=task.world.tick / TICK_HZ)
	if task.settled_ticks < HOLD_TICKS:
		return result | dict(outcome="pre_push_unsettled", elapsed_seconds=PUSH_SECONDS)
	task.world.add_impulses(task.uids[:, task.settings.bone_index(bone)], [impulse])
	task.settled_ticks = 0
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
		result["step_count"] += int(np.count_nonzero(landed & moved))
		last_contact[contact] = position[contact]
		seen_contact[:] |= contact
		planted = contact
		if task.settled_ticks >= HOLD_TICKS:
			result["recovery_seconds"] = (task.world.tick - PUSH_TICK) / TICK_HZ

	while True:
		observation, _, fallen, completed, _ = task.step(policy(observation), after_tick=after_tick)
		recovery = result["recovery_seconds"]
		if recovery is not None:
			return result | dict(outcome="recovered", elapsed_seconds=PUSH_SECONDS + recovery)
		if fallen or completed:
			return result | dict(outcome="post_push_fall" if fallen else "timeout", elapsed_seconds=task.world.tick / TICK_HZ)


def summarize(trials):
	recoveries = [trial["recovery_seconds"] for trial in trials if trial["outcome"] == "recovered"]
	return dict(trials=len(trials), recovered=len(recoveries), success_rate=len(recoveries) / len(trials),
		mean_recovery_seconds=float(np.mean(recoveries)) if recoveries else None,
		mean_step_count=float(np.mean([trial["step_count"] for trial in trials])))


def benchmark(*, seed, episodes, impulses, checkpoint=None):
	project = default_project()
	options = dict(episode_steps=(PUSH_SECONDS + RECOVERY_SECONDS) * CONTROL_HZ, tuned_stance=True, vary_start=True,
		initial_angular_speed=MAX_START_SPIN, max_start_tilt=DEFAULT_START_TILT, support_area=True)
	baseline = StandingTask(project, ankle_balance=True, **options)
	fixed = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), len(project.settings) - 1)
	controllers = {"ankle_balance": (baseline, lambda _: fixed)}
	learned = None
	try:
		if checkpoint is not None:
			import torch
			from stable_baselines3 import PPO

			torch.set_num_threads(1)
			learned = StandingTask(project, residual=True, **options)
			model = PPO.load(checkpoint, device="cpu")
			observation, _ = learned.reset(seed=seed)
			if model.action_space.shape != learned.action_shape or model.observation_space.shape != observation.shape:
				raise ValueError("checkpoint must use the residual action space and support-area observations")
			controllers["ppo_residual"] = (learned, lambda obs: model.predict(obs, deterministic=True)[0])
		cells = []
		for bone, magnitude, (direction, axis) in product(PUSH_BONES, impulses, DIRECTIONS.items()):
			vector = (magnitude * np.asarray(axis)).tolist()
			results = {}
			for name, (task, policy) in controllers.items():
				trials = [run_trial(task, policy, seed=start, bone=bone, impulse=vector) for start in range(seed, seed + episodes)]
				results[name] = summarize(trials) | dict(episodes=trials)
			cells.append(dict(bone=bone, magnitude_ns=magnitude, direction=direction, impulse_ns=vector, controllers=results))
		return dict(rig="built-in humanoid", stance="tuned", tick_hz=TICK_HZ, control_hz=CONTROL_HZ, first_seed=seed,
			episodes_per_cell=episodes, max_start_tilt_rad=DEFAULT_START_TILT, max_start_spin_rad_s=MAX_START_SPIN,
			push_seconds=PUSH_SECONDS, recovery_window_seconds=RECOVERY_SECONDS, hold_seconds=SETTLE_SECONDS, step_distance_m=STEP_DISTANCE,
			ppo_checkpoint=str(checkpoint) if checkpoint is not None else None, ppo_training_steps=model.num_timesteps if checkpoint is not None else None,
			summary={name: summarize([trial for cell in cells for trial in cell["controllers"][name]["episodes"]]) for name in controllers}, cells=cells)
	finally:
		baseline.close()
		if learned is not None:
			learned.close()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--checkpoint", type=Path, help="include a residual PPO checkpoint trained with --support-area")
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
		report = benchmark(seed=args.seed, episodes=args.episodes, impulses=args.impulses, checkpoint=args.checkpoint)
		if args.report is not None:
			args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
	except (ImportError, OSError, ValueError) as error:
		parser.error(str(error))
	print(json.dumps(report if args.report is None else report["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
	main()
