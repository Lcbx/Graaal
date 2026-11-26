"""Search bounded, oracle-timed joint targets for Genesis recovery diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.ragdoll_genesis_steps import StepTracker, motion_gated_actions, rescale_base_actions
from tools.ragdoll_genesis_train import StandingEnv
from tools.ragdoll_project import default_project


def candidates(count, seed, around=None):
	"""Sample normalized swing/stance targets, timing and hold lengths."""
	rng = np.random.default_rng(seed)
	if around is None:
		return np.column_stack((rng.uniform(0.3, 1, count), rng.uniform(-1, 0, count), rng.uniform(-1, 1, (count, 4)), rng.integers(8, 23, count), rng.integers(0, 10, count), rng.integers(5, 31, count)))
	centers = np.asarray(around, dtype=float)
	if centers.ndim != 2 or centers.shape[1] != 9 or not len(centers) or not np.isfinite(centers).all():
		raise ValueError("reference candidates must have nine finite parameters each")
	values = centers[np.arange(count) % len(centers)].copy()
	values[:, :6] += rng.normal(0, 0.18, (count, 6))
	values[:, 0] = values[:, 0].clip(0, 1)
	values[:, 1:6] = values[:, 1:6].clip(-1, 1)
	values[:, 6:] += rng.integers(-5, 6, (count, 3))
	values[:, 6] = values[:, 6].clip(4, 40)
	values[:, 7] = values[:, 7].clip(0, 30)
	values[:, 8] = values[:, 8].clip(0, 60)
	return values


def run(run_path, checkpoint, *, num_envs, seconds, impulse, seed, around=None, fixed=None, max_start_tilt=0, max_start_spin=0, swing_enabled=True, dataset_path=None, ground_contacts=False, student_run=None, student_checkpoint=None, trigger_speed=0.12, hold_seconds=0.8):
	import genesis as gs
	import torch
	from rsl_rl.runners import OnPolicyRunner
	from tensordict import TensorDict

	config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
	if config.get("foot_features", False) or not config.get("balance_features", False):
		raise ValueError("the diagnostic requires the 133-observation balance policy")
	gs.init(backend=gs.cuda, precision="32", logging_level="warning", seed=seed)
	env = StandingEnv(run_path / "rig.xml", config["actions"], default_project().settings, num_envs=num_envs, episode_seconds=seconds, push_impulse=impulse, balance_features=True, motor_kp=config.get("motor_kp", 100), motor_kd=config.get("motor_kd", 5), single_push=True, push_direction=3, limb_scale=0.5, measure_settled=True, max_start_tilt=max_start_tilt, max_start_spin=max_start_spin, foot_features=dataset_path is not None, ground_contacts=ground_contacts)
	if dataset_path is not None:
		original_observations = env.get_observations
		def teacher_observations():
			features = original_observations()["policy"]
			return TensorDict({"policy": torch.cat((features[:, :-18], features[:, -4:]), dim=1)}, batch_size=[num_envs])
		env.get_observations = teacher_observations
	runner = OnPolicyRunner(env, config["ppo"], log_dir=None, device=gs.device)
	runner.load(str(checkpoint), map_location=str(gs.device))
	policy = runner.get_inference_policy(device=gs.device)
	student_policy = None
	if student_run is not None:
		student_config = json.loads((student_run / "config.json").read_text(encoding="utf-8"))
		if not student_config.get("foot_features", False) or student_config["actions"] != config["actions"] or student_config.get("motor_kp", 100) != config.get("motor_kp", 100) or student_config.get("motor_kd", 5) != config.get("motor_kd", 5) or student_config.get("limb_scale", 0.22) != 0.5 or (student_run / "rig.xml").read_text(encoding="utf-8") != (run_path / "rig.xml").read_text(encoding="utf-8"):
			raise ValueError("student policy must use the same rig, motors and 0.5-rad limb scale")
		env.get_observations = original_observations
		student_runner = OnPolicyRunner(env, student_config["ppo"], log_dir=None, device=gs.device)
		student_runner.load(str(student_checkpoint), map_location=str(gs.device))
		student_policy = student_runner.get_inference_policy(device=gs.device)
		env.get_observations = teacher_observations
	values = np.tile(np.asarray(fixed, dtype=float), (num_envs, 1)) if fixed is not None else candidates(num_envs, seed, around)
	if values.shape != (num_envs, 9) or not np.isfinite(values).all():
		raise ValueError("each fixed candidate must contain nine finite values")
	p = torch.tensor(values, device=gs.device, dtype=torch.float32)
	obs = env.reset()
	active = torch.ones(num_envs, dtype=torch.bool, device=gs.device)
	settle_ticks = torch.zeros(num_envs, dtype=torch.int32, device=gs.device)
	success = torch.zeros_like(active)
	survival = torch.zeros(num_envs, device=gs.device)
	tracker = StepTracker(num_envs, gs.device)
	observations_history, actions_history, swing_history, active_history = [], [], [], []
	gate_ticks = torch.zeros(num_envs, dtype=torch.int32, device=gs.device)
	with torch.inference_mode():
		for tick in range(env.max_episode_length):
			if dataset_path is not None:
				full_observations = original_observations()["policy"]
				target_obs = TensorDict({"policy": torch.cat((full_observations[:, :-18], full_observations[:, -4:]), dim=1)}, batch_size=[num_envs])
			else:
				target_obs = obs
			base_action = policy(target_obs).clamp(-1, 1)
			action = rescale_base_actions(base_action, env.non_ankle, base_scale=config.get("limb_scale", 0.22), step_scale=env.limb_scale)
			elapsed = tick - 120 - p[:, 7]
			phase = (elapsed / p[:, 6]).clamp(0, 1)
			return_phase = ((elapsed - p[:, 6] - p[:, 8]) / 24).clamp(0, 1)
			swing = (elapsed >= 0) & (return_phase < 1)
			hip = p[:, 0] * torch.minimum(phase * 2, torch.ones_like(phase)) * (1 - return_phase)
			knee = p[:, 1] * torch.sin(torch.pi * phase)
			if swing_enabled:
				action[:, 24] = torch.where(swing, hip, action[:, 24])
				action[:, 26] = torch.where(swing, knee, action[:, 26])
			for index, column in ((17, 2), (19, 3), (1, 4), (18, 5)):
				action[:, index] = torch.where(swing, p[:, column] * (1 - return_phase), action[:, index])
			if dataset_path is not None:
				observations_history.append(full_observations.clone())
				actions_history.append(action.clone())
				swing_history.append(swing.clone())
				active_history.append(active.clone())
			if student_policy is not None:
				student_obs = TensorDict({"policy": full_observations}, batch_size=[num_envs])
				action, gate_ticks = motion_gated_actions(full_observations, base_action, student_policy(student_obs), gate_ticks, env.non_ankle, trigger_speed=trigger_speed, hold_ticks=round(hold_seconds / env.dt), base_scale=config.get("limb_scale", 0.22), step_scale=0.5)
			obs, _, done, info = env.step(action)
			tracker.update(info["foot_positions"], info["foot_contact"], info["settle_diagnostics"]["com_speed"], active, info["push"], info["push_direction"])
			survival += active
			settle_ticks = torch.where(info["settled"], settle_ticks + 1, 0)
			success |= active & info["time_outs"] & (settle_ticks >= 60)
			active &= ~done
			gate_ticks.masked_fill_(done, 0)
	landed, braked = tracker.landed.any(dim=1), tracker.braked.any(dim=1)
	if dataset_path is not None:
		selected = success & braked
		if student_policy is None and not bool(selected.any()):
			raise RuntimeError("no braking-step recoveries were available for the dataset")
		if student_policy is None:
			mask = selected[None].expand(len(observations_history), -1)
			episode_count = int(selected.sum())
			episode_rows = np.tile(np.arange(episode_count, dtype=np.int32), len(observations_history))
		else:
			mask = torch.stack(active_history)
			episode_count = num_envs
			episode_matrix = torch.arange(num_envs, device=gs.device)[None].expand_as(mask)
			episode_rows = episode_matrix[mask].cpu().numpy().astype(np.int32)
		observations = torch.stack(observations_history)[mask].reshape(-1, 147).cpu().numpy()
		actions = torch.stack(actions_history)[mask].reshape(-1, env.num_actions).cpu().numpy()
		swing_rows = torch.stack(swing_history)[mask].reshape(-1).cpu().numpy()
		metadata = json.dumps(dict(seed=seed, impulse=impulse, seconds=seconds, episodes=episode_count, successful_episodes=int(selected.sum()), collection="student_states" if student_policy is not None else "reference", oracle_timing=True, standing_action_scale_corrected=True, standing_limb_scale=config.get("limb_scale", 0.22), trial_limb_scale=env.limb_scale))
		with dataset_path.open("xb") as output:
			np.savez_compressed(output, observations=observations, actions=actions, swing=swing_rows, episode=episode_rows, metadata=metadata)
	survival_rows, success_rows, landed_rows, braked_rows = [value.cpu().numpy() for value in (survival, success, landed, braked)]
	trials = [dict(params=values[i].tolist(), survival=float(survival_rows[i]) / 60, success=bool(success_rows[i]), landing=bool(landed_rows[i]), braking=bool(braked_rows[i])) for i in range(num_envs)]
	return dict(seed=seed, seconds=seconds, impulse=impulse, num_envs=num_envs, max_start_tilt=max_start_tilt, max_start_spin=max_start_spin, oracle_timing=True, standing_action_scale_corrected=True, standing_limb_scale=config.get("limb_scale", 0.22), trial_limb_scale=env.limb_scale, swing_enabled=swing_enabled, ground_contacts=ground_contacts, student_run=str(student_run) if student_run else None, student_checkpoint=str(student_checkpoint) if student_checkpoint else None, trigger_speed=trigger_speed if student_run else None, hold_seconds=hold_seconds if student_run else None, successes=int(success.sum()), landed=int(landed.sum()), braked=int(braked.sum()), settled_with_step=int((success & landed).sum()), settled_with_braking_step=int((success & braked).sum()), trials=trials)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("run", type=Path)
	parser.add_argument("checkpoint", type=Path)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--around", type=Path, help="JSON list of candidates or a previous search report")
	parser.add_argument("--candidate-index", type=int, help="repeat this trial from --around instead of sampling")
	parser.add_argument("--without-swing", action="store_true", help="ablate swing hip/knee commands while retaining stance commands")
	parser.add_argument("--ground-contacts", action="store_true", help="measure foot contacts specifically against the floor")
	parser.add_argument("--max-start-tilt", type=float, default=0)
	parser.add_argument("--max-start-spin", type=float, default=0)
	parser.add_argument("--dataset", type=Path, help="save successful braking-step reference trajectories to a new NPZ")
	parser.add_argument("--student-run", type=Path, help="collect reference labels on a motion-gated student's states")
	parser.add_argument("--student-checkpoint", type=Path)
	parser.add_argument("--trigger-speed", type=float, default=0.12)
	parser.add_argument("--hold-seconds", type=float, default=0.8)
	parser.add_argument("--envs", type=int, default=512)
	parser.add_argument("--seconds", type=float, default=8)
	parser.add_argument("--impulse", type=float, default=16)
	parser.add_argument("--seed", type=int, default=16100)
	args = parser.parse_args()
	if args.envs < 1 or not np.isfinite((args.seconds, args.impulse, args.max_start_tilt, args.max_start_spin, args.trigger_speed, args.hold_seconds)).all() or args.seconds <= 3 or args.impulse <= 0 or not 0 <= args.max_start_tilt <= np.pi or args.max_start_spin < 0 or min(args.trigger_speed, args.hold_seconds) <= 0 or (args.candidate_index is not None and (args.around is None or args.candidate_index < 0)) or ((args.student_run is None) != (args.student_checkpoint is None)) or (args.student_run is not None and args.dataset is None):
		parser.error("environments and impulse must be positive and seconds must exceed three")
	if args.dataset is not None and (args.candidate_index is None or args.without_swing):
		parser.error("dataset collection requires one fixed candidate with swing enabled")
	around = fixed = None
	if args.around:
		data = json.loads(args.around.read_text(encoding="utf-8"))
		trials = data["trials"] if isinstance(data, dict) else data
		if args.candidate_index is not None:
			if args.candidate_index >= len(trials):
				parser.error("candidate index is outside the report")
			fixed = trials[args.candidate_index]["params"]
		else:
			trials = sorted(trials, key=lambda trial: (trial["success"] and trial["braking"], trial["success"], trial["braking"], trial["survival"]), reverse=True)
			around = [trial["params"] for trial in trials[:2]]
	report = run(args.run, args.checkpoint, num_envs=args.envs, seconds=args.seconds, impulse=args.impulse, seed=args.seed, around=around, fixed=fixed, max_start_tilt=args.max_start_tilt, max_start_spin=args.max_start_spin, swing_enabled=not args.without_swing, dataset_path=args.dataset, ground_contacts=args.ground_contacts, student_run=args.student_run, student_checkpoint=args.student_checkpoint, trigger_speed=args.trigger_speed, hold_seconds=args.hold_seconds)
	with args.output.open("x", encoding="utf-8") as output:
		json.dump(report, output, indent=2)
	print({key: value for key, value in report.items() if key != "trials"})


if __name__ == "__main__":
	main()
