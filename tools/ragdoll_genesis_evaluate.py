"""Compare a Genesis checkpoint against zero residual actions."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tools.ragdoll_genesis_train import StandingEnv
from tools.ragdoll_genesis_steps import StepTracker, motion_gated_actions
from tools.ragdoll_project import default_project


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("run", type=Path)
	parser.add_argument("checkpoint", type=Path)
	parser.add_argument("--compare", type=Path, nargs="*", default=(), help="additional compatible checkpoints")
	parser.add_argument("--envs", type=int, default=64)
	parser.add_argument("--seconds", type=float, default=5)
	parser.add_argument("--push-impulse", type=float, default=1)
	parser.add_argument("--push-grid", type=float, nargs="+", help="evaluate each listed impulse using paired starts")
	parser.add_argument("--seed", type=int, default=10001)
	parser.add_argument("--max-start-tilt", type=float, default=0.025)
	parser.add_argument("--max-start-spin", type=float, default=0.1)
	parser.add_argument("--body", choices=("chest", "pelvis"), default="chest")
	parser.add_argument("--direction", choices=("+X", "-X", "+Z", "-Z"), help="use one direction instead of sampling all four")
	parser.add_argument("--ground-contacts", action="store_true", help="measure foot contacts specifically against the floor")
	parser.add_argument("--output", type=Path, help="write the evaluation report to a new JSON file")
	parser.add_argument("--base-run", type=Path, help="standing run for a motion-triggered policy blend")
	parser.add_argument("--base-checkpoint", type=Path)
	parser.add_argument("--without-base", action="store_true", help="evaluate the step actor without its configured standing blend")
	parser.add_argument("--trigger-speed", type=float, help="backward COM speed in m/s that triggers the step actor")
	parser.add_argument("--hold-seconds", type=float, help="minimum time to retain the step actor after a trigger")
	args = parser.parse_args()
	impulses = args.push_grid if args.push_grid is not None else [args.push_impulse]
	if args.envs < 1 or not np.isfinite((args.seconds, *impulses, args.max_start_tilt, args.max_start_spin)).all() or args.seconds < 1 or (any(impulses) and args.seconds <= 2) or min(*impulses, args.max_start_tilt, args.max_start_spin) < 0 or args.max_start_tilt > np.pi or ((args.base_run is None) != (args.base_checkpoint is None)) or (args.without_base and args.base_run is not None):
		parser.error("environments and duration must be positive and impulse nonnegative")
	config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
	if args.base_run is None and not args.without_base and config.get("gate_base"):
		args.base_checkpoint = Path(config["gate_base"])
		args.base_run = args.base_checkpoint.parent
	args.trigger_speed = args.trigger_speed if args.trigger_speed is not None else config.get("gate_trigger_speed", 0.12)
	args.hold_seconds = args.hold_seconds if args.hold_seconds is not None else config.get("gate_hold_seconds", 0.8)
	if not np.isfinite((args.trigger_speed, args.hold_seconds)).all() or min(args.trigger_speed, args.hold_seconds) <= 0:
		parser.error("trigger speed and hold duration must be positive and finite")
	import genesis as gs
	import torch
	from rsl_rl.runners import OnPolicyRunner

	gs.init(backend=gs.cuda, precision="32", logging_level="warning", seed=args.seed)
	direction_index = ("+X", "-X", "+Z", "-Z").index(args.direction) if args.direction else None
	env = StandingEnv(args.run / "rig.xml", config["actions"], default_project().settings, num_envs=args.envs, episode_seconds=args.seconds, push_impulse=args.push_impulse, balance_features=config.get("balance_features", False), motor_kp=config.get("motor_kp", 100), motor_kd=config.get("motor_kd", 5), measure_settled=True, single_push=True, push_body=args.body, push_direction=direction_index, limb_scale=config.get("limb_scale", 0.22), foot_features=config.get("foot_features", False), ground_contacts=args.ground_contacts)
	runner = OnPolicyRunner(env, config["ppo"], log_dir=None, device=gs.device)
	runner.load(str(args.checkpoint), map_location=str(gs.device))
	policies = {"policy": copy.deepcopy(runner.get_inference_policy(device=gs.device))}
	for checkpoint in args.compare:
		if checkpoint.stem in policies or checkpoint.stem == "zero_residual":
			parser.error("comparison checkpoints need unique filenames")
		runner.load(str(checkpoint), map_location=str(gs.device))
		policies[checkpoint.stem] = copy.deepcopy(runner.get_inference_policy(device=gs.device))
	base_policy = None
	if args.base_run is not None:
		from tensordict import TensorDict

		base_config = json.loads((args.base_run / "config.json").read_text(encoding="utf-8"))
		if not config.get("foot_features", False) or not base_config.get("balance_features", False) or base_config.get("foot_features", False) or base_config["actions"] != config["actions"] or base_config.get("motor_kp", 100) != config.get("motor_kp", 100) or base_config.get("motor_kd", 5) != config.get("motor_kd", 5) or base_config.get("limb_scale", 0.22) > config.get("limb_scale", 0.22) or (args.base_run / "rig.xml").read_text(encoding="utf-8") != (args.run / "rig.xml").read_text(encoding="utf-8"):
			parser.error("the standing and step actors require the same rig, actions and motor gains")
		original_observations = env.get_observations
		def base_observations():
			features = original_observations()["policy"]
			return TensorDict({"policy": torch.cat((features[:, :-18], features[:, -4:]), dim=1)}, batch_size=[args.envs])
		env.get_observations = base_observations
		base_runner = OnPolicyRunner(env, base_config["ppo"], log_dir=None, device=gs.device)
		base_runner.load(str(args.base_checkpoint), map_location=str(gs.device))
		base_policy = base_runner.get_inference_policy(device=gs.device)
		env.get_observations = original_observations
	result = {}
	with torch.inference_mode():
		for impulse in impulses:
			env.push_impulse = impulse
			result[str(impulse)] = {}
			for name in ("zero_residual", *policies):
				torch.manual_seed(args.seed)
				observations = env.reset()
				angle = torch.rand(args.envs, device=gs.device) * args.max_start_tilt
				direction = torch.rand(args.envs, device=gs.device) * (2 * np.pi)
				qpos = env.initial_qpos.expand(args.envs, -1).clone()
				qpos[:, 3] = torch.cos(angle / 2)
				qpos[:, 4] = torch.sin(angle / 2) * torch.cos(direction)
				qpos[:, 5] = 0
				qpos[:, 6] = torch.sin(angle / 2) * torch.sin(direction)
				env.robot.set_qpos(qpos, zero_velocity=True)
				spin = torch.randn((args.envs, 3), device=gs.device)
				spin = spin / spin.norm(dim=1, keepdim=True).clamp_min(1e-8) * (torch.rand((args.envs, 1), device=gs.device) * args.max_start_spin)
				env.robot.set_dofs_velocity(spin, [3, 4, 5])
				observations = env.get_observations()
				active = torch.ones(args.envs, dtype=torch.bool, device=gs.device)
				completed = torch.zeros_like(active)
				settled_completed = torch.zeros_like(active)
				settled_ticks = torch.zeros(args.envs, dtype=torch.int32, device=gs.device)
				push_directions = torch.full((args.envs,), -1, dtype=torch.int64, device=gs.device)
				gate_ticks = torch.zeros(args.envs, dtype=torch.int32, device=gs.device)
				gate_seen = torch.zeros(args.envs, dtype=torch.bool, device=gs.device)
				initial_feet = torch.stack([foot.get_pos() for foot in env.feet], dim=1)
				peak_foot_displacement = torch.zeros(args.envs, device=gs.device)
				step_tracker = StepTracker(args.envs, gs.device)
				steps = torch.zeros(args.envs, device=gs.device)
				for _ in range(env.max_episode_length):
					actions = torch.zeros((args.envs, env.num_actions), device=gs.device) if name == "zero_residual" else policies[name](observations)
					if base_policy is not None and name != "zero_residual":
						features = observations["policy"]
						base_input = TensorDict({"policy": torch.cat((features[:, :-18], features[:, -4:]), dim=1)}, batch_size=[args.envs])
						base_action = base_policy(base_input)
						actions, gate_ticks = motion_gated_actions(features, base_action, actions, gate_ticks, env.non_ankle, trigger_speed=args.trigger_speed, hold_ticks=round(args.hold_seconds / env.dt), base_scale=base_config.get("limb_scale", 0.22), step_scale=config.get("limb_scale", 0.22))
						gate_seen |= active & (gate_ticks > 0)
					observations, _, done, info = env.step(actions)
					steps += active
					feet = info["foot_positions"]
					distance = (feet[:, :, [0, 2]] - initial_feet[:, :, [0, 2]]).norm(dim=-1).max(dim=1).values
					peak_foot_displacement = torch.where(active, torch.maximum(peak_foot_displacement, distance), peak_foot_displacement)
					if impulse:
						push_directions = torch.where(info["push"] & active, info["push_direction"], push_directions)
						step_tracker.update(feet, info["foot_contact"], info["settle_diagnostics"]["com_speed"], active, info["push"], info["push_direction"])
					settled_ticks = torch.where(info["settled"], settled_ticks + 1, 0)
					settled_completed |= active & info["time_outs"] & (settled_ticks >= 60)
					completed |= active & info["time_outs"]
					active &= ~done
					gate_ticks.masked_fill_(done, 0)
				by_direction = {label: dict(received=int((push_directions == index).sum().item()), settled=int((settled_completed & (push_directions == index)).sum().item())) for index, label in enumerate(("+X", "-X", "+Z", "-Z"))} if impulse else {}
				result[str(impulse)][name] = dict(completed=int(completed.sum().item()), settled_completed=int(settled_completed.sum().item()), episodes=args.envs, mean_survival_seconds=float((steps.mean() * env.dt).item()), mean_peak_foot_displacement_m=float(peak_foot_displacement.mean().item()), triggered=int(gate_seen.sum().item()), relocated_landings=int(step_tracker.landed.any(dim=1).sum().item()), settled_with_step=int((settled_completed & step_tracker.landed.any(dim=1)).sum().item()), settled_with_braking_step=int((settled_completed & step_tracker.braked.any(dim=1)).sum().item()), by_direction=by_direction)
	report = json.dumps(dict(seed=args.seed, seconds=args.seconds, max_start_tilt=args.max_start_tilt, max_start_spin=args.max_start_spin, body=args.body, direction=args.direction, ground_contacts=args.ground_contacts, base_run=str(args.base_run) if args.base_run else None, trigger_speed=args.trigger_speed if args.base_run else None, hold_seconds=args.hold_seconds if args.base_run else None, results=result), indent=2)
	if args.output is not None:
		with args.output.open("x", encoding="utf-8") as output:
			output.write(report + "\n")
	print(report)


if __name__ == "__main__":
	main()
