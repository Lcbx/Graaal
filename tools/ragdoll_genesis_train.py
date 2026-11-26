"""Train an experimental Genesis standing and push-recovery policy with RSL-RL."""

import argparse
import json
from pathlib import Path

import numpy as np

from scenes.ragdoll_control import STANDING_ANGLES, STANDING_MOTOR_CAPS
from tools.ragdoll_genesis_rig import export_rig
from tools.ragdoll_project import default_project, read_project


class StandingEnv:
	"""GPU-batched hinge-rig prototype; its controls are not yet Jolt compatible."""

	def __init__(self, rig_path, actions, settings, *, num_envs, episode_seconds, push_impulse, balance_features=False, motor_kp=100, motor_kd=5, random_pushes=False, measure_settled=False, single_push=False, max_start_tilt=0, max_start_spin=0, hard_push_probability=0, hard_min_impulse=12, push_body="chest", push_direction=None, limb_scale=0.22, foot_features=False, step_reward=0, ground_contacts=False):
		if step_reward and push_impulse <= 0:
			raise ValueError("step reward requires a positive push impulse")
		import genesis as gs
		import torch
		from genesis.utils.geom import inv_quat, transform_by_quat

		self.gs, self.torch = gs, torch
		self.inv_quat, self.transform_by_quat = inv_quat, transform_by_quat
		self.device, self.num_envs, self.num_actions = gs.device, num_envs, len(actions)
		self.dt, self.max_episode_length = 1 / 60, round(episode_seconds * 60)
		self.cfg = {}
		self.push_impulse = push_impulse
		self.balance_features = balance_features
		self.foot_features = foot_features
		self.random_pushes = random_pushes
		self.measure_settled = measure_settled
		self.single_push = single_push
		self.max_start_tilt, self.max_start_spin = max_start_tilt, max_start_spin
		self.hard_push_probability, self.hard_min_impulse = hard_push_probability, hard_min_impulse
		self.push_direction = push_direction
		self.step_reward = step_reward
		self.ground_contacts = ground_contacts
		self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self.dt, substeps=2, gravity=(0, -9.81, 0)), rigid_options=gs.options.RigidOptions(enable_self_collision=True), show_viewer=False)
		self.floor = self.scene.add_entity(gs.morphs.Box(pos=(0, -0.5, 0), size=(20, 1, 20), fixed=True))
		self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(rig_path)))
		self.scene.build(n_envs=num_envs)
		self.indices = torch.tensor([self.robot.get_joint(action["name"]).dof_start for action in actions], dtype=torch.int64, device=self.device)
		self.non_ankle = torch.tensor([not (action["bone"] in ("foot_l", "foot_r") and action["axis_index"] in (1, 2)) for action in actions], device=self.device)
		self.limb_scale = limb_scale
		self.action_scale = torch.where(self.non_ankle, limb_scale, 0.22)
		bone_indices = [settings.bone_index(action["bone"]) for action in actions]
		caps = torch.tensor([STANDING_MOTOR_CAPS[index] for index in bone_indices], dtype=torch.float32, device=self.device)
		target = [float(np.dot(action["axis"], [1, 0, 0]) * STANDING_ANGLES[index]) for action, index in zip(actions, bone_indices)]
		self.target = torch.tensor(target, dtype=torch.float32, device=self.device)
		self.robot.set_dofs_kp(torch.full_like(caps, motor_kp), self.indices)
		self.robot.set_dofs_kv(torch.full_like(caps, motor_kd), self.indices)
		self.robot.set_dofs_force_range(-caps, caps, self.indices)
		self.robot.set_dofs_position(self.target, self.indices, zero_velocity=True)
		self.initial_qpos = self.robot.get_qpos()[0].clone()
		self.initial_height = float(self.initial_qpos[1])
		self.chest = self.robot.get_link("chest")
		self.push_link = self.robot.get_link(push_body)
		self.feet = [self.robot.get_link(name) for name in ("foot_l", "foot_r")]
		self.masses = self.robot.get_links_mass()
		self.masses = self.masses / self.masses.sum()
		self.gravity = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.device).expand(num_envs, -1)
		self.previous = torch.zeros((num_envs, self.num_actions), device=self.device)
		self.foot_contacts = torch.zeros((num_envs, 2), device=self.device)
		self.base_policy = None
		self.gate_ticks = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
		self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
		if step_reward:
			from tools.ragdoll_genesis_steps import StepTracker

			self.step_tracker = StepTracker(num_envs, self.device)
			self.step_age = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
			self.step_hold = torch.zeros_like(self.step_age)
		self.reset()

	def _observations(self):
		q, dq = self.robot.get_dofs_position(self.indices), self.robot.get_dofs_velocity(self.indices)
		inverse = self.inv_quat(self.robot.get_quat())
		gravity = self.transform_by_quat(self.gravity, inverse)
		angular = self.transform_by_quat(self.robot.get_ang(), inverse)
		velocity = self.transform_by_quat(self.robot.get_vel(), inverse)
		values = [gravity, angular, velocity, q, dq, self.target.expand(self.num_envs, -1), self.previous]
		if self.foot_features:
			root_pos, root_vel = self.robot.get_pos(), self.robot.get_vel()
			foot_pos = [self.transform_by_quat(foot.get_pos() - root_pos, inverse) for foot in self.feet]
			foot_vel = [self.transform_by_quat(foot.get_vel() - root_vel, inverse) for foot in self.feet]
			values.extend((*foot_pos, *foot_vel, self.foot_contacts))
		if self.balance_features:
			com = self.robot.get_links_pos(ref=self.gs.link_ref_frame.root_COM)[:, 0]
			feet = (self.feet[0].get_pos() + self.feet[1].get_pos()) / 2
			link_velocities = self.robot.get_links_vel(ref=self.gs.link_ref_frame.link_COM)
			com_velocity = (link_velocities * self.masses[None, :, None]).sum(dim=1)
			values.append(self.torch.stack((com[:, 2] - feet[:, 2], com[:, 0] - feet[:, 0], com_velocity[:, 2], com_velocity[:, 0]), dim=-1))
		return self.torch.cat(values, dim=-1)

	def get_observations(self):
		from tensordict import TensorDict

		self.latest_observations = self._observations()
		return TensorDict({"policy": self.latest_observations}, batch_size=[self.num_envs])

	def _reset_world(self, mask=None):
		if not (self.max_start_tilt or self.max_start_spin):
			self.robot.set_qpos(self.initial_qpos, envs_idx=mask, zero_velocity=True)
			return
		torch = self.torch
		angle = torch.rand(self.num_envs, device=self.device) * self.max_start_tilt
		direction = torch.rand(self.num_envs, device=self.device) * (2 * np.pi)
		qpos = self.initial_qpos.expand(self.num_envs, -1).clone()
		qpos[:, 3] = torch.cos(angle / 2)
		qpos[:, 4] = torch.sin(angle / 2) * torch.cos(direction)
		qpos[:, 5] = 0
		qpos[:, 6] = torch.sin(angle / 2) * torch.sin(direction)
		spin = torch.randn((self.num_envs, 3), device=self.device)
		spin = spin / spin.norm(dim=1, keepdim=True).clamp_min(1e-8) * (torch.rand((self.num_envs, 1), device=self.device) * self.max_start_spin)
		self.robot.set_qpos(qpos if mask is None else qpos[mask], envs_idx=mask, zero_velocity=True)
		self.robot.set_dofs_velocity(spin if mask is None else spin[mask], [3, 4, 5], envs_idx=mask)

	def reset(self):
		self._reset_world()
		self.episode_length_buf.zero_()
		self.previous.zero_()
		self.foot_contacts.zero_()
		self.gate_ticks.zero_()
		if self.step_reward:
			self.step_tracker.reset(self.torch.ones(self.num_envs, dtype=self.torch.bool, device=self.device))
			self.step_age.zero_()
			self.step_hold.zero_()
		return self.get_observations()

	def step(self, actions):
		torch = self.torch
		actions = actions.clamp(-1, 1)
		if self.base_policy is not None:
			from tensordict import TensorDict
			from tools.ragdoll_genesis_steps import motion_gated_actions

			features = self.latest_observations
			base_input = TensorDict({"policy": torch.cat((features[:, :-18], features[:, -4:]), dim=1)}, batch_size=[self.num_envs])
			with torch.no_grad():
				base_action = self.base_policy(base_input)
			actions, self.gate_ticks = motion_gated_actions(features, base_action, actions, self.gate_ticks, self.non_ankle, trigger_speed=self.gate_trigger_speed, hold_ticks=self.gate_hold_ticks, base_scale=self.base_limb_scale, step_scale=self.limb_scale)
		self.robot.control_dofs_position(self.target + self.action_scale * actions, self.indices)
		if self.push_impulse:
			push = (self.episode_length_buf == 119) if self.single_push else (self.episode_length_buf.remainder(120) == 119)
			direction = torch.randint(0, 4, (self.num_envs,), device=self.device) if self.push_direction is None else torch.full((self.num_envs,), self.push_direction, device=self.device)
			magnitude = self.push_impulse * torch.rand(self.num_envs, device=self.device) * (torch.rand(self.num_envs, device=self.device) >= 0.25) if self.random_pushes else self.push_impulse
			if self.hard_push_probability:
				hard = torch.rand(self.num_envs, device=self.device) < self.hard_push_probability
				direction = torch.where(hard, 3, direction)
				hard_magnitude = self.hard_min_impulse + (self.push_impulse - self.hard_min_impulse) * torch.rand(self.num_envs, device=self.device)
				magnitude = torch.where(hard, hard_magnitude, magnitude)
			forces = torch.zeros((self.num_envs, 3), device=self.device)
			forces[:, 0] = ((direction == 0).float() - (direction == 1).float()) * push * (magnitude / self.dt)
			forces[:, 2] = ((direction == 2).float() - (direction == 3).float()) * push * (magnitude / self.dt)
			self.push_link.apply_external_force(forces)
		self.scene.step()
		foot_forces = None
		if self.foot_features:
			foot_forces = self.robot.get_links_net_contact_force()[:, [foot.idx_local for foot in self.feet], 1]
			self.foot_contacts = (foot_forces > 5).float()
		self.episode_length_buf += 1
		q, dq = self.robot.get_dofs_position(self.indices), self.robot.get_dofs_velocity(self.indices)
		gravity = self.transform_by_quat(self.gravity, self.inv_quat(self.robot.get_quat()))
		height = self.robot.get_pos()[:, 1]
		pose = torch.exp(-torch.square(q - self.target).sum(dim=-1))
		upright = (-gravity[:, 1]).clamp(-1, 1)
		reward = self.dt * (2 * pose + 2 * upright - 2 * torch.square(height - self.initial_height) - 0.002 * torch.square(dq).sum(dim=-1) - 0.01 * torch.square(actions - self.previous).sum(dim=-1))
		fallen = (height < 0.7 * self.initial_height) | (upright < 0.5) | self.scene.get_error_envs_mask()
		timeouts = (self.episode_length_buf >= self.max_episode_length) & ~fallen
		done = fallen | timeouts
		info = {"time_outs": timeouts}
		if self.push_impulse:
			info["push"] = push
			info["push_direction"] = direction
		if self.measure_settled or self.step_reward:
			velocity = (self.robot.get_links_vel(ref=self.gs.link_ref_frame.link_COM) * self.masses[None, :, None]).sum(dim=1)
			if self.ground_contacts:
				pairs = self.robot.get_contacts(with_entity=self.floor, is_padded=True)
				if pairs is None or pairs["force_a"] is None:
					raise RuntimeError("Genesis floor contact forces are unavailable after the physics step")
				contacts = torch.stack([((pairs["link_a"] == foot.idx) * pairs["force_a"][:, :, 1] + (pairs["link_b"] == foot.idx) * pairs["force_b"][:, :, 1]).mul(pairs["valid_mask"]).sum(dim=1) for foot in self.feet], dim=1)
			else:
				contacts = foot_forces if foot_forces is not None else self.robot.get_links_net_contact_force()[:, [foot.idx_local for foot in self.feet], 1]
			chest_up = -self.transform_by_quat(self.gravity, self.inv_quat(self.chest.get_quat()))[:, 1]
			info["settled"] = (height >= 0.9 * self.initial_height) & (upright >= 0.95) & (chest_up >= 0.95) & (velocity[:, [0, 2]].norm(dim=1) < 0.05) & (contacts.sum(dim=1) > 5)
			info["settle_diagnostics"] = {"height": height, "pelvis_up": upright, "chest_up": chest_up, "com_speed": velocity[:, [0, 2]].norm(dim=1), "foot_force": contacts.sum(dim=1)}
			info["foot_positions"] = torch.stack([foot.get_pos() for foot in self.feet], dim=1)
			info["foot_contact"] = contacts > 5
		if self.step_reward:
			tracked = push & (direction == 3) & (magnitude >= self.hard_min_impulse)
			previous_landing = self.step_tracker.landed.any(dim=1).clone()
			previous_braking = self.step_tracker.braked.any(dim=1).clone()
			self.step_tracker.update(info["foot_positions"], info["foot_contact"], info["settle_diagnostics"]["com_speed"], ~fallen, tracked, direction)
			self.step_age += self.step_tracker.push_seen
			lift = (info["foot_positions"][:, :, 1] - self.step_tracker.origin[:, :, 1]).clamp(0, 0.08) / 0.08
			backward = (self.step_tracker.origin[:, :, 2] - info["foot_positions"][:, :, 2]).clamp(0, 0.15) / 0.15
			movement = (lift * backward).max(dim=1).values * self.step_tracker.push_seen * (self.step_age <= 36)
			landed = self.step_tracker.landed.any(dim=1)
			braked = self.step_tracker.braked.any(dim=1)
			self.step_hold = torch.where(info["settled"] & landed & ~fallen, self.step_hold + 1, 0)
			reward += self.step_reward * (self.dt * movement + (landed & ~previous_landing).float() + 2 * (braked & ~previous_braking).float() + 3 * (self.step_hold == 60).float())
			info["log"] = {"step/movement": movement.mean(), "step/landings": (landed & ~previous_landing).float().mean(), "step/braking": (braked & ~previous_braking).float().mean(), "step/holds": (self.step_hold == 60).float().mean()}
		self.previous = actions.clone()
		self._reset_world(done)
		self.episode_length_buf.masked_fill_(done, 0)
		self.previous.masked_fill_(done[:, None], 0)
		self.foot_contacts.masked_fill_(done[:, None], 0)
		self.gate_ticks.masked_fill_(done, 0)
		if self.step_reward:
			self.step_tracker.reset(done)
			self.step_age.masked_fill_(done, 0)
			self.step_hold.masked_fill_(done, 0)
		return self.get_observations(), reward, done, info


def train_config(balance_init=False):
	return {
		"algorithm": {"class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.01, "gamma": 0.99, "lam": 0.95, "learning_rate": 0.00003 if balance_init else 0.001, "max_grad_norm": 1.0, "num_learning_epochs": 5, "num_mini_batches": 4, "schedule": "fixed" if balance_init else "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0},
		"actor": {"class_name": "MLPModel", "hidden_dims": [256, 256], "activation": "elu", "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.03 if balance_init else 0.5, "std_type": "scalar"}},
		"critic": {"class_name": "MLPModel", "hidden_dims": [256, 256], "activation": "elu"},
		"obs_groups": {"actor": ["policy"], "critic": ["policy"]}, "num_steps_per_env": 24, "save_interval": 10, "logger": "tensorboard",
	}


def initialize_balance(policy, actions):
	"""Initialize the actor near the measured COM feedback law."""
	import torch

	first, _, second, _, output = policy.mlp
	with torch.no_grad():
		for layer in (first, second, output):
			layer.weight.zero_()
			layer.bias.zero_()
		indices = torch.arange(4, device=first.weight.device)
		first.weight[indices, first.in_features - 4 + indices] = 0.1
		second.weight[indices, indices] = 1
		for index, action in enumerate(actions):
			if action["bone"] in ("foot_l", "foot_r") and action["axis_index"] in (1, 2):
				direction = 0 if action["axis_index"] == 1 else 1
				output.weight[index, direction] = 40
				output.weight[index, direction + 2] = 10


def warm_start_foot_features(runner, checkpoint):
	"""Expand the first layers while preserving the prior policy at startup."""
	import torch

	state = torch.load(checkpoint, map_location="cpu", weights_only=False)
	for model, key in ((runner.alg.actor, "actor_state_dict"), (runner.alg.critic, "critic_state_dict")):
		weights = state[key]
		old = weights["mlp.0.weight"]
		new = model.state_dict()["mlp.0.weight"].clone().zero_()
		if old.shape[0] != new.shape[0] or old.shape[1] + 14 != new.shape[1]:
			raise ValueError("warm-start checkpoint does not match the 14 added foot features")
		new[:, :old.shape[1] - 4] = old[:, :-4]
		new[:, -4:] = old[:, -4:]
		weights["mlp.0.weight"] = new
		model.load_state_dict(weights)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--output", type=Path, required=True, help="new directory for rig, configuration and checkpoints")
	parser.add_argument("--tuning", type=Path)
	parser.add_argument("--envs", type=int, default=64)
	parser.add_argument("--iterations", type=int, default=10)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--seconds", type=float, default=5)
	parser.add_argument("--push-impulse", type=float, default=0)
	parser.add_argument("--random-pushes", action="store_true")
	parser.add_argument("--single-push", action="store_true", help="apply only the first scheduled push in each episode")
	parser.add_argument("--balance-init", action="store_true", help="initialize the actor from COM feedback")
	parser.add_argument("--resume", type=Path, help="continue an RSL-RL checkpoint with the same policy configuration")
	parser.add_argument("--warm-start-feet", type=Path, help="initialize a foot-observing actor from a compatible checkpoint")
	parser.add_argument("--motor-kp", type=float, default=100)
	parser.add_argument("--motor-kd", type=float, default=5)
	parser.add_argument("--max-start-tilt", type=float, default=0)
	parser.add_argument("--max-start-spin", type=float, default=0)
	parser.add_argument("--hard-push-probability", type=float, default=0, help="fraction of push events sampled as hard backward chest pushes")
	parser.add_argument("--hard-min-impulse", type=float, default=12)
	parser.add_argument("--limb-scale", type=float, default=0.22, help="maximum non-ankle target offset in radians")
	parser.add_argument("--limb-std", type=float, help="override non-ankle policy exploration standard deviation")
	parser.add_argument("--learning-rate", type=float, help="override PPO learning rate, including on resume")
	parser.add_argument("--step-reward", type=float, default=0, help="weight for backward airborne movement, landing and braking rewards")
	parser.add_argument("--gate-base", type=Path, help="standing checkpoint for a backward-COM-speed-triggered policy blend")
	parser.add_argument("--without-gate", action="store_true", help="resume a gated checkpoint without its standing blend")
	parser.add_argument("--gate-trigger-speed", type=float)
	parser.add_argument("--gate-hold-seconds", type=float)
	args = parser.parse_args()
	if args.resume is not None and args.warm_start_feet is not None:
		parser.error("choose resume or warm-start-feet")
	if args.warm_start_feet is not None and not args.balance_init:
		parser.error("warm-start-feet requires --balance-init")
	if args.step_reward and (not args.hard_push_probability or args.seconds <= 3 or args.push_impulse <= 0):
		parser.error("step reward requires a positive push impulse, hard backward pushes and episodes longer than three seconds")
	if args.without_gate and args.gate_base is not None:
		parser.error("choose gate-base or without-gate")
	if args.envs < 1 or args.iterations < 1 or not np.isfinite(args.seconds) or args.seconds < 1 or not np.isfinite(args.push_impulse) or args.push_impulse < 0 or not np.isfinite((args.motor_kp, args.motor_kd, args.max_start_tilt, args.max_start_spin, args.hard_push_probability, args.hard_min_impulse, args.limb_scale, args.step_reward)).all() or min(args.motor_kp, args.motor_kd, args.max_start_tilt, args.max_start_spin, args.hard_min_impulse, args.step_reward) < 0 or args.limb_scale <= 0 or args.max_start_tilt > np.pi or not 0 <= args.hard_push_probability <= 1 or (args.hard_push_probability and (not args.random_pushes or args.hard_min_impulse > args.push_impulse)) or (args.step_reward and (args.seconds <= 3 or not args.hard_push_probability)) or (args.limb_std is not None and (not np.isfinite(args.limb_std) or not 0 < args.limb_std <= 1)) or (args.learning_rate is not None and (not np.isfinite(args.learning_rate) or args.learning_rate <= 0)):
		parser.error("environments and iterations must be positive, seconds at least one, and push impulse finite and nonnegative")
	project = default_project() if args.tuning is None else read_project(args.tuning)
	from scenes.passive_ragdoll_example import HUMANOID_NAMES, HUMANOID_PARENTS
	if tuple(project.settings.names) != HUMANOID_NAMES or tuple(project.settings.parents) != HUMANOID_PARENTS:
		parser.error("the prototype requires the built-in humanoid bone names and order")
	xml, actions = export_rig(project)
	foot_features = args.warm_start_feet is not None
	source = args.resume or args.warm_start_feet
	previous = {}
	if source is not None:
		previous = json.loads((source.parent / "config.json").read_text(encoding="utf-8"))
		if previous["actions"] != actions or previous.get("balance_features", False) != args.balance_init or previous.get("motor_kp", 100) != args.motor_kp or previous.get("motor_kd", 5) != args.motor_kd or (source.parent / "rig.xml").read_text(encoding="utf-8").strip() != xml:
			parser.error("checkpoint requires the same rig, action mapping, balance features and motor gains")
		if args.warm_start_feet is not None and previous.get("foot_features", False):
			parser.error("warm-start-feet requires a checkpoint without foot features")
		if args.resume is not None:
			foot_features = previous.get("foot_features", False)
	if args.resume is not None and args.gate_base is None and not args.without_gate and previous.get("gate_base"):
		args.gate_base = Path(previous["gate_base"])
	args.gate_trigger_speed = args.gate_trigger_speed if args.gate_trigger_speed is not None else previous.get("gate_trigger_speed", 0.12)
	args.gate_hold_seconds = args.gate_hold_seconds if args.gate_hold_seconds is not None else previous.get("gate_hold_seconds", 0.8)
	if not np.isfinite((args.gate_trigger_speed, args.gate_hold_seconds)).all() or min(args.gate_trigger_speed, args.gate_hold_seconds) <= 0:
		parser.error("gate speed and hold duration must be positive and finite")
	gate_config = None
	if args.gate_base is not None:
		gate_config = json.loads((args.gate_base.parent / "config.json").read_text(encoding="utf-8"))
		if not foot_features or not gate_config.get("balance_features", False) or gate_config.get("foot_features", False) or gate_config["actions"] != actions or gate_config.get("motor_kp", 100) != args.motor_kp or gate_config.get("motor_kd", 5) != args.motor_kd or gate_config.get("limb_scale", 0.22) > args.limb_scale or (args.gate_base.parent / "rig.xml").read_text(encoding="utf-8").strip() != xml:
			parser.error("gate base requires compatible standing and foot-observing policies")
	args.output.mkdir(parents=True, exist_ok=False)
	rig_path = args.output / "rig.xml"
	rig_path.write_text(xml + "\n", encoding="utf-8")
	(args.output / "config.json").write_text(json.dumps(dict(envs=args.envs, iterations=args.iterations, seed=args.seed, seconds=args.seconds, push_impulse=args.push_impulse, random_pushes=args.random_pushes, single_push=args.single_push or bool(args.step_reward), hard_push_probability=args.hard_push_probability, hard_min_impulse=args.hard_min_impulse, step_reward=args.step_reward, balance_features=args.balance_init, foot_features=foot_features, motor_kp=args.motor_kp, motor_kd=args.motor_kd, max_start_tilt=args.max_start_tilt, max_start_spin=args.max_start_spin, limb_scale=args.limb_scale, limb_std=args.limb_std, learning_rate=args.learning_rate, gate_base=str(args.gate_base) if args.gate_base else None, gate_trigger_speed=args.gate_trigger_speed, gate_hold_seconds=args.gate_hold_seconds, resume=str(args.resume) if args.resume else None, warm_start_feet=str(args.warm_start_feet) if args.warm_start_feet else None, actions=actions, ppo=train_config(args.balance_init)), indent=2) + "\n", encoding="utf-8")
	import genesis as gs
	from rsl_rl.runners import OnPolicyRunner

	gs.init(backend=gs.cuda, precision="32", logging_level="warning", seed=args.seed)
	env = StandingEnv(rig_path, actions, project.settings, num_envs=args.envs, episode_seconds=args.seconds, push_impulse=args.push_impulse, balance_features=args.balance_init, motor_kp=args.motor_kp, motor_kd=args.motor_kd, random_pushes=args.random_pushes, max_start_tilt=args.max_start_tilt, max_start_spin=args.max_start_spin, hard_push_probability=args.hard_push_probability, hard_min_impulse=args.hard_min_impulse, limb_scale=args.limb_scale, foot_features=foot_features, step_reward=args.step_reward, single_push=args.single_push or bool(args.step_reward))
	runner = OnPolicyRunner(env, train_config(args.balance_init), str(args.output), device=gs.device)
	if args.warm_start_feet is not None:
		warm_start_foot_features(runner, args.warm_start_feet)
	elif args.resume is not None:
		runner.load(str(args.resume), map_location=str(gs.device))
	elif args.balance_init:
		initialize_balance(runner.alg.get_policy(), actions)
	if args.limb_std is not None:
		import torch

		with torch.no_grad():
			runner.alg.get_policy().distribution.std_param[env.non_ankle] = args.limb_std
	if args.learning_rate is not None:
		runner.alg.learning_rate = args.learning_rate
		for group in runner.alg.optimizer.param_groups:
			group["lr"] = args.learning_rate
	if gate_config is not None:
		import torch
		from tensordict import TensorDict

		original_observations = env.get_observations
		def base_observations():
			features = original_observations()["policy"]
			return TensorDict({"policy": torch.cat((features[:, :-18], features[:, -4:]), dim=1)}, batch_size=[env.num_envs])
		env.get_observations = base_observations
		base_runner = OnPolicyRunner(env, gate_config["ppo"], log_dir=None, device=gs.device)
		base_runner.load(str(args.gate_base), map_location=str(gs.device))
		env.base_policy = base_runner.get_inference_policy(device=gs.device)
		env.get_observations = original_observations
		env.base_limb_scale = gate_config.get("limb_scale", 0.22)
		env.gate_trigger_speed = args.gate_trigger_speed
		env.gate_hold_ticks = round(args.gate_hold_seconds / env.dt)
	runner.learn(num_learning_iterations=args.iterations)


if __name__ == "__main__":
	main()
