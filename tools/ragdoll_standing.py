"""Evaluate or train a joint-controlled ragdoll to stand on a flat floor."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import joint_limit_torques, joint_spring_torques
from graaal.PhysicsRagdollControl import apply_ragdoll_motors
from scenes.passive_ragdoll_example import HUMANOID_NAMES, HUMANOID_PARENTS
from scenes.physics_geometry import collider_bounds, collider_poses
from scenes.ragdoll_control import STANDING_MOTOR_CAPS, make_ankle_balance, observe_ragdoll, standing_pose
from tools.ragdoll_project import default_project, read_project


TICK_HZ = 120
TICKS_PER_ACTION = 2
CONTROL_HZ = TICK_HZ // TICKS_PER_ACTION
MAX_OFFSET = 0.22
MAX_RESIDUAL_TORQUE = 10.0
DEFAULT_START_TILT = 0.025
SETTLE_SECONDS = 1


def compact_axes(limits):
	"""Select constraint-frame axes with enough travel for standing corrections."""
	travel = np.column_stack((np.diff(limits["twist"], axis=1), 2 * limits["swing"]))
	# An axis needs at least one full offset's worth of travel across both sides.
	bones, axes = np.nonzero(travel[1:] >= MAX_OFFSET)
	return bones + 1, Quaternion.matrix_batch(limits["child_frames"])[bones + 1, :, axes]


def _support_margin(points, center):
	"""Return the signed distance to the boundary of a planar convex hull."""
	left, right = np.triu_indices(len(points), k=1)
	start = points[left]
	edges = points[right] - start
	normals = edges[:, [1, 0]] * [-1, 1]
	sides = np.einsum("evi,ei->ev", points - start[:, None], normals)
	positive = sides.max(axis=1) > 1e-9
	negative = sides.min(axis=1) < -1e-9
	boundary = positive ^ negative
	length_squared = np.einsum("ei,ei->e", edges, edges)
	if not boundary.any():
		if not length_squared.any():
			return -float(np.linalg.norm(points[0] - center))
		selected = length_squared > 0
	else:
		selected = boundary
	edges, start = edges[selected], start[selected]
	delta = center - start
	fraction = np.clip(np.einsum("ei,ei->e", delta, edges) / length_squared[selected], 0, 1)
	distance = np.linalg.norm(delta - edges * fraction[:, None], axis=1).min()
	inside = boundary.any() and np.all(np.einsum("ei,ei->e", delta, normals[boundary]) * np.where(positive[boundary], 1, -1) >= -1e-9)
	return float(distance if inside else -distance)


class StandingTask:
	"""Run one episode in a fresh native world, exposing only joint actions."""

	def __init__(self, project, *, episode_steps=5 * CONTROL_HZ, initial_angular_speed=0.0, tuned_stance=False, compact=False, vary_start=False, max_start_tilt=DEFAULT_START_TILT, support_area=False, ankle_balance=False, residual=False):
		import pyjoltphysics as pj

		if episode_steps < 1 or not np.isfinite(initial_angular_speed) or initial_angular_speed < 0:
			raise ValueError("episode steps must be positive and initial angular speed must be finite and nonnegative")
		if not np.isfinite(max_start_tilt) or not 0 <= max_start_tilt <= np.pi:
			raise ValueError("maximum start tilt must be in [0, pi] radians")
		if compact and residual:
			raise ValueError("compact and residual actions are mutually exclusive")
		settings = pj.RagdollSettings.from_dict(project.settings.as_dict())
		if tuple(bone.name for bone in settings.bones) != HUMANOID_NAMES or tuple(settings.parents) != HUMANOID_PARENTS:
			raise ValueError("standing requires the passive humanoid's bone names, parents and order")
		settings.root_force = settings.root_torque = 0
		settings.gravity_factor = 1.0
		settings.flags |= pj.BodyFlag.REPORT
		self.settings = settings
		self.parents = settings.parents
		self.reference = standing_pose(settings) if tuned_stance else settings.rest_rotations
		self.neutral, self.spring, self.limits = project.neutral, project.spring, project.limit_parameters()
		self.feet = np.array([settings.bone_index(name) for name in ("foot_l", "foot_r")])
		self.foot_offsets = np.array([settings.bones[i].shape_offset for i in self.feet])
		self.support_area = support_area
		self.ankle_balance = ankle_balance or residual
		self.residual = residual
		if support_area:
			feet = [settings.bones[i] for i in self.feet]
			if any(foot.shape.kind.name != "BOX" for foot in feet):
				raise ValueError("support-area observations require box foot colliders")
			self.foot_rotations = np.array([foot.shape_rotation for foot in feet])
			half_sizes = np.array([foot.shape.params[:3] for foot in feet]) / 2
			self.sole_corners = half_sizes[:, None] * [[-1, -1, -1], [1, -1, -1], [-1, -1, 1], [1, -1, 1]]
			self.support_scale = float(half_sizes[:, [0, 2]].min())
		self.nonfeet = ~np.isin(np.arange(len(settings)), self.feet)
		self.chest = settings.bone_index("chest")
		self.full_action_shape = ((len(settings) - 1) * 4,)
		self.action_shape = self.full_action_shape
		self.compact = compact
		if compact:
			self.action_bones, self.action_axes = compact_axes(self.limits)
			if not len(self.action_bones):
				raise ValueError("the compact policy needs at least one usable joint axis")
			self.action_shape = (len(self.action_bones),)
		elif residual:
			self.action_shape = (2 * len(self.feet),)
			self.fixed_pose_action = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), len(settings) - 1)
		self.episode_steps = episode_steps
		self.initial_angular_speed = initial_angular_speed
		self.vary_start = vary_start
		self.max_start_tilt = max_start_tilt
		self.tuned_stance = tuned_stance
		self.positions, self.orientations = self._spawn_pose(self.reference)
		self.spawn_height = float(self.positions[0, 1] - settings.rest_translations[0][1])
		self.balance_torques = make_ankle_balance(settings, self.positions, self.orientations, self.feet) if self.ankle_balance else None
		self.world = None

	def _spawn_pose(self, local_turns):
		from pyjoltphysics.ragdolls import forward_kinematics

		positions, orientations = forward_kinematics(self.settings, local_turns, (0, 0, 0), (0, 0, 0, 1))
		lower, _ = collider_bounds(self.settings.bones, positions, orientations)
		positions[:, 1] += 0.001 - lower[:, 1].min()
		return positions, orientations

	def reset(self, *, seed=None):
		import pyjoltphysics as pj

		self.close()
		self.world = pj.World(pj.WorldSettings(tick_hz=TICK_HZ, threads=1, allow_sleeping=False))
		try:
			self.floor, _ = self.world.add_bodies(pj.BodyBatch.boxes([(0, -0.5, 0)], size=(8, 1, 8), motion=pj.MotionType.STATIC, friction=0.8))
			self.handle = self.world.add_ragdoll(self.settings, (0, self.spawn_height, 0), mode=pj.RagdollMode.POWERED)
			self.uids = self.handle.bones[None]
			self.handle.set_strength(np.zeros(len(self.settings)))
			rng = np.random.default_rng(seed)
			if self.vary_start:
				# Tilt the entire pose, preserving the authored local joint angles.
				direction = rng.uniform(0, 2 * np.pi)
				axis = (np.cos(direction), 0, np.sin(direction))
				turns = self.reference.copy()
				turns[0] = Quaternion.multiply_batch(Quaternion.from_axis_rotation(axis, rng.uniform(0, self.max_start_tilt)), turns[0])
				positions, orientations = self._spawn_pose(turns)
				self.world.set_transforms(self.uids.ravel(), positions, orientations)
			elif self.tuned_stance:
				self.world.set_transforms(self.uids.ravel(), self.positions, self.orientations)
			if self.initial_angular_speed:
				angle = rng.uniform(0, 2 * np.pi)
				speed = rng.uniform(0, self.initial_angular_speed) if self.vary_start else self.initial_angular_speed
				spin = speed * np.array([np.cos(angle), 0, np.sin(angle)])
				self.world.set_velocities(self.uids[:, 0], angular=spin)
			self.world.flush()
			self.action = np.zeros(self.full_action_shape, dtype=np.float32)
			self.action[3::4] = 1 if self.compact else -1
			if self.residual:
				self.residual_action = np.zeros(self.action_shape, dtype=np.float32)
				self.residual_effort = 0.0
			self.motor_action = np.zeros((1, len(self.settings), 4))
			self.state = self._observe()
			self.start_height = float(self.state.position[0, 0, 1])
			self.start_center = self.state.center_of_mass[0, [0, 2]]
			self.minimum_height = self.start_height
			self.settled_ticks = 0
			self.peak_drift = 0.0
			self.foot_slip = 0.0
			self.tracking_error = 0.0
			self.support_margin = None
			self.minimum_support_margin = None
			self.first_support_exit = None
			self.done = False
			return self._features(), {}
		except BaseException:
			self.close()
			raise

	def _observe(self):
		return observe_ragdoll(self.world, self.uids, self.settings, self.world.drain_events(), self.reference, self.motor_action, floor=self.floor)

	def _features(self):
		state = self.state
		root, chest = Quaternion.matrix_batch(state.orientation[0, [0, self.chest]])[..., 1]
		joints = state.joint_rotation[0, 1:]
		joints = np.where(joints[:, 3:] < 0, -joints, joints)
		hips = state.position[0, 0]
		height = hips[1] / self.start_height
		feet = (state.position[0, self.feet] - hips) / self.start_height
		center = state.center_of_mass[0, [0, 2]] - self.start_center
		return np.concatenate((
			[height], root, chest, state.linear_velocity[0, 0], state.angular_velocity[0, 0] / 5,
			center * 5, self._support_error() * 5, state.center_of_mass_velocity[0], feet.ravel(), state.linear_velocity[0, self.feet].ravel(),
			joints.ravel(), state.joint_angular_velocity[0, 1:].ravel() / 5, state.contacts[0], self.residual_action if self.residual else self.action,
			((self.support_margin or 0) / self.support_scale,) if self.support_area else (),
		), dtype=np.float32)

	def _support_error(self):
		state = self.state
		axes = Quaternion.matrix_batch(state.orientation[0, self.feet])
		centers = state.position[0, self.feet] + (axes @ self.foot_offsets[..., None])[..., 0]
		return state.center_of_mass[0, [0, 2]] - centers.mean(axis=0)[[0, 2]]

	def _measure_support(self):
		state = self.state
		planted = state.contacts[0, self.feet]
		if not planted.any():
			return None
		centers, turns = collider_poses(state.position[0, self.feet], state.orientation[0, self.feet], self.foot_offsets, self.foot_rotations)
		soles = centers[:, None] + self.sole_corners @ Quaternion.matrix_batch(turns).transpose(0, 2, 1)
		return _support_margin(soles[planted].reshape(-1, 3)[:, [0, 2]], state.center_of_mass[0, [0, 2]])

	def _posture(self):
		height = float(self.state.position[0, 0, 1])
		self.minimum_height = min(self.minimum_height, height)
		upright = Quaternion.matrix_batch(self.state.orientation[0, [0, self.chest]])[:, 1, 1]
		height_ratio = height / self.start_height
		tilt = upright.min()
		speed = np.linalg.norm(self.state.center_of_mass_velocity[0, [0, 2]])
		fallen = bool(self.state.contacts[0, self.nonfeet].any() or height_ratio < 0.7 or tilt < 0.5)
		settled = not fallen and tilt >= 0.95 and height >= 0.9 * self.positions[0, 1] and speed < 0.05 and self.state.contacts[0, self.feet].any()
		self.settled_ticks = self.settled_ticks + 1 if settled else 0
		return height_ratio, upright, speed, fallen

	def step(self, action, *, after_tick=None):
		if self.world is None or self.done:
			raise RuntimeError("reset the standing task before stepping")
		action = np.asarray(action, dtype=np.float32)
		if action.shape != self.action_shape or not np.isfinite(action).all():
			raise ValueError(f"standing action must be finite with shape {self.action_shape}")
		action = np.clip(action, -1, 1)
		residual_torques, residual_change = None, 0.0
		if self.residual:
			turns = action.reshape(len(self.feet), 2)
			turns /= np.maximum(1, np.linalg.norm(turns, axis=1, keepdims=True))
			residual_torques = turns * MAX_RESIDUAL_TORQUE
			residual_change = float(np.mean((action - self.residual_action) ** 2))
			self.residual_action = action
			action = self.fixed_pose_action.copy()
		if self.compact:
			controls = np.zeros((len(self.settings), 4), dtype=np.float32)
			np.add.at(controls[:, :3], self.action_bones, self.action_axes * action[:, None])
			controls[:, 3] = 1
			action = controls[1:].ravel()
		if self.ankle_balance:
			action.reshape(-1, 4)[self.feet - 1] = [0, 0, 0, -1]
		changes = float(np.mean((action - self.action) ** 2))
		self.action = action
		controls = action.reshape(-1, 4)
		self.motor_action[0, 1:, :3] = controls[:, :3] * MAX_OFFSET
		self.motor_action[0, 1:, 3] = (controls[:, 3] + 1) * 0.5
		apply_ragdoll_motors([self.handle], self.reference, self.motor_action, max_offset=MAX_OFFSET, max_strength=STANDING_MOTOR_CAPS)
		for tick in range(1, TICKS_PER_ACTION + 1):
			torque = joint_spring_torques(self.parents, self.state.orientation, self.state.angular_velocity, self.neutral, **self.spring)
			torque += joint_limit_torques(orientations=self.state.orientation, angular_velocity=self.state.angular_velocity, **self.limits)
			if self.balance_torques is not None:
				torque[0] += self.balance_torques(self.state, residual_torques)
			self.world.add_torques(self.uids.ravel(), torque[0])
			self.world.run_ticks(1)
			self.state = self._observe()
			height, upright, speed, fallen = self._posture()
			if after_tick is not None:
				after_tick()
			if fallen:
				break
		elapsed_ticks = self.world.tick
		completed = elapsed_ticks >= self.episode_steps * TICKS_PER_ACTION and not fallen
		self.done = fallen or completed
		foot_speed = self.state.linear_velocity[0, self.feet][:, [0, 2]]
		sliding = (np.linalg.norm(foot_speed, axis=1) * self.state.contacts[0, self.feet]).mean()
		drift = np.linalg.norm(self._support_error())
		if self.support_area:
			self.support_margin = self._measure_support()
			drift = 0.0
			if self.support_margin is not None:
				if self.minimum_support_margin is None or self.support_margin < self.minimum_support_margin:
					self.minimum_support_margin = self.support_margin
				if self.first_support_exit is None and self.support_margin < 0:
					self.first_support_exit = elapsed_ticks / TICK_HZ
				# The quiet-standing cost reaches its cap at the support boundary.
				drift = 0.2 * np.clip(1 - 4 * self.support_margin / self.support_scale, 0, 1)
		self.peak_drift = max(self.peak_drift, np.linalg.norm(self.state.center_of_mass[0, [0, 2]] - self.start_center))
		self.foot_slip += sliding * tick / TICK_HZ
		alignment = np.abs(np.einsum("ij,ij->i", self.reference[1:], self.state.joint_rotation[0, 1:]))
		self.tracking_error += np.arccos(alignment.clip(0, 1)).mean() * 2 * tick
		reward = 0.5 * min(height, 1.0) + 0.5 * upright.mean() - 0.6 * min(drift, 1) - 0.25 * min(speed, 1)
		reward -= 0.05 * sliding + 0.02 * changes + 0.005 * residual_change
		if fallen:
			reward -= 5
		info = dict(elapsed_seconds=elapsed_ticks / TICK_HZ, minimum_pelvis_m=self.minimum_height, success=completed,
			settled_success=completed and self.settled_ticks >= TICK_HZ * SETTLE_SECONDS, peak_drift_m=self.peak_drift,
			mean_foot_slip_m=self.foot_slip, mean_reference_error_rad=self.tracking_error / elapsed_ticks)
		if self.support_area:
			info["minimum_support_margin_m"] = self.minimum_support_margin
			info["first_support_exit_seconds"] = self.first_support_exit
		if self.residual:
			self.residual_effort += float(np.linalg.norm(residual_torques, axis=1).mean()) * tick
			info["mean_residual_command_nm"] = self.residual_effort / elapsed_ticks
		return self._features(), float(reward), fallen, completed, info

	def close(self):
		if self.world is not None:
			self.world.stop()
			self.world = None


def evaluate(task, policy, *, seed, episodes):
	if episodes < 1:
		raise ValueError("evaluation requires at least one episode")
	results = []
	for i in range(episodes):
		observation, _ = task.reset(seed=seed + i)
		total_reward = 0.0
		while True:
			observation, reward, fallen, completed, info = task.step(policy(observation))
			total_reward += reward
			if fallen or completed:
				results.append(info | {"reward": total_reward})
				break
	report = {}
	for key in results[0]:
		values = [result[key] for result in results if result[key] is not None]
		report[key] = float(np.mean(values)) if values else None
	return report


def _standing_env(project, task_options):
	import gymnasium as gym

	class StandingEnv(gym.Env):
		def __init__(self):
			task = StandingTask(project, **task_options)
			try:
				observation, _ = task.reset(seed=0)
			finally:
				task.close()
			self.task = task
			self.observation_space = gym.spaces.Box(-np.inf, np.inf, observation.shape, np.float32)
			self.action_space = gym.spaces.Box(-1, 1, task.action_shape, np.float32)

		def reset(self, *, seed=None, options=None):
			super().reset(seed=seed)
			return self.task.reset(seed=int(self.np_random.integers(2**32)))

		def step(self, action):
			return self.task.step(action)

		def close(self):
			self.task.close()

	return StandingEnv()


def train_policy(project, *, steps, seed, workers=1, resume=None, **task_options):
	import torch
	from stable_baselines3 import PPO
	from stable_baselines3.common.env_util import make_vec_env
	from stable_baselines3.common.vec_env import SubprocVecEnv

	if workers < 1:
		raise ValueError("workers must be positive")
	factory = partial(_standing_env, project, task_options)
	parallel = dict(vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "spawn"}) if workers > 1 else {}
	environment = make_vec_env(factory, n_envs=workers, seed=seed, **parallel)
	try:
		# Keep complete minibatches when changing the number of workers.
		batch_size = 128
		n_steps = batch_size * max(1, 8 // workers)
		options = dict(seed=seed, device="cpu", n_steps=n_steps, batch_size=batch_size)
		if resume is None:
			model = PPO("MlpPolicy", environment, gamma=0.997, target_kl=0.03, learning_rate=0.00015,
				policy_kwargs={"log_std_init": -2.5}, verbose=1, **options)
			with torch.no_grad():
				if task_options.get("residual"):
					model.policy.action_net.weight.zero_()
					model.policy.action_net.bias.zero_()
				elif not task_options.get("compact"):
					model.policy.action_net.bias[3::4].fill_(1 if task_options.get("tuned_stance") else 0.8)
		else:
			model = PPO.load(resume, env=environment, **options)
		return model.learn(total_timesteps=steps, reset_num_timesteps=resume is None)
	finally:
		environment.close()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--tuning", type=Path)
	parser.add_argument("--seconds", type=int, default=5, help="time limit per episode")
	parser.add_argument("--initial-angular-speed", type=float, default=0, help="starting pelvis spin in rad/s about a seeded horizontal axis")
	parser.add_argument("--vary-start", action="store_true", help="vary start tilt and sample spin between zero and the requested speed")
	parser.add_argument("--max-start-tilt", type=float, help=f"maximum tilt in radians with --vary-start (default {DEFAULT_START_TILT})")
	parser.add_argument("--stance", choices=("rest", "tuned"), default="rest", help="initial pose and motor reference")
	parser.add_argument("--action-space", choices=("full", "compact", "residual"), default="full", help="choose full or compact motor targets, or learn bounded corrections to ankle feedback")
	parser.add_argument("--support-area", action="store_true", help="observe and reward support margin estimated from contacting box soles")
	parser.add_argument("--ankle-balance", action="store_true", help="also evaluate a joint-only ankle torque feedback baseline")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--eval-seed", type=int, help="first seed for the evaluation starts")
	parser.add_argument("--episodes", type=int, default=5, help="number of evaluation episodes")
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("--steps", type=int, help="train PPO for this many policy steps")
	mode.add_argument("--checkpoint", type=Path, help="evaluate a saved Stable-Baselines3 checkpoint")
	parser.add_argument("--resume", type=Path, help="resume training from a saved PPO checkpoint")
	parser.add_argument("--workers", type=int, default=1, help="number of parallel Jolt environments during training")
	parser.add_argument("--output", type=Path, help="save the trained policy to this .zip file")
	args = parser.parse_args()
	if args.seconds < 1 or args.episodes < 1 or not 0 <= args.seed < 2**32 or (args.steps is not None and args.steps < 1):
		parser.error("seconds, episodes and training steps must be positive and seed must fit uint32")
	if args.eval_seed is not None and not 0 <= args.eval_seed < 2**32:
		parser.error("evaluation seed must fit uint32")
	if args.max_start_tilt is not None and not args.vary_start:
		parser.error("--max-start-tilt requires --vary-start")
	if args.output is not None and (args.steps is None or args.output.suffix.lower() != ".zip"):
		parser.error("--output requires --steps and a .zip path")
	if args.workers < 1 or (args.steps is None and (args.resume is not None or args.workers != 1)):
		parser.error("workers must be positive, and --resume and parallel workers require --steps")
	try:
		project = default_project() if args.tuning is None else read_project(args.tuning)
		options = dict(episode_steps=args.seconds * CONTROL_HZ, initial_angular_speed=args.initial_angular_speed,
			tuned_stance=args.stance == "tuned", vary_start=args.vary_start, support_area=args.support_area)
		if args.max_start_tilt is not None:
			options["max_start_tilt"] = args.max_start_tilt
		task = StandingTask(project, compact=args.action_space == "compact", residual=args.action_space == "residual", **options)
		balance_task = StandingTask(project, ankle_balance=True, **options) if args.ankle_balance or task.residual else None
	except (OSError, ValueError) as error:
		parser.error(str(error))
	try:
		if args.steps is not None or args.checkpoint is not None:
			try:
				import torch
				from stable_baselines3 import PPO
			except ImportError as error:
				parser.error(f"{error}. Install stable-baselines3 to train or load a PPO policy")
			torch.set_num_threads(1)
		evaluation = dict(seed=args.seed + 10_000 if args.eval_seed is None else args.eval_seed, episodes=args.episodes)
		baselines = {}
		baseline_task = StandingTask(project, **options) if task.compact or task.residual else task
		try:
			for name, strength in (("passive", -1), ("tracking", 1)):
				action = np.zeros(baseline_task.action_shape, dtype=np.float32)
				action[3::4] = strength
				baselines[name] = evaluate(baseline_task, lambda _: action, **evaluation)
			if balance_task is not None:
				baselines["ankle_balance"] = evaluate(balance_task, lambda _: action, **evaluation)
		finally:
			baseline_task.close()
		print(json.dumps(baselines, indent=2))
		if args.steps is not None or args.checkpoint is not None:
			if args.steps is not None:
				model = train_policy(project, steps=args.steps, seed=args.seed, workers=args.workers, resume=args.resume, compact=task.compact, residual=task.residual, **options)
			else:
				model = PPO.load(args.checkpoint, device="cpu")
				if model.action_space.shape != task.action_shape or model.observation_space.shape != task.reset(seed=evaluation["seed"])[0].shape:
					parser.error("checkpoint observation or action shape does not match the task; check --action-space and --support-area")
			if args.output is not None:
				model.save(args.output)
			print(json.dumps({"ppo": evaluate(task, lambda obs: model.predict(obs, deterministic=True)[0], **evaluation)}, indent=2))
	finally:
		task.close()
		if balance_task is not None:
			balance_task.close()


if __name__ == "__main__":
	main()
