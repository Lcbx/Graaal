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
PUSH_SECONDS = 2
PUSH_TICK = PUSH_SECONDS * TICK_HZ
PUSH_BONES = ("pelvis", "chest")
PUSH_DIRECTIONS = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}
HARD_BACKWARD_PUSHES = (("chest", 8), ("chest", 12), ("pelvis", 12))
HIP_OFFSET = 0.45
KNEE_FORWARD_OFFSET = 0.55
KNEE_BACKWARD_OFFSET = 0.05
ARM_OFFSET = 0.55
LEG_CHANNELS = 6
LANDING_BRAKE_TICKS = TICK_HZ // 5
RECOVERY_STEP_DISTANCE = 0.02
DIRECT_REWARD_TERMS = ("braking", "support", "compliance")
DIRECT_REWARD_KEYS = tuple(f"{term}_reward" for term in (*DIRECT_REWARD_TERMS, "effort"))
BACKSTEP_CHEST_MIN_IMPULSE = 10
BACKSTEP_ACCELERATION_LIMIT = 8.0
BACKSTEP_IMPACT_KEYS = ("acceleration_penalty", "floor_impulse_penalty", "peak_com_acceleration_m_s2", "peak_floor_impulse_ns")
PPO_GAMMA = 0.997
POSE_PELVIS_UP = np.cos(np.deg2rad(15))
POSE_CHEST_UP = np.cos(np.deg2rad(18))
POSE_JOINT_RMS = np.deg2rad(15)
POSE_JOINT_MAX = np.deg2rad(30)
POSE_FALL_TICKS = TICK_HZ // 4


def compact_axes(limits):
	"""Select constraint-frame axes with enough travel for standing corrections."""
	travel = np.column_stack((np.diff(limits["twist"], axis=1), 2 * limits["swing"]))
	# An axis needs at least one full offset's worth of travel across both sides.
	bones, axes = np.nonzero(travel[1:] >= MAX_OFFSET)
	return bones + 1, Quaternion.matrix_batch(limits["child_frames"])[bones + 1, :, axes]


def _support_edges(points):
	left, right = np.triu_indices(len(points), k=1)
	start = points[left]
	edges = points[right] - start
	normals = edges[:, [1, 0]] * [-1, 1]
	sides = np.einsum("evi,ei->ev", points - start[:, None], normals)
	positive = sides.max(axis=1) > 1e-9
	negative = sides.min(axis=1) < -1e-9
	return start, edges, normals, positive, positive ^ negative


def _support_margin(points, center):
	"""Return the signed distance to the boundary of a planar convex hull."""
	start, edges, normals, positive, boundary = _support_edges(points)
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

	def __init__(self, project, *, episode_steps=5 * CONTROL_HZ, initial_angular_speed=0.0, tuned_stance=False, compact=False, vary_start=False, max_start_tilt=DEFAULT_START_TILT, support_area=False, ankle_balance=False, residual=False, stepping=False, whole_body=False, push_impulses=(), recovery=False, recovery_curriculum=False, replay_start_fraction=0.0, direct=False, without_direct_reward=(), backstep=False, backstep_swing=True, strong_backstep_reference=False, pose_recovery=False, release_backstep_hips=False, backward_chest_pushes=False, touchdown_support_reward=False, adaptive_step=False):
		import pyjoltphysics as pj

		recovery = recovery or adaptive_step
		if direct and (not tuned_stance or any((compact, ankle_balance, residual, stepping, whole_body, recovery, backstep))):
			raise ValueError("direct motor control requires the tuned stance and its own action space")
		disabled_rewards = set(without_direct_reward)
		if disabled_rewards - set(DIRECT_REWARD_TERMS) or (disabled_rewards and not direct):
			raise ValueError("only direct motor control can disable braking, support, or compliance rewards")
		if backstep and recovery:
			raise ValueError("a backstep reference and recovery requests use different controllers")
		if strong_backstep_reference and not backstep:
			raise ValueError("the strong backstep reference requires backstep control")
		if release_backstep_hips and not backstep:
			raise ValueError("releasing the hip targets requires backstep control")
		if touchdown_support_reward and not backstep:
			raise ValueError("touchdown support reward requires backstep control")
		whole_body = whole_body or recovery or backstep
		support_area = support_area or recovery or direct or backstep
		stepping = stepping or whole_body
		if episode_steps < 1 or not np.isfinite(initial_angular_speed) or initial_angular_speed < 0:
			raise ValueError("episode steps must be positive and initial angular speed must be finite and nonnegative")
		if not np.isfinite(max_start_tilt) or not 0 <= max_start_tilt <= np.pi:
			raise ValueError("maximum start tilt must be in [0, pi] radians")
		if compact and (residual or stepping):
			raise ValueError("compact and residual actions are mutually exclusive")
		if stepping and not tuned_stance:
			raise ValueError("hip and knee targets require the tuned stance")
		self.push_impulses = tuple(push_impulses)
		if (self.push_impulses or recovery_curriculum) and (episode_steps * TICKS_PER_ACTION <= PUSH_TICK or not all(0 <= value <= np.finfo(np.float32).max for value in self.push_impulses)):
			raise ValueError("push training needs an episode longer than two seconds and finite nonnegative impulses")
		if recovery_curriculum and self.push_impulses:
			raise ValueError("the recovery curriculum replaces explicit push impulses")
		if not 0 <= replay_start_fraction <= 1 or (replay_start_fraction and (not recovery or not recovery_curriculum or episode_steps < 4 * CONTROL_HZ)):
			raise ValueError("replayed starts require recovery actions, the recovery curriculum, at least four seconds, and a fraction in [0, 1]")
		self.recovery_curriculum = recovery_curriculum
		self.replay_start_fraction = replay_start_fraction
		self.direct = direct
		self.backstep = backstep
		self.touchdown_support_reward = touchdown_support_reward
		self.backward_chest_pushes = backstep or backward_chest_pushes
		self.backstep_swing = backstep_swing
		self.strong_backstep_reference = strong_backstep_reference
		self.pose_recovery = pose_recovery
		self.release_backstep_hips = release_backstep_hips
		if backstep:
			from tools.ragdoll_recovery_step import BackstepReference

			self.backstep_type = BackstepReference
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
		self.residual = residual or stepping
		self.ankle_balance = ankle_balance or self.residual
		self.stepping = stepping
		self.whole_body = whole_body
		if support_area:
			feet = [settings.bones[i] for i in self.feet]
			if any(foot.shape.kind.name != "BOX" for foot in feet):
				raise ValueError("support-area observations require box foot colliders")
			self.foot_rotations = np.array([foot.shape_rotation for foot in feet])
			self.foot_half_sizes = np.array([foot.shape.params[:3] for foot in feet]) / 2
			self.sole_corners = self.foot_half_sizes[:, None] * [[-1, -1, -1], [1, -1, -1], [-1, -1, 1], [1, -1, 1]]
			self.support_scale = float(self.foot_half_sizes[:, [0, 2]].min())
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
		elif self.residual:
			self.ankle_channels = 2 * len(self.feet)
			self.action_shape = (self.ankle_channels + (LEG_CHANNELS if stepping else 0),)
			self.fixed_pose_action = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), len(settings) - 1)
		self.motor_offset_caps = MAX_OFFSET
		self.motor_offset_scale = MAX_OFFSET
		if direct:
			self.action_bones, self.action_axes = compact_axes(self.limits)
			if not len(self.action_bones):
				raise ValueError("direct motor control needs at least one usable joint axis")
			self.action_shape = (len(self.action_bones),)
			self.motor_offset_caps = np.full(len(settings), MAX_OFFSET)
			arms = settings.bone_index("upper_arm_l")
			legs = settings.bone_index("thigh_l")
			self.landing_joints = np.array([[settings.bone_index(f"{part}_{side}") for part in ("thigh", "shin")] for side in ("l", "r")])
			self.motor_offset_caps[arms:legs] = ARM_OFFSET
			self.motor_offset_caps[legs:] = HIP_OFFSET
			self.motor_offset_caps[self.landing_joints[:, 1]] = KNEE_FORWARD_OFFSET
			self.motor_offset_scale = self.motor_offset_caps[1:, None]
			self.end_effectors = np.r_[self.feet, [settings.bone_index(name) for name in ("lower_arm_l", "lower_arm_r")]]
			self.direct_reward_weights = np.array([term not in disabled_rewards for term in (*DIRECT_REWARD_TERMS, "effort")])
		if stepping:
			self.hips = np.array([settings.bone_index(name) for name in ("thigh_l", "thigh_r")])
			self.knees = np.array([settings.bone_index(name) for name in ("shin_l", "shin_r")])
			self.motor_offset_caps = np.full(len(settings), MAX_OFFSET)
			self.motor_offset_caps[self.hips] = HIP_OFFSET
			self.motor_offset_caps[self.knees] = KNEE_FORWARD_OFFSET
			self.motor_offset_scale = self.motor_offset_caps[1:, None]
		if whole_body:
			bones, axes = compact_axes(self.limits)
			first_hip = self.hips[0]
			upper = bones < first_hip
			self.upper_bones, self.upper_axes = bones[upper], axes[upper]
			self.motor_offset_caps[settings.bone_index("upper_arm_l"):first_hip] = ARM_OFFSET
			self.action_shape = (self.action_shape[0] + len(self.upper_bones),)
		if backstep:
			self.motor_offset_caps[self.hips] = 1.0
			self.motor_offset_caps[self.knees] = 1.2
			self.backstep_zero = np.zeros(self.action_shape, dtype=np.float32)
			self.floor_impulse_limit = 0.22 * sum(bone.mass for bone in settings.bones) * 9.81 / TICK_HZ
			self.contact_kind = pj.EventKind.CONTACT
			self.contact_phases = (pj.EventPhase.BEGIN, pj.EventPhase.PERSIST)
		self.episode_steps = episode_steps
		self.initial_angular_speed = initial_angular_speed
		self.vary_start = vary_start
		self.max_start_tilt = max_start_tilt
		self.tuned_stance = tuned_stance
		self.positions, self.orientations = self._spawn_pose(self.reference)
		if pose_recovery:
			self.target_up = Quaternion.matrix_batch(self.orientations[[0, self.chest]])[:, :, 1]
		self.spawn_height = float(self.positions[0, 1] - settings.rest_translations[0][1])
		self.balance_torques = make_ankle_balance(settings, self.positions, self.orientations, self.feet) if self.ankle_balance else None
		self.recovery = None
		if recovery:
			from tools.ragdoll_recovery_step import RECOVERY_CHANNELS, AdaptiveRecoveryStep, RecoveryStep

			self.motor_channels = self.action_shape[0]
			self.action_shape = (self.motor_channels + RECOVERY_CHANNELS,)
			self.recovery = (AdaptiveRecoveryStep if adaptive_step else RecoveryStep)(self)
		self.world = None

	def _spawn_pose(self, local_turns):
		from pyjoltphysics.ragdolls import forward_kinematics

		positions, orientations = forward_kinematics(self.settings, local_turns, (0, 0, 0), (0, 0, 0, 1))
		lower, _ = collider_bounds(self.settings.bones, positions, orientations)
		positions[:, 1] += 0.001 - lower[:, 1].min()
		return positions, orientations

	def reset(self, *, seed=None, replay=False):
		observation, info = self._reset(seed, replay=replay)
		try:
			if "replay_start" in info:
				if self._replay_start(info["replay_start"]):
					return self._features(), info
				return self._reset(seed, replay=False)
			if replay and self.backstep and self.push is not None and self.backstep_type.accepts(self, *self.push):
				# The residual cannot act before the reference has swung the foot.
				while not self.done and (self.backstep_reference is None or not self.backstep_reference.residual_active()):
					observation, _, _, _, _ = self.step(self.backstep_zero)
				if self.done:
					return self._reset(seed, replay=False)
			return observation, info
		except BaseException:
			self.close()
			raise

	def _reset(self, seed, *, replay):
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
			self.push = None
			self.backstep_reference = None
			self.backstep_settled = False
			if self.backstep:
				self.backstep_push_tick = None
				self.backstep_stance_ticks = 0
				self.backstep_impact_cost = np.zeros(2)
				self.backstep_peak_impact = np.zeros(2)
				self.touchdown_tick = -LANDING_BRAKE_TICKS
				self.touchdown_support_total = 0.0
			info = {}
			if self.recovery_curriculum:
				if replay and self.replay_start_fraction and rng.random() < self.replay_start_fraction:
					info["replay_start"] = "before_landing" if rng.random() < 0.5 else "after_landing"
					bone, magnitude = ("chest", 8) if rng.random() < 0.5 else ("chest", 12)
					self.push = (self.settings.bone_index(bone), np.array([0, 0, -magnitude]))
				else:
					self.push = self._sample_recovery_push(rng)
			elif self.push_impulses:
				magnitude = rng.choice(self.push_impulses)
				if magnitude:
					# Give the disturbance that toppled the baseline extra practice.
					bone, direction = ("chest", "-Z") if self.backward_chest_pushes or rng.random() < 0.5 else (rng.choice(PUSH_BONES), rng.choice(tuple(PUSH_DIRECTIONS)))
					self.push = (self.settings.bone_index(bone), magnitude * np.asarray(PUSH_DIRECTIONS[direction]))
			self.world.flush()
			self.action = np.zeros(self.full_action_shape, dtype=np.float32)
			self.action[3::4] = 1 if self.compact else -1
			if self.residual:
				self.residual_action = np.zeros(self.ankle_channels, dtype=np.float32)
				self.residual_effort = 0.0
			self.motor_action = np.zeros((1, len(self.settings), 4))
			self.state = self._observe()
			self.start_height = float(self.state.position[0, 0, 1])
			self.start_center = self.state.center_of_mass[0, [0, 2]]
			self.minimum_height = self.start_height
			self.settled_ticks = 0
			self.pose_push_tick = None
			self.pose_hold_ticks = self.toppled_ticks = 0
			self.peak_drift = 0.0
			self.foot_slip = 0.0
			self.tracking_error = 0.0
			self.support_margin = None
			self.minimum_support_margin = None
			self.first_support_exit = None
			self.done = False
			if self.direct:
				self.direct_action = np.zeros(self.action_shape, dtype=np.float32)
				self.push_tick = None
				self.landing_tick = -LANDING_BRAKE_TICKS
				self.relocated_tick = -LANDING_BRAKE_TICKS
				self.relocated_feet = np.zeros(len(self.feet), dtype=bool)
				self.direct_rewards = np.zeros(len(DIRECT_REWARD_KEYS))
				self.target_saturation = 0
				self.max_foot_clearance = np.zeros(len(self.feet))
				self._measure_direct()
			if self.recovery is not None:
				self.recovery.reset()
			return self._features(), info
		except BaseException:
			self.close()
			raise

	def _sample_recovery_push(self, rng):
		if self.backward_chest_pushes:
			return self.settings.bone_index("chest"), np.array([0, 0, -rng.uniform(11, 13)])
		draw = rng.random()
		if draw < 0.20:
			return None
		bone = rng.choice(PUSH_BONES)
		direction = rng.choice(tuple(PUSH_DIRECTIONS))
		if draw < 0.55:
			magnitude = rng.uniform(0.5, 4)
		elif draw < 0.80:
			bone, direction = (bone if self.direct else "chest"), "-Z"
			magnitude = rng.uniform(8, 12) if self.direct else rng.uniform(4, 8)
		else:
			bone, magnitude = HARD_BACKWARD_PUSHES[rng.integers(len(HARD_BACKWARD_PUSHES))]
			direction = "-Z"
		return self.settings.bone_index(bone), magnitude * np.asarray(PUSH_DIRECTIONS[direction])

	def _replay_start(self, window):
		"""Advance a fresh native world to a swing or contact under a guided action."""
		action = np.zeros(self.action_shape, dtype=np.float32)
		limit = PUSH_TICK + TICK_HZ // 2
		step = self.recovery
		while self.world.tick < limit:
			if self.world.tick == PUSH_TICK + 18:
				action[self.motor_channels:] = [1, -0.25, -0.5, 0.2, -0.29, 0.24]
				step.request(action[self.motor_channels:], guided=True)
			_, _, fallen, completed, _ = self.step(action)
			if fallen or completed:
				return False
			if window == "before_landing" and step.phase == 1 and step.airborne and self.world.tick - step.started >= 0.67 * step.duration * TICK_HZ:
				return True
			if window == "after_landing" and step.phase == 2 and step.landing_tick is not None and self.world.tick - step.loaded >= 2:
				return True
		return False

	def apply_push(self, bone, impulse):
		self.world.add_impulses(self.uids[:, bone], [impulse])
		self.settled_ticks = 0
		if self.pose_recovery:
			self.pose_push_tick = self.world.tick
			self.pose_hold_ticks = self.toppled_ticks = 0
		if self.recovery is not None:
			self.recovery.pushed = True
		if self.backstep:
			self.backstep_reference = self.backstep_type.start_for_push(self, bone, impulse)
			self.backstep_push_tick = self.world.tick
			self.backstep_stance_ticks = 0
			self.touchdown_contact = self.state.contacts[0, self.feet].copy()
			self.touchdown_tick = -LANDING_BRAKE_TICKS
		if self.direct:
			self.push_tick = self.world.tick
			self.landing_tick = self.relocated_tick = -LANDING_BRAKE_TICKS
			self.relocated_feet.fill(False)
			self.last_contact_xz = self.state.position[0, self.feet][:, [0, 2]]
			self.seen_contact = self.state.contacts[0, self.feet]

	def _observe(self):
		self.events = self.world.drain_events()
		return observe_ragdoll(self.world, self.uids, self.settings, self.events, self.reference, self.motor_action, floor=self.floor)

	def _features(self):
		if self.direct:
			return self._direct_features()
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
			self.recovery.features() if self.recovery is not None else (),
			(self.backstep_reference.progress(),) if self.backstep_reference is not None else ((0,) if self.backstep else ()),
		), dtype=np.float32)

	def _measure_direct(self):
		state = self.state
		centers, turns = collider_poses(state.position[0, self.feet], state.orientation[0, self.feet], self.foot_offsets, self.foot_rotations)
		self.sole_axes = Quaternion.matrix_batch(turns)
		self.sole_centers = centers
		self.foot_clearance = centers[:, 1] - (np.abs(self.sole_axes[:, 1]) * self.foot_half_sizes).sum(axis=1)
		self.body_axes = Quaternion.matrix_batch(state.orientation[0, [0, self.chest]])
		x, z = self.body_axes[0, [0, 2], 2]
		length = np.hypot(x, z)
		self.heading = np.array([[z, 0, x], [0, length, 0], [-x, 0, z]]) / length if length > 1e-6 else np.eye(3)
		self.support_center = centers[state.contacts[0, self.feet]].mean(axis=0) if state.contacts[0, self.feet].any() else centers.mean(axis=0)
		self.capture = state.center_of_mass[0].copy()
		self.capture[[0, 2]] += state.center_of_mass_velocity[0, [0, 2]] * np.sqrt(max(self.capture[1], 0) / 9.81)
		points = self._support_points()
		self.capture_margin = _support_margin(points, self.capture[[0, 2]]) if points is not None else None
		self.max_foot_clearance = np.maximum(self.max_foot_clearance, self.foot_clearance)

	def _direct_features(self):
		state = self.state
		hip = state.position[0, 0]
		ends = self.end_effectors
		joints = state.joint_rotation[0, 1:]
		joints = np.where(joints[:, 3:] < 0, -joints, joints)
		return np.concatenate((
			[hip[1] / self.start_height], (self.body_axes[:, :, 1] @ self.heading).ravel(),
			state.linear_velocity[0, 0] @ self.heading, (state.angular_velocity[0, [0, self.chest]] @ self.heading).ravel() / 5,
			(state.center_of_mass[0] - hip) @ self.heading, ((state.center_of_mass[0] - self.support_center) @ self.heading)[[0, 2]] * 5,
			state.center_of_mass_velocity[0] @ self.heading, ((self.capture - self.support_center) @ self.heading)[[0, 2]] * 5,
			((state.position[0, ends] - hip) @ self.heading).ravel() / self.start_height,
			((state.linear_velocity[0, ends] - state.linear_velocity[0, 0]) @ self.heading).ravel(),
			self.foot_clearance * 5, (self.sole_axes[:, :, 1] @ self.heading).ravel(),
			joints.ravel(), state.joint_angular_velocity[0, 1:].ravel() / 5, state.contacts[0],
			[(self.support_margin or 0) / self.support_scale, (self.capture_margin or 0) / self.support_scale], self.direct_action,
		), dtype=np.float32)

	def _support_error(self):
		state = self.state
		axes = Quaternion.matrix_batch(state.orientation[0, self.feet])
		centers = state.position[0, self.feet] + (axes @ self.foot_offsets[..., None])[..., 0]
		return state.center_of_mass[0, [0, 2]] - centers.mean(axis=0)[[0, 2]]

	def _support_points(self):
		state = self.state
		planted = state.contacts[0, self.feet]
		if not planted.any():
			return None
		if self.direct:
			soles = self.sole_centers[:, None] + self.sole_corners @ self.sole_axes.transpose(0, 2, 1)
		elif self.recovery is None:
			centers, turns = collider_poses(state.position[0, self.feet], state.orientation[0, self.feet], self.foot_offsets, self.foot_rotations)
			soles = centers[:, None] + self.sole_corners @ Quaternion.matrix_batch(turns).transpose(0, 2, 1)
		else:
			soles = self.recovery.soles
		return soles[planted].reshape(-1, 3)[:, [0, 2]]

	def _measure_support(self):
		points = self._support_points()
		return _support_margin(points, self.state.center_of_mass[0, [0, 2]]) if points is not None else None

	def _posture(self):
		height = float(self.state.position[0, 0, 1])
		self.minimum_height = min(self.minimum_height, height)
		up = Quaternion.matrix_batch(self.state.orientation[0, [0, self.chest]])[:, :, 1]
		upright = up[:, 1]
		height_ratio = height / self.start_height
		tilt = upright.min()
		speed = np.linalg.norm(self.state.center_of_mass_velocity[0, [0, 2]])
		fallen = bool(self.state.contacts[0, self.nonfeet].any() or height_ratio < 0.7 or tilt < 0.5)
		if self.pose_push_tick is not None:
			fallen = self._recovery_posture(up, height / self.positions[0, 1])
		standing = not fallen and tilt >= 0.95 and height >= 0.9 * self.positions[0, 1] and self.state.contacts[0, self.feet].any()
		settled = standing and speed < 0.05
		self.settled_ticks = self.settled_ticks + 1 if settled else 0
		if self.backstep:
			self.backstep_stance_ticks = self.backstep_stance_ticks + 1 if standing else 0
		return height_ratio, upright, speed, fallen

	def _recovery_posture(self, up, height_ratio):
		state = self.state
		pelvis_up, chest_up = np.einsum("ij,ij->i", up, self.target_up)
		speed = np.linalg.norm(state.center_of_mass_velocity[0])
		spin = np.linalg.norm(state.angular_velocity[0, 0])
		trunk_contact = state.contacts[0, [0, self.chest]].any()
		toppled = (height_ratio < 0.7 or pelvis_up < 0.5) and (trunk_contact or (speed < 0.15 and spin < 0.5))
		self.toppled_ticks = self.toppled_ticks + 1 if toppled else 0
		settled = (0.9 <= height_ratio <= 1.1 and pelvis_up >= POSE_PELVIS_UP and chest_up >= POSE_CHEST_UP
			and speed < 0.05 and spin < 0.2 and state.contacts[0, self.feet].any() and not state.contacts[0, self.nonfeet].any())
		if settled:
			alignment = np.abs(np.einsum("ij,ij->i", self.reference[1:], state.joint_rotation[0, 1:]))
			error = 2 * np.arccos(alignment.clip(0, 1))
			settled = error.max() <= POSE_JOINT_MAX and np.mean(error ** 2) <= POSE_JOINT_RMS ** 2
		self.pose_hold_ticks = self.pose_hold_ticks + 1 if settled else 0
		return self.toppled_ticks >= POSE_FALL_TICKS

	def _backstep_impact(self, previous_velocity):
		# The imposed push is the disturbance. Its first native tick includes the external impulse.
		if self.backstep_push_tick is not None and self.world.tick == self.backstep_push_tick + 1:
			return 0.0, 0.0
		acceleration = np.linalg.norm(self.state.center_of_mass_velocity[0] - previous_velocity) * TICK_HZ
		events = self.events
		floor = (events["a"] == self.floor[0]) | (events["b"] == self.floor[0])
		active = floor & (events["kind"] == self.contact_kind) & np.isin(events["phase"], self.contact_phases)
		return acceleration, np.max(events["impulse"][active], initial=0)

	def step(self, action, *, after_tick=None, edit_motors=None, joint_feedback=None):
		if self.world is None or self.done:
			raise RuntimeError("reset the standing task before stepping")
		action = np.asarray(action, dtype=np.float32)
		if action.shape != self.action_shape or not np.isfinite(action).all():
			raise ValueError(f"standing action must be finite with shape {self.action_shape}")
		action = np.clip(action, -1, 1)
		if self.backstep:
			if self.push is not None and self.world.tick == PUSH_TICK:
				self.apply_push(*self.push)
			action = self.backstep_zero if self.backstep_reference is None else self.backstep_reference.learned_action(action)
			if self.backstep_reference is not None:
				edit_motors = self.backstep_reference.edit_motors
				joint_feedback = self.backstep_reference.feedback
		if self.direct:
			if edit_motors is not None or joint_feedback is not None:
				raise ValueError("direct motor control only accepts policy targets")
			previous_speed = float(np.linalg.norm(self.state.center_of_mass_velocity[0, [0, 2]]))
			previous_support, previous_capture = self.support_margin, self.capture_margin
			self.direct_action = action
			self.target_saturation += np.count_nonzero(np.abs(action) >= 0.98)
			planted = self.state.contacts[0, self.feet]
		if self.recovery is not None:
			self.recovery.request(action[self.motor_channels:])
			action = action[:self.motor_channels]
			edit_motors = self.recovery.edit_motors
			joint_feedback = self.recovery.feedback
		residual_torques, residual_change = None, 0.0
		if self.residual:
			ankle_action = action[:self.ankle_channels]
			turns = ankle_action.reshape(len(self.feet), 2)
			turns /= np.maximum(1, np.linalg.norm(turns, axis=1, keepdims=True))
			residual_torques = turns * MAX_RESIDUAL_TORQUE
			residual_change = float(np.mean((ankle_action - self.residual_action) ** 2))
			self.residual_action = ankle_action
			pose_action = action[self.ankle_channels:]
			action = self.fixed_pose_action.copy()
			if self.stepping:
				controls = action.reshape(-1, 4)
				hips = pose_action[:4].reshape(2, 2)
				hips = hips / np.maximum(1, np.linalg.norm(hips, axis=1, keepdims=True))
				controls[np.ix_(self.hips - 1, (0, 2))] = hips
				knees = pose_action[4:LEG_CHANNELS]
				controls[self.knees - 1, 0] = np.where(knees < 0, knees * (KNEE_BACKWARD_OFFSET / KNEE_FORWARD_OFFSET), knees)
				if self.whole_body:
					upper = controls[:self.hips[0] - 1, :3]
					np.add.at(upper, self.upper_bones - 1, self.upper_axes * pose_action[LEG_CHANNELS:, None])
					upper /= np.maximum(1, np.linalg.norm(upper, axis=1, keepdims=True))
		if self.compact or self.direct:
			controls = np.zeros((len(self.settings), 4), dtype=np.float32)
			np.add.at(controls[:, :3], self.action_bones, self.action_axes * action[:, None])
			if self.direct:
				controls[:, :3] /= np.maximum(1, np.linalg.norm(controls[:, :3], axis=1, keepdims=True))
			controls[:, 3] = 1
			action = controls[1:].ravel()
		if self.ankle_balance:
			action.reshape(-1, 4)[self.feet - 1] = [0, 0, 0, -1]
		changes = float(np.mean((action - self.action) ** 2))
		self.action = action
		controls = action.reshape(-1, 4)
		self.motor_action[0, 1:, :3] = controls[:, :3] * self.motor_offset_scale
		self.motor_action[0, 1:, 3] = (controls[:, 3] + 1) * 0.5
		if edit_motors is not None:
			edit_motors(self.motor_action)
		apply_ragdoll_motors([self.handle], self.reference, self.motor_action, max_offset=self.motor_offset_caps, max_strength=STANDING_MOTOR_CAPS)
		if not self.backstep and self.push is not None and self.world.tick == PUSH_TICK:
			bone, impulse = self.push
			self.apply_push(bone, impulse)
		feedback = self.balance_torques if joint_feedback is None else joint_feedback
		measure_impact = self.backstep and self.backstep_push_tick is not None
		pose_recovery = self.pose_push_tick is not None
		limit = self.episode_steps * TICKS_PER_ACTION
		pose_recovered = pose_timeout = False
		peak_acceleration = peak_impulse = 0.0
		touchdown_reward = 0.0
		for tick in range(1, TICKS_PER_ACTION + 1):
			if measure_impact:
				previous_velocity = self.state.center_of_mass_velocity[0]
				previous_center = self.state.center_of_mass[0, [0, 2]].copy()
			torque = joint_spring_torques(self.parents, self.state.orientation, self.state.angular_velocity, self.neutral, **self.spring)
			torque += joint_limit_torques(orientations=self.state.orientation, angular_velocity=self.state.angular_velocity, **self.limits)
			if feedback is not None:
				torque[0] += feedback(self.state, residual_torques)
			self.world.add_torques(self.uids.ravel(), torque[0])
			self.world.run_ticks(1)
			self.state = self._observe()
			height, upright, speed, fallen = self._posture()
			if measure_impact:
				acceleration, impulse_estimate = self._backstep_impact(previous_velocity)
				peak_acceleration = max(peak_acceleration, acceleration)
				peak_impulse = max(peak_impulse, impulse_estimate)
				if self.touchdown_support_reward:
					contact = self.state.contacts[0, self.feet]
					if (contact & ~self.touchdown_contact).any():
						self.touchdown_tick = self.world.tick
					self.touchdown_contact = contact.copy()
					if self.world.tick - self.touchdown_tick <= LANDING_BRAKE_TICKS:
						points = self._support_points()
						if points is not None:
							before = _support_margin(points, previous_center)
							after = _support_margin(points, self.state.center_of_mass[0, [0, 2]])
							touchdown_reward += 0.5 * np.clip((min(after, 0) - min(before, 0)) / self.support_scale, -0.1, 0.1)
			if self.direct and self.push_tick is not None:
				contact = self.state.contacts[0, self.feet]
				landed = contact & ~planted
				positions = self.state.position[0, self.feet][:, [0, 2]]
				if landed.any():
					self.landing_tick = self.world.tick
					relocated = landed & self.seen_contact & (np.linalg.norm(positions - self.last_contact_xz, axis=1) >= RECOVERY_STEP_DISTANCE)
					if relocated.any():
						self.relocated_tick, self.relocated_feet = self.world.tick, relocated
				self.last_contact_xz[contact] = positions[contact]
				self.seen_contact |= contact
				planted = contact
			if self.recovery is not None:
				self.recovery.after_tick()
			if after_tick is not None:
				after_tick()
			if pose_recovery:
				pose_recovered = self.pose_hold_ticks >= TICK_HZ * SETTLE_SECONDS
				pose_timeout = self.world.tick >= limit and (self.pose_hold_ticks == 0 or self.world.tick >= limit + TICK_HZ * SETTLE_SECONDS)
			if fallen or pose_recovered or pose_timeout:
				break
		elapsed_ticks = self.world.tick
		timed_out = pose_timeout if pose_recovery else elapsed_ticks >= limit
		completed = not fallen and (pose_recovered or timed_out)
		self.done = fallen or completed
		foot_speed = self.state.linear_velocity[0, self.feet][:, [0, 2]]
		sliding = (np.linalg.norm(foot_speed, axis=1) * self.state.contacts[0, self.feet]).mean()
		drift = np.linalg.norm(self._support_error())
		if self.direct:
			self._measure_direct()
		if self.recovery is not None:
			self.recovery.measure()
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
		reward = 0.5 * min(height, 1.0) + 0.5 * upright.mean() - 0.6 * min(drift, 1)
		if self.backstep:
			reward -= 0.02 * changes + 0.005 * residual_change
			impact = np.array([peak_acceleration, peak_impulse])
			cost = 0.12 * np.clip(impact / [BACKSTEP_ACCELERATION_LIMIT, self.floor_impulse_limit] - 1, 0, 2)
			self.backstep_peak_impact = np.maximum(self.backstep_peak_impact, impact)
			self.backstep_impact_cost += cost
			self.touchdown_support_total += touchdown_reward
			reward += touchdown_reward - cost.sum()
			if self.support_margin is None or self.support_margin < 0:
				self.backstep_stance_ticks = 0
		else:
			reward -= 0.25 * min(speed, 1)
			reward -= 0.05 * sliding + 0.02 * changes + 0.005 * residual_change
		if self.direct:
			braking = support = compliance = 0.0
			if self.push_tick is not None and elapsed_ticks - self.push_tick > tick:
				braking = np.clip(previous_speed - speed, -0.12, 0.12)
				if elapsed_ticks - self.landing_tick <= LANDING_BRAKE_TICKS:
					braking *= 2
				for previous, current in ((previous_support, self.support_margin), (previous_capture, self.capture_margin)):
					if previous is not None and current is not None:
						support += np.clip((min(current, 0) - min(previous, 0)) / self.support_scale, -0.25, 0.25)
			loading = self.relocated_feet & self.state.contacts[0, self.feet]
			if elapsed_ticks - self.relocated_tick <= LANDING_BRAKE_TICKS and loading.any():
				rates = self.state.joint_angular_velocity[0, self.landing_joints[loading], 0]
				flexion = np.clip(rates * [-1, 1] / 3, 0, 1).mean()
				height_weight = np.clip((height - 0.9) / 0.08, 0, 1)
				compliance = 0.04 * flexion * height_weight - 0.01 * np.clip(np.abs(rates) - 3, 0, 5).mean()
			effort = -0.02 * np.mean(self.direct_action ** 2)
			rewards = np.array([braking, support, compliance, effort]) * self.direct_reward_weights
			self.direct_rewards += rewards
			reward += rewards.sum()
		if self.recovery is not None:
			if self.recovery.phase:
				reward += 0.25 * (1 - min(height, 1.0) + 1 - upright.mean())
			reward += self.recovery.reward(TICK_HZ * SETTLE_SECONDS)
		if self.backstep and pose_recovered:
			# Credit the discounted standing reward that early success would leave uncollected.
			remaining = max(0, self.episode_steps - (elapsed_ticks + TICKS_PER_ACTION - 1) // TICKS_PER_ACTION)
			reward += 12 + PPO_GAMMA * (1 - PPO_GAMMA ** remaining) / (1 - PPO_GAMMA)
		elif not pose_recovery and self.backstep_reference is not None and not self.backstep_settled and self.backstep_stance_ticks >= TICK_HZ * SETTLE_SECONDS:
			self.backstep_settled = True
			reward += 12
		if fallen:
			reward -= 5
		info = dict(elapsed_seconds=elapsed_ticks / TICK_HZ, minimum_pelvis_m=self.minimum_height, success=completed,
			settled_success=completed and self.settled_ticks >= TICK_HZ * SETTLE_SECONDS, peak_drift_m=self.peak_drift,
			mean_foot_slip_m=self.foot_slip, mean_reference_error_rad=self.tracking_error / elapsed_ticks)
		if self.pose_recovery:
			info["pose_recovered"] = pose_recovered
			if pose_recovery:
				info.update(success=pose_recovered, settled_success=pose_recovered)
		if self.support_area:
			info["minimum_support_margin_m"] = self.minimum_support_margin
			info["first_support_exit_seconds"] = self.first_support_exit
		if self.residual:
			self.residual_effort += float(np.linalg.norm(residual_torques, axis=1).mean()) * tick
			info["mean_residual_command_nm"] = self.residual_effort / elapsed_ticks
		if self.direct:
			info.update(zip(DIRECT_REWARD_KEYS, map(float, self.direct_rewards)))
			info["target_saturation_fraction"] = self.target_saturation / ((elapsed_ticks + TICKS_PER_ACTION - 1) // TICKS_PER_ACTION * self.action_shape[0])
			info["max_foot_clearance_m"] = float(self.max_foot_clearance.max())
		if self.backstep:
			info.update(zip(BACKSTEP_IMPACT_KEYS, map(float, (*self.backstep_impact_cost, *self.backstep_peak_impact))))
			if self.touchdown_support_reward:
				info["touchdown_support_reward"] = float(self.touchdown_support_total)
		return self._features(), float(reward), fallen, completed, info

	def close(self):
		if self.world is not None:
			self.world.stop()
			self.world = None


def evaluate_episode(task, policy, *, seed, on_sample=None):
	"""Evaluate one seeded episode and optionally observe every physics sample."""
	observation, _ = task.reset(seed=seed)

	def capture():
		on_sample(task)

	if on_sample is not None:
		capture()
	after_tick = capture if on_sample is not None else None
	total_reward = 0.0
	while True:
		observation, reward, fallen, completed, info = task.step(policy(observation), after_tick=after_tick)
		total_reward += reward
		if fallen or completed:
			return info | {"reward": total_reward}


def evaluate(task, policy, *, seed, episodes):
	if episodes < 1:
		raise ValueError("evaluation requires at least one episode")
	results = [evaluate_episode(task, policy, seed=seed + i) for i in range(episodes)]
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
			return self.task.reset(seed=int(self.np_random.integers(2**32)), replay=True)

		def step(self, action):
			observation, reward, fallen, completed, info = self.task.step(action)
			recovered = info.get("pose_recovered", False)
			return observation, reward, fallen or recovered, completed and not recovered, info

		def close(self):
			self.task.close()

	return StandingEnv()


def _initialize_direct_balance(model, task):
	"""Seed the MLP with COM position and velocity feedback at both ankle motors."""
	import torch

	ankles = np.flatnonzero(np.isin(task.action_bones, task.feet))
	horizontal = np.array([0, 2])
	rows = np.arange(2)
	gains = np.zeros((2, model.observation_space.shape[0]), dtype=np.float32)
	# COM and feet are relative to the pelvis. Subtract the mean ankle position
	# from the COM position and undo the reference-height scaling of the feet.
	gains[rows, 16 + horizontal] = 8
	gains[rows, 21 + horizontal] = 3
	gains[rows, 26 + horizontal] = gains[rows, 29 + horizontal] = -4 * task.positions[0, 1]
	bias = np.array([0, -0.04], dtype=np.float32)
	# Keep the two tanh layers near their linear region at standing velocities.
	scale = 0.12
	first, _, second, _ = model.policy.mlp_extractor.policy_net
	with torch.no_grad():
		model.policy.action_net.weight.zero_()
		model.policy.action_net.bias.zero_()
		first.weight[:2].copy_(torch.from_numpy(gains * scale))
		first.bias[:2].copy_(torch.from_numpy(bias * scale))
		second.weight[:2].zero_()
		second.weight[:2, :2].copy_(torch.eye(2))
		second.bias[:2].zero_()
		axes = task.action_axes[ankles][:, [2, 0]] * [-1, 1]
		caps = task.motor_offset_caps[task.action_bones[ankles]]
		model.policy.action_net.weight[ankles, :2] = torch.tensor(axes / (scale * caps[:, None]), dtype=torch.float32)


def _require_backstep_training_push(push_impulses, recovery_curriculum):
	if not recovery_curriculum and not any(BACKSTEP_CHEST_MIN_IMPULSE <= magnitude <= np.finfo(np.float32).max for magnitude in push_impulses):
		raise ValueError(f"backstep training requires --recovery-curriculum or a push of at least {BACKSTEP_CHEST_MIN_IMPULSE} N s in --push-impulses")


def train_policy(project, *, steps, seed, workers=1, resume=None, epochs=None, direct_balance_init=False, learning_rate=None, optimizer_trace=None, direct_limb_std=None, **task_options):
	if task_options.get("backstep"):
		push_impulses = tuple(task_options.get("push_impulses", ()))
		_require_backstep_training_push(push_impulses, task_options.get("recovery_curriculum", False))
		task_options["push_impulses"] = push_impulses

	import torch
	from stable_baselines3 import PPO
	from stable_baselines3.common.callbacks import BaseCallback
	from stable_baselines3.common.env_util import make_vec_env
	from stable_baselines3.common.vec_env import SubprocVecEnv

	class RewardLog(BaseCallback):
		def _on_step(self):
			return True

		def _on_rollout_end(self):
			for key in reward_keys:
				values = [episode[key] for episode in self.model.ep_info_buffer if key in episode]
				if values:
					self.logger.record(f"rollout/{key}_mean", float(np.mean(values)))

	class TracedPPO(PPO):
		def train(self):
			batches = self.rollout_buffer.get
			epochs = []

			def trace_batches(batch_size):
				epoch = []
				epochs.append(epoch)
				for batch in batches(batch_size):
					with torch.no_grad():
						_, log_prob, _ = self.policy.evaluate_actions(batch.observations, batch.actions)
						log_ratio = log_prob - batch.old_log_prob
						approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean().item()
					record = dict(approx_kl=approx_kl if np.isfinite(approx_kl) else None, accepted=False)
					epoch.append(record)
					yield batch
					# PPO stops iterating before applying a rejected minibatch.
					record["accepted"] = True

			self.rollout_buffer.get = trace_batches
			try:
				super().train()
			finally:
				del self.rollout_buffer.get
			metrics = dict(timesteps=self.num_timesteps, epochs=epochs, learning_rate=self.lr_schedule(self._current_progress_remaining),
				accepted_epochs=sum(all(batch["accepted"] for batch in epoch) for epoch in epochs),
				action_variance=self.policy.log_std.detach().mul(2).exp().mean().item())
			for key in ("clip_fraction", "entropy_loss", "value_loss", "explained_variance"):
				value = float(self.logger.name_to_value[f"train/{key}"])
				metrics[key] = value if np.isfinite(value) else None
			with Path(optimizer_trace).open("a", encoding="utf-8") as output:
				output.write(json.dumps(metrics, allow_nan=False) + "\n")

	if workers < 1 or (epochs is not None and epochs < 1):
		raise ValueError("workers and epochs must be positive")
	if direct_balance_init and (not task_options.get("direct") or resume is not None):
		raise ValueError("balance initialization requires a new direct policy")
	if learning_rate is not None and (not np.isfinite(learning_rate) or learning_rate <= 0):
		raise ValueError("learning rate must be positive and finite")
	direct = task_options.get("direct")
	if direct_limb_std is not None and (not direct or not 0 < direct_limb_std <= 1):
		raise ValueError("limb exploration requires direct control and a standard deviation in (0, 1]")
	setup_task = StandingTask(project, **task_options) if direct_balance_init or direct_limb_std is not None else None
	algorithm = TracedPPO if optimizer_trace is not None else PPO
	factory = partial(_standing_env, project, task_options)
	parallel = dict(vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "spawn"}) if workers > 1 else {}
	reward_keys = DIRECT_REWARD_KEYS if direct else BACKSTEP_IMPACT_KEYS if task_options.get("backstep") else ()
	if task_options.get("touchdown_support_reward"):
		reward_keys += ("touchdown_support_reward",)
	monitor = {"info_keywords": reward_keys} if reward_keys else None
	environment = make_vec_env(factory, n_envs=workers, seed=seed, monitor_kwargs=monitor, **parallel)
	try:
		# Keep complete minibatches when changing the number of workers.
		batch_size = 128
		n_steps = batch_size * max(1, 8 // workers)
		options = dict(seed=seed, device="cpu", n_steps=n_steps, batch_size=batch_size)
		if epochs is not None:
			options["n_epochs"] = epochs
		if learning_rate is not None:
			options["learning_rate"] = learning_rate
		if resume is None:
			options.setdefault("n_epochs", 10)
			options.setdefault("learning_rate", 0.00001 if direct_balance_init else (0.000075 if direct else 0.00015))
			log_std = -4.2 if direct_balance_init else (-3.0 if direct else -2.5)
			model = algorithm("MlpPolicy", environment, gamma=PPO_GAMMA, target_kl=0.03, policy_kwargs={"log_std_init": log_std}, verbose=1, **options)
			if direct_balance_init:
				_initialize_direct_balance(model, setup_task)
			else:
				with torch.no_grad():
					if any(task_options.get(mode) for mode in ("residual", "stepping", "whole_body", "recovery", "direct", "backstep")):
						model.policy.action_net.weight.zero_()
						model.policy.action_net.bias.zero_()
					elif not task_options.get("compact"):
						model.policy.action_net.bias[3::4].fill_(1 if task_options.get("tuned_stance") else 0.8)
					if task_options.get("recovery"):
						from tools.ragdoll_recovery_step import RECOVERY_CHANNELS

						model.policy.log_std[-RECOVERY_CHANNELS:].fill_(-1.5)
		else:
			model = algorithm.load(resume, env=environment, **options)
		if direct_limb_std is not None:
			non_ankle_axes = ~np.isin(setup_task.action_bones, setup_task.feet)
			with torch.no_grad():
				model.policy.log_std[non_ankle_axes] = np.log(direct_limb_std)
		return model.learn(total_timesteps=steps, reset_num_timesteps=resume is None, callback=RewardLog() if reward_keys else None)
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
	parser.add_argument("--action-space", choices=("full", "compact", "residual", "stepping", "whole-body", "recovery", "direct", "backstep"), default="full", help="choose a joint control or recovery step policy layout")
	parser.add_argument("--push-impulses", type=float, nargs="+", default=(), help="sample an impulse magnitude in N s at two seconds, including zero for quiet starts; backstep uses backward chest pushes")
	parser.add_argument("--recovery-curriculum", action="store_true", help="mix quiet, light, near-threshold and hard push episodes")
	parser.add_argument("--replay-start-fraction", type=float, default=0.0, help="fraction of training starts to attempt near a replayed landing")
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
	parser.add_argument("--epochs", type=int, help="maximum PPO optimizer passes per rollout (default 10 for new policies)")
	parser.add_argument("--learning-rate", type=float, help="override the PPO learning rate")
	parser.add_argument("--optimizer-trace", type=Path, help="append per-minibatch KL and rollout optimizer metrics to a JSON Lines file")
	parser.add_argument("--direct-balance-init", action="store_true", help="initialize a new direct policy with continuous ankle motor feedback")
	parser.add_argument("--direct-limb-std", type=float, help="initial exploration standard deviation on the direct policy's non-ankle axes")
	parser.add_argument("--without-direct-reward", nargs="+", choices=DIRECT_REWARD_TERMS, default=(), help="omit selected direct reward terms for ablation")
	parser.add_argument("--strong-backstep-reference", action="store_true", help="use the longer reference step for backward chest pushes of at least 14 N s")
	parser.add_argument("--pose-recovery", action="store_true", help="end pushed backstep episodes using a slow target-pose hold and up to one second of grace")
	parser.add_argument("--release-backstep-hips", action="store_true", help="blend hip pitch back to policy control after the scripted loading phase")
	parser.add_argument("--touchdown-support-reward", action="store_true", help="reward COM movement toward foot support after touchdown")
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
	if args.epochs is not None and (args.steps is None or args.epochs < 1):
		parser.error("--epochs requires --steps and a positive value")
	if args.learning_rate is not None and (args.steps is None or not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
		parser.error("--learning-rate requires --steps and a positive finite value")
	if args.optimizer_trace is not None and args.steps is None:
		parser.error("--optimizer-trace requires --steps")
	if args.direct_balance_init and (args.steps is None or args.action_space != "direct" or args.resume is not None):
		parser.error("--direct-balance-init requires --steps, --action-space direct, and a new policy")
	if args.direct_limb_std is not None and (args.steps is None or args.action_space != "direct" or not 0 < args.direct_limb_std <= 1):
		parser.error("--direct-limb-std requires direct training and a value in (0, 1]")
	if args.replay_start_fraction and args.steps is None:
		parser.error("--replay-start-fraction requires --steps")
	backstep = args.action_space == "backstep"
	if args.pose_recovery and (not backstep or not (args.recovery_curriculum or any(value > 0 for value in args.push_impulses))):
		parser.error("--pose-recovery requires backstep control with --recovery-curriculum or positive --push-impulses")
	try:
		if backstep and args.steps is not None:
			_require_backstep_training_push(args.push_impulses, args.recovery_curriculum)
		project = default_project() if args.tuning is None else read_project(args.tuning)
		options = dict(episode_steps=args.seconds * CONTROL_HZ, initial_angular_speed=args.initial_angular_speed,
			tuned_stance=args.stance == "tuned", vary_start=args.vary_start, support_area=args.support_area, push_impulses=args.push_impulses,
			recovery_curriculum=args.recovery_curriculum, pose_recovery=args.pose_recovery, backward_chest_pushes=backstep)
		if args.max_start_tilt is not None:
			options["max_start_tilt"] = args.max_start_tilt
		action_options = {args.action_space.replace("-", "_"): True} if args.action_space != "full" else {}
		if args.without_direct_reward:
			action_options["without_direct_reward"] = args.without_direct_reward
		if args.strong_backstep_reference:
			action_options["strong_backstep_reference"] = True
		if args.release_backstep_hips:
			action_options["release_backstep_hips"] = True
		if args.touchdown_support_reward:
			action_options["touchdown_support_reward"] = True
		task = StandingTask(project, **action_options, **options, replay_start_fraction=args.replay_start_fraction)
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
		baseline_task = StandingTask(project, **options) if task.compact or task.residual or task.direct else task
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
				model = train_policy(project, steps=args.steps, seed=args.seed, workers=args.workers, resume=args.resume, epochs=args.epochs,
					direct_balance_init=args.direct_balance_init, learning_rate=args.learning_rate, optimizer_trace=args.optimizer_trace, direct_limb_std=args.direct_limb_std,
					**action_options, **options, replay_start_fraction=args.replay_start_fraction)
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
