"""A timed backward recovery step for the flat-floor humanoid benchmark."""

from dataclasses import replace

import numpy as np

from graaal.Math import Quaternion
from scenes.physics_geometry import collider_poses
from scenes.ragdoll_control import STANDING_MOTOR_CAPS
from tools.ragdoll_standing import BACKSTEP_CHEST_MIN_IMPULSE, TICK_HZ, _support_margin


class ScriptedBackstep:
	"""Swing the right foot, press into the landing, then return to ankle balance."""

	delay = 0.145
	swing_seconds = 0.16
	load_seconds = 0.30
	step_offset = (-0.035, 0, -0.195)
	clearance = 0.163
	position_gain = 1227.0
	landing_force = 427.0
	stance_action = (-0.377, -0.392, -0.544)

	def __init__(self, task, *, swing=True, release_hips=False):
		if not task.whole_body:
			raise ValueError("the scripted backstep uses the whole-body action layout")
		self.task = task
		self.swing_enabled = swing
		self.release_hips = release_hips
		self.push_tick = task.world.tick
		self.swing = np.array([task.hips[1], task.knees[1], task.feet[1]])
		self.parents = np.asarray(task.parents)[self.swing]
		self.caps = np.asarray(STANDING_MOTOR_CAPS)[self.swing]
		self.start = self.anchor = None
		task.motor_offset_caps[task.hips] = 1.0
		task.motor_offset_caps[task.knees] = 1.2

	def _seconds(self):
		return (self.task.world.tick - self.push_tick) / TICK_HZ - self.delay

	def action(self):
		time = self._seconds()
		end = self.swing_seconds + self.load_seconds
		scale = np.interp(time, [0, 0.07, end, end + 0.60], [0, 1, 1, 0])
		action = np.zeros(self.task.action_shape)
		# Stance hip pitch and roll, stance knee flexion, and chest pitch.
		action[[4, 5, 8, 11]] = np.array([*self.stance_action, 0.9]) * scale
		return action

	def edit_motors(self, action):
		time = self._seconds()
		end = self.swing_seconds + self.load_seconds
		if time >= end:
			ease = np.interp(time, [end, end + 0.5, end + 1.5], [1, 1, 0])
			hip_pitch = ease * np.array([-0.2, 0.4])
			if self.release_hips:
				blend = min((time - end) / 0.25, 1)
				hip_pitch += blend * (action[0, self.task.hips, 0] - hip_pitch)
			action[0, self.task.hips, 0] = hip_pitch
		elif self.swing_enabled and time >= 0:
			action[0, self.swing, 3] = 0

	def feedback(self, state, residual):
		time = self._seconds()
		if not self.swing_enabled or time < 0 or time >= self.swing_seconds + self.load_seconds:
			return self.task.balance_torques(state, residual)
		foot = self.swing[-1]
		position = state.position[0]
		if self.start is None:
			self.start = position[foot].copy()
		loading = time >= self.swing_seconds
		phase = min(time / self.swing_seconds, 1.0)
		travel = min(phase / 0.8, 1.0)
		travel = travel ** 2 * (3 - 2 * travel)
		goal = self.start + np.asarray(self.step_offset) * travel
		goal[1] += self.clearance * np.sin(np.pi * phase) ** 0.7 - 0.035 * phase ** 4
		if loading:
			if self.anchor is None:
				self.anchor = position[foot].copy()
				self.anchor[1] = 0.073
			goal = self.anchor
		force = self.position_gain * (goal - position[foot]) - 2.5 * np.sqrt(self.position_gain) * state.linear_velocity[0, foot]
		if loading:
			force[1] -= self.landing_force
		# Translate the foot-space command into equal and opposite joint torques.
		turns = Quaternion.matrix_batch(state.orientation[0, [0, foot]])
		pair = np.empty((3, 3))
		pair[:2] = np.cross(position[foot] - position[self.swing[:2]], force)
		pair[2] = 210 * np.cross(turns[1, :, 1], [0, 1, 0]) - 7 * state.angular_velocity[0, foot]
		if loading:
			pair[0] -= 140 * np.cross(turns[0, :, 1], [0, 1, 0]) - 35 * state.angular_velocity[0, 0]
		pair *= np.minimum(1, self.caps / np.maximum(np.linalg.norm(pair, axis=1), 1))[:, None]
		contacts = state.contacts.copy()
		contacts[0, foot] = False
		torque = self.task.balance_torques(replace(state, contacts=contacts), residual)
		torque[self.swing] += pair
		torque[self.parents] -= pair
		return torque


class BackstepReference(ScriptedBackstep):
	"""The fitted reference used to initialize touchdown and settling training."""

	delay = 0.14387688194349282
	swing_seconds = 0.1586975391393785
	load_seconds = 0.29874593509199904
	step_offset = (-0.03470655427714422, 0, -0.19524371726645115)
	clearance = 0.16262734247767616
	position_gain = 1227.430027626054
	landing_force = 427.1651872768462
	stance_action = (-0.37733810693470465, -0.39198731949498544, -0.54443129475441)

	@staticmethod
	def accepts(task, bone, impulse):
		minimum = {"chest": BACKSTEP_CHEST_MIN_IMPULSE, "pelvis": 12}.get(task.settings.bones[bone].name, np.inf)
		return tuple(impulse[:2]) == (0, 0) and -impulse[2] >= minimum

	@classmethod
	def start_for_push(cls, task, bone, impulse):
		if cls.accepts(task, bone, impulse):
			strong = task.strong_backstep_reference and bone == task.chest and -impulse[2] >= 14
			reference = StrongBackstepReference if strong else cls
			return reference(task, swing=task.backstep_swing, release_hips=task.release_backstep_hips)
		return None

	def progress(self):
		return float(np.clip((self.task.world.tick - self.push_tick) / TICK_HZ, 0, 2))

	def residual_active(self):
		return self._seconds() >= self.swing_seconds

	def learned_action(self, action):
		# Keep the reference's swing intact. Learn how to load the foot and stand up.
		weight = np.clip((self._seconds() - self.swing_seconds) / 0.10, 0, 1)
		return np.clip(self.action() + 0.5 * weight * action, -1, 1)


class StrongBackstepReference(BackstepReference):
	"""Use a farther foot placement for backward chest pushes of at least 14 N s."""

	delay = 0.17090484461244
	swing_seconds = 0.1354020580978652
	load_seconds = 0.31215124881300693
	step_offset = (-0.027452342337965695, 0, -0.23677102539348102)
	clearance = 0.14423303944384036
	position_gain = 1323.4107298406195
	landing_force = 464.0199225555542
	stance_action = (-0.38178970007916635, -0.4634828142542544, -0.595489524676176)


RECOVERY_CHANNELS = 6
STEP_REQUEST = 0.18
LANDING_REWARD_TICKS = round(0.2 * TICK_HZ)


class RecoveryStep:
	"""Execute policy-selected steps using bounded joint torques and foot contact."""
	max_load_seconds = 0.42
	cooldown_seconds = 0.2
	load_with_ankles = False

	def __init__(self, task):
		import pyjoltphysics as pj

		self.task = task
		self.legs = np.column_stack((task.hips, task.knees, task.feet))
		self.parents = np.asarray(task.parents)[self.legs]
		self.caps = np.asarray(STANDING_MOTOR_CAPS)[self.legs]
		self.half_sizes = np.array([task.settings.bones[i].shape.params[:3] for i in task.feet]) / 2
		self.foot_load_per_tick = 9.81 * sum(bone.mass for bone in task.settings.bones) / (2 * TICK_HZ)
		self.contact_kind = pj.EventKind.CONTACT
		self.contact_phases = (pj.EventPhase.BEGIN, pj.EventPhase.PERSIST)

	def reset(self):
		self.contact_age = np.zeros(2, dtype=int)
		self.impulse = np.zeros(2)
		self.command = np.zeros(RECOVERY_CHANNELS)
		self.side, self.phase = None, 0
		self.cooldown = 0
		self.landing_tick = None
		self.planned_goal = None
		self.pushed = self.settle_awarded = False
		self.measure()

	def measure(self):
		task, feet = self.task, self.task.feet
		state = task.state
		centers, turns = collider_poses(state.position[0, feet], state.orientation[0, feet], task.foot_offsets, task.foot_rotations)
		axes = Quaternion.matrix_batch(turns)
		self.soles = centers[:, None] + task.sole_corners @ axes.transpose(0, 2, 1)
		self.clearance = centers[:, 1] - (np.abs(axes[:, 1]) * self.half_sizes).sum(axis=1)
		forward = Quaternion.matrix_batch(state.orientation[0, 0])[[0, 2], 2]
		forward /= max(np.linalg.norm(forward), 1e-6)
		self.heading = np.column_stack((forward[::-1] * [1, -1], forward))
		self.sole_tilt = axes[:, [0, 2], 1] @ self.heading
		center = state.center_of_mass[0]
		velocity = state.center_of_mass_velocity[0, [0, 2]]
		self.capture_time = np.sqrt(max(center[1], 0) / 9.81)
		self.capture = center[[0, 2]] + velocity * self.capture_time
		self.speed = float(np.linalg.norm(velocity))
		planted = state.contacts[0, feet]
		self.capture_margin = _support_margin(self.soles[planted].reshape(-1, 3)[:, [0, 2]], self.capture) if planted.any() else None

	def features(self):
		task = self.task
		chest_spin = task.state.angular_velocity[0, task.chest]
		relative = (self.capture - self.soles.mean(axis=1)[:, [0, 2]]) @ self.heading
		goal = np.zeros(2)
		if self.side is not None:
			foot = task.state.position[0, task.feet[self.side], [0, 2]]
			goal = (self.goal[[0, 2]] - foot) @ self.heading
		return np.concatenate((relative.ravel() * 5, [(self.capture_margin or 0) / task.support_scale],
			self.sole_tilt.ravel(), self.clearance * 10, np.minimum(self.contact_age, TICK_HZ) / TICK_HZ,
			self.impulse, chest_spin[[0, 2]] @ self.heading / 5, [chest_spin[1] / 5],
			[self.phase == 1, self.phase == 2, 0 if self.side is None else 2 * self.side - 1],
			[min((task.world.tick - self.started) / TICK_HZ, 1) if self.side is not None else 0,
				max(self.cooldown - task.world.tick, 0) / TICK_HZ], goal * 5, self.command))

	def request(self, command, *, guided=False):
		self.command = command
		if self.phase or self.task.world.tick < self.cooldown or abs(command[0]) < STEP_REQUEST:
			return
		# Keep a stance foot and reserve swing for a disturbance outside stable support.
		side = int(command[0] > 0)
		if self.capture_margin is None:
			return
		spin = np.linalg.norm(self.task.state.angular_velocity[0, self.task.chest, [0, 2]])
		if not guided and spin < 0.65 and (self.capture_margin >= -0.015 or self.speed < 0.10):
			return
		if not self.task.state.contacts[0, self.task.feet[1 - side]]:
			side = 1 - side
		start = self.task.state.position[0, self.task.feet[side]]
		landing = self.capture - start[[0, 2]]
		landing *= min(1, 0.21 / max(np.linalg.norm(landing), 1e-6))
		landing += (command[1:3] * 0.12) @ self.heading.T
		landing *= min(1, 0.33 / max(np.linalg.norm(landing), 1e-6))
		self.start_step(side, start[[0, 2]] + landing, 0.15 + 0.07 * command[3], 0.18 + 0.07 * command[4])

	def start_step(self, side, landing_xz, clearance, duration):
		self.side, self.phase = side, 1
		self.started = self.task.world.tick
		self.start = self.task.state.position[0, self.task.feet[side]].copy()
		self.goal = self.start.copy()
		self.goal[[0, 2]] = landing_xz
		self.goal[1] -= max(self.clearance[side], 0) + 0.018
		self.planned_goal = self.goal.copy()
		self.touchdown = None
		self.clear, self.duration = clearance, duration
		self.airborne = False
		self.peak_swing_torque_ratio = 0.0

	def edit_motors(self, action):
		if self.phase:
			action[0, self.legs[self.side], 3] = 0

	def feedback(self, state, residual):
		task = self.task
		if not self.phase:
			return task.balance_torques(state, residual)
		leg = self.legs[self.side]
		foot = leg[-1]
		position = state.position[0]
		time = (task.world.tick - self.started) / TICK_HZ
		phase = min(time / self.duration, 1)
		travel = min(phase / 0.8, 1)
		travel = travel ** 2 * (3 - 2 * travel)
		goal = self.start + (self.goal - self.start) * travel
		goal[1] += self.clear * np.sin(np.pi * phase) ** 0.7
		# Press through the acquired sole while damping horizontal ankle motion.
		if self.phase == 2:
			goal = self.goal
		force = 1227 * (goal - position[foot]) - 88 * state.linear_velocity[0, foot]
		turns = Quaternion.matrix_batch(state.orientation[0, [0, foot]])
		pair = np.empty((3, 3))
		if self.phase == 2:
			force[1] -= 385 + 175 * self.command[-1]
		pair[:2] = np.cross(position[foot] - position[leg[:2]], force)
		pair[2] = 210 * np.cross(turns[1, :, 1], [0, 1, 0]) - 7 * state.angular_velocity[0, foot]
		if self.phase == 2:
			pair[0] -= 140 * np.cross(turns[0, :, 1], [0, 1, 0]) - 35 * state.angular_velocity[0, 0]
		magnitude = np.linalg.norm(pair, axis=1)
		if self.phase == 1:
			self.peak_swing_torque_ratio = max(self.peak_swing_torque_ratio, float(np.max(magnitude / self.caps[self.side])))
		pair *= np.minimum(1, self.caps[self.side] / np.maximum(magnitude, 1))[:, None]
		balance_state = state
		if self.phase == 1 or not self.load_with_ankles:
			contacts = state.contacts.copy()
			contacts[0, foot] = False
			balance_state = replace(state, contacts=contacts)
		torque = task.balance_torques(balance_state, residual)
		if self.phase == 2 and self.load_with_ankles:
			combined = torque[foot] + pair[2]
			pair[2] = combined * min(1, self.caps[self.side, 2] / max(np.linalg.norm(combined), 1)) - torque[foot]
		torque[leg] += pair
		torque[self.parents[self.side]] -= pair
		return torque

	def after_tick(self):
		task = self.task
		planted = task.state.contacts[0, task.feet]
		self.contact_age = (self.contact_age + 1) * planted
		events = task.events
		active = events[(events["kind"] == self.contact_kind) & np.isin(events["phase"], self.contact_phases)]
		a, b = active["a"][:, None], active["b"][:, None]
		feet = task.uids[0, task.feet]
		contacts = ((a == feet) & np.isin(b, task.floor)) | ((b == feet) & np.isin(a, task.floor))
		load = (contacts * active["impulse"][:, None]).sum(axis=0) / self.foot_load_per_tick
		self.impulse = (0.85 * self.impulse + 0.15 * np.clip(load, 0, 6)) * planted
		if not self.phase:
			return
		time = (task.world.tick - self.started) / TICK_HZ
		contact = planted[self.side]
		if self.phase == 1:
			self.airborne |= not contact
			if self.airborne and contact and time >= 0.055:
				self.phase = 2
				self.loaded = task.world.tick
				self.goal = task.state.position[0, task.feet[self.side]].copy()
				self.touchdown = self.goal.copy()
				if np.linalg.norm((self.goal - self.start)[[0, 2]]) >= 0.02 and self.landing_tick is None:
					self.measure()
					self.landing_tick = task.world.tick
					self.landing_error = self.error()
			elif time >= self.duration + 0.15:
				self.finish()
		else:
			load_time = (task.world.tick - self.loaded) / TICK_HZ
			if contact and load_time >= 0.12 and np.linalg.norm(task.state.center_of_mass_velocity[0, [0, 2]]) <= 0.12:
				self.measure()
				if self.capture_margin is not None and self.capture_margin >= 0:
					self.finish()
					return
			if not contact or load_time >= self.max_load_seconds:
				self.finish()

	def finish(self):
		self.side, self.phase = None, 0
		self.cooldown = self.task.world.tick + round(self.cooldown_seconds * TICK_HZ)

	def error(self):
		outside = 0.25 if self.capture_margin is None else np.clip(-self.capture_margin, 0, 0.25)
		return min(self.speed, 1.5) + 3 * outside

	def reward(self, hold_ticks):
		task = self.task
		result = -0.15 * self.error()
		if self.landing_tick is not None and task.world.tick - self.landing_tick >= LANDING_REWARD_TICKS:
			result += 2 * np.clip(self.landing_error - self.error(), -1, 1)
			self.landing_tick = None
		if self.pushed and not self.settle_awarded and task.settled_ticks >= hold_ticks:
			self.settle_awarded = True
			result += 8
		return result


class AdaptiveRecoveryStep(RecoveryStep):
	"""Choose a reachable foothold and swing time from the current balance state."""

	max_load_seconds = 0.30
	cooldown_seconds = 0.05
	stance_roll = 0.20
	trigger_ticks = 10
	load_with_ankles = True

	def __init__(self, task):
		super().__init__(task)
		self.reference_up = Quaternion.matrix_batch(task.orientations[task.chest])[..., 1]
		task.motor_offset_caps[task.hips] = 1.0
		task.motor_offset_caps[task.knees] = 1.2

	def reset(self):
		super().reset()
		self.unstable_since = None

	def request(self, command, *, guided=False):
		if guided:
			super().request(command, guided=True)
			return
		task = self.task
		if self.phase or task.world.tick < self.cooldown:
			return
		if self.capture_margin is None or self.capture_time < 0.15:
			self.unstable_since = None
			return
		state = task.state
		spin = state.angular_velocity[0, task.chest]
		if self.speed < 0.055 and np.hypot(spin[0], spin[2]) < 0.55:
			self.unstable_since = None
			return
		positions = state.position[0, task.feet][:, [0, 2]]
		feet = self.soles[:, :, [0, 2]]
		planted = state.contacts[0, task.feet]
		up = Quaternion.matrix_batch(state.orientation[0, task.chest])[..., 1]
		lean = (up - self.reference_up + 0.10 * np.cross(spin, up))[[0, 2]]
		capture = self.capture + 0.32 * lean
		if _support_margin(feet[planted].reshape(-1, 2), capture) >= 0:
			self.unstable_since = None
			return
		if self.unstable_since is None:
			self.unstable_since = task.world.tick
		if task.world.tick - self.unstable_since < self.trigger_ticks:
			return
		best_score = -np.inf
		for stance in np.flatnonzero(planted):
			side = 1 - stance
			# Approximate the stance center of pressure by the closest point on the sole.
			support = feet[stance]
			origin = support[0]
			edges = support[[1, 2]] - origin
			pivot = origin + np.clip((edges @ (capture - origin)) / np.maximum((edges ** 2).sum(axis=1), 1e-8), 0, 1) @ edges
			separation = ((positions[side] - positions[stance]) @ self.heading)[0]
			outward = np.copysign(1, separation)
			for duration in (0.16, 0.20, 0.24):
				predicted = pivot + (capture - pivot) * np.exp(duration / self.capture_time)
				delta = predicted - feet[side].mean(axis=0)
				# Keep the swing on its own side and limit travel by the time available.
				local = delta @ self.heading
				local[0] = outward * max(min(abs(separation), 0.14) - abs(separation), outward * local[0])
				delta = self.heading @ local
				delta *= min(1, 1.45 * duration / max(np.linalg.norm(delta), 1e-8))
				if np.linalg.norm(delta) < 0.04:
					continue
				margin = _support_margin(np.concatenate((support, feet[side] + delta)), predicted)
				score = margin - 0.25 * duration - 0.005 * self.impulse[side]
				if score > best_score:
					best_score = score
					plan = side, positions[side] + delta, duration
		if best_score > -np.inf:
			side, landing, duration = plan
			self.start_step(side, landing, 0.16, duration)
			self.unstable_since = None
			self.command[-1] = 0.24

	def start_step(self, side, landing_xz, clearance, duration):
		super().start_step(side, landing_xz, clearance, duration)
		travel = (landing_xz - self.start[[0, 2]]) @ self.heading
		self.step_direction = travel / max(np.linalg.norm(travel), 1e-8)

	def edit_motors(self, action):
		super().edit_motors(action)
		if self.phase:
			task = self.task
			stance = 1 - self.side
			time = (task.world.tick - self.started) / TICK_HZ
			weight = min(time / 0.07, 1)
			action[0, task.hips[stance], [0, 2]] += weight * np.array([0.38 * self.step_direction[1], (2 * stance - 1) * self.stance_roll])
			action[0, task.knees[stance], 0] -= weight * 0.54
			action[0, task.chest, 0] -= weight * 0.20 * self.step_direction[1]
