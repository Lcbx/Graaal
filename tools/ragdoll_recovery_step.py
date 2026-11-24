"""A timed backward recovery step for the flat-floor humanoid benchmark."""

from dataclasses import replace

import numpy as np

from graaal.Math import Quaternion
from scenes.ragdoll_control import STANDING_MOTOR_CAPS
from tools.ragdoll_standing import TICK_HZ


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

	def __init__(self, task):
		if not task.whole_body:
			raise ValueError("the scripted backstep uses the whole-body action layout")
		self.task = task
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
		if 0 <= time < end:
			action[0, self.swing, 3] = 0
		elif time >= end:
			ease = np.interp(time, [end, end + 0.5, end + 1.5], [1, 1, 0])
			action[0, self.task.hips, 0] = ease * np.array([-0.2, 0.4])

	def feedback(self, state, residual):
		time = self._seconds()
		if time < 0 or time >= self.swing_seconds + self.load_seconds:
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
