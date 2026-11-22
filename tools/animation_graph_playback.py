"""Adapt batched graph playback to the authoring preview's scalar controls."""

import numpy as np

from graaal.Animation import AnimationGraphPlayback as runtime
from graaal.Animation.SkeletalAnimation import blend_poses


def sample_times(graph, ids, time, sync=None):
	return runtime.sample_times(graph, np.asarray(ids), np.array([time]), sync)


def state_sample(graph, state, time, parameters):
	return runtime.state_sample(graph, state, np.array([time]), np.asarray(parameters)[None])


def state_pose(graph, state, time, parameters):
	return runtime.state_pose(graph, state, np.array([time]), np.asarray(parameters)[None])


def transition_pose(graph, source, target, source_time, target_time, weight, parameters):
	"""Blend complete state poses before authored masks and additive layers."""
	skeleton = graph.plan.skeleton
	base = state_pose(graph, source, source_time, parameters)
	if target is None: return base
	overlay = state_pose(graph, target, target_time, parameters)
	return blend_poses(skeleton, base, overlay, [weight])


class GraphPlayback:
	"""Present a single runtime row through the editor's scalar interface."""
	def __init__(self, graph):
		self.graph = graph
		self._player = runtime.AnimationGraphPlayback(graph, 1)

	@property
	def state(self):
		return int(self._player.states[0])

	@property
	def time(self):
		return float(self._player.times[0])

	@property
	def speed(self):
		return float(self._player.speeds[0])

	@property
	def target_state(self):
		target = int(self._player.target_states[0])
		return None if target < 0 else target

	@property
	def target_time(self):
		return float(self._player.target_times[0])

	@property
	def target_speed(self):
		return float(self._player.target_speeds[0])

	@property
	def elapsed(self):
		return float(self._player.elapsed[0])

	@property
	def duration(self):
		return float(self._player.durations[0])

	def reset(self, state=None, time=0.0):
		"""Seek a state, cancel its fade and restore unit playback speed."""
		if state is not None and state not in self.graph.state_ids.values(): raise ValueError("unknown graph state")
		if not np.isscalar(time) or not np.isfinite(time) or time < 0: raise ValueError("time must be a finite nonnegative scalar")
		self._player.reset(None if state is None else int(state), time)

	def advance(self, dt, parameters):
		"""Advance seconds using one applied parameter row."""
		if not np.isscalar(dt): raise ValueError("dt must be a finite nonnegative scalar")
		self._player.advance(dt, np.asarray(parameters, dtype=np.float64)[None])

	def sample(self):
		"""Return a clip-only transition triple; use pose() for blend states."""
		if self.state not in self.graph.state_clips or (self.target_state is not None and self.target_state not in self.graph.state_clips):
			raise ValueError("blend-state transitions require pose()")
		source = self.graph.state_clips[self.state]
		target = source if self.target_state is None else self.graph.state_clips[self.target_state]
		weight = self.elapsed / self.duration if self.target_state is not None else 0.0
		target_time = self.time if self.target_state is None else self.target_time
		return np.array([[source, target]]), np.array([[self.time, target_time]]), np.array([[1 - weight, weight]])

	def pose(self, parameters):
		"""Return an independent local pose, including any active state fade."""
		return self._player.pose(np.asarray(parameters, dtype=np.float64)[None])
