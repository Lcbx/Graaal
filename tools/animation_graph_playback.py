"""Single-instance graph playback for the standalone authoring preview."""

import numpy as np

from graaal.Animation.SkeletalAnimation import blend_poses, sample_clip_pairs


def state_sample(graph, state, time, parameters):
	"""Select a state's clip pair at a shared seconds clock, wrapping each clip."""
	if state in graph.state_blend_spaces:
		ids, weights = graph.blend_spaces[graph.state_blend_spaces[state]].select(np.asarray(parameters)[None])
	else:
		ids, weights = np.array([[graph.state_clips[state]] * 2]), np.array([[1.0, 0.0]])
	times = np.array([[time % graph.clips[int(clip)].duration if graph.clips[int(clip)].duration > 0 else 0.0 for clip in ids[0]]])
	return ids, times, weights


def transition_pose(graph, source, target, source_time, target_time, weight, parameters):
	"""Blend complete state poses before authored masks and additive layers."""
	skeleton = graph.plan.skeleton
	base = sample_clip_pairs(skeleton, graph.clips, *state_sample(graph, source, source_time, parameters))
	if target is None: return base
	overlay = sample_clip_pairs(skeleton, graph.clips, *state_sample(graph, target, target_time, parameters))
	return blend_poses(skeleton, base, overlay, [weight])


class GraphPlayback:
	"""Advance clip and 1D blend states without ECS, events or root motion.

	All clips loop. Transitions cannot be interrupted; each advance selects at
	most one edge. Marker synchronization ends at promotion, retaining the target
	speed at the fade endpoint, as in the ECS playback adapter.
	"""
	def __init__(self, graph):
		self.graph = graph
		self.reset()

	def _time(self, state, time):
		if not np.isfinite(time): raise ValueError("playback time must be finite")
		# Keep a blend's clock unwrapped so unequal clip loops remain continuous.
		if state in self.graph.state_blend_spaces: return float(time)
		duration = self.graph.clips[self.graph.state_clips[state]].duration
		return float(time % duration) if duration > 0 else 0.0

	def reset(self, state=None, time=0.0):
		"""Seek a state, cancel its fade and restore unit playback speed."""
		state = self.graph.initial_state if state is None else state
		if state not in self.graph.state_ids.values(): raise ValueError("unknown graph state")
		if not np.isscalar(time) or not np.isfinite(time) or time < 0: raise ValueError("time must be a finite nonnegative scalar")
		time = self._time(state, time)
		self.state, self.time, self.speed = state, time, 1.0
		self.target_state, self.target_time, self.target_speed = None, 0.0, 1.0
		self.elapsed, self.duration = 0.0, 0.0

	def advance(self, dt, parameters):
		"""Advance seconds using applied parameters; failed updates preserve state."""
		if not np.isscalar(dt) or not np.isfinite(dt) or dt < 0: raise ValueError("dt must be a finite nonnegative scalar")
		parameters = np.asarray(parameters, dtype=np.float64)
		if parameters.shape != self.graph.parameter_defaults.shape or not np.isfinite(parameters).all(): raise ValueError("parameters must be a finite graph parameter row")
		state, time, speed = self.state, self.time, self.speed
		target, target_time, target_speed = self.target_state, self.target_time, self.target_speed
		elapsed, duration = self.elapsed, self.duration
		if target is None:
			selected, targets, durations = self.graph.machine.select(np.array([state]), parameters[None], np.array([False]))
			if selected[0]:
				target, target_time, target_speed = int(targets[0]), 0.0, speed
				elapsed, duration = 0.0, float(durations[0])
		if target is None:
			time = self._time(state, time + speed * dt)
		else:
			step = min(dt, max(duration - elapsed, 0.0))
			source_end = time + speed * step
			pair = self.graph.state_clips.get(state), self.graph.state_clips.get(target)
			mapping = self.graph.sync_mappings.get(pair)
			if mapping is not None:
				mapped, rates = mapping.map([source_end])
				target_time, target_speed = float(mapped[0]), speed * float(rates[0])
			else: target_time += target_speed * step
			target_time = self._time(target, target_time + target_speed * (dt - step))
			elapsed = min(elapsed + dt, duration)
			if elapsed >= duration:
				state, time, speed = target, target_time, target_speed
				target, target_time, target_speed = None, 0.0, 1.0
				elapsed, duration = 0.0, 0.0
			else: time = self._time(state, source_end)
		self.state, self.time, self.speed = state, time, speed
		self.target_state, self.target_time, self.target_speed = target, target_time, target_speed
		self.elapsed, self.duration = elapsed, duration

	def sample(self):
		"""Return a clip-only transition triple; use pose() for blend states."""
		if self.state in self.graph.state_blend_spaces or self.target_state in self.graph.state_blend_spaces:
			raise ValueError("blend-state transitions require pose()")
		source = self.graph.state_clips[self.state]
		target = source if self.target_state is None else self.graph.state_clips[self.target_state]
		weight = self.elapsed / self.duration if self.target_state is not None else 0.0
		target_time = self.time if self.target_state is None else self.target_time
		return np.array([[source, target]]), np.array([[self.time, target_time]]), np.array([[1 - weight, weight]])

	def pose(self, parameters):
		"""Return an independent local pose, including any active state fade."""
		parameters = np.asarray(parameters, dtype=np.float64)
		if parameters.shape != self.graph.parameter_defaults.shape or not np.isfinite(parameters).all(): raise ValueError("parameters must be a finite graph parameter row")
		weight = self.elapsed / self.duration if self.target_state is not None else 0.0
		return transition_pose(self.graph, self.state, self.target_state, self.time, self.target_time, weight, parameters)
