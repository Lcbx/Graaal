"""Batched graph clocks and local poses, independent of ECS and graphics."""

from collections.abc import Iterator, Mapping
from copy import copy
from dataclasses import dataclass
from operator import index
from typing import SupportsIndex

import numpy as np
from numpy.typing import ArrayLike

from .AnimationEvents import EventTrack
from .AnimationGraph import CompiledAnimationGraph
from .AnimationSync import BlendSpaceSync, SyncMapping
from .RootMotion import RootMotion, _blend_deltas
from .SkeletalAnimation import blend_poses, sample_clip_pairs


@dataclass(frozen=True, slots=True)
class GraphEvent:
	"""A state event with its elapsed seconds from the start of the tick."""
	row: int
	state: int
	name: str
	offset: float
	role: str
	weight: float


def _event_tracks(graph: CompiledAnimationGraph, clock_clips: np.ndarray, tracks: Mapping[int, EventTrack] | None) -> dict[int, EventTrack]:
	if tracks is None: return {}
	if not isinstance(tracks, Mapping): raise ValueError("event_tracks must map compiled state IDs to EventTracks")
	result = {}
	for state, track in tracks.items():
		if isinstance(state, (bool, np.bool_)): raise ValueError("event tracks require valid compiled state IDs")
		try: state = index(state)
		except TypeError as exc: raise ValueError("event tracks require valid compiled state IDs") from exc
		if not 0 <= state < len(clock_clips): raise ValueError("event tracks require valid compiled state IDs")
		if not isinstance(track, EventTrack): raise ValueError("event tracks must be EventTracks")
		snapshot = EventTrack(track.duration, track.times, track.names)
		clip = clock_clips[state]
		if clip >= 0 and snapshot.duration != graph.clips[clip].duration:
			raise ValueError("event track duration must match its state clock clip")
		if snapshot.names: result[state] = snapshot
	return result


def _crossings(track: EventTrack, rows: np.ndarray, starts: np.ndarray, ends: np.ndarray, speeds: np.ndarray, spans: np.ndarray) -> Iterator[tuple[int, str, float]]:
	for row in rows:
		for event in track.between(starts[row], ends[row]):
			offset = spans[row] if event.time == ends[row] else (event.time - starts[row]) / speeds[row]
			yield int(row), event.name, float(np.clip(offset, 0, spans[row]))


def _synced_crossings(
	track: EventTrack, mapping: SyncMapping, inverse: SyncMapping, rows: np.ndarray,
	source_start: np.ndarray, source_end: np.ndarray, target_end: np.ndarray,
	speeds: np.ndarray, spans: np.ndarray,
) -> Iterator[tuple[int, str, float]]:
	starts, _ = mapping.map(source_start[rows])
	ends = target_end[rows]
	unwrapped = mapping.advance(source_start[rows], source_end[rows], starts)
	cycle_ends = np.rint((unwrapped - ends) / track.duration) * track.duration
	target_markers = inverse.source.times
	source_markers, _ = inverse.map(target_markers)
	for row, start, end, cycle_end in zip(rows, starts, ends, cycle_ends):
		# Compare the last cycle in local time so the fade endpoint agrees
		# with the pose clock even when unwrapping would round the event time.
		loops = track._between(start, cycle_end) if cycle_end > 0 else []
		last_start = 0.0 if cycle_end > 0 else start
		last = track._between(last_start, end, loop=False) if end > last_start else []
		if not loops and not last: continue
		times = np.array([time for time, _ in loops] + [cycle_end + time for time, _ in last])
		indices = np.array([i for _, i in loops + last])
		distances = inverse.advance(np.full_like(times, start), times, np.zeros_like(times))
		offsets = np.clip(distances / speeds[row], 0, spans[row])
		offsets[times == cycle_end + end] = spans[row]
		# Use the same authored source time and cycle as source events at sync
		# markers. Inverse interpolation alone can put tied events out of order.
		markers = np.minimum(np.searchsorted(target_markers, track.times[indices]), len(target_markers) - 1)
		on_marker = target_markers[markers] == track.times[indices]
		exact_markers = source_markers[markers[on_marker]]
		duration = inverse.target.duration
		cycles = np.rint((source_start[row] + distances[on_marker] - exact_markers) / duration)
		crossings = exact_markers + cycles * duration
		offsets[on_marker] = np.clip((crossings - source_start[row]) / speeds[row], 0, spans[row])
		offsets[on_marker] = np.where(crossings == source_end[row], spans[row], offsets[on_marker])
		for index, offset in zip(indices, offsets):
			yield int(row), track.names[index], float(offset)


def _row(value: ArrayLike, count: int, name: str) -> np.ndarray:
	value = np.asarray(value, dtype=np.float64)
	if not np.isfinite(value).all() or np.any(value < 0):
		raise ValueError(f"{name} must be finite and nonnegative")
	if value.ndim == 0: value = np.full(count, value)
	if value.shape != (count,):
		raise ValueError(f"{name} must be a finite nonnegative scalar or instance row")
	return value


def sample_times(graph: CompiledAnimationGraph, ids: np.ndarray, times: np.ndarray, sync: BlendSpaceSync | None = None) -> np.ndarray:
	"""Map one state time per row to selected clip times, wrapping as needed."""
	if sync is not None: return sync.map(ids, times)[0]
	clocks = np.broadcast_to(times.reshape((len(times),) + (1,) * (ids.ndim - 1)), ids.shape)
	result = np.empty(ids.shape, dtype=np.float64)
	for clip in np.unique(ids):
		mask = ids == clip
		duration = graph.clips[int(clip)].duration
		result[mask] = np.mod(clocks[mask], duration) if duration > 0 else 0.0
	return result


def state_clock_clip(graph: CompiledAnimationGraph, state: int | None) -> int | None:
	"""Return the clip backing a state's clock for transition synchronization."""
	if state is None: return None
	if state in graph.state_clips: return graph.state_clips[state]
	if state in graph.state_blend_spaces:
		space = graph.blend_spaces[graph.state_blend_spaces[state]]
		return None if space.sync is None else space.sync.reference
	if state in graph.state_blend_spaces_2d:
		grid = graph.blend_spaces_2d[graph.state_blend_spaces_2d[state]]
		return None if grid.sync is None else grid.sync.reference
	return None


def _state_pair(graph: CompiledAnimationGraph, state: int, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, BlendSpaceSync | None]:
	if state in graph.state_blend_spaces:
		space = graph.blend_spaces[graph.state_blend_spaces[state]]
		ids, weights = space.select(parameters)
		return ids, weights, space.sync
	ids = np.full((len(parameters), 2), graph.state_clips[state], dtype=np.int64)
	return ids, np.tile([1.0, 0.0], (len(parameters), 1)), None


def state_sample(graph: CompiledAnimationGraph, state: int, times: np.ndarray, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Select clip pairs and sample times for a clip or 1D blend state."""
	if state in graph.state_blend_spaces_2d: raise ValueError("2D blend states require state_pose()")
	ids, weights, sync = _state_pair(graph, state, parameters)
	return ids, sample_times(graph, ids, times, sync), weights


def state_pose(graph: CompiledAnimationGraph, state: int, times: np.ndarray, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Evaluate rows of one state, blending 2D poses horizontally then vertically."""
	skeleton = graph.plan.skeleton
	if state in graph.state_blend_spaces_2d:
		space = graph.blend_spaces_2d[graph.state_blend_spaces_2d[state]]
		ids, horizontal, vertical = space.select(parameters)
		times = sample_times(graph, ids, times, space.sync)
		lower, upper = (sample_clip_pairs(skeleton, graph.clips, ids[:, row], times[:, row], horizontal) for row in (0, 1))
		return blend_poses(skeleton, lower, upper, vertical)
	return sample_clip_pairs(skeleton, graph.clips, *state_sample(graph, state, times, parameters))


def _sample_deltas(motions: Mapping[int, RootMotion], ids: np.ndarray, start: np.ndarray, end: np.ndarray, sync: BlendSpaceSync | None) -> np.ndarray:
	result = np.empty(ids.shape + (4, 4), dtype=np.float64)
	shape = (len(ids),) + (1,) * (ids.ndim - 1)
	starts, ends = (np.broadcast_to(times.reshape(shape), ids.shape) for times in (start, end))
	for clip in np.unique(ids):
		selected = ids == clip
		lo, hi = starts[selected], ends[selected]
		if sync is not None:
			mapping = sync.mappings[int(clip)]
			mapped = mapping.map(lo)[0]
			lo, hi = mapped, mapping.advance(lo, hi, mapped)
		result[selected] = motions[int(clip)].character_delta(lo, hi)
	return result


class AnimationGraphPlayback:
	"""Own a fixed batch of clip/1D/2D state clocks and noninterruptible fades.

	Each advance selects at most one edge per instance. All clips loop; blend
	clocks retain unwrapped seconds. Explicit transition sync uses clip or fixed
	reference clocks, stops at promotion and retains the endpoint target speed.
	Optional event tracks are keyed by compiled state ID and read its clock.
	Optional root tracks extract character-space motion. Applications own
	accepted movement, layer composition and entity identity.
	"""
	states: np.ndarray
	times: np.ndarray
	speeds: np.ndarray
	target_states: np.ndarray
	target_times: np.ndarray
	target_speeds: np.ndarray
	elapsed: np.ndarray
	durations: np.ndarray
	_event_times: np.ndarray
	_target_event_times: np.ndarray
	root_motion: np.ndarray | None = None

	def __init__(self, graph: CompiledAnimationGraph, count: SupportsIndex, *, event_tracks: Mapping[int, EventTrack] | None = None, root_motions: Mapping[int, RootMotion] | None = None) -> None:
		if isinstance(count, (bool, np.bool_)): raise ValueError("count must be a nonnegative integer")
		try: count = index(count)
		except TypeError as exc: raise ValueError("count must be a nonnegative integer") from exc
		if count < 0: raise ValueError("count must be a nonnegative integer")
		self.graph, self.count = graph, count
		self._clock_clips = np.full(len(graph.state_ids), -1, dtype=np.int64)
		for state in graph.state_ids.values():
			clip = state_clock_clip(graph, state)
			if clip is not None: self._clock_clips[state] = clip
		self._event_tracks = _event_tracks(graph, self._clock_clips, event_tracks)
		event_clips = {self._clock_clips[state] for state in self._event_tracks}
		self._inverse_sync = {pair: SyncMapping(mapping.target, mapping.source)
			for pair, mapping in graph.sync_mappings.items() if pair[1] in event_clips}
		self._motions = {} if root_motions is None else dict(root_motions)
		self._motion_reference: RootMotion | None = None
		if root_motions is not None:
			participants = set(graph.state_clips.values())
			for spaces, states in ((graph.blend_spaces, graph.state_blend_spaces), (graph.blend_spaces_2d, graph.state_blend_spaces_2d)):
				for name in states.values(): participants.update(spaces[name].clip_ids.flat)
			for clip in sorted(participants):
				motion = self._motions.get(clip)
				if not isinstance(motion, RootMotion) or motion.clip is not graph.clips[clip] or motion.skeleton is not graph.plan.skeleton:
					raise ValueError(f"Missing or incompatible root motion for clip {clip}")
				if self._motion_reference is not None and self._motion_reference.root != motion.root:
					raise ValueError("Graph root motion must use the same root for each state clip")
				self._motion_reference = motion
			if self._motion_reference is not None: self._motion_reference._reference()
		self.reset()

	def _install(self, values: Mapping[str, np.ndarray]) -> None:
		# Read-only snapshots prevent accidental edits and survive later updates.
		for name, value in values.items():
			value.flags.writeable = False
			setattr(self, name, value)

	def _times(self, states: np.ndarray, times: np.ndarray) -> np.ndarray:
		if not np.isfinite(times).all(): raise ValueError("playback clock overflow")
		times = times.copy()
		for state, clip in self.graph.state_clips.items():
			mask = states == state
			duration = self.graph.clips[clip].duration
			times[mask] = np.mod(times[mask], duration) if duration > 0 else 0.0
		return times

	def _parameters(self, parameters: ArrayLike | None) -> np.ndarray:
		if parameters is None: parameters = np.broadcast_to(self.graph.parameter_defaults, (self.count, len(self.graph.parameter_defaults)))
		parameters = np.asarray(parameters, dtype=np.float64)
		if parameters.shape != (self.count, len(self.graph.parameter_defaults)) or not np.isfinite(parameters).all():
			raise ValueError("parameters must be finite (instances, graph parameters) rows")
		return parameters

	def _defaults(self, count: int) -> dict[str, np.ndarray]:
		values = {
			"states": np.full(count, self.graph.initial_state, dtype=np.int64),
			"times": np.zeros(count), "speeds": np.ones(count),
			"target_states": np.full(count, -1, dtype=np.int64),
			"target_times": np.zeros(count), "target_speeds": np.ones(count),
			"elapsed": np.zeros(count), "durations": np.zeros(count),
		}
		if self._event_tracks:
			values.update(_event_times=np.zeros(count), _target_event_times=np.zeros(count))
		if self._motion_reference is not None:
			values["root_motion"] = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
		return values

	def remap(self, rows: ArrayLike) -> "AnimationGraphPlayback":
		"""Copy clocks in the requested row order, using -1 for new instances."""
		rows = np.asarray(rows)
		if rows.shape == (0,): rows = rows.astype(np.int64)
		if rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < -1) or np.any(rows >= self.count):
			raise ValueError("rows must contain instance indices or -1")
		result = copy(self)
		result.count = len(rows)
		values = self._defaults(result.count)
		keep = rows >= 0
		for name, value in values.items(): value[keep] = getattr(self, name)[rows[keep]]
		result._install(values)
		return result

	def reset(self, states: ArrayLike | None = None, times: ArrayLike = 0.0, *, speeds: ArrayLike = 1.0, rows: ArrayLike | None = None) -> None:
		"""Seek all or selected rows, cancel their fades and set nonnegative speeds.

		States are compiled integer IDs. States/times/speeds may be scalars or
		arrays aligned with the selected rows; omitted states use initial_state.
		"""
		if rows is None: rows = np.arange(self.count)
		else:
			rows = np.asarray(rows)
			if rows.shape == (0,): rows = rows.astype(np.int64)
			if rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0) or np.any(rows >= self.count) or len(np.unique(rows)) != len(rows):
				raise ValueError("rows must be unique valid instance indices")
		states = np.asarray(self.graph.initial_state if states is None else states)
		if states.shape == (0,): states = states.astype(np.int64)
		if states.dtype.kind not in "iu" or np.any(states < 0) or np.any(states >= len(self.graph.state_ids)):
			raise ValueError("states must contain valid compiled state IDs")
		if states.ndim == 0: states = np.full(len(rows), states, dtype=states.dtype)
		if states.shape != (len(rows),):
			raise ValueError("states must contain valid compiled state IDs")
		times = self._times(states, _row(times, len(rows), "times"))
		speeds = _row(speeds, len(rows), "speeds")
		values = self._defaults(self.count)
		if hasattr(self, "states"): values = {name: getattr(self, name).copy() for name in values}
		for name, value in (("states", states), ("times", times), ("speeds", speeds), ("target_states", -1), ("target_times", 0), ("target_speeds", 1), ("elapsed", 0), ("durations", 0)):
			values[name][rows] = value
		if self._event_tracks:
			values["_event_times"][rows] = times
			values["_target_event_times"][rows] = 0.0
		if self._motion_reference is not None: values["root_motion"][rows] = np.eye(4)
		self._install(values)

	def advance(self, dt: ArrayLike, parameters: ArrayLike | None = None) -> tuple[GraphEvent, ...]:
		"""Advance scalar or per-instance seconds; failures preserve every row.

		Parameters have one row per instance, or default to the compiled defaults.
		Return a tuple of GraphEvents ordered by row, then tick offset, with the
		source first at ties. A zero dt can select a transition without crossings.
		"""
		dt = _row(dt, self.count, "dt")
		parameters = self._parameters(parameters)
		states, speeds = self.states.copy(), self.speeds.copy()
		targets, target_speeds = self.target_states.copy(), self.target_speeds.copy()
		# Keep event endpoints in the same cycle on both sides of a tick boundary.
		source_start = self._event_times if self._event_tracks else self.times
		target_times = (self._target_event_times if self._event_tracks else self.target_times).copy()
		elapsed, durations = self.elapsed.copy(), self.durations.copy()
		selected, next_states, next_durations = self.graph.machine.select(states, parameters, targets >= 0)
		targets[selected], durations[selected] = next_states[selected], next_durations[selected]
		target_times[selected], target_speeds[selected], elapsed[selected] = 0.0, speeds[selected], 0.0
		active = targets >= 0
		step = np.where(active, np.minimum(dt, np.maximum(durations - elapsed, 0)), dt)
		track_intervals = bool(self._event_tracks) or self._motion_reference is not None
		if self._event_tracks:
			target_start = target_times.copy()
			sync_rows = np.zeros(self.count, dtype=bool)
		if self._motion_reference is not None: motion_target_start = target_times.copy()
		with np.errstate(over="ignore", invalid="ignore"):
			source_end = source_start + speeds * step
			target_times[active] += target_speeds[active] * step[active]
			if self._motion_reference is not None: motion_target_end = target_times.copy()
			for (source, target), mapping in self.graph.sync_mappings.items():
				mask = active & (self._clock_clips[states] == source) & (self._clock_clips[np.maximum(targets, 0)] == target)
				if np.any(mask):
					mapped, rates = mapping.map(source_end[mask])
					target_times[mask], target_speeds[mask] = mapped, speeds[mask] * rates
					if self._event_tracks: sync_rows |= mask
					if self._motion_reference is not None:
						start = mapping.map(source_start[mask])[0]
						motion_target_start[mask] = start
						motion_target_end[mask] = mapping.advance(source_start[mask], source_end[mask], start)
			if track_intervals: target_fade_end = target_times.copy()
			target_times[active] += target_speeds[active] * (dt[active] - step[active])
			elapsed[active] = np.minimum(elapsed[active] + dt[active], durations[active])
		if not np.isfinite(target_speeds).all(): raise ValueError("playback speed overflow")
		times = self._times(states, source_end)
		events: tuple[GraphEvent, ...] = ()
		if self._event_tracks:
			if not np.isfinite(target_times).all(): raise ValueError("playback clock overflow")
			events = self._tick_events(
				states, targets, dt, step, durations, source_start, source_end, target_start,
				target_fade_end, target_times, target_speeds, sync_rows,
			)
			target_event_end = target_times.copy()
		if self._motion_reference is not None:
			motion = self._state_deltas(states, source_start, source_end, parameters)
			if np.any(active):
				target = targets[active]
				fade_delta = self._state_deltas(target, motion_target_start[active], motion_target_end[active], parameters[active])
				weight = (self.elapsed[active] + step[active] * 0.5) / durations[active]
				motion[active] = _blend_deltas(motion[active], fade_delta, weight)
				remaining = active & (dt > step)
				motion[remaining] = motion[remaining] @ self._state_deltas(targets[remaining], target_fade_end[remaining], target_times[remaining], parameters[remaining])
			if not np.isfinite(motion).all(): raise ValueError("root motion overflow")
		target_times[active] = self._times(targets[active], target_times[active])
		promoted = active & (elapsed >= durations)
		states[promoted], times[promoted], speeds[promoted] = targets[promoted], target_times[promoted], target_speeds[promoted]
		targets[promoted], target_times[promoted], target_speeds[promoted] = -1, 0.0, 1.0
		elapsed[promoted], durations[promoted] = 0.0, 0.0
		values = dict(states=states, times=times, speeds=speeds, target_states=targets, target_times=target_times, target_speeds=target_speeds, elapsed=elapsed, durations=durations)
		if self._event_tracks:
			source_end[promoted] = target_event_end[promoted]
			target_event_end[promoted] = 0.0
			values.update(_event_times=source_end, _target_event_times=target_event_end)
		if self._motion_reference is not None: values["root_motion"] = motion
		self._install(values)
		return events

	def _state_deltas(self, states: np.ndarray, start: np.ndarray, end: np.ndarray, parameters: np.ndarray) -> np.ndarray:
		motion = np.broadcast_to(np.eye(4), (len(states), 4, 4)).copy()
		moving = start != end
		for state in np.unique(states[moving]):
			rows = (states == state) & moving
			lo, hi, values = start[rows], end[rows], parameters[rows]
			if state in self.graph.state_clips:
				motion[rows] = self._motions[self.graph.state_clips[state]].character_delta(lo, hi)
			elif state in self.graph.state_blend_spaces_2d:
				space = self.graph.blend_spaces_2d[self.graph.state_blend_spaces_2d[state]]
				ids, horizontal, vertical = space.select(values)
				deltas = _sample_deltas(self._motions, ids, lo, hi, space.sync)
				lower, upper = (_blend_deltas(deltas[:, row, 0], deltas[:, row, 1], horizontal[:, 1]) for row in (0, 1))
				motion[rows] = _blend_deltas(lower, upper, vertical)
			else:
				ids, weights, sync = _state_pair(self.graph, int(state), values)
				deltas = _sample_deltas(self._motions, ids, lo, hi, sync)
				motion[rows] = _blend_deltas(deltas[:, 0], deltas[:, 1], weights[:, 1])
		return motion

	def _tick_events(
		self, states: np.ndarray, targets: np.ndarray, dt: np.ndarray, step: np.ndarray,
		durations: np.ndarray, source_start: np.ndarray, source_end: np.ndarray, target_start: np.ndarray,
		target_fade_end: np.ndarray, target_end: np.ndarray, target_speeds: np.ndarray, sync_rows: np.ndarray,
	) -> tuple[GraphEvent, ...]:
		events = []
		source_clips = self._clock_clips[states]
		remainder = dt - step
		for state, track in self._event_tracks.items():
			source_rows = np.flatnonzero((states == state) & (source_end > source_start))
			for row, name, offset in _crossings(track, source_rows, source_start, source_end, self.speeds, step):
				weight = 1.0 if targets[row] < 0 else max(0.0, 1.0 - (self.elapsed[row] + offset) / durations[row])
				events.append(GraphEvent(row, state, name, offset, "source", float(weight)))

			target_rows = targets == state
			independent = np.flatnonzero(target_rows & ~sync_rows & (target_end > target_start))
			for row, name, offset in _crossings(track, independent, target_start, target_end, target_speeds, dt):
				weight = min(1.0, (self.elapsed[row] + offset) / durations[row])
				events.append(GraphEvent(row, state, name, offset, "target", float(weight)))

			synced = target_rows & sync_rows
			moving = synced & (source_end > source_start)
			for clip in np.unique(source_clips[moving]):
				pair = clip, self._clock_clips[state]
				mapping, inverse = self.graph.sync_mappings[pair], self._inverse_sync[pair]
				rows = np.flatnonzero(moving & (source_clips == clip))
				crossings = _synced_crossings(track, mapping, inverse, rows, source_start, source_end, target_fade_end, self.speeds, step)
				for row, name, offset in crossings:
					weight = min(1.0, (self.elapsed[row] + offset) / durations[row])
					events.append(GraphEvent(row, state, name, offset, "target", float(weight)))

			continued = np.flatnonzero(synced & (target_end > target_fade_end))
			for row, name, offset in _crossings(track, continued, target_fade_end, target_end, target_speeds, remainder):
				events.append(GraphEvent(row, state, name, float(step[row] + offset), "target", 1.0))
		events.sort(key=lambda event: (event.row, event.offset, event.role))
		return tuple(events)

	def _poses(self, states: np.ndarray, times: np.ndarray, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		shape = (len(states), len(self.graph.plan.skeleton.parents))
		pose = (np.empty(shape + (3,), dtype=np.float32), np.empty(shape + (4,), dtype=np.float32), np.empty(shape + (3,), dtype=np.float32))
		for state in np.unique(states):
			rows = states == state
			local = state_pose(self.graph, int(state), times[rows], parameters[rows])
			for output, values in zip(pose, local): output[rows] = values
		return pose

	def pose(self, parameters: ArrayLike | None = None, *, in_place: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return independently owned batched local TRS before authored layers.

		Pass this pose in sample-slot order to graph.input_plan.evaluate(), with
		application-owned layer poses, weights and root-motion inputs as needed.
		Set in_place to anchor the root when root motion has been configured.
		"""
		parameters = self._parameters(parameters)
		pose = self._poses(self.states, self.times, parameters)
		active = self.target_states >= 0
		if np.any(active):
			target = self._poses(self.target_states[active], self.target_times[active], parameters[active])
			mixed = blend_poses(self.graph.plan.skeleton, tuple(values[active] for values in pose), target, self.elapsed[active] / self.durations[active])
			for output, values in zip(pose, mixed): output[active] = values
		if in_place and self._motion_reference is not None: return self._motion_reference.in_place(*pose)
		return pose
