"""Marker-based phase mapping for looping animation clips."""

import numpy as np

from .AnimationEvents import EventTrack


class SyncMapping:
	"""Compile a phase mapping between compatible cyclic event tracks.

	The tracks must have the same unique marker names in cyclic order, with at
	least two markers and no coincident times. Tracks may start at different
	markers and use different interval lengths.
	"""
	def __init__(self, source, target):
		if not isinstance(source, EventTrack) or not isinstance(target, EventTrack): raise ValueError("sync tracks must be event tracks")
		if len(source.names) < 2 or len(source.names) != len(target.names): raise ValueError("sync tracks require the same number of at least two markers")
		if len(set(source.names)) != len(source.names) or len(set(target.names)) != len(target.names): raise ValueError("sync marker names must be unique")
		if np.any(np.diff(source.times) <= 0) or np.any(np.diff(target.times) <= 0): raise ValueError("sync marker times must be distinct")
		if set(source.names) != set(target.names): raise ValueError("sync tracks must have the same marker names")
		start = target.names.index(source.names[0])
		indices = (np.arange(len(target.names)) + start) % len(target.names)
		if tuple(target.names[i] for i in indices) != source.names: raise ValueError("sync markers must have the same cyclic order")
		self.source, self.target = source, target
		self._target_times = target.times[indices]
		self._source_spans = np.mod(np.roll(source.times, -1) - source.times, source.duration)
		self._target_spans = np.mod(np.roll(self._target_times, -1) - self._target_times, target.duration)
		with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
			self._rates = self._target_spans / self._source_spans
		if not np.isfinite(self._rates).all() or np.any(self._rates <= 0): raise ValueError("sync intervals must have finite positive rate ratios")
		self._target_offsets = np.mod(self._target_times - self._target_times[0], target.duration)

	def map(self, times, *, reverse=False):
		"""Return wrapped target times and local time-rate ratios for source times.

		Input is a finite 1D array of source times, including negative or unwrapped
		values. The ratio converts source playback speed to target speed until the
		next marker; reverse playback uses the preceding rate at a marker. The
		mapping itself does not advance a player or fire events.
		"""
		times = np.asarray(times, dtype=np.float64)
		if times.ndim != 1 or not np.isfinite(times).all(): raise ValueError("source times must be a finite 1D array")
		if not isinstance(reverse, (bool, np.bool_)): raise ValueError("reverse must be boolean")
		local = np.mod(times, self.source.duration)
		segment = (np.searchsorted(self.source.times, local, side="left" if reverse else "right") - 1) % len(self.source.times)
		elapsed = np.mod(local - self.source.times[segment], self.source.duration)
		phase = elapsed / self._source_spans[segment]
		start = self._target_times[segment]
		delta = phase * self._target_spans[segment]
		remaining = self.target.duration - start
		wrapped = delta >= remaining
		target = np.empty_like(delta)
		target[wrapped] = delta[wrapped] - remaining[wrapped]
		target[~wrapped] = start[~wrapped] + delta[~wrapped]
		target = np.mod(target, self.target.duration)
		return target, self._rates[segment].copy()

	def advance(self, start, end, target_start):
		"""Return unwrapped target end times for aligned source intervals.

		The three inputs are equal-length finite 1D arrays. Target starts must
		already be aligned to the source starts. The interval may cross markers
		and loops in either direction; callers can traverse events and root motion
		using the returned unwrapped target endpoints.
		"""
		start, end, target_start = (np.asarray(value, dtype=np.float64) for value in (start, end, target_start))
		if start.ndim != 1 or end.shape != start.shape or target_start.shape != start.shape or not all(np.isfinite(value).all() for value in (start, end, target_start)):
			raise ValueError("interval endpoints and target starts must be equal-length finite 1D arrays")
		positions, cycles = [], []
		for times in (start, end):
			with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
				cycle = np.floor(times / self.source.duration)
			if not np.isfinite(cycle).all() or np.any(np.abs(cycle) >= 2**52): raise ValueError("sync interval exceeds supported loop count")
			local = np.mod(times, self.source.duration)
			segment = (np.searchsorted(self.source.times, local, side="right") - 1) % len(self.source.times)
			elapsed = np.mod(local - self.source.times[segment], self.source.duration)
			phase = elapsed / self._source_spans[segment]
			position = self._target_offsets[segment] + phase * self._target_spans[segment]
			position -= np.where(local < self.source.times[0], self.target.duration, 0)
			positions.append(position)
			cycles.append(cycle)
		with np.errstate(over="ignore", invalid="ignore"):
			result = target_start + (cycles[1] - cycles[0]) * self.target.duration + (positions[1] - positions[0])
		if not np.isfinite(result).all(): raise ValueError("target interval exceeds finite time range")
		return result
