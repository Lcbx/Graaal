"""Marker-based phase mapping for looping animation clips."""
from __future__ import annotations

from collections.abc import Mapping
from operator import index
from types import MappingProxyType

from numpy.typing import ArrayLike
import numpy as np

from .AnimationEvents import EventTrack


class BlendSpaceSync:
	"""Map selected samples from a fixed reference clip's marker clock.

	Tracks map nonnegative integer clip IDs to compatible EventTracks. This
	resource does not advance clocks, select weights, emit events or extract motion.
	"""
	def __init__(self, reference: int, tracks: Mapping[int, EventTrack]) -> None:
		if not isinstance(tracks, Mapping) or not tracks: raise ValueError("sync tracks must be a nonempty clip-ID mapping")
		for clip in (reference, *tracks):
			if isinstance(clip, (bool, np.bool_)): raise ValueError("sync clip IDs must be nonnegative integers")
			try: valid = index(clip) >= 0
			except TypeError as exc: raise ValueError("sync clip IDs must be nonnegative integers") from exc
			if not valid: raise ValueError("sync clip IDs must be nonnegative integers")
		reference = index(reference)
		if reference not in tracks: raise ValueError("reference clip must have a sync track")
		if any(not isinstance(track, EventTrack) for track in tracks.values()): raise ValueError("sync tracks must be EventTracks")
		tracks = {int(clip): EventTrack(track.duration, track.times, track.names) for clip, track in tracks.items()}
		self.reference = int(reference)
		mappings = {}
		for clip, track in tracks.items():
			try: mappings[clip] = SyncMapping(tracks[self.reference], track)
			except ValueError as exc: raise ValueError(f"clip {clip}: {exc}") from exc
		self.mappings = MappingProxyType(mappings)

	def map(self, clip_ids: ArrayLike, times: ArrayLike, *, reverse: bool=False) -> tuple[np.ndarray, np.ndarray]:
		"""Return sample times and rates for (instances, ...) selected clip IDs.

		Times are finite reference seconds shaped (instances,), including negative
		or unwrapped values. Outputs match the IDs' shape and own their storage.
		"""
		ids, times = np.asarray(clip_ids), np.asarray(times, dtype=np.float64)
		if ids.ndim < 2 or ids.dtype.kind not in "iu": raise ValueError("clip IDs must be an integer (instances, ...) array")
		if times.shape != (len(ids),) or not np.isfinite(times).all(): raise ValueError("reference times must be finite and match instances")
		if not isinstance(reverse, (bool, np.bool_)): raise ValueError("reverse must be boolean")
		selected = np.unique(ids)
		if any(int(clip) not in self.mappings for clip in selected): raise ValueError("selected clip has no sync track")
		clocks = np.broadcast_to(times.reshape((len(times),) + (1,) * (ids.ndim - 1)), ids.shape)
		mapped, rates = np.empty(ids.shape, dtype=np.float64), np.empty(ids.shape, dtype=np.float64)
		for clip in selected:
			mask = ids == clip
			mapped[mask], rates[mask] = self.mappings[int(clip)].map(clocks[mask], reverse=reverse)
		return mapped, rates


class SyncMapping:
	"""Compile a phase mapping between compatible cyclic event tracks.

	The tracks must have the same unique marker names in cyclic order, with at
	least two markers and no coincident times. Tracks may start at different
	markers and use different interval lengths.
	"""
	def __init__(self, source: EventTrack, target: EventTrack) -> None:
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

	def _local_clock(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		duration = self.source.duration
		local = np.mod(times, duration)
		with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
			cycles = np.floor(times / duration)
		rows = np.flatnonzero((cycles != 0) & (np.abs(cycles) < 2**52))
		if len(rows):
			markers = self.source.times
			right = np.searchsorted(markers, local[rows], side="right")
			# A wrapped timestamp can be off by an ulp at an authored marker.
			# Match the actual unwrapped crossing, without rounding nearby times.
			for candidate in (right - 1, right):
				phase = markers[candidate % len(markers)]
				with np.errstate(over="ignore", invalid="ignore"):
					cycle = np.rint((times[rows] - phase) / duration)
					exact = cycle * duration + phase == times[rows]
				local[rows[exact]], cycles[rows[exact]] = phase[exact], cycle[exact]
		return local, cycles

	def map(self, times: ArrayLike, *, reverse: bool=False) -> tuple[np.ndarray, np.ndarray]:
		"""Return wrapped target times and local time-rate ratios for source times.

		Input is a finite 1D array of source times, including negative or unwrapped
		values. The ratio converts source playback speed to target speed until the
		next marker; reverse playback uses the preceding rate at a marker. The
		mapping itself does not advance a player or fire events.
		"""
		times = np.asarray(times, dtype=np.float64)
		if times.ndim != 1 or not np.isfinite(times).all(): raise ValueError("source times must be a finite 1D array")
		if not isinstance(reverse, (bool, np.bool_)): raise ValueError("reverse must be boolean")
		local, _ = self._local_clock(times)
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

	def advance(self, start: ArrayLike, end: ArrayLike, target_start: ArrayLike) -> np.ndarray:
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
			local, cycle = self._local_clock(times)
			if not np.isfinite(cycle).all() or np.any(np.abs(cycle) >= 2**52): raise ValueError("sync interval exceeds supported loop count")
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
