"""Animation event tracks and interval traversal, independent of ECS."""
from dataclasses import dataclass
import math
from operator import index

import numpy as np


@dataclass(frozen=True)
class EventOccurrence:
	"""A track event and the unwrapped playback time at which it occurred."""
	name: str
	time: float


class EventTrack:
	"""Named events in the half-open interval [0, duration)."""
	def __init__(self, duration, times, names):
		if not np.isscalar(duration) or not np.isfinite(duration) or duration <= 0:
			raise ValueError("event track duration must be positive and finite")
		values = np.array(times, dtype=np.float64, copy=True)
		names = tuple(names)
		if values.ndim != 1 or len(values) != len(names) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= duration):
			raise ValueError("event times must be finite, in [0, duration), and match names")
		if any(not isinstance(name, str) or not name for name in names):
			raise ValueError("event names must be nonempty strings")
		order = np.argsort(values, kind="stable")
		self.duration = float(duration)
		self.times = values[order]
		self.times.flags.writeable = False
		self.names = tuple(names[i] for i in order)

	def between(self, start, end, *, loop=True):
		"""Return events crossed from start to end, excluding start and including end.

		Looping accepts unwrapped times and reports every crossing, including
		multiple loops. Non-looping traversal clamps to the clip interval.
		"""
		if not np.isscalar(start) or not np.isscalar(end) or not np.isfinite(start) or not np.isfinite(end):
			raise ValueError("event interval endpoints must be finite scalars")
		start, end = float(start), float(end)
		if not loop:
			start, end = np.clip([start, end], 0, self.duration)
		if start == end: return []
		forward = end > start
		found = []
		for index, (time, name) in enumerate(zip(self.times, self.names)):
			if loop:
				if forward:
					first = math.floor((start - time) / self.duration) + 1
					last = math.floor((end - time) / self.duration)
				else:
					first = math.ceil((end - time) / self.duration)
					last = math.ceil((start - time) / self.duration) - 1
				for cycle in range(first, last + 1):
					found.append((float(time + cycle * self.duration), index, name))
			elif (start < time <= end) if forward else (end <= time < start):
				found.append((float(time), index, name))
		found.sort(key=lambda item: (item[0] if forward else -item[0], item[1]))
		return [EventOccurrence(name, time) for time, _, name in found]


class EventTimeline:
	"""Edit event markers with stable IDs and produce independent event tracks."""
	def __init__(self, track):
		if not isinstance(track, EventTrack): raise ValueError("timeline requires an EventTrack")
		track = EventTrack(track.duration, track.times, track.names)
		self._duration = float(track.duration)
		self._events = [(i, float(time), name) for i, (time, name) in enumerate(zip(track.times, track.names))]
		self._next_id = len(self._events)

	@property
	def duration(self): return self._duration

	def _find(self, event_id):
		if isinstance(event_id, (bool, np.bool_)): raise ValueError("event ID must be a nonnegative integer")
		try: event_id = index(event_id)
		except TypeError as exc: raise ValueError("event ID must be a nonnegative integer") from exc
		if event_id < 0: raise ValueError("event ID must be a nonnegative integer")
		for position, event in enumerate(self._events):
			if event[0] == event_id: return position
		raise KeyError(event_id)

	def _track(self, events):
		return EventTrack(self.duration, [event[1] for event in events], [event[2] for event in events])

	def snapshot(self):
		"""Return (event IDs, EventTrack), ordered by time and stable at ties."""
		ordered = sorted(self._events, key=lambda event: event[1])
		return tuple(event[0] for event in ordered), self._track(ordered)

	def add(self, time, name):
		"""Insert a marker and return its stable ID."""
		event = (self._next_id, time, name)
		self._track([event])
		self._events.append((event[0], float(time), name))
		self._next_id += 1
		return event[0]

	def update(self, event_id, *, time=None, name=None):
		"""Move or rename a marker; omitted fields retain their values."""
		position = self._find(event_id)
		old_id, old_time, old_name = self._events[position]
		event = (old_id, old_time if time is None else time, old_name if name is None else name)
		self._track([event])
		self._events[position] = (old_id, float(event[1]), event[2])

	def remove(self, event_id):
		"""Remove one marker by ID; removed IDs are not reused."""
		del self._events[self._find(event_id)]
