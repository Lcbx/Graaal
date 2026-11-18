"""Animation event tracks and interval traversal, independent of ECS."""
from dataclasses import dataclass
import math

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
