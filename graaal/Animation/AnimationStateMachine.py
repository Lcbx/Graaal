"""Batched conditional transition selection for animation controllers."""
from __future__ import annotations

from operator import index
from typing import Iterable, Sequence, SupportsIndex

from numpy.typing import ArrayLike
import numpy as np


def _nonnegative_integer(value: SupportsIndex, name: str) -> int:
	if isinstance(value, (bool, np.bool_)): raise ValueError(f"{name} must be a nonnegative integer")
	try: result = index(value)
	except TypeError as exc: raise ValueError(f"{name} must be a nonnegative integer") from exc
	if result < 0: raise ValueError(f"{name} must be a nonnegative integer")
	if result > np.iinfo(np.uint64).max: raise ValueError(f"{name} exceeds uint64")
	return result


class AnimationStateMachine:
	"""Compile ordered conditional edges without owning playback state.

	Edges are (source, target, parameter, comparison, threshold, duration), or
	(source, target, conditions, duration), with conditions containing one or
	more (parameter, comparison, threshold) tuples. All conditions must match;
	the first matching edge wins per instance.
	"""
	def __init__(self, transitions: Iterable[tuple[int, int, Sequence[tuple[int, str, float]], float]]) -> None:
		operations = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal, "!=": np.not_equal}
		compiled = []
		for edge in transitions:
			if not isinstance(edge, (tuple, list)) or len(edge) not in (4, 6): raise ValueError("invalid transition")
			source, target = (_nonnegative_integer(value, name) for value, name in zip(edge[:2], ("source state", "target state")))
			if source == target: raise ValueError("transition states must differ")
			conditions = (edge[2:5],) if len(edge) == 6 else edge[2]
			if not isinstance(conditions, (tuple, list)) or not conditions: raise ValueError("transition requires at least one condition")
			compiled_conditions = []
			for condition in conditions:
				if not isinstance(condition, (tuple, list)) or len(condition) != 3: raise ValueError("invalid transition condition")
				parameter = _nonnegative_integer(condition[0], "parameter column")
				if not isinstance(condition[1], str) or condition[1] not in operations: raise ValueError("invalid transition comparison")
				try: threshold = float(condition[2])
				except (TypeError, ValueError, OverflowError) as exc: raise ValueError("threshold must be finite") from exc
				if not np.isfinite(threshold): raise ValueError("threshold must be finite")
				compiled_conditions.append((parameter, operations[condition[1]], threshold))
			try: duration = float(edge[-1])
			except (TypeError, ValueError, OverflowError) as exc: raise ValueError("duration must be finite and positive") from exc
			if not np.isfinite(duration) or duration <= 0: raise ValueError("duration must be finite and positive")
			compiled.append((source, target, tuple(compiled_conditions), duration))
		self._transitions = tuple(compiled)

	def select(self, states: ArrayLike, parameters: ArrayLike, transitioning: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return (selected, target states, durations) without changing inputs.

		Existing transitions are never interrupted. For unselected rows, target is
		the current state and duration is zero. Playback and state promotion are
		owned by the caller.
		"""
		states = np.asarray(states)
		parameters = np.asarray(parameters, dtype=np.float64)
		transitioning = np.asarray(transitioning)
		if states.ndim != 1 or states.dtype.kind not in "iu" or np.any(states < 0): raise ValueError("states must be a 1D array of nonnegative integers")
		if parameters.ndim != 2 or parameters.shape[0] != len(states) or not np.isfinite(parameters).all(): raise ValueError("parameters must be a finite (instances, columns) array")
		if transitioning.shape != states.shape or transitioning.dtype.kind != "b": raise ValueError("transitioning must be a boolean array aligned with states")
		if any(parameter >= parameters.shape[1] for _, _, conditions, _ in self._transitions for parameter, _, _ in conditions): raise ValueError("missing parameter column")
		# uint64 preserves all nonnegative input state IDs and compiled targets.
		targets = states.astype(np.uint64, copy=True)
		durations = np.zeros(len(states), dtype=np.float64)
		selected = np.zeros(len(states), dtype=bool)
		for source, target, conditions, duration in self._transitions:
			rows = ~transitioning & ~selected & (states == source)
			for parameter, compare, threshold in conditions: rows &= compare(parameters[:, parameter], threshold)
			targets[rows] = target
			durations[rows] = duration
			selected |= rows
		return selected, targets, durations
