"""Route graph rows and pack clip samples for the model-space crowd evaluator."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .AnimationGraph import CompiledAnimationGraph
from .AnimationGraphPlayback import AnimationGraphPlayback, sample_times, state_sample


if TYPE_CHECKING:
	from .CrowdGraphValidation import CrowdGraphEligibility


@dataclass(frozen=True, slots=True)
class CrowdGraphBatch:
	"""Playback row indices and the corresponding inputs to the crowd pass."""
	rows: np.ndarray
	clip_ids: np.ndarray
	times: np.ndarray
	weights: np.ndarray


def state_crowd_samples(graph: CompiledAnimationGraph, state: int, times: np.ndarray, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if state in graph.state_blend_spaces_2d:
		grid = graph.blend_spaces_2d[graph.state_blend_spaces_2d[state]]
		ids, horizontal, vertical = grid.select(parameters)
		clip_times = sample_times(graph, ids, times, grid.sync)
		weights = np.stack((1 - vertical, vertical), axis=1)[:, :, None] * horizontal[:, None, :]
		return ids.reshape(-1, 4), clip_times.reshape(-1, 4), weights.reshape(-1, 4)
	ids, clip_times, weights = state_sample(graph, state, times, parameters)
	if state in graph.state_clips: return ids[:, :1], clip_times[:, :1], weights[:, :1]
	return ids, clip_times, weights


def pack_crowd_samples(ids: np.ndarray, times: np.ndarray, weights: np.ndarray) -> CrowdGraphBatch:
	"""Discard zero contributions, merge identical samples and keep rows that fit the GPU pass."""
	weights = weights.copy()
	for column in range(weights.shape[1] - 1):
		later = slice(column + 1, None)
		same = (weights[:, column, None] > 0) & (ids[:, column, None] == ids[:, later]) & (times[:, column, None] == times[:, later])
		weights[:, column] += np.sum(np.where(same, weights[:, later], 0), axis=1)
		weights[:, later] = np.where(same, 0, weights[:, later])
	active = weights > 0
	counts = active.sum(axis=1)
	rows = np.flatnonzero(counts <= 4)
	width = int(counts[rows].max(initial=1))
	order = np.argsort(~active[rows], axis=1, kind="stable")[:, :width]
	return CrowdGraphBatch(rows, *(np.take_along_axis(values[rows], order, axis=1) for values in (ids, times, weights)))


def playback_crowd_samples(playback: AnimationGraphPlayback, parameters: np.ndarray, rows: np.ndarray) -> CrowdGraphBatch:
	"""Pack selected base poses from the current clocks without advancing playback."""
	states, targets = playback.states[rows], playback.target_states[rows]
	active = targets >= 0
	selected = parameters[rows]
	fade = np.divide(playback.elapsed[rows], playback.durations[rows], out=np.zeros(len(rows)), where=active)
	width = 8 if np.any(active) else 4
	ids = np.zeros((len(rows), width), dtype=np.int64)
	times, weights = np.zeros_like(ids, dtype=np.float64), np.zeros_like(ids, dtype=np.float64)
	for start, state_ids, clocks, blend in ((0, states, playback.times[rows], 1 - fade), (4, targets, playback.target_times[rows], fade)):
		for state in np.unique(state_ids[blend > 0]):
			mask = (state_ids == state) & (blend > 0)
			clips, phase, mix = state_crowd_samples(playback.graph, int(state), clocks[mask], selected[mask])
			slots = slice(start, start + clips.shape[1])
			ids[mask, slots], times[mask, slots], weights[mask, slots] = clips, phase, mix * blend[mask, None]
	batch = pack_crowd_samples(ids, times, weights)
	return replace(batch, rows=rows[batch.rows])


class CrowdGraphRouter:
	"""Partition a graph's playback rows using measured state and transition eligibility."""
	def __init__(self, eligibility: CrowdGraphEligibility) -> None:
		self.graph = eligibility.graph
		self.pool = eligibility.pool
		self._states = np.zeros(len(self.graph.state_ids), dtype=bool)
		for name, state in self.graph.state_ids.items(): self._states[state] = eligibility.states[name].eligible
		edges: dict[tuple[int, int, float], bool] = {}
		for edge in eligibility.transitions:
			key = (self.graph.state_ids[edge.source], self.graph.state_ids[edge.target], edge.duration)
			edges[key] = edges.get(key, True) and edge.result.eligible
		self._transitions = tuple(key for key, eligible in edges.items() if eligible)

	def split(self, playback: AnimationGraphPlayback, parameters: ArrayLike | None = None) -> tuple[np.ndarray, CrowdGraphBatch]:
		"""Return exact CPU row indices and a disjoint crowd batch from the same clocks.

		Evaluate application layers and root anchoring on the CPU when they are
		required. The supplied parameters use playback row order.
		"""
		if playback.graph is not self.graph: raise ValueError("crowd eligibility belongs to a different graph")
		shape = (playback.count, len(self.graph.parameter_defaults))
		values = np.broadcast_to(self.graph.parameter_defaults, shape) if parameters is None else np.asarray(parameters, dtype=np.float64)
		if values.shape != shape or not np.isfinite(values).all(): raise ValueError("parameters must be finite (instances, graph parameters) rows")
		states, targets = playback.states, playback.target_states
		candidates = self._states[states] & (targets < 0)
		for source, target, duration in self._transitions:
			candidates |= (states == source) & (targets == target) & (playback.durations == duration)
		batch = playback_crowd_samples(playback, values, np.flatnonzero(candidates))
		cpu = np.ones(playback.count, dtype=bool)
		cpu[batch.rows] = False
		return np.flatnonzero(cpu), batch
