"""Measure graph poses against the baked model-space crowd evaluator on the CPU."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from math import prod
from operator import index
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from .AnimationGraph import CompiledAnimationGraph
from .AnimationGraphPlayback import AnimationGraphPlayback, state_clock_clip, state_pose
from .AnimationStateMachine import AnimationStateMachine
from .BakedClipPool import BakedClipPool
from .CrowdGraphSamples import CrowdGraphBatch, pack_crowd_samples, playback_crowd_samples, state_crowd_samples
from .CrowdSkeletonLod import CrowdSkeletonLod, _skin_inputs
from .SkeletalAnimation import model_matrices, skin_matrices


_BATCH_SIZE = 256
_PoseErrors = Callable[[tuple[np.ndarray, np.ndarray, np.ndarray], CrowdGraphBatch], tuple[int, np.ndarray]]


@dataclass(frozen=True, slots=True)
class CrowdEligibility:
	"""Sampled state or transition eligibility and its maximum joint or vertex position error."""
	samples: int
	unsupported_samples: int
	max_position_error: float | None
	eligible: bool


@dataclass(frozen=True, slots=True)
class CrowdTransitionEligibility:
	source: str
	target: str
	duration: float
	result: CrowdEligibility


@dataclass(frozen=True, slots=True)
class CrowdGraphEligibility:
	"""State results by name and transition results in authored order."""
	error_budget: float
	states: Mapping[str, CrowdEligibility]
	transitions: tuple[CrowdTransitionEligibility, ...]
	graph: CompiledAnimationGraph = field(repr=False, compare=False)
	pool: BakedClipPool = field(repr=False, compare=False)
	joint_map: np.ndarray | None = field(default=None, repr=False, compare=False)


def _sample_count(value: int, minimum: int, name: str) -> int:
	if isinstance(value, (bool, np.bool_)): raise ValueError(f"{name} must be an integer >= {minimum}")
	try: value = index(value)
	except TypeError as exc: raise ValueError(f"{name} must be an integer >= {minimum}") from exc
	if value < minimum: raise ValueError(f"{name} must be an integer >= {minimum}")
	return value


def _state_axes(graph: CompiledAnimationGraph, state: int) -> dict[int, np.ndarray]:
	if state in graph.state_blend_spaces:
		space = graph.blend_spaces[graph.state_blend_spaces[state]]
		return {space.parameter: space.positions}
	if state in graph.state_blend_spaces_2d:
		grid = graph.blend_spaces_2d[graph.state_blend_spaces_2d[state]]
		return {grid.x_parameter: grid.x_positions, grid.y_parameter: grid.y_positions}
	return {}


def _parameter_axes(graph: CompiledAnimationGraph, blend_samples: int, *states: int) -> dict[int, np.ndarray]:
	knots: dict[int, list[np.ndarray]] = {}
	for state in states:
		for column, points in _state_axes(graph, state).items(): knots.setdefault(column, []).append(points)
	steps = np.linspace(0, 1, blend_samples)[1:-1]
	result = {}
	for column, arrays in knots.items():
		points = np.unique(np.concatenate(arrays))
		interior = points[:-1, None] * (1 - steps) + points[1:, None] * steps
		default = np.clip(graph.parameter_defaults[column], points[0], points[-1])
		result[column] = np.unique(np.r_[points, interior.ravel(), default])
	return result


def _phase_times(graph: CompiledAnimationGraph, state: int, phase_samples: int) -> np.ndarray:
	clock_clip = state_clock_clip(graph, state)
	if clock_clip is not None:
		durations = [graph.clips[clock_clip].duration]
	else:
		if state in graph.state_blend_spaces:
			ids = graph.blend_spaces[graph.state_blend_spaces[state]].clip_ids
		else:
			ids = graph.blend_spaces_2d[graph.state_blend_spaces_2d[state]].clip_ids
		durations = sorted({graph.clips[clip].duration for clip in ids.flat} - {0.0})
		# Sample the shorter cycles as well as relative phases beyond the first long cycle.
		if len(durations) > 1: durations.append(2 * durations[-1])
	if not durations: return np.zeros(1)
	return np.unique(np.concatenate([np.linspace(0, duration, phase_samples, endpoint=False) for duration in durations]))


def _sample_batches(graph: CompiledAnimationGraph, columns: Mapping[int, np.ndarray], *axes: np.ndarray) -> Iterator[tuple[np.ndarray, tuple[np.ndarray, ...]]]:
	values = axes + tuple(columns.values())
	shape = tuple(map(len, values))
	count = prod(shape)
	for start in range(0, count, _BATCH_SIZE):
		end = min(start + _BATCH_SIZE, count)
		coordinates = np.unravel_index(np.arange(start, end), shape)
		selected = tuple(axis[rows] for axis, rows in zip(values, coordinates))
		parameters = np.tile(graph.parameter_defaults, (end - start, 1))
		for column, data in zip(columns, selected[len(axes):]): parameters[:, column] = data
		yield parameters, selected[:len(axes)]


def _position_errors(pool: BakedClipPool, exact_pose: tuple[np.ndarray, np.ndarray, np.ndarray], batch: CrowdGraphBatch) -> tuple[int, np.ndarray]:
	skipped = len(exact_pose[0]) - len(batch.rows)
	if not len(batch.rows): return skipped, np.empty(0)
	approx = pool.model(batch.clip_ids, batch.times, batch.weights)[..., :3, 3]
	exact = model_matrices(pool.skeleton, *(values[batch.rows] for values in exact_pose))[..., :3, 3]
	return skipped, np.max(np.linalg.norm(approx - exact, axis=-1), axis=1)


def _measure(batches: Iterable[tuple[int, np.ndarray]], budget: float) -> CrowdEligibility:
	count, unsupported, maximum = 0, 0, None
	for skipped, errors in batches:
		count += skipped + len(errors)
		unsupported += skipped
		if len(errors):
			batch_max = float(errors.max())
			maximum = batch_max if maximum is None else max(maximum, batch_max)
	eligible = unsupported == 0 and maximum is not None and maximum <= budget
	return CrowdEligibility(count, unsupported, maximum, eligible)


def _state_errors(graph: CompiledAnimationGraph, measure_pose: _PoseErrors, state: int, phases: np.ndarray, axes: Mapping[int, np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
	for parameters, (times,) in _sample_batches(graph, axes, phases):
		exact = state_pose(graph, state, times, parameters)
		batch = pack_crowd_samples(*state_crowd_samples(graph, state, times, parameters))
		yield measure_pose(exact, batch)


def _transition_errors(graph: CompiledAnimationGraph, measure_pose: _PoseErrors, source: int, target: int, duration: float,
	phases: np.ndarray, axes: Mapping[int, np.ndarray], fade_samples: int, rates: np.ndarray,
) -> Iterator[tuple[int, np.ndarray]]:
	# Guards select an edge, but parameters can change freely once a fade starts.
	machine = AnimationStateMachine([(source, target, [(0, "<=", np.finfo(np.float64).max)], duration)])
	isolated = replace(graph, machine=machine)
	for parameters, (starts, elapsed, speeds) in _sample_batches(graph, axes, phases, np.linspace(0, duration, fade_samples), rates):
		playback = AnimationGraphPlayback(isolated, len(parameters))
		playback.reset(source, times=starts, speeds=speeds)
		playback.advance(elapsed, parameters)
		batch = playback_crowd_samples(playback, parameters, np.arange(len(parameters)))
		yield measure_pose(playback.pose(parameters), batch)


def validate_crowd_graph(graph: CompiledAnimationGraph, pool: BakedClipPool, *, error_budget: float,
	phase_samples: int = 16, blend_samples: int = 5, fade_samples: int = 9, playback_rates: ArrayLike = (1.0,),
) -> CrowdGraphEligibility:
	"""Sample base graph playback and report maximum bone-position errors in rig units.

	Use a pool for this skeleton with clips in compiled clip-ID order. The grid
	includes blend knots and defaults, each fade endpoint, and evenly spaced
	clock phases. Edges are evaluated independently of guards and priority.
	An entry is eligible when all sampled rows fit four samples and its maximum
	position error is within budget. This finite sweep covers base poses before
	application-owned layers, root anchoring, IK or other pose modifications.
	"""
	if pool.skeleton is not graph.plan.skeleton or len(pool.clips) != len(graph.clips) or any(clip is not graph.clips[i] for i, clip in enumerate(pool.clips)):
		raise ValueError("pool must contain the graph's skeleton and clips in compiled ID order")
	return _validate(graph, pool, lambda exact, batch: _position_errors(pool, exact, batch), error_budget, phase_samples, blend_samples, fade_samples, playback_rates)


def validate_reduced_crowd_graph(graph: CompiledAnimationGraph, lod: CrowdSkeletonLod, *, error_budget: float,
	fps: float = 30, phase_samples: int = 16, blend_samples: int = 5, fade_samples: int = 9, playback_rates: ArrayLike = (1.0,),
) -> CrowdGraphEligibility:
	"""Bake the reduced clips and measure their skinned vertex displacement against full CPU poses.

	Use the same sampled graph grid as validate_crowd_graph. The returned
	pool uses the reduced mesh's joint indices. The report combines reduction,
	bake and model-space blending error in source units.
	"""
	rig = graph.plan.skeleton
	if lod.source_skeleton is not rig or len(lod.source_clips) != len(graph.clips) or any(clip is not graph.clips[i] for i, clip in enumerate(lod.source_clips)):
		raise ValueError("LOD must use the graph's source skeleton and clips in compiled ID order")
	pool = BakedClipPool(lod.skeleton, lod.clips, fps=fps)
	bones = len(rig.parents)
	points, joints, weights = _skin_inputs(lod.source_primitives, bones)
	# Build weighted bind positions once. Skinning differences can then use a
	# matrix product without gathering a transform per vertex and influence.
	weighted_points = np.zeros((len(points), bones, 4), dtype=np.float32)
	np.add.at(weighted_points, (np.arange(len(points))[:, None], joints), weights[..., None] * points[:, None])
	basis = weighted_points.reshape(len(points), bones * 4)

	def measure_pose(pose: tuple[np.ndarray, np.ndarray, np.ndarray], batch: CrowdGraphBatch) -> tuple[int, np.ndarray]:
		count = len(batch.rows)
		exact = skin_matrices(rig, model_matrices(rig, *(values[batch.rows] for values in pose)))
		reduced = skin_matrices(lod.skeleton, pool.model(batch.clip_ids, batch.times, batch.weights))
		difference = (reduced[:, lod.joint_map] - exact).transpose(0, 2, 1, 3).reshape(count * 3, bones * 4)
		errors = np.zeros(count, dtype=np.float32)
		for start in range(0, count, 16):
			end = min(start + 16, count)
			displacement = (difference[3 * start:3 * end] @ basis.T).reshape(end - start, 3, len(points))
			errors[start:end] = np.sqrt(np.max(np.sum(displacement ** 2, axis=1), axis=1))
		return len(pose[0]) - count, errors

	report = _validate(graph, pool, measure_pose, error_budget, phase_samples, blend_samples, fade_samples, playback_rates)
	return replace(report, joint_map=lod.joint_map)


def _validate(graph: CompiledAnimationGraph, pool: BakedClipPool, measure_pose: _PoseErrors, error_budget: float,
	phase_samples: int, blend_samples: int, fade_samples: int, playback_rates: ArrayLike,
) -> CrowdGraphEligibility:
	if not np.isfinite(error_budget) or error_budget < 0: raise ValueError("error_budget must be finite and nonnegative")
	phase_samples = _sample_count(phase_samples, 1, "phase_samples")
	blend_samples = _sample_count(blend_samples, 3, "blend_samples")
	fade_samples = _sample_count(fade_samples, 3, "fade_samples")
	rates = np.asarray(playback_rates, dtype=np.float64)
	if rates.ndim != 1 or not len(rates) or not np.isfinite(rates).all() or np.any(rates < 0):
		raise ValueError("playback_rates must be a nonempty row of finite nonnegative rates")
	phase_times = {state: _phase_times(graph, state, phase_samples) for state in graph.state_ids.values()}
	states = {name: _measure(_state_errors(graph, measure_pose, state, phase_times[state], _parameter_axes(graph, blend_samples, state)), error_budget)
		for name, state in graph.state_ids.items()}
	names = {state: name for name, state in graph.state_ids.items()}
	transitions = []
	for source, target, duration in graph.machine.transitions:
		axes = _parameter_axes(graph, blend_samples, source, target)
		errors = _transition_errors(graph, measure_pose, source, target, duration, phase_times[source], axes, fade_samples, rates)
		transitions.append(CrowdTransitionEligibility(names[source], names[target], duration, _measure(errors, error_budget)))
	return CrowdGraphEligibility(float(error_budget), MappingProxyType(states), tuple(transitions), graph, pool)
