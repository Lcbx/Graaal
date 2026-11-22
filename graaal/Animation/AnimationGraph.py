"""Headless authoring of state rules and layered pose plans, without playback."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast, Any, Iterable

from numpy.typing import ArrayLike
import numpy as np

from .AnimationEvents import EventTrack
from .AnimationStateMachine import AnimationStateMachine
from .AnimationSync import BlendSpaceSync, SyncMapping
from .PoseTasks import PoseOp, PosePlan
from .SkeletalAnimation import Clip, Skeleton, blend_space_1d, blend_space_2d, subtree_mask


def _record(value: Any, allowed: Iterable[str], required: Iterable[str], path: str) -> Mapping[str, Any]:
	if not isinstance(value, Mapping): raise ValueError(f"{path}: expected a mapping")
	for key in value:
		if key not in allowed: raise ValueError(f"{path}.{key}: unsupported field")
	for key in required:
		if key not in value: raise ValueError(f"{path}.{key}: missing field")
	return value


def _named(value: Any, path: str) -> Mapping[str, Any]:
	if not isinstance(value, Mapping): raise ValueError(f"{path}: expected a mapping of names")
	if any(not isinstance(name, str) or not name for name in value): raise ValueError(f"{path}: names must be nonempty strings")
	return value


def _slots(value: Any, path: str) -> dict[str, int]:
	if not isinstance(value, (list, tuple)): raise ValueError(f"{path}: expected a list of names")
	result = {}
	for i, name in enumerate(value):
		if not isinstance(name, str) or not name or name in result: raise ValueError(f"{path}[{i}]: expected a unique nonempty name")
		result[name] = i
	return result


def _reference(value: Any, names: Mapping[str, Any], path: str) -> Any:
	if not isinstance(value, str) or value not in names: raise ValueError(f"{path}: unknown reference {value!r}")
	return names[value]


def _number(value: Any, path: str) -> float:
	try: result = float(value)
	except (TypeError, ValueError, OverflowError) as exc: raise ValueError(f"{path}: expected a finite number") from exc
	if not np.isscalar(value) or not np.isfinite(result): raise ValueError(f"{path}: expected a finite number")
	return result


def _compile_masks(skeleton: Skeleton, definitions: Any) -> dict[str, np.ndarray]:
	masks = {}
	for name, value in _named(definitions, "masks").items():
		path = f"masks.{name}"
		try:
			if isinstance(value, Mapping):
				_record(value, {"root"}, ("root",), path)
				masks[name] = subtree_mask(skeleton, value["root"])
			else:
				with np.errstate(over="ignore"): mask = np.array(value, dtype=np.float32, copy=True)
				if mask.shape != (len(skeleton.parents),) or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
					raise ValueError("expected one finite weight in [0, 1] per bone")
				masks[name] = mask
		except (ValueError, TypeError, OverflowError) as exc: raise ValueError(f"{path}: {exc}") from exc
	return masks


def _compile_transitions(definitions: Any, states: Mapping[str, int], parameters: Mapping[str, int]) -> AnimationStateMachine:
	if not isinstance(definitions, (list, tuple)): raise ValueError("transitions: expected a list")
	edges = []
	for i, edge in enumerate(definitions):
		path = f"transitions[{i}]"
		_record(edge, {"source", "target", "conditions", "duration"}, ("source", "target", "conditions", "duration"), path)
		source = _reference(edge["source"], states, path + ".source")
		target = _reference(edge["target"], states, path + ".target")
		if not isinstance(edge["conditions"], (list, tuple)): raise ValueError(f"{path}.conditions: expected a list")
		conditions = []
		for j, condition in enumerate(edge["conditions"]):
			condition_path = f"{path}.conditions[{j}]"
			_record(condition, {"parameter", "comparison", "threshold"}, ("parameter", "comparison", "threshold"), condition_path)
			parameter = _reference(condition["parameter"], parameters, condition_path + ".parameter")
			conditions.append((parameter, condition["comparison"], _number(condition["threshold"], condition_path + ".threshold")))
		edge = (source, target, conditions, _number(edge["duration"], path + ".duration"))
		try: AnimationStateMachine([edge])
		except ValueError as exc: raise ValueError(f"{path}: {exc}") from exc
		with np.errstate(over="ignore", under="ignore"): duration = np.float32(edge[-1])
		if not np.isfinite(duration) or duration <= 0: raise ValueError(f"{path}.duration: fade must fit positive float32 playback storage")
		edges.append(edge)
	return AnimationStateMachine(edges)


def _compile_tasks(skeleton: Skeleton, definitions: Any, samples: Mapping[str, int], weights: Mapping[str, int], motions: Mapping[str, int], masks: Mapping[str, np.ndarray]) -> tuple[PosePlan, PosePlan]:
	if not isinstance(definitions, (list, tuple)) or not definitions: raise ValueError("tasks: expected a nonempty list")
	names: dict[str, int] = {}
	tasks: list[tuple[Any, ...]] = []
	compiled: tuple[Any, ...]
	for i, task in enumerate(definitions):
		path = f"tasks[{i}]"
		if not isinstance(task, Mapping): raise ValueError(f"{path}: expected a mapping")
		name = task.get("name")
		if not isinstance(name, str) or not name or name in names: raise ValueError(f"{path}.name: expected a unique nonempty name")
		path += f" ({name})"
		try: op = PoseOp(cast(str, task.get("op")))
		except (ValueError, TypeError) as exc: raise ValueError(f"{path}.op: unsupported operation") from exc
		if op is PoseOp.SAMPLE:
			_record(task, {"name", "op", "slot"}, ("slot",), path)
			compiled = (op, _reference(task["slot"], samples, path + ".slot"))
		elif op is PoseOp.IN_PLACE:
			_record(task, {"name", "op", "base", "slot"}, ("base", "slot"), path)
			compiled = (op, _reference(task["base"], names, path + ".base (must name an earlier task)"), _reference(task["slot"], motions, path + ".slot"))
		elif op in (PoseOp.BLEND, PoseOp.ADDITIVE):
			_record(task, {"name", "op", "base", "overlay", "weight", "mask"}, ("base", "overlay", "weight"), path)
			base = _reference(task["base"], names, path + ".base (must name an earlier task)")
			overlay = _reference(task["overlay"], names, path + ".overlay (must name an earlier task)")
			weight = _reference(task["weight"], weights, path + ".weight")
			mask = None if task.get("mask") is None else _reference(task["mask"], masks, path + ".mask")
			compiled = (op, base, overlay, weight, mask)
		else: raise ValueError(f"{path}.op: operation {op.value!r} is not supported by graph definitions")
		tasks.append(compiled)
		names[name] = i
	input_tasks = [(PoseOp.INPUT, task[1]) if task[0] is PoseOp.SAMPLE else task for task in tasks]
	return PosePlan(skeleton, tasks), PosePlan(skeleton, input_tasks)


def _compile_sync(definitions: Any, tracks: Mapping[str, EventTrack] | None, clips: Mapping[str, Clip], clip_ids: Mapping[str, int]) -> Mapping[tuple[int, int], SyncMapping]:
	if not isinstance(definitions, (list, tuple)): raise ValueError("sync_mappings: expected a list")
	tracks = _named({} if tracks is None else tracks, "sync_tracks")
	mappings = {}
	for i, entry in enumerate(definitions):
		path = f"sync_mappings[{i}]"
		_record(entry, {"source", "target"}, ("source", "target"), path)
		source = _reference(entry["source"], clip_ids, path + ".source")
		target = _reference(entry["target"], clip_ids, path + ".target")
		if source == target or (source, target) in mappings: raise ValueError(f"{path}: expected a distinct, unique source/target clip pair")
		markers = []
		for role in ("source", "target"):
			name = entry[role]
			track = tracks.get(name)
			if not isinstance(track, EventTrack): raise ValueError(f"{path}.{role}: missing EventTrack for clip {name!r}")
			if track.duration != clips[name].duration: raise ValueError(f"{path}.{role}: marker duration must match clip duration")
			markers.append(track)
		try: mappings[source, target] = SyncMapping(*markers)
		except ValueError as exc: raise ValueError(f"{path}: {exc}") from exc
	return MappingProxyType(mappings)


@dataclass(frozen=True)
class CompiledBlendSpace1D:
	"""A named 1D selector; callers supply playback times for selected clips."""
	parameter: int
	positions: np.ndarray
	clip_ids: np.ndarray
	sync: BlendSpaceSync | None = None

	def select(self, parameters: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
		values = np.asarray(parameters, dtype=np.float64)
		if values.ndim != 2 or self.parameter >= values.shape[1]: raise ValueError("parameters must contain the blend-space column")
		return blend_space_1d(self.positions, self.clip_ids, values[:, self.parameter])


def _compile_space_sync(value: Any, path: str, ids: np.ndarray, clips: Mapping[str, int], resources: Mapping[str, Clip], tracks: Mapping[str, EventTrack] | None) -> BlendSpaceSync | None:
	if "sync_reference" not in value: return None
	path += ".sync_reference"
	reference = _reference(value["sync_reference"], clips, path)
	participants = set(int(clip) for clip in np.asarray(ids).flat)
	if reference not in participants: raise ValueError(f"{path}: reference must be a clip in the blend space")
	markers = {}
	for name, clip in clips.items():
		if clip not in participants: continue
		track = ({} if tracks is None else tracks).get(name)
		if not isinstance(track, EventTrack): raise ValueError(f"{path}: missing EventTrack for clip {name!r}")
		if track.duration != resources[name].duration: raise ValueError(f"{path}: marker duration must match clip {name!r}")
		markers[clip] = track
	try: return BlendSpaceSync(reference, markers)
	except ValueError as exc: raise ValueError(f"{path}: {exc}") from exc


def _compile_blend_spaces(definitions: Any, parameters: Mapping[str, int], clips: Mapping[str, int], resources: Mapping[str, Clip], tracks: Mapping[str, EventTrack] | None) -> Mapping[str, CompiledBlendSpace1D]:
	result = {}
	for name, value in _named(definitions, "blend_spaces").items():
		path = f"blend_spaces.{name}"
		_record(value, {"parameter", "points", "sync_reference"}, ("parameter", "points"), path)
		parameter = _reference(value["parameter"], parameters, path + ".parameter")
		points = value["points"]
		if not isinstance(points, (list, tuple)) or not points: raise ValueError(f"{path}.points: expected a nonempty list")
		positions: list[float] = []
		ids: list[int] = []
		for i, point in enumerate(points):
			item = f"{path}.points[{i}]"
			_record(point, {"position", "clip"}, ("position", "clip"), item)
			position = _number(point["position"], item + ".position")
			if positions and position <= positions[-1]: raise ValueError(f"{item}.position: positions must increase strictly")
			positions.append(position)
			ids.append(_reference(point["clip"], clips, item + ".clip"))
		position_array = np.array(positions, dtype=np.float64)
		id_array = np.array(ids, dtype=np.int64)
		position_array.flags.writeable = False
		id_array.flags.writeable = False
		sync = _compile_space_sync(value, path, id_array, clips, resources, tracks)
		result[name] = CompiledBlendSpace1D(parameter, position_array, id_array, sync)
	return MappingProxyType(result)


@dataclass(frozen=True)
class CompiledBlendSpace2D:
	"""Two horizontal sample pairs, optionally bound to pose-plan slots."""
	x_parameter: int
	y_parameter: int
	x_positions: np.ndarray
	y_positions: np.ndarray
	clip_ids: np.ndarray
	sample_slots: tuple
	weight_slot: int | None
	sync: BlendSpaceSync | None = None

	def select(self, parameters: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		values = np.asarray(parameters, dtype=np.float64)
		if values.ndim != 2 or max(self.x_parameter, self.y_parameter) >= values.shape[1]: raise ValueError("parameters must contain both blend-space columns")
		return blend_space_2d(self.x_positions, self.y_positions, self.clip_ids, values[:, [self.x_parameter, self.y_parameter]])


def _compile_blend_spaces_2d(definitions: Any, parameters: Mapping[str, int], clips: Mapping[str, int], samples: Mapping[str, int], weights: Mapping[str, int], resources: Mapping[str, Clip], tracks: Mapping[str, EventTrack] | None) -> Mapping[str, CompiledBlendSpace2D]:
	result = {}
	weight_owners: dict[int, str] = {}
	for name, value in _named(definitions, "blend_spaces_2d").items():
		path = f"blend_spaces_2d.{name}"
		_record(value, {"x_parameter", "y_parameter", "x_positions", "y_positions", "clips", "sample_slots", "weight_slot", "sync_reference"}, ("x_parameter", "y_parameter", "x_positions", "y_positions", "clips"), path)
		x_parameter = _reference(value["x_parameter"], parameters, path + ".x_parameter")
		y_parameter = _reference(value["y_parameter"], parameters, path + ".y_parameter")
		if x_parameter == y_parameter: raise ValueError(f"{path}.y_parameter: axis parameters must differ")
		axes = []
		for axis in ("x_positions", "y_positions"):
			positions = value[axis]
			if not isinstance(positions, (list, tuple)) or not positions: raise ValueError(f"{path}.{axis}: expected a nonempty list")
			array = np.array([_number(point, f"{path}.{axis}[{i}]") for i, point in enumerate(positions)], dtype=np.float64)
			if np.any(array[1:] <= array[:-1]): raise ValueError(f"{path}.{axis}: positions must increase strictly")
			array.flags.writeable = False
			axes.append(array)
		grid = value["clips"]
		if not isinstance(grid, (list, tuple)) or len(grid) != len(axes[1]): raise ValueError(f"{path}.clips: expected one row per y position")
		rows = []
		for y, row in enumerate(grid):
			if not isinstance(row, (list, tuple)) or len(row) != len(axes[0]): raise ValueError(f"{path}.clips[{y}]: expected one clip per x position")
			rows.append([_reference(clip, clips, f"{path}.clips[{y}][{x}]") for x, clip in enumerate(row)])
		ids = np.array(rows, dtype=np.int64)
		ids.flags.writeable = False
		sync = _compile_space_sync(value, path, ids, clips, resources, tracks)
		if "sample_slots" not in value and "weight_slot" not in value:
			result[name] = CompiledBlendSpace2D(x_parameter, y_parameter, axes[0], axes[1], ids, (), None, sync)
			continue
		for field in ("sample_slots", "weight_slot"):
			if field not in value: raise ValueError(f"{path}.{field}: sample_slots and weight_slot must be supplied together")
		slots = value["sample_slots"]
		if not isinstance(slots, (list, tuple)) or len(slots) != 2: raise ValueError(f"{path}.sample_slots: expected two distinct sample slots")
		compiled_slots = tuple(_reference(slot, samples, f"{path}.sample_slots[{i}]") for i, slot in enumerate(slots))
		if compiled_slots[0] == compiled_slots[1]: raise ValueError(f"{path}.sample_slots: expected two distinct sample slots")
		weight = _reference(value["weight_slot"], weights, path + ".weight_slot")
		if weight in weight_owners: raise ValueError(f"{path}.weight_slot: already used by blend_spaces_2d.{weight_owners[weight]}")
		weight_owners[weight] = name
		result[name] = CompiledBlendSpace2D(x_parameter, y_parameter, axes[0], axes[1], ids, compiled_slots, weight, sync)
	return MappingProxyType(result)


@dataclass(frozen=True)
class CompiledAnimationGraph:
	"""Runtime resources plus read-only name-to-column/ID bindings.

	input_plan mirrors plan with each sample slot replaced by a local-pose input
	at the same index, for callers composing state poses before authored layers.
	"""
	machine: AnimationStateMachine
	plan: PosePlan
	clips: Mapping
	clip_ids: Mapping
	state_ids: Mapping
	state_clips: Mapping
	initial_state: int
	parameter_columns: Mapping
	parameter_defaults: np.ndarray
	sample_slots: Mapping
	weight_columns: tuple
	root_motion_slots: Mapping
	sync_mappings: Mapping
	blend_spaces: Mapping
	blend_spaces_2d: Mapping
	state_blend_spaces: Mapping
	input_plan: PosePlan
	state_blend_spaces_2d: Mapping


class AnimationGraphDefinition:
	"""An editable copied dictionary; compilation never changes live resources.

	Version 1 supports single-clip and 1D/2D blend states, named selectors and
	sample/in-place/blend/additive tasks. The last task is the output;
	task inputs must refer backward.
	Playback, sample times, events, physics and ECS integration remain external.
	"""
	def __init__(self, data: Mapping[str, Any]) -> None:
		self.data = deepcopy(data)

	def compile(self, skeleton: Skeleton, clips: Mapping[str, Clip], *, sync_tracks: Mapping[str, EventTrack] | None=None) -> CompiledAnimationGraph:
		"""Validate the current draft and return independent compiled resources.

		Clips are a name-to-Clip mapping for one skeleton. Clip objects are shared;
		compiled rules, metadata and pose buffers are independent of later edits.
		Optional sync_tracks maps clip names to EventTracks used by sync_mappings.
		"""
		fields = {"version", "parameters", "states", "initial_state", "transitions", "sample_slots", "weight_slots", "root_motion_slots", "masks", "tasks", "sync_mappings", "blend_spaces", "blend_spaces_2d"}
		data = _record(self.data, fields, ("version", "states", "initial_state", "sample_slots", "tasks"), "graph")
		if type(data["version"]) is not int or data["version"] != 1: raise ValueError("graph.version: supported version is integer 1")
		if not isinstance(skeleton, Skeleton): raise ValueError("skeleton: expected a Skeleton")
		clips = _named(clips, "clips")
		clip_ids = {name: i for i, name in enumerate(clips)}
		for name, clip in clips.items():
			if not isinstance(clip, Clip) or clip.translation.shape[1] != len(skeleton.parents): raise ValueError(f"clips.{name}: expected a Clip with matching skeleton bone count")
		parameters = _named(data.get("parameters", {}), "parameters")
		parameter_columns = {name: i for i, name in enumerate(parameters)}
		defaults = np.array([_number(value, f"parameters.{name}") for name, value in parameters.items()], dtype=np.float64)
		defaults.flags.writeable = False
		states = _named(data["states"], "states")
		if not states: raise ValueError("states: expected at least one state")
		state_ids = {name: i for i, name in enumerate(states)}
		initial = _reference(data["initial_state"], state_ids, "initial_state")
		samples = _slots(data["sample_slots"], "sample_slots")
		weights = _slots(data.get("weight_slots", []), "weight_slots")
		motions = _slots(data.get("root_motion_slots", []), "root_motion_slots")
		weight_columns = tuple(_reference(name, parameter_columns, f"weight_slots[{slot}]") for name, slot in weights.items())
		for name, column in zip(weights, weight_columns):
			if not 0 <= defaults[column] <= 1: raise ValueError(f"parameters.{name}: layer weight default must be in [0, 1]")
		masks = _compile_masks(skeleton, data.get("masks", {}))
		machine = _compile_transitions(data.get("transitions", []), state_ids, parameter_columns)
		plan, input_plan = _compile_tasks(skeleton, data["tasks"], samples, weights, motions, masks)
		sync = _compile_sync(data.get("sync_mappings", []), sync_tracks, clips, clip_ids)
		blend_spaces = _compile_blend_spaces(data.get("blend_spaces", {}), parameter_columns, clip_ids, clips, sync_tracks)
		blend_spaces_2d = _compile_blend_spaces_2d(data.get("blend_spaces_2d", {}), parameter_columns, clip_ids, samples, weights, clips, sync_tracks)
		state_clips, state_blend_spaces, state_blend_spaces_2d = {}, {}, {}
		for name, state in states.items():
			path = f"states.{name}"
			_record(state, {"clip", "blend_space", "blend_space_2d"}, (), path)
			if not state: raise ValueError(f"{path}: expected exactly one of clip, blend_space or blend_space_2d")
			if len(state) != 1:
				field = "blend_space_2d" if "blend_space_2d" in state else "blend_space"
				raise ValueError(f"{path}.{field}: cannot be combined with another state source")
			if "clip" in state: state_clips[state_ids[name]] = _reference(state["clip"], clip_ids, path + ".clip")
			elif "blend_space" in state:
				_reference(state["blend_space"], blend_spaces, path + ".blend_space")
				state_blend_spaces[state_ids[name]] = state["blend_space"]
			else:
				_reference(state["blend_space_2d"], blend_spaces_2d, path + ".blend_space_2d")
				state_blend_spaces_2d[state_ids[name]] = state["blend_space_2d"]
		return CompiledAnimationGraph(
			machine, plan, MappingProxyType({clip_ids[name]: clip for name, clip in clips.items()}),
			MappingProxyType(clip_ids), MappingProxyType(state_ids), MappingProxyType(state_clips), initial,
			MappingProxyType(parameter_columns), defaults, MappingProxyType(samples), weight_columns, MappingProxyType(motions), sync, blend_spaces, blend_spaces_2d, MappingProxyType(state_blend_spaces), input_plan, MappingProxyType(state_blend_spaces_2d),
		)
