"""Headless authoring of state rules and layered pose plans, without playback."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .AnimationStateMachine import AnimationStateMachine
from .AnimationEvents import EventTrack
from .AnimationSync import SyncMapping
from .PoseTasks import PoseOp, PosePlan
from .SkeletalAnimation import Clip, Skeleton, subtree_mask


def _record(value, allowed, required, path):
	if not isinstance(value, Mapping): raise ValueError(f"{path}: expected a mapping")
	for key in value:
		if key not in allowed: raise ValueError(f"{path}.{key}: unsupported field")
	for key in required:
		if key not in value: raise ValueError(f"{path}.{key}: missing field")
	return value


def _named(value, path):
	if not isinstance(value, Mapping): raise ValueError(f"{path}: expected a mapping of names")
	if any(not isinstance(name, str) or not name for name in value): raise ValueError(f"{path}: names must be nonempty strings")
	return value


def _slots(value, path):
	if not isinstance(value, (list, tuple)): raise ValueError(f"{path}: expected a list of names")
	result = {}
	for i, name in enumerate(value):
		if not isinstance(name, str) or not name or name in result: raise ValueError(f"{path}[{i}]: expected a unique nonempty name")
		result[name] = i
	return result


def _reference(value, names, path):
	if not isinstance(value, str) or value not in names: raise ValueError(f"{path}: unknown reference {value!r}")
	return names[value]


def _number(value, path):
	try: result = float(value)
	except (TypeError, ValueError, OverflowError) as exc: raise ValueError(f"{path}: expected a finite number") from exc
	if not np.isscalar(value) or not np.isfinite(result): raise ValueError(f"{path}: expected a finite number")
	return result


def _compile_masks(skeleton, definitions):
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


def _compile_transitions(definitions, states, parameters):
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


def _compile_tasks(skeleton, definitions, samples, weights, motions, masks):
	if not isinstance(definitions, (list, tuple)) or not definitions: raise ValueError("tasks: expected a nonempty list")
	names, tasks = {}, []
	for i, task in enumerate(definitions):
		path = f"tasks[{i}]"
		if not isinstance(task, Mapping): raise ValueError(f"{path}: expected a mapping")
		name = task.get("name")
		if not isinstance(name, str) or not name or name in names: raise ValueError(f"{path}.name: expected a unique nonempty name")
		path += f" ({name})"
		try: op = PoseOp(task.get("op"))
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
	return PosePlan(skeleton, tasks)


def _compile_sync(definitions, tracks, clips, clip_ids):
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
class CompiledAnimationGraph:
	"""Existing runtime resources plus read-only name-to-column/ID bindings."""
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


class AnimationGraphDefinition:
	"""An editable copied dictionary; compilation never changes live resources.

	Version 1 supports single-clip states and sample/in-place/blend/additive
	tasks. The last task is the output; task inputs must refer backward.
	Playback, sample times, events, physics and ECS integration remain external.
	"""
	def __init__(self, data):
		self.data = deepcopy(data)

	def compile(self, skeleton, clips, *, sync_tracks=None):
		"""Validate the current draft and return independent compiled resources.

		Clips are a name-to-Clip mapping for one skeleton. Clip objects are shared;
		compiled rules, metadata and pose buffers are independent of later edits.
		Optional sync_tracks maps clip names to EventTracks used by sync_mappings.
		"""
		fields = {"version", "parameters", "states", "initial_state", "transitions", "sample_slots", "weight_slots", "root_motion_slots", "masks", "tasks", "sync_mappings"}
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
		state_clips = {}
		for name, state in states.items():
			_record(state, {"clip"}, ("clip",), f"states.{name}")
			state_clips[state_ids[name]] = _reference(state["clip"], clip_ids, f"states.{name}.clip")
		initial = _reference(data["initial_state"], state_ids, "initial_state")
		samples = _slots(data["sample_slots"], "sample_slots")
		weights = _slots(data.get("weight_slots", []), "weight_slots")
		motions = _slots(data.get("root_motion_slots", []), "root_motion_slots")
		weight_columns = tuple(_reference(name, parameter_columns, f"weight_slots[{slot}]") for name, slot in weights.items())
		for name, column in zip(weights, weight_columns):
			if not 0 <= defaults[column] <= 1: raise ValueError(f"parameters.{name}: layer weight default must be in [0, 1]")
		masks = _compile_masks(skeleton, data.get("masks", {}))
		machine = _compile_transitions(data.get("transitions", []), state_ids, parameter_columns)
		plan = _compile_tasks(skeleton, data["tasks"], samples, weights, motions, masks)
		sync = _compile_sync(data.get("sync_mappings", []), sync_tracks, clips, clip_ids)
		return CompiledAnimationGraph(
			machine, plan, MappingProxyType({clip_ids[name]: clip for name, clip in clips.items()}),
			MappingProxyType(clip_ids), MappingProxyType(state_ids), MappingProxyType(state_clips), initial,
			MappingProxyType(parameter_columns), defaults, MappingProxyType(samples), weight_columns, MappingProxyType(motions), sync,
		)
