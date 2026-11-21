"""Compiled, batched local-pose tasks built on the standalone animation API."""

from enum import StrEnum
from operator import index

import numpy as np

from .SkeletalAnimation import _array, _rotations, additive_pose, blend_poses, sample_clip_pairs
from .RootMotion import RootMotion
from .InverseKinematics import _validate_axis, _validate_chain, align_bone_axis, two_bone_pose


def _index(value, name):
	if isinstance(value, (bool, np.bool_)): raise ValueError(f"{name} must be a nonnegative integer")
	try: result = index(value)
	except TypeError as exc: raise ValueError(f"{name} must be a nonnegative integer") from exc
	if result < 0: raise ValueError(f"{name} must be a nonnegative integer")
	return result


class PoseOp(StrEnum):
	"""Operations accepted by a compiled pose plan."""
	SAMPLE = "sample"
	INPUT = "input"
	IN_PLACE = "in_place"
	TWO_BONE_IK = "two_bone_ik"
	ALIGN_AXIS = "align_axis"
	BLEND = "blend"
	ADDITIVE = "additive"


class PosePlan:
	"""Compile sample, blend and additive tasks into reusable pose buffers.

	Tasks are tuples: ("sample", sample_slot), ("input", pose_slot),
	("in_place", base_task, motion_slot), ("two_bone_ik", base_task, chain,
	goal_slot, weight_slot), ("align_axis", base_task, bone, goal_slot,
	weight_slot, axis), or
	("blend" | "additive", base_task, overlay_task, weight_slot, mask).
	Task references must point backward. The last task is the output.
	"""
	def __init__(self, skeleton, tasks):
		self.skeleton = skeleton
		compiled = []
		for task in tasks:
			if not isinstance(task, (tuple, list)) or not task: raise ValueError("invalid pose task")
			try: op = PoseOp(task[0])
			except (ValueError, TypeError) as exc: raise ValueError("invalid pose task") from exc
			if op in (PoseOp.SAMPLE, PoseOp.INPUT) and len(task) == 2:
				compiled.append((op, _index(task[1], "input slot")))
			elif op is PoseOp.IN_PLACE and len(task) == 3:
				base, motion = _index(task[1], "base task"), _index(task[2], "motion slot")
				if base >= len(compiled): raise ValueError("pose task inputs must reference earlier tasks")
				compiled.append((op, base, motion))
			elif op is PoseOp.TWO_BONE_IK and len(task) == 5:
				base = _index(task[1], "base task")
				if base >= len(compiled): raise ValueError("pose task inputs must reference earlier tasks")
				compiled.append((op, base, _validate_chain(skeleton, task[2]), _index(task[3], "goal slot"), _index(task[4], "weight slot")))
			elif op is PoseOp.ALIGN_AXIS and len(task) == 6:
				base, bone = _index(task[1], "base task"), _index(task[2], "bone")
				if base >= len(compiled): raise ValueError("pose task inputs must reference earlier tasks")
				if bone >= len(skeleton.parents): raise ValueError("bone must be a valid index")
				compiled.append((op, base, bone, _index(task[3], "goal slot"), _index(task[4], "weight slot"), tuple(_validate_axis(task[5]))))
			elif op in (PoseOp.BLEND, PoseOp.ADDITIVE) and len(task) == 5:
				base, overlay = _index(task[1], "base task"), _index(task[2], "overlay task")
				weight = _index(task[3], "weight slot")
				if base >= len(compiled) or overlay >= len(compiled): raise ValueError("pose task inputs must reference earlier tasks")
				mask = None if task[4] is None else np.array(task[4], dtype=np.float32, copy=True)
				if mask is not None:
					if mask.shape != (len(skeleton.parents),) or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
						raise ValueError("mask must contain one finite value in [0, 1] per bone")
					mask.flags.writeable = False
				compiled.append((op, base, overlay, weight, mask))
			else: raise ValueError("invalid pose task")
		if not compiled: raise ValueError("pose plan must contain at least one task")
		last_use = list(range(len(compiled)))
		for task_id, task in enumerate(compiled):
			if task[0] not in (PoseOp.SAMPLE, PoseOp.INPUT):
				for source in task[1:3] if task[0] in (PoseOp.BLEND, PoseOp.ADDITIVE) else task[1:2]: last_use[source] = task_id
		last_use[-1] = len(compiled)
		free, slots, allocated = [], [], 0
		for task_id, task in enumerate(compiled):
			if free: slot = free.pop()
			else:
				slot = allocated
				allocated += 1
			slots.append(slot)
			if task[0] not in (PoseOp.SAMPLE, PoseOp.INPUT):
				for source in set(task[1:3] if task[0] in (PoseOp.BLEND, PoseOp.ADDITIVE) else task[1:2]):
					if last_use[source] == task_id: free.append(slots[source])
			if last_use[task_id] == task_id: free.append(slot)
		self._tasks = tuple(compiled)
		self._slots = tuple(slots)
		self._buffer_count = allocated
		self._buffers = None

	def evaluate(self, clips, samples, weights, *, root_motions=(), poses=(), goals=()):
		"""Evaluate with sample triples and a (instances, weight slots) array.

		The returned local TRS arrays are reused by the next evaluation, including
		one that fails. Copy them if they must survive that call.
		"""
		weights = np.asarray(weights, dtype=np.float32)
		if weights.ndim != 2 or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
			raise ValueError("weights must be a finite (instances, slots) array in [0, 1]")
		count, weight_count = weights.shape
		input_poses = {}
		for task in self._tasks:
			if task[0] is PoseOp.SAMPLE:
				if task[1] >= len(samples): raise ValueError("missing sample slot")
				value = samples[task[1]]
				if not isinstance(value, (tuple, list)) or len(value) != 3 or np.shape(value[0]) != (count, 2):
					raise ValueError("sample slots must contain clip IDs, times and weights for each instance")
			elif task[0] is PoseOp.INPUT:
				if task[1] >= len(poses): raise ValueError("missing pose input slot")
				value = poses[task[1]]
				if not isinstance(value, (tuple, list)) or len(value) != 3: raise ValueError("pose slots must contain translation, rotation and scale")
				shape = (count, len(self.skeleton.parents))
				input_poses[task[1]] = (_array(value[0], shape + (3,), "pose translation"), _rotations(value[1], shape + (4,), "pose rotation"), _array(value[2], shape + (3,), "pose scale"))
			elif task[0] is PoseOp.IN_PLACE:
				if task[2] >= len(root_motions): raise ValueError("missing root motion slot")
				value = root_motions[task[2]]
				if not isinstance(value, (tuple, list)) or len(value) != 2: raise ValueError("root motion slots require a motion and instance mask")
				mask = np.asarray(value[1])
				if mask.shape != (count,) or mask.dtype.kind != "b": raise ValueError("root motion instance mask must be a boolean array")
				if value[0] is not None and (not isinstance(value[0], RootMotion) or value[0].skeleton is not self.skeleton): raise ValueError("root motion must match the plan skeleton")
				if value[0] is None and np.any(mask): raise ValueError("selected instances require root motion")
			elif task[0] in (PoseOp.TWO_BONE_IK, PoseOp.ALIGN_AXIS):
				if task[3] >= len(goals): raise ValueError("missing IK goal slot")
				value = goals[task[3]]
				expected = 2 if task[0] is PoseOp.TWO_BONE_IK else 1
				if not isinstance(value, (tuple, list)) or len(value) != expected or any(np.shape(item) != (count, 3) or not np.isfinite(item).all() for item in value): raise ValueError("goal slots must contain finite (instances, 3) arrays")
				if task[4] >= weight_count: raise ValueError("missing weight slot")
			elif task[3] >= weight_count: raise ValueError("missing weight slot")
		if self._buffers is None or self._buffers[0][0].shape[0] < count:
			capacity = max(count, 1 if self._buffers is None else len(self._buffers[0][0]) * 2)
			shape = (capacity, len(self.skeleton.parents))
			self._buffers = tuple(tuple(np.empty(shape + (size,), dtype=np.float32) for size in (3, 4, 3)) for _ in range(self._buffer_count))
		buffers = tuple(tuple(array[:count] for array in pose) for pose in self._buffers)
		for task, slot in zip(self._tasks, self._slots):
			if task[0] is PoseOp.SAMPLE:
				sample_clip_pairs(self.skeleton, clips, *samples[task[1]], out=buffers[slot])
			elif task[0] is PoseOp.INPUT:
				for output, values in zip(buffers[slot], input_poses[task[1]]): output[:] = values
			elif task[0] is PoseOp.IN_PLACE:
				motion, mask = root_motions[task[2]]
				mask = np.asarray(mask)
				base = buffers[self._slots[task[1]]]
				for output, values in zip(buffers[slot], base): output[:] = values
				if np.any(mask):
					pose = motion.in_place(*(values[mask] for values in base))
					for output, values in zip(buffers[slot], pose): output[mask] = values
			elif task[0] is PoseOp.TWO_BONE_IK:
				pose = two_bone_pose(self.skeleton, buffers[self._slots[task[1]]], task[2], *goals[task[3]], weight=weights[:, task[4]])
				for output, values in zip(buffers[slot], pose): output[:] = values
			elif task[0] is PoseOp.ALIGN_AXIS:
				pose = align_bone_axis(self.skeleton, buffers[self._slots[task[1]]], task[2], task[5], goals[task[3]][0], weight=weights[:, task[4]])
				for output, values in zip(buffers[slot], pose): output[:] = values
			else:
				operation = blend_poses if task[0] is PoseOp.BLEND else additive_pose
				pose = operation(self.skeleton, buffers[self._slots[task[1]]], buffers[self._slots[task[2]]], weights[:, task[3]], mask=task[4])
				for output, values in zip(buffers[slot], pose): output[:] = values
		return buffers[self._slots[-1]]
