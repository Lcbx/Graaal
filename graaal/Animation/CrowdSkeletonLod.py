"""Generate a compact crowd skeleton and matching skin weights from clip poses."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .SkeletalAnimation import Clip, Skeleton, model_matrices, skin_matrices
from .SkeletalGltf import SkinnedPrimitive


@dataclass(frozen=True, slots=True)
class CrowdSkeletonLod:
	"""Reduced resources in source clip order, with compact indices for every source joint."""
	skeleton: Skeleton
	clips: tuple[Clip, ...]
	primitives: tuple[SkinnedPrimitive, ...]
	source_skeleton: Skeleton
	source_clips: tuple[Clip, ...]
	source_primitives: tuple[SkinnedPrimitive, ...]
	joint_map: np.ndarray
	max_position_error_bound: float


def _skin_inputs(primitives: Sequence[SkinnedPrimitive], bones: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	positions, joints, weights = [], [], []
	if not primitives: raise ValueError("skeleton LOD needs at least one skinned primitive")
	for primitive in primitives:
		vertices = primitive.vertices
		if vertices.ndim != 1 or not len(vertices) or not {"position", "joints", "weights"} <= set(vertices.dtype.names or ()):
			raise ValueError("expected skinned vertices with position, joints and weights")
		p, j, w = vertices["position"], vertices["joints"], vertices["weights"]
		if p.shape != (len(vertices), 3) or j.shape != (len(vertices), 4) or w.shape != j.shape:
			raise ValueError("expected vec3 positions and four skin influences per vertex")
		if j.dtype.kind not in "iu" or np.any(j < 0) or np.any(j >= bones):
			raise ValueError("vertex joints must index the source skeleton")
		if not np.isfinite(p).all() or not np.isfinite(w).all() or np.any(w < 0) or not np.allclose(w.sum(axis=1), 1, rtol=0, atol=1e-5):
			raise ValueError("expected finite positions and nonnegative skin weights summing to one")
		positions.append(p)
		joints.append(j)
		weights.append(w)
	points = np.ones((sum(map(len, positions)), 4), dtype=np.float32)
	points[:, :3] = np.concatenate(positions)
	return points, np.concatenate(joints), np.concatenate(weights).astype(np.float32, copy=False)


def _remap_primitive(primitive: SkinnedPrimitive, joint_map: np.ndarray) -> SkinnedPrimitive:
	vertices = primitive.vertices.copy()
	joints, weights = vertices["joints"], vertices["weights"]
	joints[:] = joint_map[joints]
	for slot in range(joints.shape[1] - 1):
		later = slice(slot + 1, None)
		same = joints[:, slot, None] == joints[:, later]
		weights[:, slot] += np.sum(np.where(same, weights[:, later], 0), axis=1)
		weights[:, later] = np.where(same, 0, weights[:, later])
	joints[weights == 0] = 0
	vertices.flags.writeable = False
	return SkinnedPrimitive(vertices, primitive.indices, primitive.material_id)


def _merge_displacement(points: np.ndarray, matrices: np.ndarray, weight: np.ndarray) -> np.ndarray:
	# Bound each vertex over the authored frames and keep scratch independent of clip length.
	maximum = np.zeros(len(points), dtype=np.float32)
	for start in range(0, len(matrices), 16):
		displacement = np.einsum("fij,vj->fvi", matrices[start:start + 16], points)
		np.maximum(maximum, np.max(np.sum(displacement ** 2, axis=-1), axis=0), out=maximum)
	return np.sqrt(maximum) * weight


def reduce_crowd_skeleton(skeleton: Skeleton, clips: Sequence[Clip], primitives: Sequence[SkinnedPrimitive], *, error_budget: float) -> CrowdSkeletonLod | None:
	"""Collapse leaf influences to their parents within an authored-frame position budget.

	The generator retains roots and ancestors of surviving joints. It tests
	weighted vertex displacement in source units and sums the bounds across
	merges. A separate graph sweep is needed to cover blended/baked poses.
	Returns None when the budget allows no joint to be removed.
	"""
	if not clips or not np.isfinite(error_budget) or error_budget < 0:
		raise ValueError("skeleton LOD needs clips and a finite nonnegative position budget")
	parents = skeleton.parents
	points, joints, weights = _skin_inputs(primitives, len(parents))
	matrices = np.concatenate([skin_matrices(skeleton, model_matrices(skeleton, clip.translation, clip.rotation, clip.scale)) for clip in clips])
	joint_map = np.arange(len(parents), dtype=np.intp)
	active = np.ones(len(parents), dtype=bool)
	child_counts = np.bincount(parents[parents >= 0], minlength=len(parents))
	error = np.zeros(len(points), dtype=np.float32)
	while True:
		best: tuple[float, int, np.ndarray, np.ndarray] | None = None
		mapped = joint_map[joints]
		for joint in np.flatnonzero(active & (child_counts == 0) & (parents >= 0)):
			weight = np.sum(np.where(mapped == joint, weights, 0), axis=1)
			rows = np.flatnonzero(weight)
			delta = matrices[:, parents[joint]] - matrices[:, joint]
			bound = error[rows] + _merge_displacement(points[rows], delta, weight[rows])
			peak = float(bound.max(initial=0))
			if peak <= error_budget and (best is None or peak < best[0]):
				best = peak, int(joint), rows, bound
				if peak == 0: break
		if best is None: break
		_, removed, rows, bound = best
		parent = parents[removed]
		error[rows] = bound
		joint_map[joint_map == removed] = parent
		active[removed] = False
		child_counts[parent] -= 1
	if active.all(): return None
	retained = np.flatnonzero(active)
	compact = np.cumsum(active, dtype=np.intp) - 1
	joint_map = compact[joint_map]
	joint_map.flags.writeable = False
	new_parents = np.where(parents[retained] < 0, -1, compact[parents[retained]])
	rig = Skeleton(new_parents, skeleton.translation[retained], skeleton.rotation[retained], skeleton.scale[retained], skeleton.inverse_bind[retained])
	reduced_clips = tuple(Clip(clip.translation[:, retained], clip.rotation[:, retained], clip.scale[:, retained], clip.fps) for clip in clips)
	reduced_primitives = tuple(_remap_primitive(primitive, joint_map) for primitive in primitives)
	return CrowdSkeletonLod(rig, reduced_clips, reduced_primitives, skeleton, tuple(clips), tuple(primitives), joint_map, float(error.max()))
