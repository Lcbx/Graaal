"""Bind compiled graph playback rows to ECS entity handles."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..Animation.AnimationGraphPlayback import AnimationGraphPlayback
from ..ECS import component


if TYPE_CHECKING:
	from ..Animation.AnimationEvents import EventTrack
	from ..Animation.AnimationGraph import CompiledAnimationGraph
	from ..Animation.RootMotion import RootMotion
	from ..ECS import ECS


@component
class GraphPlayer:
	graph_id: np.uint32


@dataclass(frozen=True, slots=True)
class EntityGraphEvent:
	entity: int
	state: int
	name: str
	offset: float
	role: str
	weight: float


class EcsAnimationGraphAdapter:
	"""Run one compiled graph for the entities referencing its graph ID.

	Evaluate the parameter rows in `entities()` order each tick. `update()`
	returns local TRS in the same order and publishes character-space movement
	requests and crossed state events. The application owns physics and layers.
	"""
	def __init__(self, world: ECS, graph_id: int, graph: CompiledAnimationGraph, *, event_tracks: Mapping[int, EventTrack] | None = None, root_motions: Mapping[int, RootMotion] | None = None) -> None:
		self.query = world.query(GraphPlayer)
		self.graph_id = graph_id
		self.playback = AnimationGraphPlayback(graph, 0, event_tracks=event_tracks, root_motions=root_motions)
		self._owners = np.empty(0, dtype=np.uint64)
		self._owners.flags.writeable = False
		self.events: tuple[EntityGraphEvent, ...] = ()

	@property
	def owners(self) -> np.ndarray:
		"""Read-only entity order of the most recently published outputs."""
		return self._owners

	def entities(self) -> np.ndarray:
		"""Return the current full entity handles in graph parameter-row order."""
		result = self.query.fetch()
		return result.entities[result[GraphPlayer].selection.graph_id == self.graph_id]

	def update(self, dt: ArrayLike, parameters: ArrayLike | None = None, *, in_place: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		owners = self.entities()
		if np.array_equal(self._owners, owners):
			playback = copy(self.playback)
		else:
			previous_rows = {entity: row for row, entity in enumerate(self._owners)}
			playback = self.playback.remap([previous_rows.get(entity, -1) for entity in owners])
		events = playback.advance(dt, parameters)
		pose = playback.pose(parameters, in_place=in_place)
		owners.flags.writeable = False
		self.playback, self._owners = playback, owners
		self.events = tuple(EntityGraphEvent(int(owners[event.row]), event.state, event.name, event.offset, event.role, event.weight) for event in events)
		return pose
