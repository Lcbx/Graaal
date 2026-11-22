"""Optional ECS-to-array drawing adapter; application schemas remain caller-owned."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from numpy.typing import ArrayLike
import numpy as np

if TYPE_CHECKING:
	from ..ECS import ComponentSelection
	from ..DrawBatches import DrawBatches
	from ..ECS import ComponentAccessor


class EcsDrawAdapter:
	def __init__(self, draws: DrawBatches, mesh_refs: ComponentAccessor, *, lod_field: str="lod_id", shader_field: str="shader_id") -> None:
		self.draws, self.mesh_refs = draws, mesh_refs
		self.lod_field, self.shader_field = lod_field, shader_field
		self._token: tuple[Any, ...] | None
		self._entities: np.ndarray | None
		self._selection: ComponentSelection | None
		self._token, self._entities, self._selection = None, None, None
		self._order = np.empty(0, dtype=np.intp)

	def sync(self, entities: ArrayLike) -> bool:
		entities = np.asarray(entities)
		token = self.mesh_refs.layout_version, self.mesh_refs.version(self.lod_field, self.shader_field)
		changed = token != self._token or self._entities is None or not np.array_equal(entities, self._entities)
		if changed:
			selection = self.mesh_refs[entities]
			labels = self.mesh_refs.owner_ids(selection)
			self.draws.sync(selection[self.lod_field], selection[self.shader_field], labels=labels)
			# A retained checked selection also expires after layout-only changes.
			self._selection, self._order = selection, self.draws.instance_order.copy()
			self._entities, self._token = entities.copy(), token
			self.draws.invalidate_instances()
		self.draws.sync_resources()
		return changed

	def mesh_field(self, name: str) -> np.ndarray:
		if self._selection is None: raise RuntimeError("Synchronize the ECS adapter first")
		return self._selection[name][self._order]
