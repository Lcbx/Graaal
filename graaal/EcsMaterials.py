"""Optional ECS extraction for material rendering; ECS owns entity state."""
from dataclasses import field
import numpy as np

from .ecs_types import component
from .MaterialTransform import MaterialTransform
from .Materials import MaterialBinding
from .InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_scale


@component(multiples=True)
class MaterialInstance:
	mesh: object
	material: object = None
	overrides: object = field(default_factory=dict)
	shader: object = None
	surface_id: np.uint32 = 0


class EcsMaterialAdapter:
	"""Versioned extraction with stable component IDs and explicit dict invalidation.

	In-place dictionary/default edits require touch_materials(), or touch() on the
	material component accessor. Transform field writes use normal ECS versions.
	Empty selections and multiple material instances per entity are supported.
	"""
	def __init__(self, renderer, world, *, material_type=MaterialInstance, transform_type=MaterialTransform, transform_source=None):
		self.renderer = renderer
		self.materials, self.transforms = world.get(material_type), world.get(transform_type)
		self.query = world.query(material_type, transform_type)
		self.material_type = material_type
		self.transform_source = transform_source
		self._source_token = None
		self._material_token = self._transform_token = None
		self._revision = 0
		self._dirty_materials = []
		self._next_surface = 1
		self._owners = np.empty(0, dtype=np.uint64)

	def touch_materials(self, *materials):
		"""Notify edited dictionaries; no arguments invalidates all resolved records."""
		self._revision += 1
		if materials:
			self._dirty_materials.extend(materials)
		else:
			self._material_token = None  # Force a full refresh even if targeted edits follow.
			self._dirty_materials = []

	def _tokens(self):
		return ((self.materials.layout_version, self.materials.version(), self._revision),
			(self.transforms.layout_version, self.transforms.version()))

	def _transforms(self, owners):
		if self.transform_source is None:
			selection = self.transforms[owners]
			position, rotation, scale = selection.position, selection.rotation, selection.scale
		else:
			position, rotation, scale = map(np.asarray, self.transform_source(owners))
			if position.shape != (len(owners), 3) or rotation.shape != (len(owners), 4) or scale.shape != (len(owners), 3):
				raise ValueError("transform_source must return position (n, 3), rotation (n, 4), scale (n, 3) in owner order")
		values = np.zeros(len(owners), dtype=MESH_INSTANCE_DTYPE)
		values["iPosition"] = position
		values["iRotation"] = pack_quaternion(rotation)
		values["iScale"] = pack_scale(scale)
		return values

	def sync(self):
		material_token, transform_token = self._tokens()
		# Membership in either component changes the renderable selection.
		membership = (self.materials.membership_version, self.transforms.membership_version)
		material_changed = material_token != self._material_token or membership != getattr(self, "_membership", None)
		if not material_changed:
			if transform_token == self._transform_token and self.transform_source is None and self._source_token is None: return False
			self.renderer.update_transforms(self._transforms(self._owners))
			self._transform_token = transform_token
			self._source_token = self.transform_source
			return True
		selection = self.query.fetch()[self.material_type].selection
		owners = self.materials.owner_ids(selection)
		ids = selection.surface_id.copy()
		occupied = ids[ids != 0]
		if len(np.unique(occupied)) != len(occupied): raise ValueError("Material surface IDs must be unique")
		self._next_surface = max(self._next_surface, int(occupied.max(initial=0))+1)
		missing = ids == 0
		end = self._next_surface + int(missing.sum())
		if end > 0x100000000: raise OverflowError("Material surface IDs exhausted uint32")
		if missing.any(): ids[missing] = np.arange(self._next_surface, end, dtype=np.uint32)
		self._next_surface = end
		meshes, shaders = selection.mesh, selection.shader
		bindings = [MaterialBinding(shader if shader is not None else mesh.shader,
			material if material is not None else mesh.material, overrides)
			for mesh, shader, material, overrides in zip(meshes, shaders, selection.material, selection.overrides)]
		self.renderer.update(np.array([self.renderer.mesh_ids[mesh] for mesh in meshes], dtype=np.uint32),
			bindings, self._transforms(owners), owners=owners, surface_ids=ids,
			dirty_materials=self._dirty_materials if self._material_token is not None
			and material_token[:2] == self._material_token[:2] and membership == getattr(self, "_membership", None) else None)
		if missing.any(): selection.surface_id = ids
		self._owners = owners.copy()
		self._membership = membership
		self._material_token, self._transform_token = self._tokens()
		self._dirty_materials = []
		self._source_token = self.transform_source
		return True
