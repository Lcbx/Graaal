"""Map independently skinned poses to material meshes without depending on ECS."""
from operator import index

import numpy as np

from ..Animation.SkeletalAnimation import skin_vertices


def _owner(value):
	if isinstance(value, (bool, np.bool_)):
		raise ValueError("owner must be a uint64-compatible integer")
	try: value = index(value)
	except TypeError as exc: raise ValueError("owner must be a uint64-compatible integer") from exc
	if not 0 <= value <= np.iinfo(np.uint64).max:
		raise ValueError("owner must be a uint64-compatible integer")
	return value


class MaterialSkinningAdapter:
	"""Skin registered material meshes for caller-supplied persistent owner IDs.

	Each bound owner must have a distinct MaterialMesh; the caller manages owner
	lifetimes and removes bindings when renderables are removed. Source vertices
	are copied so repeated updates always start from the bind pose.
	"""
	def __init__(self, renderer):
		self.renderer = renderer
		self._bindings = {}

	def bind(self, owner, mesh, vertices):
		"""Bind an owner to a registered mesh and its unskinned vertex data."""
		owner = _owner(owner)
		if mesh not in self.renderer.mesh_ids: raise ValueError("Mesh is not registered")
		if mesh.bounds is None: raise ValueError("Skinned meshes require explicit conservative bounds")
		if any(other_mesh is mesh and other_owner != owner for other_owner, (other_mesh, _) in self._bindings.items()):
			raise ValueError("Independent owners require distinct material meshes")
		vertices = np.asarray(vertices)
		required = ("position", "normal", "uv", "joints", "weights")
		if vertices.ndim != 1 or len(vertices) != len(mesh.vertices) or not vertices.dtype.names or any(name not in vertices.dtype.names for name in required):
			raise ValueError("Expected original vertices with position, normal, uv, joints and weights")
		if vertices["uv"].shape != (len(vertices), 2) or not np.isfinite(vertices["uv"]).all():
			raise ValueError("Vertex UVs must be finite vec2 values")
		self._bindings[owner] = mesh, vertices.copy()

	def unbind(self, owner):
		"""Remove a binding; the last uploaded mesh geometry is left unchanged."""
		del self._bindings[_owner(owner)]

	def update(self, owners, matrices):
		"""Skin a batch aligned with owner IDs and upload it before frame recording."""
		owners = np.asarray(owners)
		matrices = np.asarray(matrices, dtype=np.float32)
		if owners.ndim != 1 or owners.dtype.kind not in "iu" or np.any(owners < 0):
			raise ValueError("owners must be a one-dimensional array of nonnegative integers")
		if len(np.unique(owners)) != len(owners): raise ValueError("owner IDs must be distinct")
		if matrices.ndim != 4 or matrices.shape[0] != len(owners) or matrices.shape[2:] != (3, 4):
			raise ValueError("matrices must have shape (owners, bones, 3, 4)")
		prepared = []
		for owner, pose in zip(owners, matrices):
			key = int(owner)
			if key not in self._bindings: raise ValueError(f"No skinned mesh bound for owner {key}")
			mesh, source = self._bindings[key]
			vertices = source.copy()
			vertices["position"], vertices["normal"] = skin_vertices(source["position"], source["normal"], source["joints"], source["weights"], pose)
			prepared.append((mesh, vertices))
		for mesh, vertices in prepared:
			self.renderer.update_mesh_vertices(mesh, vertices)
