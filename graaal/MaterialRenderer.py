"""DrawBatches integration for dictionary materials and inherited shading code."""
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from .DrawBatches import DrawBatches, FrustumCuller, RenderShader
from .InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_scale
from .Materials import MaterialBinding, MaterialLibrary, MaterialShader
from .EcsMaterials import MaterialInstance, MaterialTransform, EcsMaterialAdapter
from .RenderContext import BufferUsage, GpuBuffer, Mesh, RenderContext, Texture, TextureUsage
from .TransparencyData import VERTEX_DTYPE, SHADING_VERTEX_DTYPE, SKINNED_VERTEX_DTYPE, SKINNED_SHADING_VERTEX_DTYPE, MATERIAL_DTYPE
from .TransparencyPass import TransparencyPass, record_transparency

_DRAW_IDS = count(1)


@dataclass(eq=False)
class MaterialMesh:
	vertices: np.ndarray
	shader: MaterialShader
	indices: np.ndarray | None = None
	material: dict | None = None
	bounds: tuple | None = None


@dataclass(eq=False)
class DrawInstance:
	"""Optional direct-draw input; ECS applications use MaterialInstance components."""
	mesh: MaterialMesh
	material: dict | None = None
	overrides: dict = field(default_factory=dict)
	position: tuple = (0, 0, 0)
	rotation: tuple = (0, 0, 0, 1)
	scale: tuple = (1, 1, 1)
	shader: MaterialShader | None = None
	surface_id: int = field(default_factory=lambda: next(_DRAW_IDS))


class MaterialRenderer:
	"""Registered mesh resources rendered from explicit material/transform columns.

	All shader families share winner, metadata and vertex addressing. Per-instance
	values do not split draw batches. EcsMaterialAdapter supplies versioned extraction;
	DrawInstance lists are an optional direct-draw convenience. record() uses frustum
	culling and caller-owned commands/output; presentation/submission remain external.
	"""

	def __init__(self, instances=None, *, meshes=None, shaders=(), blue_noise=None, base_textures=None, frame_fields=None, skin_palette=None, **transparency_options):
		self.instances = None if instances is None else list(instances)
		self.meshes = list(dict.fromkeys(meshes if meshes is not None else (i.mesh for i in self.instances or ())))
		if not self.meshes: raise ValueError("Register at least one mesh")
		self.mesh_ids = {mesh: index for index, mesh in enumerate(self.meshes)}
		self.skin_palette = skin_palette
		self._skin_owners = np.empty(0, dtype=np.uint64)
		self._skin_meshes = np.empty(0, dtype=np.uint32)
		self._skin_codes = np.empty(0, dtype=np.uint32)
		self._mesh_max_joint = []
		self._raster_dtype = SKINNED_VERTEX_DTYPE if skin_palette is not None else VERTEX_DTYPE
		self._shading_dtype = SKINNED_SHADING_VERTEX_DTYPE if skin_palette is not None else SHADING_VERTEX_DTYPE
		self.library = MaterialLibrary([mesh.shader for mesh in self.meshes] + list(shaders)
			+ [i.shader for i in self.instances or () if i.shader is not None])
		self.draws = DrawBatches(MESH_INSTANCE_DTYPE)
		self.culler = FrustumCuller()
		vertices, self.offsets, self._mesh_gpu = [], [], []
		self._mesh_indices, self._mesh_vertex_counts, self._mesh_has_bounds = [], [], []
		for index, mesh in enumerate(self.meshes):
			indices = None if mesh.indices is None else np.asarray(mesh.indices).copy()
			self._mesh_indices.append(indices)
			self._mesh_vertex_counts.append(len(mesh.vertices))
			self._mesh_has_bounds.append(mesh.bounds is not None)
			source = mesh.vertices if indices is None else mesh.vertices[indices]
			if not len(source) or len(source) % 3: raise ValueError("Mesh must contain complete triangles")
			self.offsets.append(sum(len(v) for v in vertices)//3)
			raster = np.zeros(len(source), dtype=self._raster_dtype)
			stored = np.zeros(len(source), dtype=self._shading_dtype)
			for name, width in (("position", 3), ("normal", 3), ("uv", 2)):
				raster[name] = source[name]
				stored[name][:, :width] = source[name]
			fields = set(source.dtype.names or ())
			has_skin = {"joints", "weights"} <= fields
			if skin_palette is not None and bool(fields & {"joints", "weights"}) != has_skin: raise ValueError("Mesh needs both joints and weights")
			if has_skin and skin_palette is not None:
				joints, weights = source["joints"], source["weights"]
				if mesh.bounds is None: raise ValueError("Skinned meshes require explicit conservative bounds")
				if joints.shape != (len(source), 4) or joints.dtype.kind not in "iu" or np.any(joints < 0) or np.any(joints > np.iinfo(np.uint32).max):
					raise ValueError("Skinned joints must contain four uint32-compatible indices per vertex")
				if weights.shape != (len(source), 4) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, atol=1e-5):
					raise ValueError("Skinned weights must be finite, nonnegative and sum to one")
				raster["joints"], stored["joints"] = joints, joints
				raster["weights"], stored["weights"] = weights, weights
				self._mesh_max_joint.append(int(np.max(joints)))
			else: self._mesh_max_joint.append(None)
			raster["triangle"] = np.repeat(np.arange(len(source)//3), 3)
			raster["barycentric"] = np.tile(((1, 0), (0, 1), (0, 0)), (len(source)//3, 1))
			vertices.append(stored)
			gpu_mesh = Mesh(raster, np.arange(len(source), dtype=np.uint32))
			self._mesh_gpu.append(gpu_mesh)
			self.draws.register_mesh(index, gpu_mesh, bounds=mesh.bounds)
			self.draws.register_lod_group(index, (index,))
		usage = BufferUsage.STORAGE | BufferUsage.COPY_DST
		headers, shared, per_instance, _, _ = self.library.pack([])
		self.headers = GpuBuffer(headers, usage)
		self.parameters = dict(material_parameters=GpuBuffer(shared, usage), instance_parameters=GpuBuffer(per_instance, usage),
			material_headers=self.headers)
		if skin_palette is not None: self.parameters["skin_matrices"] = skin_palette.buffer
		self.metadata = GpuBuffer(np.zeros((1, 4), dtype=np.uint32), usage)
		if base_textures is None:
			white = Texture((1, 1, 1), format="rgba8unorm-srgb", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST)
			RenderContext.device.queue.write_texture({"texture": white.handle}, np.full(4, 255, dtype=np.uint8), {"bytes_per_row": 4}, (1, 1, 1))
			base_textures = white.view(dimension="2d-array")
		self.transparency = TransparencyPass(materials=GpuBuffer(np.zeros(1, dtype=MATERIAL_DTYPE), usage), base_textures=base_textures,
			vertices=GpuBuffer(np.concatenate(vertices), usage), blue_noise=blue_noise,
			material_library=self.library, frame_fields=frame_fields, gpu_skinning=skin_palette is not None,
			**transparency_options)
		self.geometry = [self.transparency.create_geometry_passes(self.metadata, transparent=transparent,
			instances=self.draws.buffers["instances"], geometry_resources=self.parameters,
			shading_resources=self.parameters) for transparent in (False, True)]
		self._register_passes()
		self.visibility = self.draws.create_visibility()
		if self.instances is not None: self.sync()

	def _bindings(self, instances):
		return [MaterialBinding(i.shader or i.mesh.shader,
			i.material if i.material is not None else i.mesh.material, i.overrides) for i in instances]

	def _register_passes(self):
		opaque, transparent = self.geometry
		self.draws.register_shader(0, RenderShader({"opaque": opaque.render_shader["main"]}))
		passes = dict(transparent.render_shader.passes)
		if transparent.nearest is not None: passes["nearest"] = transparent.nearest
		self.draws.register_shader(1, RenderShader(passes))


	@staticmethod
	def _write(buffer, content):
		buffer.write_changed(content)

	def sync(self):
		"""Convenience for direct DrawInstance lists; ECS extraction calls update()."""
		if self.instances is None: return
		mesh_ids = np.array([self.mesh_ids[i.mesh] for i in self.instances], dtype=np.uint32)
		transforms = np.zeros(len(self.instances), dtype=MESH_INSTANCE_DTYPE)
		if self.instances:
			transforms["iPosition"] = [i.position for i in self.instances]
			transforms["iRotation"] = pack_quaternion([i.rotation for i in self.instances])
			transforms["iScale"] = pack_scale([i.scale for i in self.instances])
		ids = np.array([i.surface_id for i in self.instances], dtype=np.uint64)
		self.update(mesh_ids, self._bindings(self.instances), transforms, owners=ids, surface_ids=ids)

	def update(self, mesh_ids, bindings, transforms, *, owners, surface_ids, dirty_materials=None):
		"""Upload explicit columns. IDs are persistent renderable identities, not rows."""
		mesh_ids, owners, surface_ids = np.asarray(mesh_ids), np.asarray(owners), np.asarray(surface_ids)
		n = len(bindings)
		if (mesh_ids.shape != (n,) or owners.shape != (n,) or surface_ids.shape != (n,)
			or transforms.shape != (n,) or transforms.dtype != MESH_INSTANCE_DTYPE):
			raise ValueError("Supply one mesh, binding, transform, owner and surface ID per draw")
		if mesh_ids.dtype.kind not in "iu" or np.any(mesh_ids < 0) or np.any(mesh_ids >= len(self.meshes)):
			raise ValueError("Unknown mesh ID")
		if surface_ids.dtype.kind not in "iu" or np.any(surface_ids <= 0) or np.any(surface_ids > 0xffffffff) or len(np.unique(surface_ids)) != n:
			raise ValueError("Surface IDs must be distinct positive uint32 values")
		codes = self._skin_codes_for(owners, mesh_ids) if self.skin_palette is not None else None
		# Pack once, before mutations. Instance offsets remain valid after regrouping.
		headers, shared, instance, ids, offsets = self.library.pack(bindings, dirty_materials=dirty_materials)
		routes = np.array([int(binding.shader.is_blended(self.library.material_values(binding), binding.overrides)) for binding in bindings], dtype=np.uint32)
		self.draws.sync(mesh_ids, routes, labels=owners.astype(np.uint64))
		order = self.draws.instance_order
		if codes is not None:
			self._skin_owners, self._skin_meshes, self._skin_codes = owners[order].copy(), mesh_ids[order].copy(), codes[order].copy()
		self._write(self.headers, headers)
		self._write(self.parameters["material_parameters"], shared)
		self._write(self.parameters["instance_parameters"], instance)
		metadata = np.zeros((max(1, n), 4), dtype=np.uint32)
		metadata[:n, 0] = np.asarray(self.offsets, dtype=np.uint32)[mesh_ids[order]]
		metadata[:n, 1], metadata[:n, 2], metadata[:n, 3] = ids[order], surface_ids[order], offsets[order]
		self._write(self.metadata, metadata)
		self.update_transforms(transforms)

	def update_transforms(self, transforms):
		"""Transform-only ECS edits do not repack or validate material dictionaries."""
		ordered = transforms[self.draws.instance_order]
		if self.skin_palette is not None:
			ordered = ordered.copy()
			ordered["iTint"][:, 0] = self._skin_codes
		if not np.array_equal(self.draws.buffers["instances"].content[:len(ordered)], ordered):
			self.draws.write_instances(ordered)

	def _skin_codes_for(self, owners, mesh_ids):
		codes = np.zeros(len(owners), dtype=np.uint32)
		for row, (owner, mesh_id) in enumerate(zip(owners, mesh_ids)):
			maximum = self._mesh_max_joint[int(mesh_id)]
			if maximum is None: continue
			try: base, count = self.skin_palette.range(int(owner))
			except KeyError as exc: raise ValueError(f"No skin palette range for owner {int(owner)}") from exc
			if maximum >= count: raise ValueError("Skin palette range does not cover mesh joints")
			codes[row] = base + 1
		return codes

	def sync_skinning(self):
		"""Refresh palette offsets after owner ranges change, before recording."""
		if self.skin_palette is None: return
		codes = self._skin_codes_for(self._skin_owners, self._skin_meshes)
		if np.array_equal(codes, self._skin_codes): return
		self._skin_codes = codes
		values = self.draws.buffers["instances"].content[:len(codes)].copy()
		values["iTint"][:, 0] = codes
		self.draws.write_instances(values)

	def update_mesh_vertices(self, mesh, vertices):
		"""Upload same-topology mesh vertices for both raster and deferred shading.

		The registered bounds must enclose all updated positions. All instances of
		this mesh share the update; use separate meshes for independent poses.
		Call before recording commands that consume these vertices.
		"""
		if mesh not in self.mesh_ids: raise ValueError("Mesh is not registered")
		mesh_id = self.mesh_ids[mesh]
		if not self._mesh_has_bounds[mesh_id]: raise ValueError("Updated meshes require explicit conservative bounds")
		vertices = np.asarray(vertices)
		if vertices.ndim != 1 or len(vertices) != self._mesh_vertex_counts[mesh_id] or not vertices.dtype.names or any(name not in vertices.dtype.names for name in ("position", "normal", "uv")):
			raise ValueError("Expected one position, normal and uv per original vertex")
		indices = self._mesh_indices[mesh_id]
		source = vertices if indices is None else vertices[indices]
		gpu_mesh = self._mesh_gpu[mesh_id]
		if len(source) != gpu_mesh.vertex_range[1] or any(source[name].shape != (len(source), width) or not np.isfinite(source[name]).all() for name, width in (("position", 3), ("normal", 3), ("uv", 2))):
			raise ValueError("Vertex fields must be finite and preserve mesh topology")
		raster = gpu_mesh.vertices.copy()
		offset = self.offsets[self.mesh_ids[mesh]] * 3
		stored = self.transparency.vertices.content[offset:offset + len(source)].copy()
		for name, width in (("position", 3), ("normal", 3), ("uv", 2)):
			raster[name] = source[name]
			stored[name][:, :width] = source[name]
		start, count = gpu_mesh.vertex_range
		gpu_mesh.vertex_buffer.write_range(raster, offset=start * raster.dtype.itemsize)
		self.transparency.vertices.write_range(stored, offset=offset * stored.dtype.itemsize)

	def set_sample_count(self, count):
		if self.transparency.set_sample_count(count):
			self._register_passes()

	def set_ambient_occlusion_settings(self, settings):
		if self.transparency.set_ambient_occlusion_settings(settings): self._register_passes()

	def set_banded_options(self, **options):
		if self.transparency.set_banded_options(**options): self._register_passes()

	def record(self, commands, size, camera, output, *, frame=None, lighting=None):
		"""Record the default frame sequence; callers may record the stages themselves."""
		record_material_frame(self, commands, size, camera, output, frame=frame, lighting=lighting)


def record_material_frame(renderer, commands, size, camera, output, *, frame=None, lighting=None, culler=None):
	"""Record the default material frame; culler may override the renderer's culler."""
	renderer.sync()
	if getattr(renderer, "skin_palette", None) is not None: renderer.sync_skinning()
	oit = renderer.transparency
	if size[0] <= 0 or size[1] <= 0: return
	oit.resize(size)
	vp = oit.update_camera(camera, frame=frame, lighting=lighting)
	(culler if culler is not None else renderer.culler).record(commands, renderer.visibility, camera_position=camera.position, view_projection=vp,
		fovy_deg=camera.fovy_deg if camera.perspective else 90.0)
	external = oit.has_external_opaque_inputs
	if not external or any(batch.shader_id == 0 and batch.mesh_id is not None for batch in renderer.draws.draw_batches):
		with oit.opaque_pass(commands, load=external) as render:
			renderer.draws.draw(render, renderer.visibility, "opaque")
	def draw_transparent(render, name):
		renderer.draws.draw(render, renderer.visibility, name)
	record_transparency(commands, oit, renderer.geometry[1], draw_transparent)
	oit.compose(commands, output)
