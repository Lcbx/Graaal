"""Banded transparency with exact nearest capture and dual Kawase tail blur.

The caller owns geometry, camera, command recording, output and submission.
No window, asset file, ECS world, or ImGui context is created by this module.
"""
from dataclasses import dataclass, asdict
from operator import index
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .postprocessing_settings import AmbientOcclusionSettings
from .DrawBatches import RenderShader, ShaderPass
from .RenderContext import BindGroup, BufferUsage, GpuBuffer, ComputePipeline, RenderContext, RenderPipeline, Sampler, Shader, Texture, TextureUsage
from .TransparencyData import MATERIAL_DTYPE, SHADING_VERTEX_DTYPE, VERTEX_DTYPE
from .TextureProcessing import get_default_blue_noise
from .material_layout import material_pipeline_layout
from .settings_validation import checked_dataclass, float_range


SHADERS = Path(__file__).with_name("shaders")
DEBUG_VIEWS = ("final", "transmittance", "winner", "reconstruction", "nearest", "nearest_mask", "tail_transmittance")
MSAA_SAMPLE_COUNTS = (1, 4)


@checked_dataclass(frozen=True)
class GrazingFade:
	"""BLEND opacity and MASK coverage fade by geometric angle, 0 face-on to 90 edge-on."""
	enabled: bool = True
	start_angle: float_range(0, 90) = 83.0
	end_angle: float_range(0, 90) = 86.0

	def __post_init__(self):
		if not (self.start_angle < self.end_angle and np.float32(self.start_angle) < np.float32(self.end_angle)):
			raise ValueError("Grazing fade angles must satisfy 0 <= start_angle < end_angle <= 90 degrees")


@dataclass
class GeometryPasses:
	"""DrawBatches passes and deferred bindings for caller-owned geometry.

	metadata is a vec4u buffer in compact instance order: triangle offset,
	material index, stable surface ID, reserved. All transparent draws share
	this metadata/vertex address space. Use nearest as a pass override.
	"""
	render_shader: RenderShader
	shading: BindGroup
	owner: object
	shading_bindings: tuple[BindGroup, ...] = ()
	creation_args: dict | None = None
	nearest: ShaderPass | None = None
	nearest_shading: BindGroup | None = None


@dataclass(frozen=True)
class TransparencyTargets:
	main_color: Texture
	depth: Texture
	transmittance: Texture
	winner_color: Texture
	reconstruction_color: Texture
	nearest_payload: Texture
	nearest_alpha: Texture
	nearest_depth: Texture
	nearest_color: Texture
	nearest_extra: Texture | None = None


class TransparencyPass:
	"""Exact nearest capture, complementary band allocation and Kawase composition.

	Inputs are GPU resources, independent of their source (glTF or procedural).
	Geometry uses VERTEX_DTYPE and MESH_INSTANCE_DTYPE. Material hooks support
	custom shading, opacity and matched raster/deferred vertex deformation.
	"""

	def __init__(self, *, materials, base_textures, vertices, blue_noise=None,
		material_sampler=None, target_format=None, noise_scale=1.0, grazing_fade=None, sample_count=1,
		opaque_color=None, opaque_depth=None, material_source=None, material_library=None, frame_fields=None, gpu_skinning=False,
		kawase_passes=1, banded_proxy=False, banded_subset=False, banded_tail_msaa=False):
		if any(type(value) is not bool for value in (banded_proxy, banded_subset, banded_tail_msaa)): raise ValueError("Banded options must be boolean")
		if type(kawase_passes) is not int or not 0 <= kawase_passes <= 8: raise ValueError("kawase_passes must be an integer from 0 through 8")
		if type(sample_count) is not int or sample_count not in MSAA_SAMPLE_COUNTS: raise ValueError("Transparency sample_count must be 1 or 4")
		self.banded_proxy, self.banded_subset, self.banded_tail_msaa = banded_proxy, banded_subset, banded_tail_msaa
		self.kawase_passes = kawase_passes
		self.band_counts = self.band_payloads = None
		self.kawase_stages = []
		self.target_format = target_format or getattr(RenderContext, "presentation_format", "rgba8unorm")
		self.sample_count = sample_count
		self._material_source = material_source
		self._material_library = material_library
		self._gpu_skinning = bool(gpu_skinning)
		if frame_fields is None: frame_fields = {}
		if not isinstance(frame_fields, Mapping):
			raise TypeError("frame_fields must map names to (WGSL type, default) pairs")
		reserved = {"view", "proj", "camera_position", "clip", "grazing"}
		self.frame_fields = {}
		for name, spec in frame_fields.items():
			if not isinstance(name, str) or not name.isascii() or not name.isidentifier() or name in reserved:
				raise ValueError(f"Invalid or reserved frame field {name!r}")
			if not isinstance(spec, (tuple, list)) or len(spec) != 2 or not isinstance(spec[0], str) or not spec[0].strip():
				raise ValueError(f"Frame field {name!r} requires a WGSL type and default")
			self.frame_fields[name] = (spec[0], spec[1])
		self._frame_types = {name: spec[0] for name, spec in self.frame_fields.items()}
		self.ambient_occlusion_settings = AmbientOcclusionSettings()
		self._shader_sets = {}
		self._geometry_shaders = {}
		self._layout_cache, self._pipeline_layouts = {}, {}
		self._nearest_shaders = {}
		self._activate_shaders(sample_count)
		self.uniforms = self.shader.uniform_buffer("frame")
		for name, (_, default) in self.frame_fields.items():
			self.uniforms.content[name] = self._frame_value(name, default)
		self.lighting_uniforms = self.shader.uniform_buffer("lighting") if "lighting" in self.shader.info.uniforms else None
		if self.lighting_uniforms is not None and "light_dir" in self.lighting_uniforms.dtype.names:
			self.lighting_uniforms.content["light_dir"] = (0.4, 0.8, 0.3, 0)
			self.lighting_uniforms.upload_changed()
		if blue_noise is None:
			blue_noise = get_default_blue_noise()
		self.noise = blue_noise
		self._transparent_geometries = []
		self._geometries = []
		self.size = None
		self.set_noise_scale(noise_scale)
		self.set_grazing_fade(grazing_fade if grazing_fade is not None else GrazingFade())
		self.compose_uniforms = self.compose_shader.uniform_buffer("compose_settings")
		self.debug_view = "final"
		self.ambient_occlusion = False
		self._update_compose_settings()
		self._create_pipelines()
		self.vertices = vertices
		self.shared = dict(materials=materials, base_textures=base_textures,
			material_sampler=material_sampler or Sampler(address_mode_u="repeat", address_mode_v="repeat", mag_filter="linear", min_filter="linear", mipmap_filter="linear"))
		self.shade_frame = self._frame_group(self.shading_shader)
		self._opaque_inputs = None
		self._targets = None
		if opaque_color is not None or opaque_depth is not None:
			self.set_opaque_inputs(opaque_color, opaque_depth)

	def _geometry_shader(self, count):
		key = (count, self._banded_flags)
		if key not in self._geometry_shaders:
			self._geometry_shaders[key] = Shader(filepath=str(SHADERS / "stochastic_geometry.shader"),
				params={**self._banded_params(count), "material_source": self._material_source, "sample_count": count,
					"material_library": self._material_library, "frame_fields": self._frame_types, "gpu_skinning": self._gpu_skinning})
		return self._geometry_shaders[key]


	@property
	def depth_shader(self):
		"""The auxiliary 1x geometry module is created only when requested."""
		return self._geometry_shader(1)

	def _frame_group(self, shader):
		if "frame" in shader.info.uniforms and shader.info.uniforms["frame"].dtype != self.uniforms.dtype:
			raise ValueError("Shader Frame uniform layout differs from the configured frame_fields")
		resources = dict(frame=self.uniforms)
		if "lighting" in shader.info.bindings: resources["lighting"] = self.lighting_uniforms
		return shader.bind_group(0, **resources)

	def _pipeline_layout(self, shader, *entries):
		key = (shader, entries)
		if key not in self._pipeline_layouts:
			self._pipeline_layouts[key] = material_pipeline_layout(shader, entries, self._layout_cache)
		return self._pipeline_layouts[key]

	def _shader_set(self, count):
		key = (count, self._banded_flags, self.ambient_occlusion_settings)
		if key not in self._shader_sets:
			params = {**self._banded_params(count), "material_source": self._material_source,
				"material_library": self._material_library, "frame_fields": self._frame_types, "gpu_skinning": self._gpu_skinning}
			self._shader_sets[key] = (
				self._geometry_shader(count),
				Shader(filepath=str(SHADERS / "stochastic_shade.shader"), params=dict(params, sample_count=count if self.banded_tail_msaa else 1)),
				Shader(filepath=str(SHADERS / "stochastic_compose.shader"), params={**self._banded_params(count),
					"ao": asdict(self.ambient_occlusion_settings), "srgb_surface": self.target_format.endswith("-srgb"),
					"sample_count": count, "frame_fields": self._frame_types}),
			)
		return self._shader_sets[key]

	def _activate_shaders(self, count):
		self.shader, self.shading_shader, self.compose_shader = self._shader_set(count)
		self.tail_shader = self.shader
		if count > 1 and not self.banded_tail_msaa:
			key = ("tail", count, self._banded_flags)
			if key not in self._geometry_shaders:
				params = dict(self._banded_params(count), sample_count=1, guide_sample_count=count, banded_statistics=True,
					material_source=self._material_source, material_library=self._material_library, frame_fields=self._frame_types, gpu_skinning=self._gpu_skinning)
				self._geometry_shaders[key] = Shader(filepath=str(SHADERS / "stochastic_geometry.shader"), params=params)
			self.tail_shader = self._geometry_shaders[key]

	def _create_pipelines(self):
		key = self.sample_count
		if key not in self._nearest_shaders:
			self._nearest_shaders[key] = Shader(filepath=str(SHADERS / "stochastic_shade.shader"),
				params={"material_source": self._material_source, "sample_count": self.sample_count,
					"material_library": self._material_library, "frame_fields": self._frame_types, "shade_nearest": True, "gpu_skinning": self._gpu_skinning})
		self.nearest_shading_shader = self._nearest_shaders[key]
		self.nearest_shade_pipeline = ComputePipeline(self.nearest_shading_shader, entry="shade_nearest", layout=self._pipeline_layout(self.nearest_shading_shader, "shade_nearest"))
		self.shade_pipeline = ComputePipeline(self.shading_shader, entry="shade_bands", layout=self._pipeline_layout(self.shading_shader, "shade_bands"))
		self.compose_pipeline = RenderPipeline(self.compose_shader, vertex_entry="fullscreen", fragment_entry="compose",
			depth_test=None, depth_write=False, cull_mode="none", layout=self._pipeline_layout(self.compose_shader, "fullscreen", "compose"))


	@property
	def _banded_flags(self):
		return self.banded_proxy, self.banded_subset, self.banded_tail_msaa

	def _banded_params(self, samples):
		mixed = samples > 1 and not self.banded_tail_msaa
		return dict(banded_proxy=self.banded_proxy, banded_subset=self.banded_subset,
			banded_mixed=mixed, banded_statistics=not mixed, guide_sample_count=samples)


	@property
	def tail_sample_count(self):
		return 1 if not self.banded_tail_msaa else self.sample_count


	@property
	def mixed_tail(self):
		return self.tail_sample_count != self.sample_count

	def set_banded_options(self, *, proxy=None, subset=None, tail_msaa=None):
		"""Configure proxy residuals, one selected pixel per 2x2 block, and tail MSAA."""
		values = tuple(current if value is None else value for current, value in zip(self._banded_flags, (proxy, subset, tail_msaa)))
		if any(type(value) is not bool for value in values): raise ValueError("Banded options must be boolean")
		if values == self._banded_flags: return False
		self.banded_proxy, self.banded_subset, self.banded_tail_msaa = values
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry()
		return True

	def set_kawase_passes(self, count):
		"""Set 0-8 down/up pairs for the banded tail."""
		if type(count) is not int or not 0 <= count <= 8: raise ValueError("kawase_passes must be an integer from 0 through 8")
		if count == self.kawase_passes: return False
		self.kawase_passes = count
		if self.size is not None:
			self._setup_kawase()
			for layout in self._layout_cache.values(): layout.clear_cache()
		return True

	def set_sample_count(self, count):
		"""Switch 1x/4x MSAA, preserving geometry registration and live settings."""
		if type(count) is not int or count not in MSAA_SAMPLE_COUNTS:
			raise ValueError("Transparency sample_count must be 1 or 4")
		if count == self.sample_count:
			return False
		if self._opaque_inputs and self._opaque_inputs[0].sample_count != count:
			raise ValueError("External opaque inputs must match the selected sample count")
		self._activate_shaders(count)
		self.sample_count = count
		self.set_noise_scale(self.noise_scale)
		self._rebuild_geometry()
		return True

	def _rebuild_geometry(self, *, preserve_targets=False):
		size, geometries = self.size, self._geometries
		self._create_pipelines()
		targets = self._targets if preserve_targets and size is not None else None
		self.size = None
		self._geometries, self._transparent_geometries = [], []
		for geometry in geometries:
			rebuilt = self.create_geometry_passes(**geometry.creation_args)
			geometry.render_shader.passes.clear()
			geometry.render_shader.passes.update(rebuilt.render_shader.passes)
			geometry.shading = rebuilt.shading
			geometry.nearest, geometry.nearest_shading = rebuilt.nearest, rebuilt.nearest_shading
			self._geometries[-1] = geometry
			if geometry.creation_args["transparent"]:
				self._transparent_geometries[-1] = geometry
		self.shade_frame = self._frame_group(self.shading_shader)
		if size is not None:
			self.resize(size, targets=targets)

	def create_geometry_passes(self, metadata, *, transparent, instances,
		geometry_resources=None, shading_resources=None, geometry_bindings=(), shading_bindings=()):
		"""Build DrawBatches passes bound to caller-owned per-instance metadata."""
		if metadata.binding_size % 16 or not metadata.binding_size: raise ValueError("Metadata must contain nonempty vec4u records")
		if instances.binding_size % 32 or instances.binding_size // 32 < metadata.binding_size // 16: raise ValueError("Supply an instance transform for every metadata record")
		geometry_resources, shading_resources = dict(geometry_resources or {}), dict(shading_resources or {})
		if set(geometry_resources) & (set(self.shared) | {"metadata"}) or set(shading_resources) & (set(self.shared) | {"metadata", "vertices", "shading_instances"}):
			raise ValueError("Additional resources must not replace the geometry addressing resources")
		if any(group.group < 4 for group in geometry_bindings) or any(group.group < 3 for group in shading_bindings): raise ValueError("Extra geometry groups start at 4; extra shading groups start at 3")
		bindings = (self._frame_group(self.shader), self.shader.bind_group(2, **self.shared, metadata=metadata, **geometry_resources), *geometry_bindings)
		shading = self.shading_shader.bind_group(2, **self.shared, metadata=metadata, vertices=self.vertices, shading_instances=instances, **shading_resources)
		passes = {}
		nearest = nearest_shading = None
		if transparent:
			multiply = {"color": {"src_factor": "zero", "dst_factor": "src", "operation": "add"}, "alpha": {"src_factor": "zero", "dst_factor": "src-alpha", "operation": "add"}}
			add = {"color": {"src_factor": "one", "dst_factor": "one", "operation": "add"}, "alpha": {"src_factor": "one", "dst_factor": "one", "operation": "add"}}
			transmittance = RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="transmittance",
				cull_mode="none", depth_write=False, blend=multiply, label="transmittance", sample_count=self.sample_count,
				targets=({"blend": multiply}, {"blend": add}) if self.banded_proxy and not self.mixed_tail else None,
				layout=self._pipeline_layout(self.shader, "vertex", "transmittance"))
			passes["transmittance"] = ShaderPass(transmittance, bindings)
			nearest = ShaderPass(RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="nearest_layer", cull_mode="none", sample_count=self.sample_count, label="nearest_layer", layout=self._pipeline_layout(self.shader, "vertex", "nearest_layer")), bindings)
			nearest_shading = self.nearest_shading_shader.bind_group(2, **self.shared, metadata=metadata, vertices=self.vertices, shading_instances=instances, **shading_resources)
			tail_bindings = (self._frame_group(self.tail_shader), self.tail_shader.bind_group(2, **self.shared, metadata=metadata, **geometry_resources), *geometry_bindings)
			bands = ShaderPass(RenderPipeline(self.tail_shader, vertex_entry="vertex", fragment_entry="select_bands", cull_mode="none", depth_write=False, depth_test=None if self.mixed_tail else "always", sample_count=self.tail_sample_count, targets=({"write_mask": 0},) if self.mixed_tail else None, label="banded_tail", layout=self._pipeline_layout(self.tail_shader, "vertex", "select_bands")), tail_bindings)
			passes.update(main=bands, bands=bands)
			if self.mixed_tail:
				passes["tail_statistics"] = ShaderPass(RenderPipeline(self.tail_shader, vertex_entry="vertex", fragment_entry="tail_statistics", cull_mode="none", depth_write=False, depth_test=None, sample_count=1, blend=add, label="tail_statistics", layout=self._pipeline_layout(self.tail_shader, "vertex", "tail_statistics")), tail_bindings)
		else:
			opaque = ShaderPass(RenderPipeline(self.shader, vertex_entry="vertex", fragment_entry="opaque", cull_mode="none", label="opaque", sample_count=self.sample_count, layout=self._pipeline_layout(self.shader, "vertex", "opaque")), bindings)
			passes.update(main=opaque, prepass=opaque)
		geometry = GeometryPasses(RenderShader(passes), shading, self, tuple(shading_bindings), nearest=nearest, nearest_shading=nearest_shading)
		geometry.creation_args = dict(metadata=metadata, transparent=transparent, instances=instances,
			geometry_resources=geometry_resources, shading_resources=shading_resources, geometry_bindings=tuple(geometry_bindings), shading_bindings=tuple(shading_bindings))
		self._geometries.append(geometry)
		if transparent:
			self._transparent_geometries.append(geometry)
			if self.size is not None: self._bind_targets(geometry)
		return geometry

	def _bind_targets(self, geometry):
		bindings = tuple(b for b in geometry.render_shader["transmittance"].bindings if b.group != 3)
		nearest = dict(nearest_ids=self.nearest_payload)
		if self.sample_count > 1: nearest["nearest_ids_extra"] = self.nearest_extra
		geometry.nearest.bindings = bindings + (self.shader.bind_group(3, opaque_depth=self.depth),)
		geometry.render_shader["transmittance"].bindings = bindings + (self.shader.bind_group(3, opaque_depth=self.depth, band_counts=self.band_counts, **nearest),)
		tail_bindings = tuple(b for b in geometry.render_shader["bands"].bindings if b.group != 3)
		geometry.render_shader["bands"].bindings = tail_bindings + (self.tail_shader.bind_group(3, opaque_depth=self.depth, blue_noise=self.noise, band_counts=self.band_counts, band_payloads=self.band_payloads.view(dimension="2d-array"), **nearest),)
		if self.mixed_tail: geometry.render_shader["tail_statistics"].bindings = tail_bindings + (self.tail_shader.bind_group(3, opaque_depth=self.depth, band_counts=self.band_counts, **nearest),)

	def _update_compose_settings(self):
		self.compose_uniforms.content["options"] = (DEBUG_VIEWS.index(self.debug_view), int(self.ambient_occlusion), 0, 0)
		self.compose_uniforms.upload_changed()

	def set_ambient_occlusion_settings(self, settings):
		"""Set the composed ambient occlusion sampling and strength controls."""
		if not isinstance(settings, AmbientOcclusionSettings): raise TypeError("Expected AmbientOcclusionSettings")
		if settings == self.ambient_occlusion_settings: return False
		self.ambient_occlusion_settings = settings
		self._activate_shaders(self.sample_count)
		self._rebuild_geometry(preserve_targets=True)
		return True

	def set_ambient_occlusion(self, enabled):
		"""Enable screen-space ambient occlusion in the composed output."""
		if type(enabled) is not bool: raise ValueError("Ambient occlusion enabled must be boolean")
		if enabled == self.ambient_occlusion: return False
		self.ambient_occlusion = enabled
		self._update_compose_settings()
		return True


	@property
	def effective_noise_scale(self):
		"""Approximate cell width after MSAA sample-density adjustment."""
		return self.noise_scale / self.sample_count**0.5

	def set_noise_scale(self, pixels):
		"""Set approximate 1x screen noise-cell width in pixels (default 1).

		Larger cells trade coarser grain for fewer threshold changes under motion.
		4x MSAA halves the effective width; the chosen base value is preserved.
		"""
		limits = np.finfo(np.float32)
		if not np.isfinite(pixels) or not float(limits.tiny) <= pixels <= float(limits.max):
			raise ValueError("Noise scale must be a finite, positive normalized float32")
		self.noise_scale = float(pixels)
		self.uniforms.content["clip"][..., 2] = self.effective_noise_scale
		self.uniforms.upload_changed()

	def set_grazing_fade(self, options):
		"""Change BLEND opacity and MASK coverage fade without rebuilding GPU resources."""
		self.uniforms.content["grazing"] = (options.start_angle, options.end_angle, options.enabled, 0)
		self.uniforms.upload_changed()
		self.grazing_fade = options

	def set_debug_view(self, view):
		"""Inspect transmission, raw tail winners, reconstruction or the nearest layer."""
		if view not in DEBUG_VIEWS: raise ValueError(f"Debug view must be one of {DEBUG_VIEWS}")
		self.debug_view = view
		self._update_compose_settings()


	@property
	def has_external_opaque_inputs(self):
		return self._opaque_inputs is not None

	def set_opaque_inputs(self, color, depth):
		"""Use another renderer's color/depth; pass both as None to restore defaults."""
		if (color is None) != (depth is None): raise ValueError("Supply both opaque color and depth")
		if color is not None:
			if color.size != depth.size or not depth.format.startswith("depth"):
				raise ValueError("Opaque inputs require matching dimensions and a depth format")
			for texture in (color, depth):
				if texture.sample_count != self.sample_count or not texture.usage & TextureUsage.TEXTURE_BINDING:
					raise ValueError("Opaque inputs must match sample_count and have TEXTURE_BINDING")
		self._opaque_inputs = None if color is None else (color, depth)
		self.size = None

	def resize(self, size: tuple[int, int], *, targets: TransparencyTargets | None = None) -> bool:
		"""Return whether targets changed; retain targets for minimized sizes."""
		if len(size) != 2:
			raise ValueError("Expected transparency target width and height")
		try:
			size = tuple(index(value) for value in size)
		except TypeError:
			raise ValueError("Expected integer transparency target dimensions") from None
		if size[0] <= 0 or size[1] <= 0:
			return False
		if size == self.size and (targets is None or targets is getattr(self, "_targets", None)):
			return False
		inputs = getattr(self, "_opaque_inputs", None)
		if inputs and inputs[0].size[:2] != size: raise ValueError("Opaque input dimensions do not match transparency targets")
		formats = dict(main_color="rgba16float", depth="depth32float", transmittance="rgba16float",
			winner_color="rgba16float", reconstruction_color="rgba16float", nearest_depth="depth32float", nearest_alpha="rgba16float",
			nearest_payload="rgba16uint" if self.sample_count > 1 else "rgba32uint", nearest_color="rgba16float")
		if self.sample_count > 1: formats["nearest_extra"] = "rgba16uint"
		values = {}
		for name, format in formats.items():
			shaded = name in ("winner_color", "nearest_color", "reconstruction_color")
			usage = TextureUsage.TEXTURE_BINDING | (TextureUsage.STORAGE_BINDING if shaded else TextureUsage.RENDER_ATTACHMENT)
			count = 1 if shaded else self.sample_count
			layers = (self.sample_count if name == "nearest_color" else self.tail_sample_count) if shaded else 1
			if targets is not None:
				texture = getattr(targets, name)
				if texture is None or texture.size != (*size, layers) or texture.format != format or texture.sample_count != count or texture.usage & usage != usage:
					raise ValueError(f"Invalid transparency target {name}: expected {size}, {format}, {count} samples, {layers} layers, usage {usage}")
			elif inputs and name in ("main_color", "depth"):
				texture = inputs[0 if name == "main_color" else 1]
			else:
				texture = Texture((*size, layers) if shaded else size, format=format, usage=usage, label=name, sample_count=count)
			values[name] = texture
		if targets is None: targets = TransparencyTargets(**values)
		for name, texture in values.items(): setattr(self, name, texture)
		self.nearest_extra = targets.nearest_extra
		self._targets = targets
		resources = dict(winners=self.nearest_payload, winner_color=self.nearest_color.view(dimension="2d-array") if self.sample_count > 1 else self.nearest_color)
		if self.sample_count > 1: resources["winners_extra"] = self.nearest_extra
		self.nearest_shade_targets = self.nearest_shading_shader.bind_group(1, **resources)
		self._setup_banded(size)
		self.size = size
		# Native bind groups retain their textures. Drop old target bindings on
		# resize rather than keeping many full-resolution target sets alive.
		for layout in self._layout_cache.values(): layout.clear_cache()
		return True

	def _setup_banded(self, size):
		count = size[0]*size[1]*self.tail_sample_count*2
		if count*4 > RenderContext.device.limits["max-storage-buffer-binding-size"]:
			raise ValueError("Banded tail counters exceed max-storage-buffer-binding-size at this resolution/sample count")
		self.band_counts = GpuBuffer(None, BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC, dtype=np.uint32, count=count, label="band_counts")
		payload_size = tuple((value+1)//2 for value in size) if self.banded_subset else size
		self.band_payloads = Texture((*payload_size, self.tail_sample_count*2), format="rgba32uint", usage=TextureUsage.STORAGE_BINDING | TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_SRC, label="band_payloads")
		self.band_proxy = Texture(size, format="rgba16float", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING, sample_count=self.tail_sample_count, label="band_proxy") if self.banded_proxy or self.mixed_tail else None
		self.band_clear_shader = Shader(filepath=str(SHADERS / "banded_clear.shader"))
		self.band_clear_pipeline = ComputePipeline(self.band_clear_shader, entry="clear", layout=self._pipeline_layout(self.band_clear_shader, "clear"))
		self.band_clear_targets = self.band_clear_shader.bind_group(1, payloads=self.band_payloads.view(dimension="2d-array"))
		for geometry in self._transparent_geometries: self._bind_targets(geometry)
		color = self.winner_color.view(dimension="2d-array") if self.tail_sample_count > 1 else self.winner_color
		resources = dict(band_counts=self.band_counts, band_payloads=self.band_payloads.view(dimension="2d-array"), winner_color=color)
		if self.banded_proxy: resources["band_proxy"] = self.band_proxy
		self.shade_targets = self.shading_shader.bind_group(1, **resources)
		color = self.winner_color.view(dimension="2d-array") if self.sample_count > 1 else self.winner_color
		self.compose_frame = self._frame_group(self.compose_shader)
		self.compose_resources = self.compose_shader.bind_group(1, main_color=self.main_color, transmittance=self.transmittance,
			winner_color=color, opaque_depth=self.depth, compose_settings=self.compose_uniforms, nearest_alpha=self.nearest_alpha,
			nearest_color=self.nearest_color.view(dimension="2d-array") if self.sample_count > 1 else self.nearest_color,
			reconstructed=self.reconstruction_color.view(dimension="2d-array") if self.sample_count > 1 else self.reconstruction_color)
		self._setup_kawase()

	def _setup_kawase(self):
		self.kawase_shader = Shader(filepath=str(SHADERS / "stochastic_kawase.shader"), params={"sample_count": self.tail_sample_count, "banded_proxy": self.banded_proxy, "banded_mixed": self.mixed_tail})
		self.kawase_pipelines = {entry: ComputePipeline(self.kawase_shader, entry=entry, layout=self._pipeline_layout(self.kawase_shader, entry)) for entry in ("down", "up", "finish")}
		self.kawase_stages, self.kawase_textures = [], []
		self.kawase_dummy = Texture((1, 1, self.tail_sample_count), format="rgba16float", usage=TextureUsage.STORAGE_BINDING)
		def stage(entry, source, destination):
			resources = dict(source_pair=source.view(dimension="2d-array"))
			if not self.mixed_tail: resources["transmission"] = self.transmittance
			if self.banded_proxy: resources.update(band_proxy=self.band_proxy, band_counts=self.band_counts, local_pair=self.winner_color.view(dimension="2d-array"))
			inputs = self.kawase_shader.bind_group(0, **resources)
			pair = self.kawase_dummy if entry == "finish" else destination
			color = destination if entry == "finish" else self.kawase_dummy
			resources = dict(output_pair=pair.view(dimension="2d-array"), output_color=color.view(dimension="2d-array") if self.tail_sample_count > 1 else color.view(dimension="2d"))
			outputs = self.kawase_shader.bind_group(1, **resources)
			self.kawase_stages.append((entry, inputs, outputs, destination.size[:2]))
		def texture(size):
			result = Texture((*size, self.tail_sample_count), format="rgba16float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.STORAGE_BINDING, label="kawase_pair")
			self.kawase_textures.append(result)
			return result
		current = self.winner_color
		sizes = [current.size[:2]]
		for _ in range(self.kawase_passes):
			if sizes[-1] == (1, 1): break
			size = tuple((value+1)//2 for value in sizes[-1])
			destination = texture(size)
			stage("down", current, destination)
			current = destination
			sizes.append(size)
		for size in reversed(sizes[:-1]):
			destination = texture(size)
			stage("up", current, destination)
			current = destination
		stage("finish", current, self.reconstruction_color)

	def _require_targets(self) -> tuple[int, int]:
		if self.size is None:
			raise RuntimeError("Resize transparency targets before recording passes or updating frame uniforms")
		return self.size


	@staticmethod
	def _uniform_value(buffer, name, value, kind):
		"""Check one uniform field before mutating its buffer."""
		field_dtype = buffer.dtype.fields[name][0]
		dtype, shape = field_dtype.subdtype or (field_dtype, ())
		try: supplied = np.asarray(value)
		except (TypeError, ValueError) as exc: raise ValueError(f"Invalid {kind} field {name!r}") from exc
		if supplied.shape != shape or supplied.dtype.kind not in "fiu" or (dtype.kind in "iu" and supplied.dtype.kind not in "iu"):
			raise ValueError(f"{kind.capitalize()} field {name!r} must have shape {shape} and numeric type {dtype}")
		try:
			with np.errstate(over="ignore", invalid="ignore"):
				converted = supplied.astype(dtype)
			if not np.all(np.isfinite(supplied)) or not np.all(np.isfinite(converted)):
				raise ValueError()
			if dtype.kind in "iu" and not np.all(supplied == converted):
				raise ValueError()
		except (TypeError, OverflowError, ValueError) as exc:
			raise ValueError(f"{kind.capitalize()} field {name!r} is out of range or non-finite") from exc
		return converted

	def _frame_value(self, name, value):
		return self._uniform_value(self.uniforms, name, value, "frame")

	def update_frame(self, *, view: ArrayLike, projection: ArrayLike,
		camera_position: ArrayLike, near: float, far: float,
		frame=None, lighting=None) -> NDArray[np.float32]:
		"""Upload explicit camera values and return view @ projection for culling.

		Matrices follow the existing row-vector convention. No camera class is
		required; resize targets first. Invalid inputs leave uniforms unchanged.
		"""
		self._require_targets()
		view, projection = np.asarray(view, dtype=np.float32), np.asarray(projection, dtype=np.float32)
		position = np.asarray(camera_position, dtype=np.float32)
		if (view.shape != (4, 4) or projection.shape != (4, 4)
			or position.shape != (3,)
			or not all(np.all(np.isfinite(value)) for value in (view, projection, position))):
			raise ValueError("Expected finite 4x4 view/projection matrices and vec3 camera position")
		if not (np.isfinite(near) and np.isfinite(far) and 0 < near < far):
			raise ValueError("Camera clip distances must satisfy 0 < near < far")
		if frame is None: frame = {}
		if not isinstance(frame, Mapping): raise ValueError("frame must be a mapping of field values")
		unknown = set(frame) - self.frame_fields.keys()
		if unknown: raise ValueError(f"Unknown frame fields: {sorted(map(str, unknown))}")
		custom = {name: self._frame_value(name, value) for name, value in frame.items()}
		if lighting is None: lighting = {}
		if not isinstance(lighting, Mapping): raise ValueError("lighting must be a mapping of field values")
		fields = set(self.lighting_uniforms.dtype.names) if self.lighting_uniforms is not None else set()
		unknown = set(lighting) - fields
		if unknown: raise ValueError(f"Unknown lighting fields: {sorted(map(str, unknown))}")
		light_values = {}
		for name, value in lighting.items():
			if name == "light_dir":
				try: value = np.asarray(value)
				except (TypeError, ValueError) as exc: raise ValueError("Invalid lighting field 'light_dir'") from exc
				if value.shape != (3,) or value.dtype.kind not in "fiu":
					raise ValueError("Lighting field 'light_dir' must be a numeric vec3")
				value = np.append(value, 0)
			light_values[name] = self._uniform_value(self.lighting_uniforms, name, value, "lighting")
		values = dict(view=view, proj=projection,
			camera_position=(*position, 1), clip=(near, far, self.effective_noise_scale, 0))
		values.update(custom)
		for name, value in values.items(): self.uniforms.content[name] = value
		if light_values:
			for name, value in light_values.items(): self.lighting_uniforms.content[name] = value
			self.lighting_uniforms.upload_changed()
		self.uniforms.upload_changed()
		return view @ projection

	def update_camera(self, camera, *, frame=None, lighting=None):
		"""Convenience adapter for camera objects; update_frame accepts plain arrays."""
		width, height = self._require_targets()
		return self.update_frame(view=camera.view(), projection=camera.projection(width/height),
			camera_position=camera.position, near=camera.near, far=camera.far, frame=frame, lighting=lighting)

	def opaque_pass(self, commands, *, clear=(0.035, 0.05, 0.08, 1), load=False):
		"""Draw opaque geometry; load=True preserves existing color and depth."""
		self._require_targets()
		return commands.render_pass(color=self.main_color.color_attachment(clear=None if load else clear),
			depth=self.depth.depth_attachment(clear=None if load else 1), label="1_opaque")

	def nearest_pass(self, commands):
		self._require_targets()
		colors = [self.nearest_payload.color_attachment(clear=(0, 0, 0, 0)), self.nearest_alpha.color_attachment(clear=(0, 0, 0, 0))]
		if self.sample_count > 1: colors.append(self.nearest_extra.color_attachment(clear=(0, 0, 0, 0)))
		return commands.render_pass(color=colors, depth=self.nearest_depth.depth_attachment(clear=1), label="2_nearest_layer")

	def shade_nearest(self, commands, geometry):
		width, height = self._require_targets()
		if geometry.owner is not self: raise ValueError("Geometry passes belong to another transparency pass")
		with commands.compute_pass(label="2_shade_nearest") as compute:
			compute.set_pipeline(self.nearest_shade_pipeline)
			for bindings in (self._frame_group(self.nearest_shading_shader), self.nearest_shade_targets, geometry.nearest_shading, *geometry.shading_bindings): compute.set_bind_group(bindings)
			compute.dispatch((width+7)//8, (height+7)//8, self.sample_count)

	def transmittance_pass(self, commands):
		self._require_targets()
		commands.clear_buffer(self.band_counts)
		width, height = self.band_payloads.size[:2]
		with commands.compute_pass(label="2_clear_bands") as compute:
			compute.set_pipeline(self.band_clear_pipeline)
			compute.set_bind_group(self.band_clear_targets)
			compute.dispatch((width+7)//8, (height+7)//8, self.tail_sample_count*2)
		colors = [self.transmittance.color_attachment(clear=(1, 1, 1, 1))]
		if self.banded_proxy and not self.mixed_tail: colors.append(self.band_proxy.color_attachment(clear=(0, 0, 0, 0)))
		return commands.render_pass(color=colors, depth=self.depth.depth_attachment(read_only=True), label="2_transmittance")

	def banded_pass(self, commands):
		"""Allocate both banded payloads after recording transmittance once."""
		self._require_targets()
		if self.mixed_tail: return commands.render_pass(color=self.band_proxy.color_attachment(), label="3_banded_tail")
		return commands.render_pass(depth=self.depth.depth_attachment(read_only=True), label="3_banded_tail")

	def tail_statistics_pass(self, commands):
		"""Accumulate full-resolution single-sample S/Q for a mixed-sample tail."""
		self._require_targets()
		if not self.mixed_tail: raise RuntimeError("Separate tail statistics require mixed tail sampling")
		return commands.render_pass(color=self.band_proxy.color_attachment(clear=(0, 0, 0, 0)), label="2_tail_statistics")

	def shade_winner(self, commands, geometry):
		"""Shade both banded tail winners in one dispatch."""
		width, height = self._require_targets()
		if geometry.owner is not self: raise ValueError("Geometry passes belong to another transparency pass")
		step = 16 if self.banded_subset else 8
		with commands.compute_pass(label="4_shade_winner") as compute:
			compute.set_pipeline(self.shade_pipeline)
			for bindings in (self.shade_frame, self.shade_targets, geometry.shading, *geometry.shading_bindings): compute.set_bind_group(bindings)
			compute.dispatch((width+step-1)//step, (height+step-1)//step, self.tail_sample_count)

	def compose(self, commands, output):
		"""Reconstruct into a caller-provided ColorAttachment; does not submit."""
		self._require_targets()
		if output.format != self.target_format: raise ValueError("Output format must match the configured transparency target_format")
		if self.debug_view in ("final", "reconstruction"):
			for entry, inputs, outputs, size in self.kawase_stages:
				with commands.compute_pass(label=f"5_kawase_{entry}") as compute:
					compute.set_pipeline(self.kawase_pipelines[entry])
					compute.set_bind_group(inputs)
					compute.set_bind_group(outputs)
					compute.dispatch((size[0]+7)//8, (size[1]+7)//8, self.tail_sample_count)
		with commands.render_pass(color=output, label="5_reconstruct_compose") as render:
			render.set_pipeline(self.compose_pipeline)
			render.set_bind_groups(*((self.compose_frame,) if self.compose_frame is not None else ()), self.compose_resources)
			render.draw(3)


def record_transparency(commands, transparency, geometry, draw):
	"""Record banded transparency using draw(render, pass_name)."""
	with transparency.nearest_pass(commands) as render:
		draw(render, "nearest")
	transparency.shade_nearest(commands, geometry)
	with transparency.transmittance_pass(commands) as render:
		draw(render, "transmittance")
	if transparency.mixed_tail:
		with transparency.tail_statistics_pass(commands) as render:
			draw(render, "tail_statistics")
	with transparency.banded_pass(commands) as render:
		draw(render, "bands")
	transparency.shade_winner(commands, geometry)
