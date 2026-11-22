"""Ordered fullscreen postprocessing, with explicitly fused effect groups."""
from __future__ import annotations

from dataclasses import asdict
from operator import index
from pathlib import Path
from typing import cast, Any, Mapping, Sequence, TYPE_CHECKING
import re

import numpy as np

from ..RenderContext import BindGroupLayout, PipelineLayout, RenderContext, RenderPipeline, Sampler, Shader, Texture, TextureUsage, wgpu
from .pass_settings import AmbientOcclusionSettings, AntialiasingSettings


if TYPE_CHECKING:
	from ..Math import Camera
	from ..RenderContext import BindingInfo, CommandContext, ColorAttachment, GpuBuffer


class PostprocessingEffect:
	"""Shader contribution for one stage of a postprocessing group.

	shader_fragment returns Jinja source and options. Templates receive INDEX,
	GROUP=1, SLOT=the next free slot and OPTIONS. Declare names and slots in the shader;
	resource and uniform hooks return mappings keyed by those names. Define
	The function named by shader_function receives uv and can call sampleN(uv),
	where N is INDEX. Override pipeline_key for shader state.
	"""
	requires_depth = False
	label = "postprocessing"

	def shader_fragment(self, index: int) -> tuple[str, dict[str, Any]]:
		raise NotImplementedError

	def shader_function(self, index: int) -> str:
		return f"effect{index}"

	def pipeline_key(self) -> tuple[Any, ...]:
		return type(self), getattr(self, "mode", None), getattr(self, "settings", None)

	def uniform_values(self, index: int) -> Mapping[str, Mapping[str, Any]]:
		return {}

	def binding_layouts(self, index: int) -> Mapping[str, dict[str, Any]]:
		"""Override reflected binding properties by name, excluding slot/visibility."""
		return {}

	def shader_paths(self) -> Sequence[str | Path]:
		"""Additional directories searched for this effect's shader includes."""
		return ()

	def resources(self, index: int, depth: Texture | None) -> Mapping[str, Any]:
		"""Return additional shader bindings for this effect."""
		return {f"effect{index}_depth": depth} if self.requires_depth else {}


class AntialiasingEffect(PostprocessingEffect):
	def shader_function(self, index: int) -> str: return "antialiasing"
	@property
	# The effect label follows its current mode.
	def label(self) -> str: return self.mode  # type: ignore[override]

	def shader_fragment(self, index: int) -> tuple[str, dict[str, Any]]:
		return '#include "antialiasing.shaderlib"\n', {"mode": self.mode, **asdict(self.settings)}

	def __init__(self, mode: str="faaa", settings: AntialiasingSettings | None=None) -> None:
		if mode not in ("lxaa", "faaa", "ddaa2"): raise ValueError("Unknown antialiasing mode")
		self.mode = mode
		self.set_settings(settings if settings is not None else AntialiasingSettings())

	def set_settings(self, settings: AntialiasingSettings) -> None:
		if not isinstance(settings, AntialiasingSettings): raise TypeError("Expected AntialiasingSettings")
		self.settings = settings


class AmbientOcclusionEffect(PostprocessingEffect):
	def shader_function(self, index: int) -> str: return "apply_ambient_occlusion"
	requires_depth = True
	label = "6_ambient_occlusion"
	def shader_fragment(self, index: int) -> tuple[str, dict[str, Any]]:
		return '#include "ambient_occlusion.shaderlib"\n', {"ao": asdict(self.settings)}

	def uniform_values(self, index: int) -> Mapping[str, Mapping[str, Any]]:
		if self.camera is None: raise RuntimeError("Update ambient occlusion camera before recording")
		return {"ao_camera": self.camera}

	def resources(self, index: int, depth: Texture | None) -> Mapping[str, Any]: return {"ao_depth": depth}

	def __init__(self, settings: AmbientOcclusionSettings | None=None) -> None:
		self.set_settings(settings if settings is not None else AmbientOcclusionSettings())
		self.camera: dict[str, Any] | None
		self.camera = None

	def set_settings(self, settings: AmbientOcclusionSettings) -> None:
		if not isinstance(settings, AmbientOcclusionSettings): raise TypeError("Expected AmbientOcclusionSettings")
		self.settings = settings

	def update_camera(self, camera: Camera, aspect: float) -> None:
		projection = np.asarray(camera.projection(aspect), dtype=np.float32)
		if projection.shape != (4, 4) or not np.all(np.isfinite(projection)): raise ValueError("Camera projection must be a finite 4x4 matrix")
		if not (np.isfinite(camera.near) and np.isfinite(camera.far) and 0 < camera.near < camera.far): raise ValueError("Camera clip distances must satisfy 0 < near < far")
		self.camera = {"projection": projection, "clip": (camera.near, camera.far, 0, 0)}


def _binding_layout(info: BindingInfo) -> dict[str, Any]:
	"""Infer WGSL properties; runtime-only properties use named overrides."""
	if info.storage == "uniform": return {"buffer": {"type": "uniform"}}
	if info.storage == "storage": return {"buffer": {"type": "storage" if info.access == "read_write" else "read-only-storage"}}
	kind = info.type_name.replace(" ", "")
	if kind in ("sampler", "sampler_comparison"): return {"sampler": {"type": "comparison" if kind == "sampler_comparison" else "filtering"}}
	match = re.fullmatch(r"texture_(depth_)?(multisampled_)?(1d|2d|3d|cube)(_array)?(?:<(f32|i32|u32)>)?", kind)
	if match:
		depth, multisampled, dimension, array, scalar = match.groups()
		return {"texture": {"sample_type": "depth" if depth else {"f32": "float", "i32": "sint", "u32": "uint"}[scalar], "view_dimension": dimension+("-array" if array else ""), "multisampled": bool(multisampled)}}
	raise ValueError(f"Supply a binding_layouts override for {info.name}: {info.type_name}")


class PostprocessingPass:
	"""Apply a list of effects sequentially; tuples request fusion into one draw."""
	def __init__(self, effects: list[PostprocessingEffect | tuple[PostprocessingEffect, ...]], *, target_format: str | None=None) -> None:
		if not isinstance(effects, list): raise TypeError("Effects must be a list")
		self.groups = tuple(item if isinstance(item, tuple) else (item,) for item in effects)
		if any(not group or any(not isinstance(effect, PostprocessingEffect) for effect in group) for group in self.groups): raise ValueError("Each group must contain postprocessing effects")
		if any(len({type(effect) for effect in group}) != len(group) for group in self.groups): raise ValueError("A fused group cannot contain the same effect type twice")
		self.target_format: str = target_format or cast(str, getattr(RenderContext, "presentation_format", "rgba8unorm"))
		self.sample_format = self.target_format.removesuffix("-srgb")
		self.manual_filter = self.sample_format == "rgba32float" and "float32-filterable" not in RenderContext.device.features
		self.sampler = None if self.manual_filter else Sampler(min_filter="linear", mag_filter="linear", address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge")
		self.size: tuple[int, int] | None
		self.size = None
		self.targets: list[Texture]
		self.targets = []
		self.sampled: list[Texture]
		self.sampled = []
		self.pipelines: dict[tuple[Any, ...], tuple[Shader, RenderPipeline, dict[str, GpuBuffer]]]
		self.pipelines = {}

	def resize(self, size: tuple[int, int]) -> bool:
		if len(size) != 2: raise ValueError("Expected width and height")
		try: size = (index(size[0]), index(size[1]))
		except TypeError: raise ValueError("Expected integer dimensions") from None
		if min(size) <= 0 or size == self.size: return False
		for pipeline in self.pipelines.values(): pipeline[1].clear_bind_group_cache()
		self.targets = [Texture(size, format=self.target_format, usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_SRC, label="postprocessing_source") for _ in range(max(1, len(self.groups)))]
		self.sampled = [Texture(size, format=self.sample_format, usage=TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING, label="postprocessing_encoded") for _ in self.targets] if self.sample_format != self.target_format else []
		self.size = size
		return True

	def color_attachment(self, *, clear: tuple[float, float, float, float] | None=None) -> ColorAttachment:
		if self.size is None: raise RuntimeError("Resize postprocessing targets first")
		return self.targets[0].color_attachment(clear=clear)

	def _pipeline(self, group: Sequence[PostprocessingEffect], sample_count: int, group_index: int) -> tuple[Shader, RenderPipeline, dict[str, GpuBuffer]]:
		# Each recorded stage needs its own uniforms even when shaders are identical.
		key = (group_index, sample_count, tuple(effect.pipeline_key() for effect in group))
		if key in self.pipelines: return self.pipelines[key]
		params: dict[str, Any]
		params = {"sample_count": sample_count, "manual_filter": self.manual_filter, "srgb_surface": self.target_format.endswith("-srgb"), "functions": [effect.shader_function(i) for i, effect in enumerate(group)], "options": []}
		parts, overrides = [], {}
		for i, effect in enumerate(group):
			part, options = effect.shader_fragment(i)
			params["options"].append(options)
			parts.append(f"#with INDEX={i}, GROUP=1, SLOT=next_binding(1), OPTIONS=PARAMS.options[{i}]\n"+part+"\n#endwith\n")
			for name, descriptor in effect.binding_layouts(i).items():
				if name in overrides: raise ValueError(f"Duplicate layout override: {name}")
				overrides[name] = descriptor
		source = '#extends "postprocessing.shader"\n#block effects\n'+"\n".join(parts)+'\n#endblock\n'
		paths = [str(path) for effect in group for path in effect.shader_paths()]
		shader = Shader(source=source, basedir=[str(Path(__file__).resolve().parents[1] / "shaders")]+paths, params=params)
		layout: str | PipelineLayout = "auto"
		if self.manual_filter: overrides["scene_color"] = {"texture": {"sample_type": "unfilterable-float", "view_dimension": "2d", "multisampled": False}}
		if overrides:
			unknown = overrides.keys()-shader.info.bindings.keys()
			if unknown: raise ValueError(f"Unknown binding layout overrides: {sorted(unknown)}")
			entries: list[list[dict[str, Any]]]
			entries = [[] for _ in range(max(info.group for info in shader.info.bindings.values())+1)]
			for name, info in shader.info.bindings.items():
				descriptor = overrides[name] if name in overrides else _binding_layout(info)
				if "binding" in descriptor or "visibility" in descriptor: raise ValueError("Layout overrides cannot change binding or visibility")
				entries[info.group].append(dict(binding=info.binding, visibility=wgpu.ShaderStage.FRAGMENT, **descriptor))
			layout = PipelineLayout([BindGroupLayout(group_entries) for group_entries in entries])
		pipeline = RenderPipeline(shader, vertex_entry="fullscreen", fragment_entry="fragment", depth_test=None, depth_write=False, cull_mode="none", layout=layout, label="postprocessing")
		buffers: dict[str, GpuBuffer]
		buffers = {}
		if len(self.pipelines) >= 32: self.pipelines.pop(next(iter(self.pipelines)))
		self.pipelines[key] = shader, pipeline, buffers
		return self.pipelines[key]

	def record(self, commands: CommandContext, output: ColorAttachment, *, depth: Texture | None=None) -> None:
		if self.size is None: raise RuntimeError("Resize postprocessing targets first")
		if output.format != self.target_format or output.sample_count != 1: raise ValueError("Output must match the target format and be single-sample")
		if any(output.view == target.binding for target in self.targets): raise ValueError("Postprocessing output cannot alias an input target")
		ao = any(effect.requires_depth for group in self.groups for effect in group)
		if ao and (depth is None or depth.size[:2] != self.size or not depth.format.startswith("depth") or depth.sample_count not in (1, 4) or not depth.usage & TextureUsage.TEXTURE_BINDING): raise ValueError("Effect requires matching sampleable depth")
		groups = self.groups or ((),)
		for group_index, group in enumerate(groups):
			shader, pipeline, buffers = self._pipeline(group, cast(Texture, depth).sample_count if ao else 1, group_index)
			color = self.targets[group_index]
			if self.sampled:
				commands.handle.copy_texture_to_texture({"texture": color.handle}, {"texture": self.sampled[group_index].handle}, (*self.size, 1))
				color = self.sampled[group_index]
			resources: dict[str, Any] = {"scene_color": color}
			if self.sampler is not None: resources["scene_sampler"] = self.sampler
			for i, effect in enumerate(group):
				additional = dict(effect.resources(i, depth))
				for name, values in effect.uniform_values(i).items():
					if name in additional: raise ValueError(f"Duplicate resource binding: {name}")
					if name not in shader.info.uniforms: raise ValueError(f"Unknown uniform binding: {name}")
					if name not in buffers: buffers[name] = shader.uniform_buffer(name)
					buffer = buffers[name]
					assert buffer.content is not None
					assert buffer.content is not None
					for field, value in values.items(): buffer.content[field] = value
					buffer.upload()
					additional[name] = buffer
				if resources.keys() & additional.keys(): raise ValueError("Duplicate postprocessing resource binding")
				resources.update(additional)
			bindings: dict[int, dict[str, Any]]
			bindings = {}
			for name, resource in resources.items():
				info = shader.info.bindings[name]
				bindings.setdefault(info.group, {})[name] = resource
			destination = self.targets[group_index+1].color_attachment() if group_index+1 < len(groups) else output
			label = group[0].label if len(group) == 1 else "postprocessing"
			with commands.render_pass(color=destination, label=label) as render:
				render.set_pipeline(pipeline)
				render.set_bind_groups(*(shader.bind_group(number, **mapping) for number, mapping in sorted(bindings.items())))
				render.draw(3)
