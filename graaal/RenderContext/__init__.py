"""GPU backend, resources, commands and shader APIs, loaded on explicit access."""
from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from .core import (
		RenderContext,
		GCManager,
		CommandContext,
		RenderPass,
		ComputePass,
		ColorAttachment,
		DepthAttachment,
		Color,
		dtype_to_vertex_format,
		GpuBuffer,
		Mesh,
		Texture,
		TextureView,
		Sampler,
		BufferUsage,
		TextureUsage,
		ShaderAttribute,
		ShaderField,
		ShaderStruct,
		BindingInfo,
		EntryPointInfo,
		UniformInfo,
		VertexInfo,
		ShaderDeclarations,
		ShaderSource,
		ShaderInterface,
		Shader,
		DefaultFalseDict,
		BindGroup,
		BindGroupLayout,
		PipelineLayout,
		RenderPipeline,
		ComputePipeline,
		create_depth_framebuffer,
		create_depth_sampler,
		wgpu,
	)


__all__ = (
	'RenderContext', 'GCManager', 'CommandContext', 'RenderPass', 'ComputePass',
	'ColorAttachment', 'DepthAttachment', 'Color', 'dtype_to_vertex_format',
	'GpuBuffer', 'Mesh', 'Texture', 'TextureView', 'Sampler', 'BufferUsage', 'TextureUsage',
	'ShaderAttribute', 'ShaderField', 'ShaderStruct', 'BindingInfo', 'EntryPointInfo',
	'UniformInfo', 'VertexInfo', 'ShaderDeclarations', 'ShaderSource', 'ShaderInterface', 'Shader',
	'DefaultFalseDict', 'BindGroup', 'BindGroupLayout', 'PipelineLayout', 'RenderPipeline',
	'ComputePipeline', 'create_depth_framebuffer', 'create_depth_sampler', 'wgpu',
)


def __getattr__(name: str) -> Any:
	if name not in __all__:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	value = getattr(_import_module('.core', __name__), name)
	globals()[name] = value
	return value
