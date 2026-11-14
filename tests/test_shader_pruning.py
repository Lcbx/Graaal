"""CPU-only shader generation checks; GPU compilation is exercised by rendering tests."""
from pathlib import Path

import pytest

from graaal.Materials import MaterialLibrary
from graaal.RenderContext import Shader, ShaderSource
from scenes.material_example import make_instances


SHADERS = Path(__file__).resolve().parents[1] / "graaal" / "shaders"

def load_shader(name, **params):
	loader = Shader.__new__(Shader)
	source, declarations = loader._render_source(None, str(SHADERS / name), None, None, params)
	return ShaderSource(source, declarations=declarations)


def test_material_library_only_emits_stage_hooks():
	instances, _, _ = make_instances()
	library = MaterialLibrary([instance.mesh.shader for instance in instances])
	geometry_info = load_shader("stochastic_geometry.shader", material_library=library)
	shading_info = load_shader("stochastic_shade.shader", material_library=library)
	assert "opaque" in geometry_info.entry_points
	assert "library_shade" in geometry_info.functions
	assert "shade_material" in geometry_info.functions
	assert "library_grazing" not in shading_info.functions
	assert "library_opacity" not in shading_info.functions
	assert "shade_0" in geometry_info.functions
	assert "grazing_0" not in shading_info.functions
	assert "opacity_0" not in shading_info.functions
	assert "shade_0" in shading_info.functions
	assert "grazing_0" in geometry_info.functions
	assert "opacity_0" in geometry_info.functions
	assert "default_lighting_3" in geometry_info.functions
	assert "default_lighting_3" in shading_info.functions
	assert "shade_3" in geometry_info.functions
	with pytest.raises(ValueError, match="Unknown material stage"):
		library.shaders[0].render(0, "invalid")


def test_material_parameter_decoder_matches_packed_lanes():
	from graaal.Materials import MaterialShader
	shader = MaterialShader(source='''
#extends "material_surface.shaderlib"
#block parameters
<MaterialParameters> color: vec3f = (0.1, 0.2, 0.3);
<MaterialParameters> gain: f32 = 0.5;
<InstanceParameters> valid: bool = true;
#endblock
''')
	library = MaterialLibrary([shader])
	info = load_shader("stochastic_geometry.shader", material_library=library)
	source = info.source
	assert "p.color = bitcast<vec3f>(material_parameters.values[material_offset+0u].xyz);" in source
	assert "p.gain = bitcast<f32>(material_parameters.values[material_offset+0u].w);" in source
	assert "p.valid = instance_parameters.values[instance_offset+0u].x != 0u;" in source
	assert "fn library_vertex(" in source and "fn library_opacity(" in source
	shading = load_shader("stochastic_shade.shader", material_library=library)
	assert "fn library_vertex(" in shading.source and "fn library_shade(" in shading.source
	assert "fn library_opacity(" not in shading.source and "fn library_grazing(" not in shading.source


def test_utility_compatibility_include_still_exports_macros():
	path = SHADERS.parents[1] / "scenes" / "shaders" / "simple.shader"
	loader = Shader.__new__(Shader)
	source, declarations = loader._render_source(None, str(path), None, None, {})
	info = ShaderSource(source, declarations=declarations)
	assert "unpack_instance_transform" in info.functions
	assert "unpack_rgba8_srgb" in info.functions
	assert "hash_u32" in info.functions  # legacy include still provides all utilities


@pytest.mark.parametrize("sample_count", [1, 4])
@pytest.mark.parametrize("proxy,subset", [(False, False), (True, False), (True, True)])
def test_material_shader_variants_expand(sample_count, proxy, subset):
	instances, _, _ = make_instances()
	library = MaterialLibrary([instance.mesh.shader for instance in instances])
	base = dict(sample_count=sample_count, banded_proxy=proxy, banded_subset=subset)
	geometry = load_shader("stochastic_geometry.shader", material_library=library, **base)
	shading = load_shader("stochastic_shade.shader", material_library=library, **base)
	assert {"select_bands", "nearest_layer", "transmittance"} <= set(geometry.entry_points)
	assert "shade_bands" in shading.entry_points
	assert "select_winner" not in geometry.entry_points
	assert "library_opacity" not in shading.functions
	nearest = load_shader("stochastic_shade.shader", material_library=library, shade_nearest=True, **base)
	assert "shade_nearest" in nearest.entry_points and "shade_bands" not in nearest.entry_points
	compose = load_shader("stochastic_compose.shader", srgb_surface=True, **base)
	assert "compose" in compose.entry_points
	assert "linear_to_srgb" not in compose.functions


def test_custom_frame_schema_is_shared_across_render_stages():
	frame_fields = {"time": "f32", "wind": "vec3f", "seed": "u32"}
	params = dict(frame_fields=frame_fields, srgb_surface=False, sample_count=4,
		banded_proxy=True)
	modules = [load_shader(name, **params) for name in (
		"stochastic_geometry.shader", "stochastic_shade.shader", "stochastic_compose.shader")]
	dtype = modules[0].uniforms["frame"].dtype
	assert dtype.names[-3:] == ("time", "wind", "seed")
	assert all(module.uniforms["frame"].dtype == dtype for module in modules)
	assert "time" not in load_shader("stochastic_geometry.shader").uniforms["frame"].dtype.names
