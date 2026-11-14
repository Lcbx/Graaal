#set MATERIAL_STAGE = "shading"
#include "stochastic_common.shaderlib"
#include "draw_common.shaderlib"
#include "utils_transform.shaderlib"
#include "instance_transform.shaderlib"

<ShadingVertex> position: vec4f;
<ShadingVertex> normal: vec4f;
<ShadingVertex> uv: vec4f;
storage<2, 4, read> vertices: array<ShadingVertex>;
storage<2, 5, read> shading_instances: array<MeshInstance>;
#if PARAMS.get("sample_count", 1) > 1
texture<1, 1> winner_color: texture_storage_2d_array<rgba16float, write>;
#else
texture<1, 1> winner_color: texture_storage_2d<rgba16float, write>;
#endif
#if PARAMS.get("shade_nearest", false)
sample_texture<1, 0> winners: u32;
#if PARAMS.get("sample_count", 1) > 1
texture<1, 2> winners_extra: texture_multisampled_2d<u32>;
#endif


#endif

fn shade_payload(payload: vec4u, pixel: vec2i, sample: u32) -> vec4f {
	// Empty pixels are zero samples, including pixels where all low-alpha
	// fragments lost the stochastic test. Never leave last frame's value here.
	if payload.x == 0u {
		return vec4f(0.0);
	}
	let instance_id = payload.x - 1u;
	if instance_id >= arrayLength(&metadata) || instance_id >= arrayLength(&shading_instances) {
		return vec4f(0.0);
	}
	let info = metadata[instance_id];
	let triangle_count = arrayLength(&vertices) / 3u;
#if PARAMS.get("material_library") is not none
	let material_count = library_material_count();
#else
	let material_count = arrayLength(&materials);
#endif
	if info.x >= triangle_count || payload.y >= triangle_count - info.x || info.y >= material_count {
		return vec4f(0.0);
	}
	let triangle = (info.x + payload.y) * 3u;
	let raw_a = vertices[triangle];
	let raw_b = vertices[triangle + 1u];
	let raw_c = vertices[triangle + 2u];
	// Deform each original vertex before interpolating, exactly as rasterization.
	let a = deform_vertex(instance_id, raw_a.position.xyz, raw_a.normal.xyz, raw_a.uv.xy);
	let b = deform_vertex(instance_id, raw_b.position.xyz, raw_b.normal.xyz, raw_b.uv.xy);
	let c = deform_vertex(instance_id, raw_c.position.xyz, raw_c.normal.xyz, raw_c.uv.xy);
	let xy = unpack2x16unorm(payload.z);
	var bary = max(vec3f(xy, 1.0-xy.x-xy.y), vec3f(0.0));
	bary /= max(bary.x+bary.y+bary.z, 1e-6);
	let inst = shading_instances[instance_id];
	let position = instance_position(inst, a.position.xyz*bary.x + b.position.xyz*bary.y + c.position.xyz*bary.z);
	let normal = instance_normal(inst, a.normal.xyz*bary.x + b.normal.xyz*bary.y + c.normal.xyz*bary.z);
	let uv = a.uv.xy*bary.x + b.uv.xy*bary.y + c.uv.xy*bary.z;
	let lod_face = unpack2x16float(payload.w);
	let color = shade_material(instance_id, get_material(instance_id), uv, position, select(-normal, normal, lod_face.y > 0.0), lod_face.x);
	return vec4f(color, 1.0);
}

#if PARAMS.get("shade_nearest", false)
fn winner_payload(id: vec3u) -> vec4u {
	let pixel = vec2i(id.xy);
#if PARAMS.get("sample_count", 1) > 1
	let low = sample_texture_load(winners, pixel, id.z);
	let high = textureLoad(winners_extra, pixel, i32(id.z));
	return vec4u(low.x | (low.y << 16u), low.z | (low.w << 16u),
		high.x | (high.y << 16u), high.z | (high.w << 16u));
#else
	return sample_texture_load(winners, pixel, id.z);
#endif
}

@compute @workgroup_size(8, 8)
fn shade_nearest(@builtin(global_invocation_id) id: vec3u) {
	if any(id.xy >= textureDimensions(winners)) { return; }
	store_winner(id, shade_payload(winner_payload(id), vec2i(id.xy), id.z));
}
#endif

fn store_winner(id: vec3u, color: vec4f) {
#if PARAMS.get("sample_count", 1) > 1
	textureStore(winner_color, vec2i(id.xy), i32(id.z), color);
#else
	textureStore(winner_color, vec2i(id.xy), color);
#endif
}


#if not PARAMS.get("shade_nearest", false)
#include "banded_sampling.shaderlib"
<BandCount> total: u32;
<BandCount> running: u32;
storage<1, 11, read> band_counts: array<BandCount>;
texture<1, 10> band_payloads: texture_2d_array<u32>;
#if PARAMS.get("banded_proxy", false)
sample_texture<1, 12> band_proxy: f32;
#endif

@compute @workgroup_size(8, 8)
fn shade_bands(@builtin(global_invocation_id) id: vec3u) {
	let size = textureDimensions(winner_color);
#if PARAMS.get("banded_subset", false)
	let origin = id.xy*2u;
	if any(origin >= size) { return; }
	let selected = band_selected_pixel(id.xy);
	for (var y = 0u; y < 2u; y += 1u) {
		for (var x = 0u; x < 2u; x += 1u) {
			let q = origin+vec2u(x, y);
			if all(q < size) && any(q != selected) { store_winner(vec3u(q, id.z), vec4f(0.0)); }
		}
	}
	if any(selected >= size) { return; }
	let actual_id = vec3u(selected, id.z);
	let payload_pixel = vec2i(id.xy);
#else
	if any(id.xy >= size) { return; }
	let actual_id = id;
	let payload_pixel = vec2i(id.xy);
#endif
	let pixel = vec2i(actual_id.xy);
	let a = textureLoad(band_payloads, payload_pixel, i32(id.z*2u), 0);
	let b = textureLoad(band_payloads, payload_pixel, i32(id.z*2u+1u), 0);
	let ca = shade_payload(a, pixel, id.z);
	var cb = ca;
	if any(a != b) { cb = shade_payload(b, pixel, id.z); }
	let index = (actual_id.y*size.x+actual_id.x)*{{PARAMS.get("sample_count", 1)}}u+id.z;
	let s = f32(band_counts[index].total)/65536.0;
	let den = min(1.0, s);
	var residual = 0.5*(ca.rgb+cb.rgb);
#if PARAMS.get("banded_proxy", false)
	residual -= sample_texture_load(band_proxy, pixel, id.z).rgb/max(1.0, s);
#endif
	store_winner(actual_id, vec4f(select(vec3f(0.0), residual, den > 0.0), den));
}
#endif
