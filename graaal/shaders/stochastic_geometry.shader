#include "draw_common.shaderlib"
#set MATERIAL_STAGE = "geometry"
#include "stochastic_common.shaderlib"
#include "utils_transform.shaderlib"
#include "utils_texture.shaderlib"
#include "utils_hash.shaderlib"
#include "utils_alpha.shaderlib"
#include "instance_transform.shaderlib"

storage<1, 0, read> instances: array<MeshInstance>;
storage<1, 1, read> visible_instances: array<u32>;
#set GUIDE_SAMPLES = PARAMS.get("guide_sample_count", PARAMS.get("sample_count", 1))
#if GUIDE_SAMPLES > 1 and PARAMS.get("sample_count", 1) == 1
texture<3, 0> opaque_depth: texture_depth_multisampled_2d;
fn raster_opaque_depth(pixel: vec2i, sample: u32) -> f32 { return textureLoad(opaque_depth, pixel, i32(sample)); }
#else
sample_texture<3, 0> opaque_depth: depth;
fn raster_opaque_depth(pixel: vec2i, sample: u32) -> f32 { return sample_texture_load(opaque_depth, pixel, sample); }
#endif
texture<3, 1> blue_noise: texture_2d<f32>;
#include "banded_sampling.shaderlib"
<BandCount> total: atomic<u32>;
<BandCount> running: atomic<u32>;
storage<3, 10, read_write> band_counts: array<BandCount>;
texture<3, 11> band_payloads: texture_storage_2d_array<rgba32uint, write>;

fn band_index(pixel: vec2u, sample: u32) -> u32 {
	return (pixel.y*textureDimensions(opaque_depth).x+pixel.x)*{{PARAMS.get("sample_count", 1)}}u+sample;
}
#if PARAMS.get("banded_proxy", false) and PARAMS.get("banded_statistics", true)
<BandTransmission> @location(0) transmission: vec4f;
<BandTransmission> @location(1) proxy: vec4f;
#endif
#if GUIDE_SAMPLES > 1 and PARAMS.get("sample_count", 1) == 1
texture<3, 8> nearest_ids: texture_multisampled_2d<u32>;
#else
sample_texture<3, 8> nearest_ids: u32;
#endif
#if GUIDE_SAMPLES > 1
texture<3, 9> nearest_ids_extra: texture_multisampled_2d<u32>;
#endif
fn is_exact_nearest(input: VertexOutput, sample: u32) -> bool {
#if GUIDE_SAMPLES > 1 and PARAMS.get("sample_count", 1) == 1
	// Pixel-center rasterization can land on the other triangle across an
	// internal edge. Exclude any native nearest ID, avoiding a false front tail
	// along mesh diagonals without assuming cards or merging primitive IDs.
	for (var native = 0u; native < {{GUIDE_SAMPLES}}u; native += 1u) {
		let low = textureLoad(nearest_ids, vec2i(input.position.xy), i32(native));
		let ids = vec2u(low.x | (low.y << 16u), low.z | (low.w << 16u));
		if all(ids == vec2u(input.instance+1u, input.triangle)) { return true; }
	}
	return false;
#else
#if GUIDE_SAMPLES > 1
	let low = textureLoad(nearest_ids, vec2i(input.position.xy), i32(sample));
	let ids = vec2u(low.x | (low.y << 16u), low.z | (low.w << 16u));
#else
	let ids = sample_texture_load(nearest_ids, vec2i(input.position.xy), sample).xy;
#endif
	return all(ids == vec2u(input.instance+1u, input.triangle));
#endif
}


<VertexInput> position: vec3f;
<VertexInput> normal: vec3f;
<VertexInput> uv: vec2f;
<VertexInput> barycentric: vec2f;
<VertexInput> triangle: u32;

<VertexOutput> @builtin(position) position: vec4f;
#if PARAMS.get("sample_count", 1) > 1
<VertexOutput> @interpolate(perspective, sample) world: vec3f;
<VertexOutput> @interpolate(perspective, sample) normal: vec3f;
<VertexOutput> @interpolate(perspective, sample) uv: vec2f;
<VertexOutput> @interpolate(perspective, sample) barycentric: vec2f;
#else
<VertexOutput> world: vec3f;
<VertexOutput> normal: vec3f;
<VertexOutput> uv: vec2f;
<VertexOutput> barycentric: vec2f;
#endif
<VertexOutput> @interpolate(flat) triangle: u32;
<VertexOutput> @interpolate(flat) instance: u32;

@vertex
fn vertex(input: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
	let id = visible_instances[instance_index];
	let deformed = deform_vertex(id, input.position, input.normal, input.uv);
	let world = instance_position(instances[id], deformed.position);
	var output: VertexOutput;
	output.position = frame.proj * frame.view * vec4f(world, 1.0);
	output.world = world;
	// Keep inverse-transformed normals unnormalized until after interpolation.
	output.normal = instance_normal_vector(instances[id], deformed.normal);
	output.uv = deformed.uv;
	output.barycentric = input.barycentric;
	output.triangle = input.triangle;
	output.instance = id;
	return output;
}

fn surface_grazing_fade(id: u32, world: vec3f) -> f32 {
	// Geometric derivatives reflect the projected surface, independent of smooth
	// shading normals and instance scale. Evaluate them before every discard.
	let geometric_normal = cross(dpdxFine(world), dpdyFine(world));
#if PARAMS.get("material_library") is not none
	let grazing = library_grazing(id);
#else
	let grazing = frame.grazing.xyz;
#endif
	if grazing.z == 0.0 { return 1.0; }
	let orthographic_to_eye = vec3f(frame.view[0].z, frame.view[1].z, frame.view[2].z);
	let to_eye = select(frame.camera_position.xyz-world, orthographic_to_eye, frame.proj[3].w == 1.0);
	let denominator = length(geometric_normal)*length(to_eye);
	if denominator <= 1e-20 { return 1.0; }
	return grazing_fade_factor(dot(geometric_normal, to_eye)/denominator, grazing.x, grazing.y);
}

fn raster_opacity_from_alpha(id: u32, material: Material, uv: vec2f, lod: f32, world: vec3f, alpha: f32) -> vec3f {
	let fade = surface_grazing_fade(id, world);
	var result = alpha*material.opacity_rgb.rgb;
#if PARAMS.get("material_library") is not none
	result = library_opacity(id, material, result, uv, world, lod);
#endif
	return clamp(result, vec3f(0.0), vec3f(1.0))*fade;
}

fn raster_opacity(id: u32, material: Material, uv: vec2f, lod: f32, world: vec3f) -> vec3f {
	return raster_opacity_from_alpha(id, material, uv, lod, world, base_color(material, uv, lod).a);
}

fn opaque_visible(input: VertexOutput, material: Material, lod: f32, sample: u32) -> bool {
	let fade = surface_grazing_fade(input.instance, input.world);
	if material.surface.z != 1.0 { return true; }
	if base_color(material, input.uv, lod).a < material.surface.w { return false; }
	let pixel = vec2u(input.position.xy);
	let seed = hash_u32(pixel.x ^ hash_u32(pixel.y) ^ hash_u32(sample) ^ hash_u32(metadata[input.instance].z));
	return f32(seed >> 8u)*(1.0/16777216.0) < fade;
}

@fragment
fn opaque_prepass(input: VertexOutput, @builtin(front_facing) front: bool, @builtin(sample_index) sample: u32) {
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	let visible = opaque_visible(input, material, lod, sample);
	if !front && material.flags.y == 0u { discard; }
	if !visible { discard; }
}

@fragment
fn opaque(input: VertexOutput, @builtin(front_facing) front: bool, @builtin(sample_index) sample: u32) -> @location(0) vec4f {
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	let visible = opaque_visible(input, material, lod, sample);
	if !front && material.flags.y == 0u { discard; }
	if !visible { discard; }
	let signed_normal = select(-input.normal, input.normal, front);
	let normal = signed_normal / max(length(signed_normal), 1e-8);
	return vec4f(shade_material(input.instance, material, input.uv, input.world, normal, lod), 1.0);
}

@fragment
fn transmittance(input: VertexOutput, @builtin(front_facing) front: bool, @builtin(sample_index) sample: u32)
#if PARAMS.get("banded_proxy", false) and PARAMS.get("banded_statistics", true)
 -> BandTransmission {
#else
 -> @location(0) vec4f {
#endif
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
#if PARAMS.get("banded_proxy", false) and PARAMS.get("banded_statistics", true)
	let base = base_color(material, input.uv, lod);
	let alpha = raster_opacity_from_alpha(input.instance, material, input.uv, lod, input.world, base.a);
#else
	let alpha = raster_opacity(input.instance, material, input.uv, lod, input.world);
#endif
	if material.surface.z == 1.0 { discard; }
	if is_exact_nearest(input, sample) { discard; }
	if !front && material.flags.y == 0u { discard; }
	let p = max(alpha.x, max(alpha.y, alpha.z));
	let transmission = vec4f(vec3f(1.0)-alpha, 1.0-p);
	// Side effects are not protected by WebGPU early depth tests.
	if input.position.z >= raster_opaque_depth(vec2i(input.position.xy), sample) { discard; }
#if PARAMS.get("banded_statistics", true)
	let quantized_weight = band_weight(p, input, sample, front);
	atomicAdd(&band_counts[band_index(vec2u(input.position.xy), sample)].total, quantized_weight);
#endif
#if PARAMS.get("banded_proxy", false) and PARAMS.get("banded_statistics", true)
	let pq = f32(quantized_weight)/65536.0;
	return BandTransmission(transmission, vec4f(pq*base.rgb, pq));
#else
	return transmission;
#endif
}

fn raster_lod(uv: vec2f) -> f32 {
	// Derivatives are evaluated before any divergent branch or discard. Match
	// the half-float LOD stored in the winner payload exactly across all passes.
	let lod = clamp(texture_lod_from_gradients(dpdxFine(uv), dpdyFine(uv),
		vec2f(textureDimensions(base_textures))), 0.0, f32(textureNumLevels(base_textures)-1u));
	return unpack2x16float(pack2x16float(vec2f(lod, 0.0))).x;
}

fn visibility_payload(input: VertexOutput, lod: f32, front: bool) -> vec4u {
	return vec4u(input.instance+1u, input.triangle, pack2x16unorm(clamp(input.barycentric, vec2f(0.0), vec2f(1.0))), pack2x16float(vec2f(lod, select(-1.0, 1.0, front))));
}

fn band_weight(p: f32, input: VertexOutput, sample: u32, front: bool) -> u32 {
	let pixel = vec2u(input.position.xy);
	var seed = hash_u32(metadata[input.instance].z ^ 0x9e3779b9u);
	seed = hash_u32(seed ^ hash_u32(input.triangle ^ 0x85ebca6bu));
	seed = hash_u32(seed ^ hash_u32(pixel.x) ^ hash_u32(pixel.y ^ 0xc2b2ae35u));
	seed = hash_u32(seed ^ hash_u32(sample ^ select(0xa511e9b3u, 0u, front)));
	let scaled = p * 65536.0;
	let random = f32(seed >> 8u) * (1.0 / 16777216.0);
	return u32(floor(scaled)) + select(0u, 1u, random < fract(scaled));
}

@fragment
fn tail_statistics(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4f {
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	let base = base_color(material, input.uv, lod);
	let alpha = raster_opacity_from_alpha(input.instance, material, input.uv, lod, input.world, base.a);
	if material.surface.z == 1.0 { discard; }
	// Depth uses native guide sample zero; nearest exclusion uses their ID union.
	// Transmission and composition retain their actual per-hardware-sample IDs.
	if is_exact_nearest(input, 0u) || (!front && material.flags.y == 0u) { discard; }
	if input.position.z >= raster_opaque_depth(vec2i(input.position.xy), 0u) { discard; }
	let pq = band_weight(max(alpha.x, max(alpha.y, alpha.z)), input, 0u, front);
	atomicAdd(&band_counts[band_index(vec2u(input.position.xy), 0u)].total, pq);
#if PARAMS.get("banded_proxy", false)
	return vec4f(f32(pq)/65536.0*base.rgb, f32(pq)/65536.0);
#else
	return vec4f(0.0, 0.0, 0.0, f32(pq)/65536.0);
#endif
}

@fragment
fn select_bands(input: VertexOutput, @builtin(front_facing) front: bool, @builtin(sample_index) sample: u32) {
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	let alpha = raster_opacity(input.instance, material, input.uv, lod, input.world);
	if material.surface.z == 1.0 { discard; }
#if PARAMS.get("banded_subset", false)
	// Alpha/material derivatives above require uniform quad execution.
	if any(vec2u(input.position.xy) != band_selected_pixel(vec2u(input.position.xy)/2u)) { discard; }
#endif
	if is_exact_nearest(input, sample) || (!front && material.flags.y == 0u) { discard; }
	let pixel = vec2i(input.position.xy);
	if input.position.z >= raster_opaque_depth(pixel, sample) { discard; }
	let pq = band_weight(max(alpha.x, max(alpha.y, alpha.z)), input, sample, front);
	if pq == 0u { discard; }
	let index = band_index(vec2u(pixel), sample);
	let start = atomicAdd(&band_counts[index].running, pq);
	let end = start+pq;
	let total = atomicLoad(&band_counts[index].total);
	// One shared screen-space field, never a per-fragment hash. Native sample
	// positions address the finer lattice directly, preserving all four parities.
	let size = textureDimensions(blue_noise);
	let noise_pixels = frame.clip.z*{{2.0 if PARAMS.get("banded_mixed", false) else 1.0}};
	let cell = vec2u(floor(input.position.xy/max(noise_pixels, 1e-6)));
	let u = textureLoad(blue_noise, vec2i(cell%size), 0).r;
	let m = f32(max(65536u, total));
	// Clamp against conversion rounding at large totals. Complement is last.
	let a = min(u32(u*m), max(65536u, total)-1u);
	let b = min(u32((1.0-u)*m), max(65536u, total)-1u);
	let payload = visibility_payload(input, lod, front);
#if PARAMS.get("banded_subset", false)
	let destination = pixel/2;
#else
	let destination = pixel;
#endif
	if a >= start && a < end { textureStore(band_payloads, destination, i32(sample*2u), payload); }
	if b >= start && b < end { textureStore(band_payloads, destination, i32(sample*2u+1u), payload); }
}

<NearestOutput> @location(0) low: vec4u;
<NearestOutput> @location(1) alpha: vec4f;
#if PARAMS.get("sample_count", 1) > 1
<NearestOutput> @location(2) high: vec4u;
#endif
@fragment
fn nearest_layer(input: VertexOutput, @builtin(front_facing) front: bool, @builtin(sample_index) sample: u32) -> NearestOutput {
	let lod = raster_lod(input.uv);
	let material = get_material(input.instance);
	let alpha = raster_opacity(input.instance, material, input.uv, lod, input.world);
	let p = max(alpha.x, max(alpha.y, alpha.z));
	if material.surface.z == 1.0 { discard; }
	if p <= 0.0 || (!front && material.flags.y == 0u) { discard; }
	if input.position.z >= raster_opaque_depth(vec2i(input.position.xy), sample) { discard; }
	let payload = visibility_payload(input, lod, front);
#if PARAMS.get("sample_count", 1) > 1
	return NearestOutput(vec4u(payload.x & 65535u, payload.x >> 16u, payload.y & 65535u, payload.y >> 16u), vec4f(alpha, p), vec4u(payload.z & 65535u, payload.z >> 16u, payload.w & 65535u, payload.w >> 16u));
#else
	return NearestOutput(payload, vec4f(alpha, p));
#endif
}
