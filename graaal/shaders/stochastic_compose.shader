#include "frame.shaderlib"
sample_texture<1, 1> main_color: f32;
sample_texture<1, 2> transmittance: f32;
sample_texture<1, 3, layered> winner_color: f32;
sample_texture<1, 4> opaque_depth: depth;
<ComposeSettings> options: vec4u;
uniform<1, 5> compose_settings: ComposeSettings;
sample_texture<1, 11, layered> reconstructed: f32;
sample_texture<1, 13> nearest_alpha: f32;
sample_texture<1, 14, layered> nearest_color: f32;

#if not PARAMS.get("srgb_surface", false)
#include "utils_color.shaderlib"
#endif
#include "ambient_occlusion.shaderlib"

@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	return vec4f(uv*2.0-1.0, 0.0, 1.0);
}

fn present_linear(color: vec3f) -> vec4f {
#if PARAMS.get("srgb_surface", false)
	return vec4f(color, 1.0);
#else
	return vec4f(linear_to_srgb(color), 1.0);
#endif
}

fn compose_sample(pixel: vec2i, sample: u32) -> vec3f {
	let view = compose_settings.options.x;
	let t = sample_texture_load(transmittance, pixel, sample);
	let alpha = sample_texture_load(nearest_alpha, pixel, sample);
	let nearest = sample_texture_load(nearest_color, pixel, sample, layered=true).rgb;
	if view == 6u { return vec3f(t.a); }
	if view == 5u { return vec3f(select(0.0, 1.0, alpha.a > 0.0)); }
	if view == 4u { return alpha.rgb*nearest; }
	// Display scalar transmission without exposure or reconstruction.
	if view == 1u { return vec3f(t.a*(1.0-alpha.a)); }
#if PARAMS.get("banded_mixed", false)
	let tail_sample = 0u;
#else
	let tail_sample = sample;
#endif
	if view == 2u {
		let winner = sample_texture_load(winner_color, pixel, tail_sample, layered=true).rgb;
#if PARAMS.get("banded_proxy", false)
		return clamp(vec3f(0.5)+0.5*winner, vec3f(0.0), vec3f(1.0));
#else
		return max(winner, vec3f(0.0));
#endif
	}
	var main = sample_texture_load(main_color, pixel, sample).rgb;
	if compose_settings.options.y != 0u && view == 0u {
		let depth = textureLoad(opaque_depth, pixel, i32(sample));
		main *= ambient_occlusion(pixel, sample, depth, frame.proj, frame.clip.xy);
	}
	var transparent = sample_texture_load(reconstructed, pixel, tail_sample, layered=true).rgb;
#if PARAMS.get("banded_mixed", false)
	transparent *= vec3f(1.0)-t.rgb;
#endif
	// Compose in linear HDR, preserving the sharp exact nearest contribution.
	let tail = select(main*t.rgb+transparent, transparent, view == 3u);
	return max(alpha.rgb*nearest+(vec3f(1.0)-alpha.rgb)*tail, vec3f(0.0));
}

@fragment
fn compose(@builtin(position) position: vec4f) -> @location(0) vec4f {
	var color = vec3f(0.0);
	for (var sample = 0u; sample < {{PARAMS.get("sample_count", 1)}}u; sample += 1u) {
		color += compose_sample(vec2i(position.xy), sample);
	}
	color /= {{PARAMS.get("sample_count", 1)}}.0;
	let view = compose_settings.options.x;
	var tone_map = view != 1u && view != 5u && view != 6u;
#if PARAMS.get("banded_proxy", false)
	tone_map = tone_map && view != 2u;
#endif
	if tone_map { color /= vec3f(1.0)+color; }
	return present_linear(color);
}
