#include "utils_color.shaderlib"
texture<0, 0> scene_color: texture_2d<f32>;
#if not PARAMS.manual_filter
texture<0, 1> scene_sampler: sampler;
#endif

fn sample0(uv: vec2f) -> vec4f {
#if PARAMS.manual_filter
	let size = vec2i(textureDimensions(scene_color));
	let p = uv*vec2f(size)-0.5;
	let i = vec2i(floor(p));
	let f = fract(p);
	let a = textureLoad(scene_color, clamp(i, vec2i(0), size-1), 0);
	let b = textureLoad(scene_color, clamp(i+vec2i(1, 0), vec2i(0), size-1), 0);
	let c = textureLoad(scene_color, clamp(i+vec2i(0, 1), vec2i(0), size-1), 0);
	let d = textureLoad(scene_color, clamp(i+vec2i(1, 1), vec2i(0), size-1), 0);
	return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
#else
	return textureSampleLevel(scene_color, scene_sampler, uv, 0.0);
#endif
}

#block effects
#endblock

#for function in PARAMS.functions
fn sample{{ loop.index }}(uv: vec2f) -> vec4f { return {{ function }}(uv); }
#endfor

@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	return vec4f(uv*2.0-1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
	var color = sample{{ PARAMS.functions | length }}(position.xy/vec2f(textureDimensions(scene_color)));
#if PARAMS.srgb_surface
	color = vec4f(srgb_to_linear(color.rgb), color.a);
#endif
	return color;
}
