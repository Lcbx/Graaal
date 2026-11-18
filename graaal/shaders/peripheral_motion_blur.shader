#if PARAMS.srgb_surface
#include "utils_color.shaderlib"
#endif
texture<0, 0> scene_color: texture_2d<f32>;
#if not PARAMS.manual_filter
texture<0, 1> scene_sampler: sampler;
#endif
<Motion> linear: vec4f;
<Motion> angular: vec4f;
<Motion> params: vec4f;
<Motion> radii: vec4f;
uniform<0, 2> motion: Motion;

fn sample_color(uv: vec2f) -> vec4f {
#if PARAMS.manual_filter
	let size = vec2i(textureDimensions(scene_color));
	let p = uv * vec2f(size) - vec2f(0.5);
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
@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	return vec4f(uv*2.0-1.0, 0.0, 1.0);
}
@fragment
fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
	let size = vec2f(textureDimensions(scene_color));
	let uv = position.xy / size;
	let projection_scale = 2.0 * motion.params.x * vec2f(size.x/size.y, -1.0);
	let ray = vec3f((uv-0.5)*projection_scale, -1.0);
	let delta = -motion.linear.xyz / motion.params.z - cross(motion.angular.xyz, ray);
	let flow = (delta.xy + ray.xy*delta.z) / projection_scale;
	// Vignette radius: 0 at screen center and 1 at corners.
	let radius = length((uv-vec2f(0.5))*1.41421356);
	let peripheral_strength = pow(smoothstep(motion.radii.x, motion.radii.y, radius), motion.radii.w);
	let raw_offset = flow * motion.params.y * peripheral_strength;
	let magnitude = length(raw_offset);
	let pixel_length = length(raw_offset * size);
	// Apply the screen-space and 127-pixel quality limits together.
	let scale = min(1.0, min(motion.params.w/max(magnitude, 0.00001), 127.0/max(pixel_length, 0.00001)));
	let offset = raw_offset * scale;
	let distance = pixel_length * scale;
	let original = sample_color(uv);
	var color = original.rgb;
	if distance > 0.00001 {
		var previous = color;
		var integral = vec3f(0.0);
		let step = offset / distance;
		let count = min(127, i32(ceil(distance)));
		for (var i = 1; i <= count; i++) {
			let begin = f32(i-1);
			let end = min(f32(i), distance);
			let next = sample_color(uv-step*end).rgb;
			integral += (previous+next)*0.5*(end-begin);
			previous = next;
		}
		color = integral / distance;
	}
	if motion.radii.z > 0.0 && magnitude > 0.0 && motion.params.w > 0.0 {
		// Screen-space right is red; hue rotates clockwise with the motion vector.
		let hue = atan2(offset.y, offset.x) / 6.28318530718;
		let hue_rgb = clamp(abs(fract(hue + vec3f(0.0, 2.0/3.0, 1.0/3.0))*6.0-vec3f(3.0))-vec3f(1.0), vec3f(0.0), vec3f(1.0));
		let debug_opacity = 0.95 * sqrt(clamp(magnitude*scale/motion.params.w, 0.0, 1.0));
		color = mix(color, hue_rgb, debug_opacity);
	}
#if PARAMS.srgb_surface
	color = srgb_to_linear(color);
#endif
	return vec4f(color, original.a);
}


