// LXAA and FAAA adapted from shaders by Germán Rebord. See antialiasing_faaa_lxaa_LICENSE.txt
// DDAA2 adapted from shader by almarklein. See antialiasing_ddaa2_LICENSE.txt

#if PARAMS.srgb_surface
#include "utils_color.shaderlib"
#endif

texture<0, 0> scene_color: texture_2d<f32>;
#if not PARAMS.manual_filter
texture<0, 1> scene_sampler: sampler;
#endif

fn texel(pixel: vec2i) -> vec4f {
	let dimensions = vec2i(textureDimensions(scene_color));
	return textureLoad(scene_color, clamp(pixel, vec2i(0), dimensions-vec2i(1)), 0);
}

fn sample_color(uv: vec2f) -> vec4f {
#if PARAMS.manual_filter
	let position = uv*vec2f(textureDimensions(scene_color))-vec2f(0.5);
	let pixel = vec2i(floor(position));
	let fraction = fract(position);
	return mix(mix(texel(pixel), texel(pixel+vec2i(1, 0)), fraction.x), mix(texel(pixel+vec2i(0, 1)), texel(pixel+vec2i(1, 1)), fraction.x), fraction.y);
#else
	return textureSampleLevel(scene_color, scene_sampler, uv, 0.0);
#endif
}

fn sample_color_offset(uv: vec2f, offset: vec2i) -> vec4f {
	return sample_color(uv+vec2f(offset)/vec2f(textureDimensions(scene_color)));
}

fn output_color(color: vec4f) -> vec4f {
#if PARAMS.srgb_surface
	return vec4f(srgb_to_linear(color.rgb), color.a);
#else
	return color;
#endif
}

fn rgb2luma(rgb: vec3f) -> f32 {
	var luma = dot(rgb, vec3f(0.299, 0.587, 0.114));  // real luma
	return sqrt(luma);  // trick for perceived lightness, used in Bevy
	// return luma;
}


#if PARAMS.hybrid_luma_chroma

fn extract_key(rgb: vec3f) -> vec3f {
	const chroma_weight : f32 = {{ PARAMS.chroma_weight }};
	let luma = rgb2luma(rgb);
	let co = rgb.r - rgb.b;
	let cg = rgb.g - 0.5 * (rgb.r + rgb.b);
	return vec3f(luma, co * chroma_weight, cg * chroma_weight);
}
// Project keys onto the strongest local color difference so edge direction and
// polarity stay consistent through the scalar edge-search algorithms.
fn finalize_key_axis(direction: vec3f) -> vec3f {
	var axis = direction;
	if dot(axis, axis) < 1e-20 { return vec3f(1.0, 0.0, 0.0); }
	let magnitude = abs(axis);
	let dominant = select(select(axis.z, axis.y, magnitude.y >= magnitude.z), axis.x, magnitude.x >= max(magnitude.y, magnitude.z));
	axis *= select(-1.0, 1.0, dominant >= 0.0);
	return normalize(axis);
}
fn key_axis6(a: vec3f, b: vec3f, c: vec3f, d: vec3f, e: vec3f, f: vec3f) -> vec3f {
	let first = a-b;
	let second = c-d;
	let third = e-f;
	var axis = select(second, first, dot(first, first) >= dot(second, second));
	axis = select(third, axis, dot(axis, axis) >= dot(third, third));
	return finalize_key_axis(axis);
}
fn key_axis4(a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
	let first = a-b;
	let second = c-d;
	var axis = select(second, first, dot(first, first) >= dot(second, second));
	axis = select(vec3f(0.0), axis, dot(axis, axis) >= 0.0);
	return finalize_key_axis(axis);
}
fn project_key(key: vec3f, axis: vec3f) -> f32 { return dot(key, axis); }
fn key_value(rgb: vec3f, axis: vec3f) -> f32 { return project_key(extract_key(rgb), axis); }
fn sample_key(uv: vec2f, axis: vec3f) -> f32 { return key_value(sample_color(uv).rgb, axis); }
fn edge_threshold_scale(sw: vec3f, se: vec3f, ne: vec3f, nw: vec3f) -> f32 {
	return max(max(sw.r, se.r), max(ne.r, nw.r))*0.5+0.5;
}

#else
fn extract_key(rgb: vec3f) -> f32 {	return rgb2luma(rgb); }
fn key_axis6(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> f32 { return 1.0; }
fn key_axis4(a: f32, b: f32, c: f32, d: f32) -> f32 { return 1.0; }
fn project_key(key: f32, axis: f32) -> f32 { return key; }
fn key_value(rgb: vec3f, axis: f32) -> f32 { return project_key(extract_key(rgb), axis); }
fn sample_key(uv: vec2f, axis: f32) -> f32 { return key_value(sample_color(uv).rgb, axis); }
fn edge_threshold_scale(sw: f32, se: f32, ne: f32, nw: f32) -> f32 {
	return max(max(sw, se), max(ne, nw))*0.5+0.5;
}
#endif


@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
	let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	return vec4f(uv*2.0-1.0, 0.0, 1.0);
}

#if PARAMS.mode == "lxaa"
@fragment
fn lxaa(@builtin(position) position: vec4f) -> @location(0) vec4f {
	let pixel = vec2i(position.xy);
	let uv = position.xy/vec2f(textureDimensions(scene_color));
	let step = 1.0/vec2f(textureDimensions(scene_color));
	let sw_color = texel(pixel+vec2i(-1, 1)).rgb;
	let se_color = texel(pixel+vec2i(1, 1)).rgb;
	let ne_color = texel(pixel+vec2i(1, -1)).rgb;
	let nw_color = texel(pixel+vec2i(-1, -1)).rgb;
	let sw_key = extract_key(sw_color);
	let ne_key = extract_key(ne_color);
	let se_key = extract_key(se_color);
	let nw_key = extract_key(nw_color);
	let axis = key_axis4(sw_key, ne_key, se_key, nw_key);
	// Estimate the edge direction from diagonal luminance differences.
	var a = vec4f(project_key(sw_key, axis), project_key(se_key, axis), project_key(ne_key, axis), project_key(nw_key, axis));
	let gradient = vec2f(a.x-a.z, a.y-a.w);
	let direction = vec2f(gradient.x+gradient.y, gradient.x-gradient.y);
	let magnitude = abs(direction);
	let maximum = max(max(a.x, a.y), max(a.z, a.w));
	let minimum_luma = min(min(a.x, a.y), min(a.z, a.w));
	let threshold = {{ PARAMS.lxaa_edge_threshold }}* edge_threshold_scale(sw_key, se_key, ne_key, nw_key);
	let center = texel(pixel);
#if PARAMS.show_edges
	if abs(magnitude.x-magnitude.y) < threshold { return output_color(vec4f(vec3f(rgb2luma(center.rgb)*0.9), center.a)); }
	return output_color(vec4f(0.0, 1.0, 0.0, center.a));
#endif
	if abs(magnitude.x-magnitude.y) < threshold { return output_color(center); }
	let minimum = min(magnitude.x, magnitude.y);
	let offsets = min(vec2f(1.0), 0.0625*magnitude/max(minimum, 1e-20));
	var multiplier = 2.0*step*sign(direction);
#if PARAMS.lxaa_line_check
	if max(offsets.x, offsets.y) == 1.0 {
		let horizontal = offsets.x == 1.0;
		let negative = select(multiplier.y < 0.0, multiplier.x < 0.0, horizontal);
		let southwest = horizontal == negative;
		var shifted = uv;
		if horizontal { shifted.x += 2.0*multiplier.x; } else { shifted.y += 2.0*multiplier.y; }
		if southwest { a.x = sample_key(shifted+vec2f(-step.x, step.y), axis); }
		if !negative { a.y = sample_key(shifted+step, axis); }
		if !southwest { a.z = sample_key(shifted+vec2f(step.x, -step.y), axis); }
		if negative { a.w = sample_key(shifted-step, axis); }
		let g = vec2f(a.x-a.z, a.y-a.w);
		var d = vec2f(g.x+g.y, g.x-g.y);
		if !horizontal { d = d.yx; }
		if abs(d.x) > 2.0*abs(d.y) { multiplier *= 2.0; }
	}
#endif
	// Blend along the detected edge while preserving the local luma range.
	let offset = offsets*multiplier;
	let result = (sample_color(uv-offset)+sample_color(uv+offset))*0.3+center*0.4;
	let result_luma = key_value(result.rgb, axis);
	if result_luma < minimum_luma || result_luma > maximum { return output_color(center); }
	return output_color(vec4f(result.rgb, center.a));
}

#endif

#if PARAMS.mode == "faaa"
#set faaa_quality = PARAMS.faaa_quality
@fragment
fn faaa(@builtin(position) position: vec4f) -> @location(0) vec4f {
	let pixel = vec2i(position.xy);
	let uv = position.xy/vec2f(textureDimensions(scene_color));
	let step = 1.0/vec2f(textureDimensions(scene_color));
	let sw_color = texel(pixel+vec2i(-1, 1)).rgb;
	let se_color = texel(pixel+vec2i(1, 1)).rgb;
	let ne_color = texel(pixel+vec2i(1, -1)).rgb;
	let nw_color = texel(pixel+vec2i(-1, -1)).rgb;
	let sw_key = extract_key(sw_color);
	let ne_key = extract_key(ne_color);
	let se_key = extract_key(se_color);
	let nw_key = extract_key(nw_color);
	let axis = key_axis4(sw_key, ne_key, se_key, nw_key);
	// Detect edge orientation and reject low-contrast pixels.
	let sw = project_key(sw_key, axis);
	let se = project_key(se_key, axis);
	let ne = project_key(ne_key, axis);
	let nw = project_key(nw_key, axis);
	let gradient = vec2f(sw-ne, se-nw);
	let direction = abs(vec2f(gradient.x+gradient.y, gradient.x-gradient.y));
	let threshold = (10.0-{{ PARAMS.faaa_edge_threshold }})*0.0625*edge_threshold_scale(sw_key, se_key, ne_key, nw_key);
	let center = texel(pixel);
#if PARAMS.show_edges
	if abs(direction.x-direction.y) < threshold { return output_color(vec4f(vec3f(rgb2luma(center.rgb)*0.9), center.a)); }
	return output_color(vec4f(0.0, 1.0, 0.0, center.a));
#endif
	if abs(direction.x-direction.y) < threshold { return output_color(center); }
	let horizontal = direction.x > direction.y;
	let m = key_value(center.rgb, axis);
	let n = key_value(texel(pixel+select(vec2i(-1, 0), vec2i(0, -1), horizontal)).rgb, axis);
	let s = key_value(texel(pixel+select(vec2i(1, 0), vec2i(0, 1), horizontal)).rgb, axis);
	let pair_n = abs(n-m) > abs(s-m);
	let neighbor = select(s, n, pair_n);
	let gradient_scaled = abs(neighbor-m)*0.25;
	let middle = (neighbor+m)*0.5;
	var length_sign = select(step.x, step.y, horizontal);
	if pair_n { length_sign = -length_sign; }
	var base = uv;
	if horizontal { base.y += length_sign*0.5; } else { base.x += length_sign*0.5; }
	let along = select(vec2f(0.0, step.y), vec2f(step.x, 0.0), horizontal);
	var pos_n = base-along;
	var pos_p = base+along;
	var end_n = sample_key(pos_n, axis)-middle;
	var end_p = sample_key(pos_p, axis)-middle;
	var done_n = abs(end_n) > gradient_scaled;
	var done_p = abs(end_p) > gradient_scaled;
	// Follow both ends of the edge until their contrast changes.
#for i in range(1, faaa_quality+3)
	if !done_n || !done_p {
		let distance = {{ (1.0 if faaa_quality == 9 and i <= 4 else 1.5 if faaa_quality == 9 and i == 5 else 8.0 if faaa_quality <= 8 and i == faaa_quality+2 else (3.0 if faaa_quality == 4 else 4.0) if 4 <= faaa_quality <= 8 and i == faaa_quality+1 else 1.5 if i == 1 else 4.0 if i == 10 else 8.0 if i == 11 else 2.0) }};
		if !done_n { pos_n -= along*distance; end_n = sample_key(pos_n, axis)-middle; done_n = abs(end_n) > gradient_scaled; }
		if !done_p { pos_p += along*distance; end_p = sample_key(pos_p, axis)-middle; done_p = abs(end_p) > gradient_scaled; }
	}
#endfor
	// Offset toward the nearer valid endpoint to smooth the edge.
	let dst_n = select(uv.y-pos_n.y, uv.x-pos_n.x, horizontal);
	let dst_p = select(pos_p.y-uv.y, pos_p.x-uv.x, horizontal);
	let near_n = dst_n < dst_p;
	let good_span = (select(end_p, end_n, near_n) < 0.0) != (m-middle < 0.0);
	let offset = 0.5-min(dst_n, dst_p)/max(dst_n+dst_p, 1e-20);
	if offset < 0.0 || !good_span { return output_color(center); }
	var result_uv = uv;
	if horizontal { result_uv.y += offset*length_sign; } else { result_uv.x += offset*length_sign; }
	return output_color(vec4f(sample_color(result_uv).rgb, center.a));
}

#endif

#if PARAMS.mode == "ddaa2"
#set ddaa2_edge_steps = PARAMS.ddaa2_edge_steps
// DDAA2 v2.4 by Almar Klein; see ddaa2_LICENSE.txt.
const MAX_EDGE_ITERS = 5;  // length(EDGE_STEP_LIST)
const MAX_EDGE_SAMPLES = 15; // sum(EDGE_STEP_LIST), how far the algirithm can look along an edge

// The strength of the diffusion. A value of 3 seems to work well.

// Trims the algorithm from processing darks.
// low: 0.0833, medium: 0.0625, high: 0.0312, ultra: 0.0156, extreme: 0.0078

// The minimum amount of local contrast required to apply algorithm.
// low: 0.250, medium: 0.166, high: 0.125, ultra: 0.063, extreme: 0.031

// ========== Constants and helper functions ==========

@fragment
fn ddaa2(@builtin(position) position: vec4f) -> @location(0) vec4<f32> {

	let texCoord = position.xy/vec2f(textureDimensions(scene_color));
	let pixel = vec2i(position.xy);
	// Sample the center pixel
	let centerSample = texel(pixel);
	let centerKey = extract_key(centerSample.rgb);
	let northKey = extract_key(texel(pixel+vec2i(0, 1)).rgb);
	let eastKey = extract_key(texel(pixel+vec2i(1, 0)).rgb);
	let southKey = extract_key(texel(pixel+vec2i(0, -1)).rgb);
	let westKey = extract_key(texel(pixel+vec2i(-1, 0)).rgb);
	let axis = key_axis6(centerKey, eastKey, centerKey, westKey, northKey, southKey);

	let resolution = vec2f(textureDimensions(scene_color));
	let pixelStep = 1.0 / resolution.xy;

	let lumaCenter = project_key(centerKey, axis);

	// Luma at the four direct neighbors of the current fragment.
	let lumaN = project_key(northKey, axis);
	let lumaE = project_key(eastKey, axis);
	let lumaS = project_key(southKey, axis);
	let lumaW = project_key(westKey, axis);

	// Compute the range
	let lumaMin = min(lumaCenter, min(min(lumaS, lumaN), min(lumaW, lumaE)));
	let lumaMax = max(lumaCenter, max(max(lumaS, lumaN), max(lumaW, lumaE)));
	let lumaRange = lumaMax - lumaMin;

#if PARAMS.show_edges
	if lumaRange < max({{ PARAMS.ddaa2_threshold_min }}, lumaMax * {{ PARAMS.ddaa2_threshold_max }}) { return output_color(vec4f(vec3f(rgb2luma(centerSample.rgb)*0.9), centerSample.a)); }
	return output_color(vec4f(0.0, 1.0, 0.0, centerSample.a));
#endif
	// If the luma variation is lower that a threshold (or if we are in a really dark area), we are not on an edge, don't perform any AA.
	if (lumaRange < max({{ PARAMS.ddaa2_threshold_min }}, lumaMax * {{ PARAMS.ddaa2_threshold_max }})) {
		return output_color(centerSample);
	}

	// Query the 4 remaining corners lumas.
	let lumaNW = key_value(texel(pixel+vec2i(-1, 1)).rgb, axis);
	let lumaNE = key_value(texel(pixel+vec2i(1, 1)).rgb, axis);
	let lumaSW = key_value(texel(pixel+vec2i(-1, -1)).rgb, axis);
	let lumaSE = key_value(texel(pixel+vec2i(1, -1)).rgb, axis);

	// Combine the four edges lumas (using intermediary variables for future computations with the same values).
	let lumaSUp = lumaS + lumaN;
	let lumaWRight = lumaW + lumaE;
	let lumaWCorners = lumaSW + lumaNW;

	// Same for corners
	let lumaSCorners = lumaSW + lumaSE;
	let lumaECorners = lumaSE + lumaNE;
	let lumaNCorners = lumaNE + lumaNW;

	// Calculate the image gradient using the Schar kernel, which is (relatively) rotationally invariant.
	const k1 = 162.0 / 256.0; // 61
	const k2 = 47.0 / 256.0; // 17
	let imDx = (lumaW * k1 + lumaSW * k2 + lumaNW * k2) - (lumaE * k1 + lumaSE * k2 + lumaNE * k2);
	let imDy = (lumaS * k1 + lumaSW * k2 + lumaSE * k2) - (lumaN * k1 + lumaNW * k2 + lumaNE * k2);

	// Get the edge vector (orthogonal to the gradient), and calculate strength and direction.
	let edgeVector = vec2f(-imDy, imDx);
	var diffuseStrength = sqrt(length(edgeVector)) * {{ PARAMS.ddaa2_strength }};
	var diffuseDirection = normalize(edgeVector);
	if diffuseStrength < 1e-6 {
		diffuseDirection = vec2f(0.0, 0.0);
		diffuseStrength = 0.0;
	}
	diffuseStrength = min(1.0, diffuseStrength);

	// Is the local edge horizontal or vertical ?
	let edgeHorizontal = abs(-2.0 * lumaW + lumaWCorners) + abs(-2.0 * lumaCenter + lumaSUp) * 2.0 + abs(-2.0 * lumaE + lumaECorners);
	let edgeVertical = abs(-2.0 * lumaN + lumaNCorners) + abs(-2.0 * lumaCenter + lumaWRight) * 2.0 + abs(-2.0 * lumaS + lumaSCorners);
	let isHorizontal = (edgeHorizontal >= edgeVertical);
	//let isHorizontal = (abs(diffuseDirection.x) >= abs(diffuseDirection.y)); -> different, resulting in wrong ridge detection

	// Calculate gradient on both sides of the current pixel
	var luma1 = select(lumaW, lumaS, isHorizontal);
	var luma2 = select(lumaE, lumaN, isHorizontal);    // Compute gradients in this direction.
	let gradient1 = luma1 - lumaCenter;
	let gradient2 = luma2 - lumaCenter;

	// Maintain ridges and thin lines. This is inspired by AXAA's 2nd enhancement, except we also apply it to negative edges and do a smooth transition instead of a threshold.
	// Note that we can diminish quite hard, because the neighbouring pixels likely get diffused in the direction of the edge (this is one of our advantages over fxaa).
	if sign(gradient1) == sign(gradient2) {
		// This is a ridge or a valley, e.g. a thin line. We want to presereve these.
		let ridgeness = min(abs(gradient1), abs(gradient2));
		let diminish_factor = 1.0 - (min(1.0, 10 * ridgeness));
		diffuseStrength *= diminish_factor;
	}

	// For long edges, the diffusion has to be huge to remove the jaggies. What algorithms like FXAA do instead, is detect
	// the length of the edge segment (successive horizontal/vertical pixels), and the use that to calculate the subpixel
	// texture coordinate offset, perpendicular to the edge. So technically this is diffusion perpendicular to the edge,
	// but in a controlled manner to smooth the step/jaggy.
	var subpixelEdgeOffset = vec2f(0.0);
#if ddaa2_edge_steps > 0
	{

		// Choose the step size (one pixel) accordingly.
		var stepLength = select(pixelStep.x, pixelStep.y, isHorizontal);

		// Gradient in the corresponding direction, normalized.
		let gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

		// Average luma in the current direction.
		var lumaLocalAverage = 0.0;
		let gradient2IsHigher = abs(gradient2) > abs(gradient1);
		if  gradient2IsHigher {
			lumaLocalAverage = 0.5 * (luma2 + lumaCenter);
		} else {
			stepLength = -stepLength;  // switch the direction
			lumaLocalAverage = 0.5 * (luma1 + lumaCenter);
		}

		// Shift UV in the correct direction by half a pixel (orthogonal to the edge)
		var currentUv = texCoord;
		if isHorizontal {
			currentUv.y += stepLength * 0.5;
		} else {
			currentUv.x += stepLength * 0.5;
		}

		// Initialize edge search params

		var distance1 = 999.0;
		var distance2 = 999.0;

		var lumaEnd1: f32;
		var lumaEnd2: f32;

		var lumaEnd_0: f32;
		var lumaEnd_1: f32;
		var lumaEnd_2: f32;
		var lumaEnd_3: f32;
		var lumaEnd_4: f32;
		var lumaEnd_5: f32;

		// Step 0 (3 steps)

		// Read the lumas at both current extremities of the exploration segment, and compute the delta wrt to the local average luma.
		// We take 3 steps in both directions, but we get one sample for free (for each direction), so we take 4 samples.
		if isHorizontal {
			lumaEnd_0 = 0.5 * (lumaW + select(lumaSW, lumaNW, gradient2IsHigher)) - lumaLocalAverage;
			lumaEnd_3 = 0.5 * (lumaE + select(lumaSE, lumaNE, gradient2IsHigher)) - lumaLocalAverage;
			lumaEnd_1 = key_value(sample_color_offset(currentUv, -vec2i(2, 0)).rgb, axis) - lumaLocalAverage;
			lumaEnd_2 = key_value(sample_color_offset(currentUv, -vec2i(3, 0)).rgb, axis) - lumaLocalAverage;
			lumaEnd_4 = key_value(sample_color_offset(currentUv, vec2i(2, 0)).rgb, axis) - lumaLocalAverage;
			lumaEnd_5 = key_value(sample_color_offset(currentUv, vec2i(3, 0)).rgb, axis) - lumaLocalAverage;

		} else {
			lumaEnd_0 = 0.5 * (lumaS + select(lumaSW, lumaSE, gradient2IsHigher)) - lumaLocalAverage;
			lumaEnd_3 = 0.5 * (lumaN + select(lumaNW, lumaNE, gradient2IsHigher)) - lumaLocalAverage;
			lumaEnd_1 = key_value(sample_color_offset(currentUv, -vec2i(0, 2)).rgb, axis) - lumaLocalAverage;
			lumaEnd_2 = key_value(sample_color_offset(currentUv, -vec2i(0, 3)).rgb, axis) - lumaLocalAverage;
			lumaEnd_4 = key_value(sample_color_offset(currentUv, vec2i(0, 2)).rgb, axis) - lumaLocalAverage;
			lumaEnd_5 = key_value(sample_color_offset(currentUv, vec2i(0, 3)).rgb, axis) - lumaLocalAverage;
		}

		// Search for left endpoint in the current 3 samples

		if (abs(lumaEnd_2) >= gradientScaled) { distance1 = 3.0; lumaEnd1 = lumaEnd_2; }
		if (abs(lumaEnd_1) >= gradientScaled) { distance1 = 2.0; lumaEnd1 = lumaEnd_1; }
		if (abs(lumaEnd_0) >= gradientScaled) { distance1 = 1.0; lumaEnd1 = lumaEnd_0; }

		// Same for the right endpoint

		if (abs(lumaEnd_5) >= gradientScaled) { distance2 = 3.0; lumaEnd2 = lumaEnd_5; }
		if (abs(lumaEnd_4) >= gradientScaled) { distance2 = 2.0; lumaEnd2 = lumaEnd_4; }
		if (abs(lumaEnd_3) >= gradientScaled) { distance2 = 1.0; lumaEnd2 = lumaEnd_3; }


		// Now search for endpoints in a series of rounds, using a templated (i.e. unrolled) loop.
		// This is much faster (in WGSL) than a normal loop, probably due to optimization related to the texture lookups.
		// We use offset coordinates to sample farther along the edge.

		let max_distance = {{ ddaa2_edge_steps * 3 }}.0;

#for i in range(1, ddaa2_edge_steps)
		// Step {{ i }} (3 steps, offset {{ i * 3 }})
#for endpoint in (1, 2)
#set direction = "-" if endpoint == 1 else "+"
#set offset_sign = "-" if endpoint == 1 else ""
		if (distance{{ endpoint }} > 900.0) {
			if isHorizontal {
				let currentUv{{ endpoint }} = currentUv {{ direction }} vec2f({{ i * 3 + 2 }}.0, 0.0) * pixelStep;
				lumaEnd_0 = key_value(sample_color_offset(currentUv{{ endpoint }}, {{ offset_sign }}vec2i(-1, 0)).rgb, axis) - lumaLocalAverage;
				lumaEnd_1 = key_value(sample_color_offset(currentUv{{ endpoint }}, {{ offset_sign }}vec2i(0, 0)).rgb, axis) - lumaLocalAverage;
				lumaEnd_2 = key_value(sample_color_offset(currentUv{{ endpoint }}, {{ offset_sign }}vec2i(1, 0)).rgb, axis) - lumaLocalAverage;
			} else {
				let currentUv{{ endpoint }} = currentUv {{ direction }} vec2f(0.0, {{ i * 3 + 2 }}.0) * pixelStep;
				lumaEnd_0 = key_value(sample_color_offset(currentUv{{ endpoint }}, {{ offset_sign }}vec2i(0, -1)).rgb, axis) - lumaLocalAverage;
				lumaEnd_1 = key_value(sample_color_offset(currentUv{{ endpoint }}, {{ offset_sign }}vec2i(0, 0)).rgb, axis) - lumaLocalAverage;
				lumaEnd_2 = key_value(sample_color_offset(currentUv{{ endpoint }}, {{ offset_sign }}vec2i(0, 1)).rgb, axis) - lumaLocalAverage;
			}
			lumaEnd{{ endpoint }} = lumaEnd_2;
			if (abs(lumaEnd_2) >= gradientScaled) { distance{{ endpoint }} = {{ i * 3 + 3 }}.0; lumaEnd{{ endpoint }} = lumaEnd_2; }
			if (abs(lumaEnd_1) >= gradientScaled) { distance{{ endpoint }} = {{ i * 3 + 2 }}.0; lumaEnd{{ endpoint }} = lumaEnd_1; }
			if (abs(lumaEnd_0) >= gradientScaled) { distance{{ endpoint }} = {{ i * 3 + 1 }}.0; lumaEnd{{ endpoint }} = lumaEnd_0; }
		}
#endfor
#endfor

		// Clip the distance (if we did not find the end, we assume it's one pixel further)
		distance1 = min(distance1, max_distance + 1.0);
		distance2 = min(distance2, max_distance + 1.0);

		// UV offset: read in the direction of the closest side of the edge.
		let pixelOffset = - min(distance1, distance2) / (distance1 + distance2) + 0.5;

		// If the luma at center is smaller than at its neighbor, the delta luma at each end should be positive (same variation).
		let isLumaCenterSmaller = lumaCenter < lumaLocalAverage;
		var correctVariation: bool;
		if (distance1 < distance2) {
			correctVariation = (lumaEnd1 < 0.0) != isLumaCenterSmaller;
		} else {
			correctVariation = (lumaEnd2 < 0.0) != isLumaCenterSmaller;
		}

		// Set subpixel texCoord offset
		if (!correctVariation) {
			subpixelEdgeOffset = vec2f(0.0);
		} else if isHorizontal {
			subpixelEdgeOffset = vec2f(0.0, pixelOffset * stepLength);
		} else {
			subpixelEdgeOffset = vec2f(pixelOffset * stepLength, 0.0);
		}
	}
#endif

	// We mix the effects of the edge-search with the directional diffusion.
	// Basically, we allow more diffusion if the edge-effect is smaller.
	let edgeStrength = (min(1.0, length(2.0 * subpixelEdgeOffset / pixelStep)));
	diffuseStrength = diffuseStrength * (1.0 - edgeStrength);

	// The step to take for the diffusion effect (blur in the direction of the
	// edge). Note that for diagonal-ish lines, the most blur is obtained when
	// stepping halfway to the next pixel, i.e. 0.707, because then the neighbour
	// pixels are taken into account more. Let's use no more than 0.6 because then
	// for horizontal lines, the max diffusion kernel is effectively [0.6, 2 * 0.5,
	// 0.6] which is still somewhat bell-shaped. Actually, if we use 0.5, we trade a
	// bit more smoothness for perceived sharpness. Make its 0.51 so it does not
	// look like some sort of offset.
	let max_step_size = 0.51;
	let diffuseStep = diffuseDirection * pixelStep * (max_step_size * diffuseStrength);

	// Compose the three texture coordinates, combining the ede-offset with the diffusion componennt, depending on the steepness of the edge.
	let texCoord1 = texCoord - diffuseStep + subpixelEdgeOffset;
	let texCoord2 = texCoord + diffuseStep + subpixelEdgeOffset;

	// Sample the final color
	var finalColor = vec3f(0.0);
	finalColor += 0.5 * sample_color(texCoord1).rgb;
	finalColor += 0.5 * sample_color(texCoord2).rgb;

	return output_color(vec4f(finalColor, centerSample.a));

}

#endif
