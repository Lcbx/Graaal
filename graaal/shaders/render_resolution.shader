texture<0, 0> scene_color: texture_2d<f32>;
texture<0, 1> scene_sampler: sampler;

<FullscreenOutput> @builtin(position) position: vec4f;
<FullscreenOutput> @location(0) uv: vec2f;

@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOutput {
	let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
	var output: FullscreenOutput;
	output.position = vec4f(corner*2.0-1.0, 0.0, 1.0);
	output.uv = vec2f(corner.x, 1.0-corner.y);
	return output;
}

@fragment
fn present(input: FullscreenOutput) -> @location(0) vec4f {
	return textureSampleLevel(scene_color, scene_sampler, input.uv, 0.0);
}
