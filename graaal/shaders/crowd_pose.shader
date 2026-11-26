#include "skinning.shaderlib"

<BakedClip> base: u32;
<BakedClip> frames: u32;
<BakedClip> fps: f32;
<BakedClip> duration: f32;
<CrowdSample> clip: u32;
<CrowdSample> time: f32;
<CrowdSample> weight: f32;
<CrowdRow> base: u32;
<CrowdRow> count: u32;
<CrowdRow> padding: vec2u;
<CrowdRow> samples: array<CrowdSample, 4>;

storage<0, 0, read> packed: array<vec4u>;
storage<0, 1, read> clips: array<BakedClip>;
storage<0, 2, read> inverse_bind: array<mat4x4f>;
storage<0, 3, read> rows: array<CrowdRow>;
storage<0, 4, read_write> palette: array<SkinMatrix>;
{% if PARAMS.get("mapped_palette", false) %}
storage<0, 5, read> mapped_joints: array<u32>;
{% endif %}

struct Pose {
	rotation: vec4f,
	translation_scale: vec4f,
};

fn unpack_pose(index: u32) -> Pose {
	let value = packed[index];
	return Pose(vec4f(unpack2x16float(value.x), unpack2x16float(value.y)),
		vec4f(unpack2x16float(value.z), unpack2x16float(value.w)));
}

fn sample_pose(source: CrowdSample, bone: u32, bones: u32) -> Pose {
	let clip = clips[source.clip];
	let last = clip.frames - 1u;
	let frame = select(clamp(source.time * clip.fps, 0.0, f32(last)), f32(last), source.time >= clip.duration);
	let left = u32(frame);
	let right = min(left + 1u, last);
	let a = unpack_pose(clip.base + left * bones + bone);
	let b = unpack_pose(clip.base + right * bones + bone);
	let alpha = frame - f32(left);
	let adjusted = select(b.rotation, -b.rotation, dot(a.rotation, b.rotation) < 0.0);
	return Pose(normalize(mix(a.rotation, adjusted, alpha)), mix(a.translation_scale, b.translation_scale, alpha));
}

fn affine(rotation: vec4f, translation_scale: vec4f) -> mat4x4f {
	let x = rotation.x;
	let y = rotation.y;
	let z = rotation.z;
	let w = rotation.w;
	let s = translation_scale.w;
	return mat4x4f(
		vec4f(vec3f(1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y + w*z), 2.0 * (x*z - w*y)) * s, 0.0),
		vec4f(vec3f(2.0 * (x*y - w*z), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z + w*x)) * s, 0.0),
		vec4f(vec3f(2.0 * (x*z + w*y), 2.0 * (y*z - w*x), 1.0 - 2.0 * (x*x + y*y)) * s, 0.0),
		vec4f(translation_scale.xyz, 1.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
	let bones = arrayLength(&inverse_bind);
	if id.x >= bones { return; }
	let row = rows[id.y];
	var dominant = 0u;
	var poses: array<Pose, 4>;
	for (var i = 0u; i < row.count; i++) {
		poses[i] = sample_pose(row.samples[i], id.x, bones);
		if row.samples[i].weight > row.samples[dominant].weight { dominant = i; }
	}
	let anchor = poses[dominant].rotation;
	var rotation = vec4f(0.0);
	var translation_scale = vec4f(0.0);
	for (var i = 0u; i < row.count; i++) {
		let weight = row.samples[i].weight;
		let turn = poses[i].rotation;
		rotation += select(turn, -turn, dot(turn, anchor) < 0.0) * weight;
		translation_scale += poses[i].translation_scale * weight;
	}
	let skin = transpose(affine(normalize(rotation), translation_scale) * inverse_bind[id.x]);
	let matrix = SkinMatrix(skin[0], skin[1], skin[2]);
{% if PARAMS.get("mapped_palette", false) %}
	for (var i = mapped_joints[id.x]; i < mapped_joints[id.x + 1u]; i++) {
		palette[row.base + mapped_joints[bones + 1u + i]] = matrix;
	}
{% else %}
	palette[row.base + id.x] = matrix;
{% endif %}
}
