from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from graaal.RenderContext import BufferUsage, ComputePipeline, GpuBuffer, RenderContext, Shader
from graaal.Animation.SkeletalAnimation import skin_vertices
from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer


def test_skin_matrix_allocation_recycling_and_batch_validation():
	RenderContext.setup_graphics_backend(True)
	palette = SkinMatrixBuffer(5)
	assert palette.acquire(np.uint64(0), 2) == 0
	assert palette.acquire(np.uint64(1), 1) == 2
	np.testing.assert_array_equal(palette.ranges([1, 0, 1]), [[2, 1], [0, 2], [2, 1]])
	assert palette.acquire(np.uint64(2), 2) == 3
	palette.release(0)
	palette.release(1)
	replacement = np.uint64(1 << 32)
	assert palette.acquire(replacement, 3) == 0
	with pytest.raises(KeyError): palette.range(0)
	with pytest.raises(ValueError, match="insufficient"):
		palette.acquire(4, 1)
	pose = np.tile(np.eye(4, dtype=np.float32)[:3], (1, 3, 1, 1))
	pose[0, :, 0, 3] = [1, 2, 3]
	palette.write(np.array([replacement]), pose)
	before = palette.buffer.content.copy()
	with pytest.raises(ValueError, match="No skin range"):
		palette.write(np.array([replacement, 9], dtype=np.uint64), np.concatenate((pose, pose)))
	np.testing.assert_array_equal(palette.buffer.content, before)
	raw = RenderContext.device.queue.read_buffer(palette.buffer.handle)
	np.testing.assert_array_equal(np.frombuffer(raw, dtype=np.float32).reshape(5, 3, 4)[:3], pose[0])
	palette.close()


def test_gpu_skinning_matches_cpu_for_weighted_and_reflected_transforms():
	RenderContext.setup_graphics_backend(True)
	palette = SkinMatrixBuffer(2)
	palette.acquire(12, 2)
	matrices = np.tile(np.eye(4, dtype=np.float32)[:3], (1, 2, 1, 1))
	matrices[0, :, 0, 0] = [-2, -1]
	matrices[0, :, 1, 1] = [1, 3]
	matrices[0, :, 0, 3] = [1, 3]
	palette.write(np.array([12], dtype=np.uint64), matrices)
	dtype = np.dtype([("position", "f4", 4), ("normal", "f4", 4), ("joints", "u4", 4), ("weights", "f4", 4)])
	source = np.zeros(2, dtype=dtype)
	source["position"][:, :3] = [[1, 1, 0], [2, 0, 1]]
	source["normal"][:, :3] = [[1, 1, 0], [0, 1, 1]]
	source["joints"] = [0, 1, 0, 1]
	source["weights"] = [[0.25, 0.75, 0, 0], [1, 0, 0, 0]]
	shader = Shader(source='''
#include "skinning.shaderlib"
<Source> position: vec4f;
<Source> normal: vec4f;
<Source> joints: vec4u;
<Source> weights: vec4f;
<Output> position: vec4f;
<Output> normal: vec4f;
storage<0, 0, read> source: array<Source>;
storage<0, 1, read> matrices: array<SkinMatrix>;
storage<0, 2, read_write> output: array<Output>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
	if id.x >= arrayLength(&source) { return; }
	let v = source[id.x];
	let skin = skin_vertex(v.position.xyz, v.normal.xyz, v.weights,
		matrices[v.joints.x], matrices[v.joints.y], matrices[v.joints.z], matrices[v.joints.w]);
	output[id.x].position = vec4f(skin.position, 1.0);
	output[id.x].normal = vec4f(skin.normal, 0.0);
}
''', basedir=str(Path("graaal/shaders").resolve()))
	pipeline = ComputePipeline(shader)
	source_buffer = GpuBuffer(source, BufferUsage.STORAGE)
	output = GpuBuffer(np.zeros((2, 2, 4), dtype=np.float32), BufferUsage.STORAGE | BufferUsage.COPY_SRC)
	bindings = shader.bind_group(0, source=source_buffer, matrices=palette.buffer, output=output)
	commands = RenderContext.commands("skin_test")
	with commands.compute_pass() as compute:
		compute.set_pipeline(pipeline)
		compute.set_bind_group(bindings)
		compute.dispatch(1)
	RenderContext.submit(commands.finish())
	actual = np.frombuffer(RenderContext.device.queue.read_buffer(output.handle), dtype=np.float32).reshape(2, 2, 4)
	position, normal = skin_vertices(source["position"][:, :3], source["normal"][:, :3], source["joints"], source["weights"], matrices[0])
	np.testing.assert_allclose(actual[:, 0, :3], position, atol=1e-6)
	np.testing.assert_allclose(actual[:, 1, :3], normal, atol=1e-6)


def test_skin_matrix_batch_coalesces_adjacent_ranges(monkeypatch):
	RenderContext.setup_graphics_backend(True)
	palette = SkinMatrixBuffer(4)
	palette.acquire(10, 2)
	palette.acquire(11, 2)
	write_range = Mock(wraps=palette.buffer.write_range)
	monkeypatch.setattr(palette.buffer, "write_range", write_range)
	poses = np.tile(np.eye(4, dtype=np.float32)[:3], (2, 2, 1, 1))
	poses[0, :, 0, 3] = 11
	poses[1, :, 0, 3] = 10
	palette.write(np.array([11, 10], dtype=np.uint64), poses)
	write_range.assert_called_once()
	raw = RenderContext.device.queue.read_buffer(palette.buffer.handle)
	actual = np.frombuffer(raw, dtype=np.float32).reshape(4, 3, 4)
	np.testing.assert_allclose(actual[:, 0, 3], [10, 10, 11, 11])
