import numpy as np
import pytest

from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.AnimationGraphPlayback import AnimationGraphPlayback
from graaal.Animation.BakedClipPool import BakedClipPool
from graaal.Animation.CrowdGraphValidation import validate_crowd_graph
from graaal.Animation.SkeletalAnimation import Clip, Skeleton


def test_graph_passes_write_shared_palette_in_command_order():
	pytest.importorskip("wgpu")
	from graaal.RenderContext import RenderContext
	from graaal.RenderContext.SkinMatrixBuffer import SkinMatrixBuffer
	from graaal.RenderPasses.CrowdPose import GraphPosePass

	rotation = [[0, 0, 0, 1]]
	scale = [[1, 1, 1]]
	rig = Skeleton([-1], [[0, 0, 0]], rotation, scale, [np.eye(4)])
	definition = AnimationGraphDefinition({
		"version": 1,
		"states": {"idle": {"clip": "idle"}},
		"initial_state": "idle",
		"sample_slots": ["base"],
		"tasks": [{"name": "base", "op": "sample", "slot": "base"}],
	})

	def graph_at(x):
		idle = Clip([[[x, 0, 0]]], [rotation], [scale])
		return definition.compile(rig, {"idle": idle})

	RenderContext.setup_graphics_backend(True)
	palette = SkinMatrixBuffer(5)
	owners = np.array([1, 2, 3, 4, (1 << 32) + 1], dtype=np.uint64)
	for owner in owners: palette.acquire(owner, 1)
	passes = []
	try:
		commands = RenderContext.commands("independent graph pose passes")
		commands.clear_buffer(palette.buffer)
		for graph, rows in ((graph_at(1.0003), [4, 0]), (graph_at(2.0006), [3, 1])):
			eligibility = validate_crowd_graph(graph, BakedClipPool.from_graph(graph), error_budget=0.002)
			playback = AnimationGraphPlayback(graph, 2)
			pose_pass = GraphPosePass(palette, 2, eligibility=eligibility)
			passes.append(pose_pass)
			with pytest.raises(ValueError, match="distinct"):
				pose_pass.record(commands, owners[[0, 0]], playback, gpu_mask=[False, True])
			with pytest.raises(ValueError, match="gpu_mask"):
				pose_pass.record(commands, owners[rows], playback, gpu_mask=[0, 1])
			pose_pass.record(commands, owners[rows], playback, gpu_mask=[False, True])

		cpu = GraphPosePass(palette, 1)
		passes.append(cpu)
		cpu.record(commands, owners[[2]], AnimationGraphPlayback(graph_at(3.0009), 1))
		RenderContext.submit(commands.finish())
		actual = np.frombuffer(RenderContext.device.queue.read_buffer(palette.buffer.handle), dtype=np.float32).reshape(5, 3, 4)
		np.testing.assert_allclose(actual[:, 0, 3], [1, 2, 3.0009, 2.0006, 1.0003], rtol=0, atol=1e-7)
	finally:
		for pose_pass in passes: pose_pass.close()
		palette.close()
