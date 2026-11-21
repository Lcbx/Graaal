"""Compare Hi-Z mip granularity and sample counts on a fixed synthetic scene."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from graaal.DrawBatches import DrawBatches, FrustumCuller, HiZCuller, INDIRECT_DTYPE
from graaal.Diagnostics import GpuProfiler
from graaal.DrawBatches.InstanceData import MESH_INSTANCE_DTYPE, pack_quaternion, pack_scale
from graaal.RenderContext import BufferUsage, GpuBuffer, RenderContext, Texture, TextureUsage


def make_pyramid(size):
	"""Conservative max-depth mips for a near wall with narrow gaps."""
	x = np.arange(size)
	y = np.arange(size)
	wall = (x[None, :] > size//8) & (x[None, :] < size*7//8) & (y[:, None] > size//8) & (y[:, None] < size*7//8)
	gaps = ((x[None, :]//37) % 5 == 0) | ((y[:, None]//53) % 6 == 0)
	depth = np.where(wall & ~gaps, 0.35, 0.95).astype(np.float32)
	levels = [depth]
	while size > 1:
		depth = depth.reshape(size//2, 2, size//2, 2).max(axis=(1, 3))
		levels.append(depth)
		size //= 2
	return levels


def make_scene(count, size, seed):
	rng = np.random.default_rng(seed)
	positions = np.empty((count, 3), dtype=np.float32)
	positions[:, :2] = rng.uniform(-0.9, 0.9, (count, 2))
	positions[:, 2] = 0.7
	scales = rng.uniform(0.015, 0.05, (count, 3)).astype(np.float32)
	draws = DrawBatches(MESH_INSTANCE_DTYPE)
	# The benchmark dispatches culling only; a render pipeline is unnecessary.
	draws._resources.register_shader(0, prepass=False)
	draws.register_mesh(0, SimpleNamespace(index_count=3, index_range=(0, 3), vertex_range=(0, 3)), bounds=((0, 0, 0), (0.5, 0.5, 0.5)))
	draws.register_lod_group(0, (0,))
	draws.sync(np.zeros(count, dtype=np.int32), np.zeros(count, dtype=np.int32))
	instances = np.zeros(count, dtype=MESH_INSTANCE_DTYPE)
	instances["iPosition"] = positions
	instances["iRotation"] = pack_quaternion(np.array([0, 0, 0, 1], dtype=np.float32))
	instances["iScale"] = pack_scale(scales)
	draws.write_instances(instances[draws.instance_order])
	visibility = draws.create_visibility()
	for name in ("main_draw_cmd", "main_visible_instances"):
		old = visibility._view.buffers[name]
		visibility._view.buffers[name] = (GpuBuffer(old.content.copy(), old.usage | BufferUsage.COPY_SRC) if old.content is not None
			else GpuBuffer(None, old.usage | BufferUsage.COPY_SRC, dtype=old.dtype, count=old.count))
	visibility._view.bindings.clear()
	levels = make_pyramid(size)
	texture = Texture((size, size), format="r32float", usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST, mip_level_count=len(levels))
	for mip, level in enumerate(levels):
		width = level.shape[1]
		RenderContext.device.queue.write_texture({"texture": texture.handle, "mip_level": mip}, level, {"bytes_per_row": width*4}, (width, width, 1))
	return visibility, texture


def read_visible(visibility):
	queue = RenderContext.device.queue
	commands = np.frombuffer(queue.read_buffer(visibility.commands.handle), dtype=INDIRECT_DTYPE)
	count = int(commands["instance_count"].sum())
	ids = np.frombuffer(queue.read_buffer(visibility.visible_instances.handle, size=max(4, count*4)), dtype=np.uint32)[:count]
	return np.sort(ids.copy())


def run(count=20000, size=512, frames=24, warmup=4, rounds=3, seed=42, variants=((0, 4), (1, 4), (2, 4), (0, 1), (1, 1))):
	if count < 1 or size < 2 or size & (size - 1) or frames < 1 or warmup < 0 or rounds < 1:
		raise ValueError("Expected positive count/frames/rounds and a power-of-two size >= 2")
	RenderContext.setup_graphics_backend(True, required_features=("timestamp-query",))
	profiler = GpuProfiler(max_passes=3)
	frustum_culler, hiz_culler = FrustumCuller(), HiZCuller()
	try:
		visibility, texture = make_scene(count, size, seed)
		configs = list(dict.fromkeys(((0, 4), *variants)))
		measurements = {config: [] for config in configs}
		visible = {}
		rng = np.random.default_rng(seed + 1)
		for trial in range(rounds):
			repeats = frames//rounds + (trial < frames%rounds)
			if not repeats: continue
			for index in rng.permutation(len(configs)):
				bias, samples = configs[index]
				for frame in range(warmup + repeats):
					commands = profiler.wrap(RenderContext.commands("hiz_sampling"))
					visibility._stage = None
					frustum_culler.record(commands, visibility, camera_position=(0, 0, 0), view_projection=np.eye(4, dtype=np.float32))
					hiz_culler.record(commands, visibility, hzb_view=texture.view(), mip_bias=bias, samples=samples)
					RenderContext.submit(commands.finish())
					passes = profiler.read()
					if frame >= warmup: measurements[(bias, samples)].append(passes["cull_hiz"])
				visible[(bias, samples)] = read_visible(visibility)
		baseline = visible[(0, 4)]
		rows = []
		for config in configs:
			values = np.asarray(measurements[config])
			ids = visible[config]
			missing = np.setdiff1d(baseline, ids, assume_unique=True)
			rows.append(dict(mip_bias=config[0], samples=config[1], hiz_ms_median=float(np.median(values)),
				hiz_ms_p10=float(np.percentile(values, 10)), hiz_ms_p90=float(np.percentile(values, 90)),
				visible=int(len(ids)), extra_visible=int(len(np.setdiff1d(ids, baseline, assume_unique=True))), missing_baseline=int(len(missing))))
		return dict(adapter=dict(RenderContext.adapter.info), count=count, size=size, frames=frames, warmup=warmup, rounds=rounds, seed=seed,
			baseline="mip_bias=0, samples=4", timing="GPU pass timestamps; frustum reset and readback excluded", rows=rows)
	finally:
		profiler.close()
		RenderContext.cleanup()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--count", type=int, default=20000)
	parser.add_argument("--size", type=int, default=512)
	parser.add_argument("--frames", type=int, default=24)
	parser.add_argument("--warmup", type=int, default=4)
	parser.add_argument("--rounds", type=int, default=3)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output", type=Path)
	args = parser.parse_args()
	result = run(args.count, args.size, args.frames, args.warmup, args.rounds, args.seed)
	content = json.dumps(result, indent=2)
	if args.output: args.output.write_text(content + "\n", encoding="utf-8")
	print(content)


if __name__ == "__main__": main()
