"""Compare river crowd pose modes with physics and offscreen material draws."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from graaal.Animation.AnimationLod import DistancePoseLod
from graaal.Animation.CrowdGraphSamples import CrowdGraphRouter
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from scenes.river_crossing_example import FIXED_DT, LANES, POSE_LOD_FAR, POSE_LOD_NEAR, create_demo, crowd_eligibility, river_rendering


MODES = ("cpu", "gpu-eligible", "distance-lod")
PHASES = ("bank", "swim")
GPU_START = "river benchmark start"


def measure_mode(mode, row_count, assets, output, profiler, *, frames, warmup, swim_tick):
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext
	from graaal.RenderPasses.CrowdPose import GraphPosePass

	rig, primitive, clips = assets
	world, dummies, physics, schedule, animation, parameters, _ = create_demo(rig, clips, row_count=row_count)
	transforms = world.get(MaterialTransform)
	graph = animation.playback.graph
	eligibility = crowd_eligibility(graph, primitive)
	router = CrowdGraphRouter(eligibility)
	water_column = graph.parameter_columns["in_water"]
	queue = RenderContext.device.queue
	size = output.size[:2]
	result = {phase: [] for phase in PHASES}
	tick = 0
	with physics, river_rendering(world, dummies, physics, rig, primitive, lambda: 0.0) as (drawing, renderer, palette):
		poses = GraphPosePass(palette, len(dummies), eligibility=eligibility if mode != "cpu" else None)
		try:
			for phase, start_tick in zip(PHASES, (0, swim_tick)):
				while tick < start_tick:
					schedule.run(FIXED_DT, fixed_steps=1, fixed_dt=FIXED_DT)
					tick += 1
				lod = DistancePoseLod(POSE_LOD_NEAR, POSE_LOD_FAR) if mode == "distance-lod" else None
				camera = Camera((15, 10, -5), (0, 0, -2), (0, 1, 0), 50, near=0.1, far=120)
				for frame in range(warmup + frames):
					# Each mode follows the same camera sweep and one physics tick per draw.
					progress = frame / max(1, warmup + frames - 1)
					start = perf_counter()
					schedule.run(FIXED_DT, fixed_steps=1, fixed_dt=FIXED_DT)
					tick += 1
					camera.position[:] = (15 - 5 * progress, 10, -5 + 3 * progress)
					commands = RenderContext.commands("river crowd benchmark")
					if profiler is not None:
						commands = profiler.wrap(commands)
						# Bracket CPU palette copies as well as compute and draw passes.
						with commands.compute_pass(label=GPU_START): pass
					pose_start = perf_counter()
					owners, playback = animation.owners, animation.playback
					values = parameters[np.searchsorted(dummies, owners)]
					mask = None if lod is None else lod.update(owners, transforms[owners].position, camera.position, hold=playback.blending(values))
					poses.record(commands, owners, playback, values, gpu_mask=mask)
					pose_ms = (perf_counter() - pose_start) * 1000
					drawing.sync()
					renderer.record(commands, size, camera, output.color_attachment())
					RenderContext.submit(commands.finish())
					queue.on_submitted_work_done_sync()
					frame_ms = (perf_counter() - start) * 1000
					gpu_passes = profiler.read() if profiler is not None else None
					if frame < warmup: continue

					# Count eligibility outside the timed frame to avoid duplicating routing work in it.
					fallback, batch = router.split(playback, values)
					gpu_rows = batch.rows if mask is None else batch.rows[mask[batch.rows]]
					gpu_count = len(gpu_rows) if mode != "cpu" else 0
					in_water = values[:, water_column] != 0
					fading = playback.target_states[fallback] >= 0
					result[phase].append(dict(tick=tick, swimmers=int(np.count_nonzero(in_water)), gpu_poses=gpu_count,
						cpu_poses=len(owners) - gpu_count, graph_fallback=len(fallback),
						fallback_swim=int(np.count_nonzero(in_water[fallback] & ~fading)), fallback_fade=int(np.count_nonzero(fading)),
						lod_cpu_poses=len(batch.rows) - gpu_count if lod is not None else 0,
						pose_ms=pose_ms, frame_ms=frame_ms, gpu_frame_ms=profiler.last_frame_ms if profiler is not None else None,
						gpu_pose_ms=gpu_passes.get("crowd pose", 0.0) if gpu_passes is not None else None))
		finally:
			poses.close()
	return result


def summarize(samples):
	counts = ("swimmers", "gpu_poses", "cpu_poses", "graph_fallback", "fallback_swim", "fallback_fade", "lod_cpu_poses")
	times = ("pose_ms", "gpu_pose_ms", "gpu_frame_ms", "frame_ms")
	return {
		**{key: dict(zip(("min", "median", "max"), np.percentile([row[key] for row in samples], (0, 50, 100)).tolist())) for key in counts},
		**{key: None if samples[0][key] is None else dict(zip(("median", "p95"),
			np.percentile([row[key] for row in samples], (50, 95)).tolist())) for key in times},
	}


def run(*, row_counts=(12, 24), size=(640, 480), frames=30, warmup=5, rounds=3, swim_tick=950):
	from graaal.Diagnostics import GpuProfiler
	from graaal.RenderContext import RenderContext, Texture, TextureUsage
	from scenes.animation_dummy import load_dummy
	from scenes.physics_viewing_utils import has_blue_pixels, read_pixels

	assets = load_dummy(swimming=True)
	output = profiler = None
	try:
		RenderContext.setup_graphics_backend(True, required_features=("indirect-first-instance",), optional_features=("timestamp-query",))
		output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
		if "timestamp-query" in RenderContext.device.features: profiler = GpuProfiler()
		rows = []
		for row_count in row_counts:
			measurements = {(phase, mode): [] for phase in PHASES for mode in MODES}
			for trial in range(rounds):
				order = MODES[trial % len(MODES):] + MODES[:trial % len(MODES)]
				for mode in order:
					result = measure_mode(mode, row_count, assets, output, profiler, frames=frames, warmup=warmup, swim_tick=swim_tick)
					for phase in PHASES: measurements[phase, mode].extend(dict(round=trial + 1, **row) for row in result[phase])
			for (phase, mode), samples in measurements.items():
				rows.append(dict(dummies=LANES * row_count, phase=phase, mode=mode, summary=summarize(samples), samples=samples))
		if not has_blue_pixels(read_pixels(output)): raise RuntimeError("River crowd benchmark rendered no water pixels")
		return dict(adapter=dict(RenderContext.adapter.info), platform=platform.platform(), python=platform.python_version(),
			numpy=np.__version__, size=size, frames_per_round=frames, warmup_per_window=warmup, rounds=rounds,
			swim_preroll_ticks=swim_tick, fixed_dt=FIXED_DT, lod_near=POSE_LOD_NEAR, lod_far=POSE_LOD_FAR,
			gpu_timestamps=profiler is not None, rows=rows)
	finally:
		if profiler is not None: profiler.close()
		if output is not None: output.close()
		RenderContext.cleanup()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--rows", type=int, nargs="+", default=[12, 24], help=f"crowd rows at {LANES} dummies per row")
	parser.add_argument("--size", default="640x480", help="offscreen target as WIDTHxHEIGHT")
	parser.add_argument("--frames", type=int, default=30, help="measured frames per window and round")
	parser.add_argument("--warmup", type=int, default=5, help="warm-up frames before each measured window")
	parser.add_argument("--rounds", type=int, default=3, help="repetitions with a rotated mode order")
	parser.add_argument("--swim-tick", type=int, default=950, help="simulation tick before swim window warm-up")
	parser.add_argument("--output", type=Path, help="write JSON summaries and individual frames")
	args = parser.parse_args()
	try:
		width, height = map(int, args.size.lower().split("x"))
	except ValueError:
		parser.error("--size must be WIDTHxHEIGHT")
	if min(width, height, *args.rows, args.frames, args.rounds) < 1 or args.warmup < 0:
		parser.error("size, rows, frames and rounds must be positive and warmup must be nonnegative")
	if args.swim_tick < args.frames + args.warmup:
		parser.error("--swim-tick must be after the bank window")
	report = run(row_counts=tuple(dict.fromkeys(args.rows)), size=(width, height), frames=args.frames,
		warmup=args.warmup, rounds=args.rounds, swim_tick=args.swim_tick)
	if args.output is not None: args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
	print("Dummies  Window  Mode          Swim   GPU  CPU  Fallback   Pose ms   GPU frame ms   Frame ms (p50/p95)")
	for row in report["rows"]:
		summary = row["summary"]
		swimmers, gpu, cpu, fallback = (summary[key]["median"] for key in ("swimmers", "gpu_poses", "cpu_poses", "graph_fallback"))
		gpu_ms = "—" if summary["gpu_frame_ms"] is None else f"{summary['gpu_frame_ms']['median']:.3f}"
		print(f"{row['dummies']:7d}  {row['phase']:6}  {row['mode']:12}  {swimmers:5g}  {gpu:4g}  {cpu:3g}  {fallback:8g}"
			f"  {summary['pose_ms']['median']:8.3f}  {gpu_ms:>13}  {summary['frame_ms']['median']:9.3f}/{summary['frame_ms']['p95']:.3f}")


if __name__ == "__main__":
	main()
