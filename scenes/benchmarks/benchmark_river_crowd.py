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
from scenes.river_crossing_example import FIXED_DT, LANES, POSE_LOD_FAR, POSE_LOD_NEAR, create_demo, crowd_eligibility, crowd_skeleton_lod, make_animation_graph, river_rendering


MODES = ("cpu", "gpu-eligible", "distance-lod")
PHASES = ("bank", "swim")
GPU_START = "river benchmark start"


def measure_mode(mode, row_count, assets, graph_setup, eligibility, output, profiler, *, frames, warmup, swim_tick, pose_culling):
	from graaal.Math import Camera
	from graaal.RenderContext import RenderContext
	from graaal.RenderPasses.CrowdPose import GraphPosePass

	rig, primitive, clips = assets
	world, dummies, physics, schedule, animation, parameters, _ = create_demo(rig, clips, row_count=row_count, graph_setup=graph_setup)
	transforms = world.get(MaterialTransform)
	graph = animation.playback.graph
	router = None if eligibility is None else CrowdGraphRouter(eligibility)
	water_column = graph.parameter_columns["in_water"]
	queue = RenderContext.device.queue
	size = output.size[:2]
	result = {phase: [] for phase in PHASES}
	tick = 0
	with (
		physics,
		river_rendering(world, dummies, physics, rig, primitive, lambda: 0.0, pose_culling=pose_culling) as (drawing, renderer, palette, pose_visibility),
	):
		poses = GraphPosePass(palette, len(dummies), eligibility=eligibility if mode != "cpu" else None)
		try:
			for phase, start_tick in zip(PHASES, (0, swim_tick)):
				while tick < start_tick:
					schedule.run(FIXED_DT, fixed_steps=1, fixed_dt=FIXED_DT)
					tick += 1
				lod = DistancePoseLod(POSE_LOD_NEAR, POSE_LOD_FAR) if mode in ("distance-lod", "reduced-rig") else None
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
					drawing.sync()
					pose_start = perf_counter()
					owners, playback = animation.owners, animation.playback
					rows = np.searchsorted(dummies, owners)
					values = parameters[rows]
					mask = None if lod is None else lod.update(owners, transforms[owners].position, camera.position, hold=playback.blending(values))
					visible = pose_visibility(camera, size, rows)
					poses.record(commands, owners, playback, values, gpu_mask=mask, pose_mask=visible)
					pose_ms = (perf_counter() - pose_start) * 1000
					renderer.record(commands, size, camera, output.color_attachment())
					RenderContext.submit(commands.finish())
					queue.on_submitted_work_done_sync()
					frame_ms = (perf_counter() - start) * 1000
					gpu_passes = profiler.read() if profiler is not None else None
					if frame < warmup: continue

					# Count eligibility outside the timed frame to avoid duplicating routing work in it.
					if router is None:
						fallback, eligible = np.arange(len(owners)), np.empty(0, dtype=np.intp)
					else:
						fallback, batch = router.split(playback, values)
						eligible = batch.rows
					skipped = 0
					if visible is not None:
						eligible, fallback = eligible[visible[eligible]], fallback[visible[fallback]]
						skipped = len(owners) - int(np.count_nonzero(visible))
					gpu_rows = eligible if mask is None else eligible[mask[eligible]]
					gpu_count = len(gpu_rows) if mode != "cpu" else 0
					in_water = values[:, water_column] != 0
					fading = playback.target_states[fallback] >= 0
					result[phase].append(dict(tick=tick, swimmers=int(np.count_nonzero(in_water)), gpu_poses=gpu_count,
						reduced_poses=gpu_count if mode == "reduced-rig" else 0,
						cpu_poses=len(owners) - gpu_count - skipped, skipped_poses=skipped, graph_fallback=len(fallback),
						fallback_swim=int(np.count_nonzero(in_water[fallback] & ~fading)), fallback_fade=int(np.count_nonzero(fading)),
						lod_cpu_poses=len(eligible) - gpu_count if lod is not None else 0,
						pose_ms=pose_ms, frame_ms=frame_ms, gpu_frame_ms=profiler.last_frame_ms if profiler is not None else None,
						gpu_pose_ms=gpu_passes.get("crowd pose", 0.0) if gpu_passes is not None else None))
		finally:
			poses.close()
	return result


def summarize(samples):
	counts = ("swimmers", "gpu_poses", "reduced_poses", "cpu_poses", "skipped_poses", "graph_fallback", "fallback_swim", "fallback_fade", "lod_cpu_poses")
	times = ("pose_ms", "gpu_pose_ms", "gpu_frame_ms", "frame_ms")
	return {
		**{key: dict(zip(("min", "median", "max"), np.percentile([row[key] for row in samples], (0, 50, 100)).tolist())) for key in counts},
		**{key: None if samples[0][key] is None else dict(zip(("median", "p95"),
			np.percentile([row[key] for row in samples], (50, 95)).tolist())) for key in times},
	}


def run(*, row_counts=(12, 24), size=(640, 480), frames=30, warmup=5, rounds=3, swim_tick=950, pose_culling=(False, True), modes=MODES):
	from graaal.Diagnostics import GpuProfiler
	from graaal.RenderContext import RenderContext, Texture, TextureUsage
	from scenes.animation_dummy import load_dummy
	from scenes.physics_viewing_utils import has_blue_pixels, read_pixels

	assets = load_dummy(swimming=True)
	rig, primitive, clips = assets
	graph_setup = make_animation_graph(rig, clips)
	graph, _ = graph_setup
	full = crowd_eligibility(graph, primitive) if set(modes) - {"reduced-rig"} else None
	reduction_start = perf_counter()
	variant, compact = crowd_skeleton_lod(graph, primitive) if "reduced-rig" in modes else (None, None)
	reduction_seconds = perf_counter() - reduction_start
	pools = {mode: compact if mode == "reduced-rig" else full for mode in modes}
	output = profiler = None
	try:
		RenderContext.setup_graphics_backend(True, required_features=("indirect-first-instance",), optional_features=("timestamp-query",))
		output = Texture(size, format="rgba8unorm", usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC)
		if "timestamp-query" in RenderContext.device.features: profiler = GpuProfiler()
		rows = []
		for row_count in row_counts:
			measurements = {(phase, mode, culling): [] for phase in PHASES for mode in modes for culling in pose_culling}
			for trial in range(rounds):
				order = modes[trial % len(modes):] + modes[:trial % len(modes)]
				for i, mode in enumerate(order):
					for culling in pose_culling if (trial + i) % 2 == 0 else reversed(pose_culling):
						result = measure_mode(mode, row_count, assets, graph_setup, pools[mode], output, profiler,
							frames=frames, warmup=warmup, swim_tick=swim_tick, pose_culling=culling)
						for phase in PHASES: measurements[phase, mode, culling].extend(dict(round=trial + 1, **row) for row in result[phase])
			for (phase, mode, culling), samples in measurements.items():
				pool = None if mode == "cpu" or pools[mode] is None else pools[mode].pool
				rows.append(dict(dummies=LANES * row_count, phase=phase, mode=mode, pose_culling=culling,
					pool_bones=pool.bones if pool is not None else 0, pool_bytes=pool.packed.nbytes + pool.table.nbytes if pool is not None else 0,
					summary=summarize(samples), samples=samples))
		if not has_blue_pixels(read_pixels(output)): raise RuntimeError("River crowd benchmark rendered no water pixels")
		return dict(adapter=dict(RenderContext.adapter.info), platform=platform.platform(), python=platform.python_version(),
			numpy=np.__version__, size=size, frames_per_round=frames, warmup_per_window=warmup, rounds=rounds,
			swim_preroll_ticks=swim_tick, fixed_dt=FIXED_DT, lod_near=POSE_LOD_NEAR, lod_far=POSE_LOD_FAR,
			gpu_timestamps=profiler is not None, rows=rows, reduced_rig=None if compact is None else dict(error_budget=compact.error_budget,
				baked_frame_error_bound=variant.max_position_error_bound, preparation_seconds=reduction_seconds,
				states={name: result.max_position_error for name, result in compact.states.items()},
				transitions={f"{edge.source} -> {edge.target}": edge.result.max_position_error for edge in compact.transitions}))
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
	parser.add_argument("--pose-culling", choices=("both", "off", "on"), default="both", help="compare pose culling or run one setting")
	parser.add_argument("--modes", choices=MODES + ("reduced-rig",), nargs="+", default=MODES, help="pose modes to compare")
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
	pose_culling = (False, True) if args.pose_culling == "both" else (args.pose_culling == "on",)
	report = run(row_counts=tuple(dict.fromkeys(args.rows)), size=(width, height), frames=args.frames,
		warmup=args.warmup, rounds=args.rounds, swim_tick=args.swim_tick, pose_culling=pose_culling, modes=tuple(dict.fromkeys(args.modes)))
	if args.output is not None: args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
	print("Dummies  Window  Mode          Cull  Swim   GPU  LOD  CPU  Skip  Fallback   Pose ms   GPU frame ms   Frame ms (p50/p95)  Pool KiB")
	for row in report["rows"]:
		summary = row["summary"]
		swimmers, gpu, reduced, cpu, skipped, fallback = (summary[key]["median"] for key in ("swimmers", "gpu_poses", "reduced_poses", "cpu_poses", "skipped_poses", "graph_fallback"))
		culling = "on" if row["pose_culling"] else "off"
		gpu_ms = "—" if summary["gpu_frame_ms"] is None else f"{summary['gpu_frame_ms']['median']:.3f}"
		print(f"{row['dummies']:7d}  {row['phase']:6}  {row['mode']:12}  {culling:4}  {swimmers:4g}  {gpu:4g}  {reduced:3g}  {cpu:3g}  {skipped:4g}  {fallback:8g}"
			f"  {summary['pose_ms']['median']:8.3f}  {gpu_ms:>13}  {summary['frame_ms']['median']:9.3f}/{summary['frame_ms']['p95']:.3f}  {row['pool_bytes'] / 1024:8.1f}")


if __name__ == "__main__":
	main()
