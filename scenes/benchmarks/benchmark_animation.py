"""Measure ECS animation control and CPU pose evaluation without a GPU."""
import argparse
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

from graaal.ECS import ECS
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.Animation.SkeletalAnimation import Clip, Skeleton


def make_workload(entities, bones, crossfade):
	# A forest of short chains gives a repeatable, depth-first skeleton.
	parents = [-1 if i % 8 == 0 else i - 1 for i in range(bones)]
	translations = np.zeros((bones, 3), dtype=np.float32)
	rotations = np.tile([0, 0, 0, 1], (bones, 1))
	scales = np.ones((bones, 3), dtype=np.float32)
	rig = Skeleton(parents, translations, rotations, scales, np.tile(np.eye(4), (bones, 1, 1)))
	frames = np.broadcast_to(translations, (3, bones, 3)).copy()
	frames[1, :, 0] = 0.2
	clip = Clip(frames, np.broadcast_to(rotations, (3, bones, 4)), np.broadcast_to(scales, (3, bones, 3)), fps=1)
	target_frames = frames.copy()
	target_frames[1, :, 1] = 0.3
	target = Clip(target_frames, np.broadcast_to(rotations, (3, bones, 4)), np.broadcast_to(scales, (3, bones, 3)), fps=1)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer)
	ids = world.create(entities)
	world.add(ids, SkeletonRef(0), ClipPlayer(0))
	players = world.get(ClipPlayer)[ids]
	players.assign(time=np.linspace(0, 1.9, entities, dtype=np.float32))
	if crossfade:
		players.assign(transitioning=True, target_clip_id=1, target_time=0.25, transition_time=500000, transition_duration=1000000)
	return EcsAnimationAdapter(world, {0: rig}, {0: clip, 1: target})


def measure(callback, warmup, repeat):
	for _ in range(warmup): callback()
	timings = []
	for _ in range(repeat):
		start = time.perf_counter()
		callback()
		timings.append((time.perf_counter() - start) * 1000)
	tracemalloc.start()
	try:
		callback()
		_, peak = tracemalloc.get_traced_memory()
	finally: tracemalloc.stop()
	return {"median_ms": statistics.median(timings), "p95_ms": float(np.percentile(timings, 95)), "peak_traced_bytes": peak}


def benchmark(entities, bones, crossfade, warmup, repeat):
	adapter = make_workload(entities, bones, crossfade)
	dt = 1 / 60
	control = measure(lambda: adapter.update(dt, evaluate_poses=False), warmup, repeat)
	pose = measure(adapter.evaluate_samples, warmup, repeat)
	combined = measure(lambda: adapter.update(dt), warmup, repeat)
	return {"entities": entities, "bones": bones, "mode": "crossfade" if crossfade else "single", "control": control, "pose": pose, "combined": combined}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--entities", type=int, nargs="+", default=[64, 256, 1024])
	parser.add_argument("--bones", type=int, nargs="+", default=[12, 60])
	parser.add_argument("--warmup", type=int, default=3)
	parser.add_argument("--repeat", type=int, default=10)
	parser.add_argument("--output", type=Path)
	args = parser.parse_args()
	if min(args.entities) < 1 or min(args.bones) < 1 or args.warmup < 0 or args.repeat < 1:
		parser.error("entities, bones and repeat must be positive; warmup must be nonnegative")
	results = [benchmark(n, bones, crossfade, args.warmup, args.repeat) for bones in args.bones for n in args.entities for crossfade in (False, True)]
	report = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "warmup": args.warmup, "repeat": args.repeat, "results": results}
	text = json.dumps(report, indent=2)
	if args.output is not None: args.output.write_text(text + "\n", encoding="utf-8")
	else: print(text)


if __name__ == "__main__":
	main()
