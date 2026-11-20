"""Measure 120 Hz ECS/native synchronization and optional interpolation, without GPU."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.EcsPhysics import Collider, EcsPhysicsAdapter, RigidBody
from graaal.MaterialTransform import MaterialTransform


def summary(samples):
	return dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))


def measure_io(physics, owners, repeat):
	"""Isolate UID lookup, gather, conversion, and ECS scatter on one snapshot."""
	snapshot = physics.world.latest()
	uids = np.array([physics.body_uids[int(owner)] for owner in owners], dtype=np.uint32)
	selection = physics.ecs.get(MaterialTransform)[owners]
	samples = {name: [] for name in ("uid_lookup", "gather", "float64_to_float32", "scatter")}
	for _ in range(repeat):
		start = time.perf_counter_ns()
		rows = snapshot.rows_of(uids)
		looked_up = time.perf_counter_ns()
		position, rotation = snapshot.position[rows], snapshot.orientation[rows]
		gathered = time.perf_counter_ns()
		converted = position.astype(np.float32)
		cast = time.perf_counter_ns()
		selection.assign(position=converted, rotation=rotation)
		end = time.perf_counter_ns()
		for name, duration in zip(samples, (looked_up - start, gathered - looked_up, cast - gathered, end - cast)):
			samples[name].append(duration / 1e6)
	return {name: summary(values) for name, values in samples.items()}


def benchmark(count, threads, interpolate, warmup, repeat):
	import pyjoltphysics as pj

	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform)
	owners = ecs.create(count)
	ecs.add(owners, Collider(0), RigidBody(flags=int(pj.BodyFlag.NO_SLEEP)), MaterialTransform())
	i = np.arange(count)
	ecs.get(MaterialTransform)[owners].position = np.column_stack(((i % 32) * 2, (i // 32) * 2, np.zeros(count)))
	settings = pj.WorldSettings(tick_hz=120, threads=threads, gravity=(0, 0, 0))
	with EcsPhysicsAdapter(ecs, {0: pj.Shape.box()}, settings=settings, interpolate=interpolate) as physics:
		physics.set_velocity(owners, linear=(0.25, 0.1, 0), angular=(0, 0.2, 0))
		start = time.perf_counter_ns()
		physics.reconcile()
		creation_ms = (time.perf_counter_ns() - start) / 1e6
		schedule = Scheduler(ecs)
		samples = {name: [] for name in ("reconcile", "prepare_inputs", "step", "pull")}
		def timed(name, callback):
			def run(world, commands, dt):
				start = time.perf_counter_ns()
				callback(world, commands, dt)
				samples[name].append((time.perf_counter_ns() - start) / 1e6)
			return run
		for name in samples: schedule.add(name, timed(name, getattr(physics, name)), phase="fixed_update")
		total, render = [], []
		for _ in range(warmup + repeat):
			start = time.perf_counter_ns()
			schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
			total.append((time.perf_counter_ns() - start) / 1e6)
			if interpolate:
				start = time.perf_counter_ns()
				physics.render_transforms(owners, alpha=0.5)
				render.append((time.perf_counter_ns() - start) / 1e6)
			physics.world.drain_events()
		samples = {name: values[warmup:] for name, values in samples.items()}
		sync = np.sum([samples[name] for name in ("reconcile", "prepare_inputs", "pull")], axis=0)
		result = dict(bodies=count, threads=threads, interpolate=interpolate, tick_hz=120, ticks=physics.world.tick, creation_ms=creation_ms)
		result.update(stages={name: summary(values) for name, values in samples.items()}, sync=summary(sync), fixed_tick=summary(total[warmup:]), io=measure_io(physics, owners, repeat))
		if interpolate: result["render_transforms"] = summary(render[warmup:])
		return result


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--bodies", type=int, nargs="+", default=[100, 1000])
	parser.add_argument("--threads", type=int, nargs="+", default=[1])
	parser.add_argument("--warmup", type=int, default=30)
	parser.add_argument("--repeat", type=int, default=240)
	parser.add_argument("--output", type=Path)
	args = parser.parse_args()
	if min(args.bodies) < 1 or min(args.threads) < 1 or args.warmup < 0 or args.repeat < 1:
		parser.error("bodies, threads and repeat must be positive; warmup must be nonnegative")
	report = dict(python=platform.python_version(), numpy=np.__version__, pyjoltphysics=importlib.metadata.version("pyjoltphysics"), platform=platform.platform(), warmup=args.warmup, repeat=args.repeat)
	report["results"] = [benchmark(count, threads, interpolate, args.warmup, args.repeat) for count in args.bodies for threads in args.threads for interpolate in (False, True)]
	text = json.dumps(report, indent=2)
	if args.output is None: print(text)
	else: args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
