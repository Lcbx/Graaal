"""Measure headless 120 Hz synchronization, collisions, compounds and ragdolls."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from graaal.ECS import ECS, Scheduler
from graaal.Diagnostics import inspect_physics
from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter, PhysicsRagdoll, RigidBody
from graaal.EcsAdapters.MaterialTransform import MaterialTransform
from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import RagdollAsset
from graaal.Animation.SkeletalAnimation import Skeleton, model_matrices


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


def make_workload(count, workload, pj):
	"""Build deterministic declarations; collision workloads share a static floor."""
	ecs = ECS()
	ecs.register(Collider, RigidBody, MaterialTransform, PhysicsRagdoll)
	owners = ecs.create(count)
	i = np.arange(count)
	flags = int(pj.BodyFlag.NO_SLEEP | pj.BodyFlag.REPORT)
	ragdolls = None
	if workload == "ragdolls":
		native = pj.RagdollSettings.humanoid(flags=flags)
		bones = len(native)
		rig = Skeleton(native.parents, native.rest_translations, native.rest_rotations, np.ones((bones, 3)), np.tile(np.eye(4), (bones, 1, 1)))
		rig = Skeleton(rig.parents, rig.translation, rig.rotation, rig.scale, np.linalg.inv(model_matrices(rig, rig.translation[None], rig.rotation[None], rig.scale[None])[0]))
		ragdolls = {0: RagdollAsset(rig, native)}
		columns = int(np.ceil(np.sqrt(count)))
		positions = np.column_stack(((i % columns) * 1.8, np.full(count, 0.1), (i // columns) * 1.8))
		ecs.add(owners, PhysicsRagdoll(0, mode=int(pj.RagdollMode.LIMP)), MaterialTransform(rotation=Quaternion.from_z_rotation(0.35)))
		shapes = {}
		spacing = 1.8
	else:
		ecs.add(owners, RigidBody(flags=flags), MaterialTransform())
		shapes = {0: pj.Shape.box((0.5, 1, 1) if workload == "compounds" else (1, 1, 1))}
		if workload == "compounds":
			ecs.add(owners, Collider(0, position=(-0.25, 0, 0)), Collider(0, position=(0.25, 0, 0)))
		else:
			ecs.add(owners, Collider(0))
		columns = int(np.ceil(np.cbrt(count)))
		positions = np.column_stack(((i % columns) * 1.05, 0.49 + (i // (columns * columns)) * 1.02, ((i // columns) % columns) * 1.05))
		if workload == "isolated": positions = np.column_stack(((i % 32) * 2, (i // 32) * 2, np.zeros(count)))
		spacing = 1.05
	ecs.get(MaterialTransform)[owners].position = positions
	if workload != "isolated":
		floor = ecs.create(1)[0]
		width = (columns + 2) * spacing
		shapes[1] = pj.Shape.box((width, 1, width))
		center = (columns - 1) * spacing * 0.5
		ecs.add(floor, Collider(1), MaterialTransform(position=(center, -0.5, center)))
	return ecs, owners, shapes, ragdolls


def benchmark(count, threads, interpolate, warmup, repeat, *, workload="isolated"):
	import pyjoltphysics as pj

	if workload not in ("isolated", "colliding", "compounds", "ragdolls"): raise ValueError("unknown physics workload")
	if count < 1 or threads < 1 or warmup < 0 or repeat < 1: raise ValueError("counts, threads and repeat must be positive; warmup must be nonnegative")
	ecs, owners, shapes, ragdolls = make_workload(count, workload, pj)
	settings = pj.WorldSettings(tick_hz=120, threads=threads, gravity=(0, 0, 0) if workload == "isolated" else (0, -9.81, 0), allow_sleeping=False)
	with EcsPhysicsAdapter(ecs, shapes, settings=settings, interpolate=interpolate, ragdolls=ragdolls) as physics:
		if workload != "ragdolls": physics.set_velocity(owners, linear=(0.25, 0.1, 0) if workload == "isolated" else (0.1, 0, 0), angular=(0, 0.2, 0))
		start = time.perf_counter_ns()
		physics.reconcile()
		creation_ms = (time.perf_counter_ns() - start) / 1e6
		schedule = Scheduler(ecs)
		samples = {name: [] for name in ("reconcile", "prepare_inputs", "step", "pull", "dispatch_events")}
		def timed(name, callback):
			def run(world, commands, dt):
				start = time.perf_counter_ns()
				callback(world, commands, dt)
				samples[name].append((time.perf_counter_ns() - start) / 1e6)
			return run
		previous = ()
		for name in samples:
			schedule.add(name, timed(name, getattr(physics, name)), phase="fixed_update", after=previous)
			previous = name
		total, render, skin = [], [], []
		contact_events = contact_ticks = 0
		for tick in range(warmup + repeat):
			if tick == warmup:
				physics.world.reset_stats()
				baseline = physics.world.stats()
			start = time.perf_counter_ns()
			schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)
			total.append((time.perf_counter_ns() - start) / 1e6)
			if workload == "ragdolls":
				start = time.perf_counter_ns()
				palette = physics.ragdoll_skin(owners, alpha=0.5 if interpolate else None)
				skin.append((time.perf_counter_ns() - start) / 1e6)
				if not np.isfinite(palette).all(): raise RuntimeError("nonfinite benchmark ragdoll palette")
			elif interpolate:
				start = time.perf_counter_ns()
				physics.render_transforms(owners, alpha=0.5)
				render.append((time.perf_counter_ns() - start) / 1e6)
			if tick >= warmup:
				contacts = np.count_nonzero(physics.events["kind"] == pj.EventKind.CONTACT)
				contact_events += int(contacts)
				contact_ticks += int(contacts > 0)
		samples = {name: values[warmup:] for name, values in samples.items()}
		sync = np.sum([samples[name] for name in ("reconcile", "prepare_inputs", "pull", "dispatch_events")], axis=0)
		native = inspect_physics(physics.world)
		result = dict(workload=workload, instances=count, bodies=0 if ragdolls else count, ragdolls=count if ragdolls else 0, threads=threads, interpolate=interpolate, tick_hz=120, ticks=physics.world.tick, creation_ms=creation_ms)
		result.update(stages={name: summary(values) for name, values in samples.items()}, sync=summary(sync), fixed_tick=summary(total[warmup:]), native=native,
			contact_events=contact_events, contact_ticks=contact_ticks, events_dropped=native["stats"]["events_dropped"] - baseline["events_dropped"])
		if ragdolls:
			result["skinning"] = summary(skin[warmup:])
			result["bones_per_ragdoll"] = len(ragdolls[0].bones)
		else:
			result["io"] = measure_io(physics, owners, repeat)
			if interpolate: result["render_transforms"] = summary(render[warmup:])
		return result


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--workloads", choices=("isolated", "colliding", "compounds", "ragdolls"), nargs="+", default=["isolated"])
	parser.add_argument("--bodies", type=int, nargs="+", default=[100, 1000])
	parser.add_argument("--ragdolls", type=int, nargs="+", default=[10, 20])
	parser.add_argument("--threads", type=int, nargs="+", default=[1])
	parser.add_argument("--warmup", type=int, default=30)
	parser.add_argument("--repeat", type=int, default=240)
	parser.add_argument("--output", type=Path)
	args = parser.parse_args()
	if min(args.bodies) < 1 or min(args.ragdolls) < 1 or min(args.threads) < 1 or args.warmup < 0 or args.repeat < 1:
		parser.error("bodies, ragdolls, threads and repeat must be positive; warmup must be nonnegative")
	report = dict(python=platform.python_version(), numpy=np.__version__, pyjoltphysics=importlib.metadata.version("pyjoltphysics"), platform=platform.platform(), warmup=args.warmup, repeat=args.repeat)
	report["results"] = [benchmark(count, threads, interpolate, args.warmup, args.repeat, workload=workload) for workload in args.workloads
		for count in (args.ragdolls if workload == "ragdolls" else args.bodies) for threads in args.threads for interpolate in (False, True)]
	text = json.dumps(report, indent=2)
	if args.output is None: print(text)
	else: args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
