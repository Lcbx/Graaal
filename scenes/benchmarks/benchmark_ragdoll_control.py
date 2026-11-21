"""Probe batched native ragdoll observations, neutral springs and motor strengths."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from graaal.Diagnostics import inspect_physics
from graaal.Math import Quaternion
from graaal.PhysicsRagdoll import joint_limit_torques, joint_spring_torques, ragdoll_limit_parameters


def summary(samples):
	return dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))


def observations(world, uids):
	"""Copy one published snapshot in requested (instances, bones) UID order."""
	snapshot = world.latest()
	rows = snapshot.rows_of(uids.ravel())
	if np.any(rows < 0): raise RuntimeError("ragdoll observation contains missing bone UIDs")
	return {name: getattr(snapshot, name)[rows].reshape(uids.shape + (width,)) for name, width in (("position", 3), ("orientation", 4), ("linear_velocity", 3), ("angular_velocity", 3))}


def joint_error(parents, orientations, neutral):
	"""Shortest relative joint error in radians, excluding the unanchored root."""
	relative = Quaternion.multiply_batch(orientations[:, parents[1:]] * [-1, -1, -1, 1], orientations[:, 1:])
	error = Quaternion.multiply_batch(relative, neutral[None, 1:] * [-1, -1, -1, 1])
	return 2 * np.arctan2(np.linalg.norm(error[..., :3], axis=-1), np.abs(error[..., 3]))


def benchmark(count, threads, warmup, repeat, *, passive=True, mode="limp", soft_limits=False):
	import pyjoltphysics as pj

	if count < 1 or threads < 1 or warmup < 0 or repeat < 1: raise ValueError("counts, threads and repeat must be positive; warmup must be nonnegative")
	if mode not in ("limp", "powered"): raise ValueError("mode must be limp or powered")
	settings = pj.RagdollSettings.humanoid(self_collision=False, linear_damping=0, angular_damping=0, flags=pj.BodyFlag.NO_SLEEP)
	for bone in settings.bones: bone.friction = 0
	parents, neutral = settings.parents, settings.rest_rotations
	limit_parameters = ragdoll_limit_parameters(settings) if soft_limits else None
	pose = neutral.copy()
	pose[settings.bone_index("upper_arm_l")] = Quaternion.from_z_rotation(0.5)
	pose[settings.bone_index("upper_arm_r")] = Quaternion.from_z_rotation(-0.5)
	with pj.World(pj.WorldSettings(threads=threads, tick_hz=120, gravity=(0, 0, 0), allow_sleeping=False)) as world:
		handles = [world.add_ragdoll(settings, (i * 4, 0, 0), mode=pj.RagdollMode.KINEMATIC) for i in range(count)]
		uids = np.stack([handle.bones for handle in handles])
		for handle in handles: handle.set_pose(pose)
		world.run_ticks(2)
		for handle in handles:
			handle.set_pose(neutral)
			handle.set_strength(np.zeros(len(settings)))
			handle.set_mode(pj.RagdollMode.LIMP if mode == "limp" else pj.RagdollMode.POWERED)
		world.set_velocities(uids.ravel(), linear=np.zeros((uids.size, 3)), angular=np.zeros((uids.size, 3)))
		world.flush()
		world.drain_events()
		initial_error = float(np.mean(joint_error(parents, observations(world, uids)["orientation"], neutral)))
		samples = {name: [] for name in ("observations", "spring_torques", "limit_torques", "torque_submission", "strength_submission", "step", "events", "fixed_tick")}
		strength = np.zeros(len(settings))
		peak_torque = 0.0
		peak_limit_torque = 0.0
		event_count = 0
		for tick in range(warmup + repeat):
			if tick == warmup:
				world.reset_stats()
				baseline = world.stats()
			start = time.perf_counter_ns()
			state = observations(world, uids)
			observed = time.perf_counter_ns()
			torque = joint_spring_torques(parents, state["orientation"], state["angular_velocity"], neutral, stiffness=1, damping=0.05, max_torque=0.5) if passive else np.zeros(uids.shape + (3,))
			sprung = time.perf_counter_ns()
			limits = joint_limit_torques(orientations=state["orientation"], angular_velocity=state["angular_velocity"], margin=0.1, stiffness=1, damping=0.05, max_torque=0.5, **limit_parameters) if passive and soft_limits else None
			if limits is not None: torque += limits
			computed = time.perf_counter_ns()
			if passive: world.add_torques(uids.ravel(), torque.reshape(-1, 3))
			submitted = time.perf_counter_ns()
			for handle in handles: handle.set_strength(strength)
			strengthened = time.perf_counter_ns()
			world.run_ticks(1)
			stepped = time.perf_counter_ns()
			events = world.drain_events()
			end = time.perf_counter_ns()
			if tick >= warmup:
				for name, duration in zip(samples, (observed - start, sprung - observed, computed - sprung, submitted - computed, strengthened - submitted, stepped - strengthened, end - stepped, end - start)):
					samples[name].append(duration / 1e6)
				peak_torque = max(peak_torque, float(np.max(np.linalg.norm(torque, axis=-1))))
				if limits is not None: peak_limit_torque = max(peak_limit_torque, float(np.max(np.linalg.norm(limits, axis=-1))))
				event_count += len(events)
		state = observations(world, uids)
		if not all(np.isfinite(value).all() for value in state.values()): raise RuntimeError("nonfinite ragdoll state")
		native = inspect_physics(world)
		result = dict(ragdolls=count, bones_per_ragdoll=len(settings), threads=threads, passive=passive, mode=mode, tick_hz=120, ticks=world.tick,
			spring=dict(stiffness_nm_rad=1, damping_nm_s_rad=0.05, max_joint_torque_nm=0.5), motor_strength_nm=0,
			soft_limits=bool(passive and soft_limits), limit_controls=dict(margin_rad=0.1, stiffness_nm_rad=1, damping_nm_s_rad=0.05, max_joint_torque_nm=0.5) if passive and soft_limits else None,
			stages={name: summary(values) for name, values in samples.items()}, initial_joint_error_rad=initial_error, final_joint_error_rad=float(np.mean(joint_error(parents, state["orientation"], neutral))),
			peak_body_torque_nm=peak_torque, peak_limit_body_torque_nm=peak_limit_torque, event_records=event_count, events_dropped=native["stats"]["events_dropped"] - baseline["events_dropped"], native=native)
		# Isolate direct locked readback from the snapshot-copy path without stepping.
		direct = []
		for _ in range(repeat):
			start = time.perf_counter_ns()
			bodies = world.bodies_state(uids.ravel())
			direct.append((time.perf_counter_ns() - start) / 1e6)
			if not bodies.found.all(): raise RuntimeError("direct readback contains missing bone UIDs")
		result["direct_readback"] = summary(direct)
		return result


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--ragdolls", type=int, nargs="+", default=[10, 20])
	parser.add_argument("--threads", type=int, nargs="+", default=[1])
	parser.add_argument("--modes", choices=("limp", "powered"), nargs="+", default=["limp", "powered"])
	parser.add_argument("--warmup", type=int, default=60)
	parser.add_argument("--repeat", type=int, default=240)
	parser.add_argument("--soft-limits", action="store_true", help="add progressive swing/twist resistance in passive cases")
	parser.add_argument("--output", type=Path)
	args = parser.parse_args()
	if min(args.ragdolls) < 1 or min(args.threads) < 1 or args.warmup < 0 or args.repeat < 1: parser.error("counts, threads and repeat must be positive; warmup must be nonnegative")
	report = dict(python=platform.python_version(), numpy=np.__version__, pyjoltphysics=importlib.metadata.version("pyjoltphysics"), platform=platform.platform(), warmup=args.warmup, repeat=args.repeat)
	report["results"] = [benchmark(count, threads, args.warmup, args.repeat, passive=passive, mode=mode, soft_limits=args.soft_limits) for count in args.ragdolls for threads in args.threads for mode in args.modes for passive in (False, True)]
	text = json.dumps(report, indent=2, allow_nan=False)
	if args.output is None: print(text)
	else: args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
	main()
