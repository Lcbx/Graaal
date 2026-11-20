"""Standalone lockstep physics: python -m scenes.physics_headless_example."""

import argparse

import numpy as np


def run_scene(*, ticks=120, tick_hz=120.0, threads=1):
	"""Drop three unit boxes onto a floor; return owned arrays after shutdown.

	Requires the physics extra. Uses metres, seconds and xyzw quaternions.
	The caller controls simulated time through ticks; no wall-clock pacing.
	"""
	import pyjoltphysics as pj

	if not pj.selftest():
		raise RuntimeError("pyjoltphysics native selftest failed")
	positions = np.array([[-2, 2, 0], [0, 3, 0], [2, 4, 0]], dtype=np.float64)
	with pj.World(pj.WorldSettings(tick_hz=tick_hz, threads=threads)) as world:
		floor_uids, floor_seq = world.add_bodies(pj.BodyBatch.plane())
		uids, seq = world.add_bodies(pj.BodyBatch.boxes(positions, size=(1, 1, 1), mass=1.0))
		world.run_ticks(ticks)
		snapshot = world.latest()
		if snapshot is None or snapshot.applied_seq < max(floor_seq, seq):
			raise RuntimeError("Physics tick did not apply the body creation batches")
		rows = snapshot.rows_of(uids)
		if np.any(rows < 0):
			raise RuntimeError("Physics snapshot is missing a created box")
		# Copy before leaving: native snapshot views retain the world's buffer.
		result = dict(tick=snapshot.tick, time=snapshot.time, uids=uids.copy(), floor_uids=floor_uids.copy(), position=snapshot.position[rows].copy())
		result.update(events=world.drain_events().copy(), stats=world.stats())
	print(f"tick={result['tick']} time={result['time']:.3f}s heights={result['position'][:, 1].round(3)} events={len(result['events'])}")
	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--ticks", type=int, default=120)
	parser.add_argument("--tick-hz", type=float, default=120.0)
	parser.add_argument("--threads", type=int, default=1)
	args = parser.parse_args()
	run_scene(ticks=args.ticks, tick_hz=args.tick_hz, threads=args.threads)
