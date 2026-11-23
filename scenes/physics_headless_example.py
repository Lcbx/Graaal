"""Standalone lockstep physics: python -m scenes.physics_headless_example."""

import argparse

import numpy as np


def run_scene(*, ticks=120, tick_hz=120.0, threads=1, on_pose=None):
	"""Drop three unit boxes onto a floor; return owned arrays after shutdown.

	Requires the physics extra. Uses metres, seconds and xyzw quaternions.
	The caller controls simulated time through ticks; no wall-clock pacing.
	on_pose receives box positions and orientations at tick zero and after each step.
	"""
	import pyjoltphysics as pj

	if not pj.selftest():
		raise RuntimeError("pyjoltphysics native selftest failed")
	positions = np.array([[-2, 2, 0], [0, 3, 0], [2, 4, 0]], dtype=np.float64)
	with pj.World(pj.WorldSettings(tick_hz=tick_hz, threads=threads)) as world:
		floor_uids, floor_seq = world.add_bodies(pj.BodyBatch.plane())
		uids, seq = world.add_bodies(pj.BodyBatch.boxes(positions, size=(1, 1, 1), mass=1.0))
		if on_pose is None:
			world.run_ticks(ticks)
		else:
			on_pose(positions.copy(), np.tile((0, 0, 0, 1), (len(uids), 1)))
			for _ in range(ticks):
				world.run_ticks(1)
				state = world.bodies_state(uids)
				on_pose(state.position, state.orientation)
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


def run_viewer(*, ticks=120, tick_hz=120.0, threads=1, offscreen=False):
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import DrawInstance, MaterialMesh
	from graaal.Math import Camera
	from scenes.physics_geometry import cube_vertices
	from scenes.physics_viewing_utils import has_blue_pixels, view_poses

	samples = []
	result = run_scene(ticks=ticks, tick_hz=tick_hz, threads=threads, on_pose=lambda position, rotation: samples.append((position, rotation)))
	mesh = MaterialMesh(cube_vertices(), DefaultMaterialShader())
	boxes = [DrawInstance(mesh, material={"base_color": (0.12, 0.55, 0.95, 1)}) for _ in result["uids"]]
	floor = DrawInstance(mesh, material={"base_color": (0.3, 0.3, 0.34, 1)}, position=(0, -0.25, 0), scale=(12, 0.5, 8))
	camera = Camera((7, 5, 9), (0, 1.75, 0), (0, 1, 0), 42, near=0.05, far=60)
	view_poses("Falling boxes", camera, samples, boxes, sample_hz=tick_hz, static=[floor], offscreen=offscreen, check_frame=has_blue_pixels)
	return result


if __name__ == "__main__":
	from scenes.physics_viewing_utils import add_viewing_options

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--ticks", type=int, default=120)
	parser.add_argument("--tick-hz", type=float, default=120.0)
	parser.add_argument("--threads", type=int, default=1)
	add_viewing_options(parser)
	args = parser.parse_args()
	if args.headless:
		run_scene(ticks=args.ticks, tick_hz=args.tick_hz, threads=args.threads)
	else:
		run_viewer(ticks=args.ticks, tick_hz=args.tick_hz, threads=args.threads, offscreen=args.offscreen)
