"""Headless ECS pre/post-physics pose tasks over an analytic terrain query.

Run: python -m scenes.foot_ik_example [--plant]
"""
import argparse
from dataclasses import field

import numpy as np

from graaal.ECS import ECS, Scheduler, component
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.Animation.InverseKinematics import foot_plant
from graaal.Animation.PoseTasks import PoseOp, PosePlan
from graaal.Animation.RootMotion import RootMotion
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, skin_matrices
from graaal.Math import Vec3


@component
class Character:
	x: np.float32 = 0
	ik_weight: np.float32 = 1
	locked: bool = False
	anchor: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
	anchor_normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))


def terrain_height(x):
	"""Illustrative ground query in world space; no physics backend is used."""
	return 0.25 * np.sin(x)


def terrain_normal(x):
	"""Normal of the example height function in world space."""
	return np.column_stack((-0.25 * np.cos(x), np.ones_like(x), np.zeros_like(x)))


def create_demo(ground_query=None, *, plant=False):
	"""Optional ground_query(world_points) returns heights and normals per foot."""
	rig = Skeleton([-1, 0, 1], [[0, 2, 0], [0, -1, 0], [0, -1, 0]], [[0, 0, 0, 1]] * 3, np.ones((3, 3)), np.tile(np.eye(4), (3, 1, 1)))
	translations = np.tile(rig.translation, (3, 1, 1))
	translations[:, 0, 0] = [0, 0.25, 0.5]
	clip = Clip(translations, np.tile(rig.rotation, (3, 1, 1)), np.tile(rig.scale, (3, 1, 1)), fps=1)
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, Character)
	entities = world.create(2)
	world.add(entities, SkeletonRef(0), ClipPlayer(0, extract_root_motion=True), Character())
	world.get(Character)[entities].assign(x=[0, 0.5], ik_weight=[1, 0])
	animation = EcsAnimationAdapter(world, {0: rig}, {0: clip}, root_motions={(0, 0): RootMotion(rig, clip)})
	pre = PosePlan(rig, [(PoseOp.SAMPLE, 0), (PoseOp.IN_PLACE, 0, 0)])
	post = PosePlan(rig, [(PoseOp.INPUT, 0), (PoseOp.TWO_BONE_IK, 0, (0, 1, 2), 0, 0), (PoseOp.ALIGN_AXIS, 1, 2, 1, 0, (0, 1, 0))])
	frames = {"pre": [], "goals": [], "ground_hits": [], "batches": []}

	def prepare_pose(world, commands, dt):
		frames["pre"] = []
		for batch in animation.sample_batches:
			samples = [(batch.clip_ids, batch.times, batch.weights)]
			pose = pre.evaluate({0: clip}, samples, np.empty((len(batch.entities), 0)), root_motions=[animation.in_place_motion(batch)])
			frames["pre"].append((batch.entities.copy(), tuple(value.copy() for value in pose)))

	def physics(world, commands, dt):
		for request in animation.root_motion_batches:
			character = world.get(Character)[request.entities]
			character.assign(x=character.x + request.matrices[:, 0, 3])
		frames["goals"] = []
		frames["ground_hits"] = []
		for owners, pose in frames["pre"]:
			character = world.get(Character)[owners]
			model = model_matrices(rig, *pose)
			root = model[:, 0, :3, 3]
			target = model[:, 2, :3, 3].copy()
			points = target.copy()
			points[:, 0] += character.x
			query_points = points
			if plant:
				stance = (character.ik_weight > 0) & (world.get(ClipPlayer)[owners].time < 0.75)
				query_points = np.where((stance & character.locked)[:, None], character.anchor, points)
			if ground_query is None:
				height, normal = terrain_height(query_points[:, 0]), terrain_normal(query_points[:, 0])
			else:
				height, normal = ground_query(query_points)
			valid = np.isfinite(height) & np.all(np.isfinite(normal), axis=1) & (np.linalg.norm(normal, axis=1) > 1e-6)
			points[valid, 1] = height[valid]
			normal = np.where(valid[:, None], normal, [0, 1, 0])
			if plant:
				stance &= valid
				points, normal = foot_plant(points, normal, stance, character.locked, character.anchor, character.anchor_normal)
				character.assign(locked=stance, anchor=points, anchor_normal=normal)
			points[:, 0] -= character.x
			target = points
			pole = root + [0, 0, 1]
			frames["goals"].append(((target, pole), (normal,)))
			frames["ground_hits"].append(valid)

	def finalize_pose(world, commands, dt):
		frames["batches"] = []
		for (owners, base), goals, hits in zip(frames["pre"], frames["goals"], frames["ground_hits"]):
			weights = world.get(Character)[owners].ik_weight[:, None] * hits[:, None]
			pose = post.evaluate({}, [], weights, poses=[base], goals=goals)
			matrices = skin_matrices(rig, model_matrices(rig, *pose))
			frames["batches"].append((owners, matrices))

	schedule = Scheduler(world)
	schedule.add("animation", animation.advance, phase="fixed_update")
	schedule.add("pre_pose", prepare_pose, phase="fixed_update", after="animation")
	schedule.add("terrain", physics, phase="fixed_update", after="pre_pose")
	schedule.add("post_pose", finalize_pose, phase="fixed_update", after="terrain")
	return world, entities, animation, schedule, frames


def run_scene(*, plant=False):
	world, entities, animation, schedule, frames = create_demo(plant=plant)
	for tick in range(8):
		schedule.run(0.25, fixed_steps=1, fixed_dt=0.25)
		for owners, matrices in frames["batches"]:
			for entity, foot in zip(owners, matrices[:, 2, :, 3]):
				x = world.get(Character)[[entity]].x[0]
				print(f"tick={tick} entity={int(entity)} x={x:.3f} ground={terrain_height(x):.3f} foot_y={foot[1]:.3f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--plant", action="store_true", help="hold each supported foot during the stance window")
	run_scene(plant=parser.parse_args().plant)
