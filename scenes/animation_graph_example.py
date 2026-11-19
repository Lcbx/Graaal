"""Headless state-driven locomotion with a masked upper-body pose layer.

Run: python -m scenes.animation_graph_example
"""
import numpy as np

from graaal.AnimationStateMachine import AnimationStateMachine
from graaal.ECS import ECS, component
from graaal.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.EcsAnimationStateMachine import AnimationState, EcsAnimationStateAdapter
from graaal.PoseTasks import PosePlan
from graaal.SkeletalAnimation import Clip, Skeleton, model_matrices, skin_matrices, subtree_mask


@component
class AnimationControls:
	speed: np.float32 = 0
	upper_weight: np.float32 = 0


def create_scene():
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, AnimationState, AnimationControls)
	entities = world.create(2)
	world.add(entities, SkeletonRef(0), ClipPlayer(0), AnimationState(0), AnimationControls())
	world.get(AnimationControls)[entities].assign(speed=[1, 0], upper_weight=[1, 0])
	translation = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
	rotation = np.tile([0, 0, 0, 1], (3, 1))
	scale = np.ones((3, 3))
	rig = Skeleton([-1, 0, 0], translation, rotation, scale, np.tile(np.eye(4), (3, 1, 1)))
	idle, walk, upper = (translation.copy() for _ in range(3))
	walk[0, 0] = 2
	upper[1, 1] = 3
	clips = {i: Clip(pose[None], rotation[None], scale[None]) for i, pose in enumerate((idle, walk, upper))}
	machine = AnimationStateMachine([(0, 1, 0, ">", 0.5, 0.5), (1, 0, 0, "<", 0.2, 0.5)])
	control = EcsAnimationStateAdapter(world, machine, {0: 0, 1: 1})
	animation = EcsAnimationAdapter(world, {0: rig}, clips)
	plan = PosePlan(rig, [("sample", 0), ("in_place", 0, 0), ("sample", 1), ("blend", 1, 2, 0, subtree_mask(rig, 1))])
	return world, entities, rig, clips, control, animation, plan


def step(world, rig, clips, control, animation, plan, dt):
	"""Return entity handles and skin matrices for each animation batch."""
	entities = control.query.fetch().entities
	parameters = world.get(AnimationControls)[entities].speed[:, None]
	control.update(parameters)
	animation.update(dt, evaluate_poses=False)
	outputs = []
	for batch in animation.sample_batches:
		count = len(batch.entities)
		upper_ids = np.full((count, 2), 2, dtype=np.uint32)
		upper_times = np.zeros((count, 2))
		upper_weights = np.tile([1, 0], (count, 1))
		layer_weight = world.get(AnimationControls)[batch.entities].upper_weight[:, None]
		samples = [(batch.clip_ids, batch.times, batch.weights), (upper_ids, upper_times, upper_weights)]
		pose = plan.evaluate(clips, samples, layer_weight, root_motions=[animation.in_place_motion(batch)])
		matrices = skin_matrices(rig, model_matrices(rig, *pose))
		outputs.append((batch.entities.copy(), matrices))
	return outputs


def run_scene():
	world, entities, rig, clips, control, animation, plan = create_scene()
	for tick in range(4):
		if tick == 2: world.get(AnimationControls)[entities[:1]].assign(speed=0, upper_weight=0)
		for owners, matrices in step(world, rig, clips, control, animation, plan, 0.25):
			for entity, root, upper in zip(owners, matrices[:, 0, :, 3], matrices[:, 1, :, 3]):
				print(f"tick={tick} entity={int(entity)} root={root} upper={upper}")


if __name__ == "__main__":
	run_scene()
