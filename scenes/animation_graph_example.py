"""Headless state-driven locomotion with a masked upper-body pose layer.

Run: python -m scenes.animation_graph_example
"""
import numpy as np

from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.ECS import ECS, component
from graaal.EcsAdapters.EcsAnimation import ClipPlayer, EcsAnimationAdapter, SkeletonRef
from graaal.EcsAdapters.EcsAnimationStateMachine import AnimationState, EcsAnimationStateAdapter
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, skin_matrices


@component
class AnimationControls:
	speed: np.float32 = 0
	upper_weight: np.float32 = 0


def create_scene():
	world = ECS()
	world.register(SkeletonRef, ClipPlayer, AnimationState, AnimationControls)
	entities = world.create(2)
	translation = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32)
	rotation = np.tile([0, 0, 0, 1], (3, 1))
	scale = np.ones((3, 3))
	rig = Skeleton([-1, 0, 0], translation, rotation, scale, np.tile(np.eye(4), (3, 1, 1)))
	idle, walk, upper = (translation.copy() for _ in range(3))
	walk[0, 0] = 2
	upper[1, 1] = 3
	clips = {name: Clip(pose[None], rotation[None], scale[None]) for name, pose in zip(("idle", "walk", "upper"), (idle, walk, upper))}
	definition = AnimationGraphDefinition({
		"version": 1,
		"parameters": {"speed": 0, "upper_weight": 0},
		"states": {"idle": {"clip": "idle"}, "walk": {"clip": "walk"}},
		"initial_state": "idle",
		"transitions": [
			{"source": "idle", "target": "walk", "conditions": [{"parameter": "speed", "comparison": ">", "threshold": 0.5}], "duration": 0.5},
			{"source": "walk", "target": "idle", "conditions": [{"parameter": "speed", "comparison": "<", "threshold": 0.2}], "duration": 0.5},
		],
		"sample_slots": ["locomotion", "upper"],
		"weight_slots": ["upper_weight"],
		"root_motion_slots": ["locomotion"],
		"masks": {"upper_body": {"root": 1}},
		"tasks": [
			{"name": "locomotion", "op": "sample", "slot": "locomotion"},
			{"name": "in_place", "op": "in_place", "base": "locomotion", "slot": "locomotion"},
			{"name": "upper", "op": "sample", "slot": "upper"},
			{"name": "output", "op": "blend", "base": "in_place", "overlay": "upper", "weight": "upper_weight", "mask": "upper_body"},
		],
	})
	graph = definition.compile(rig, clips)
	world.add(entities, SkeletonRef(0), ClipPlayer(graph.state_clips[graph.initial_state]), AnimationState(graph.initial_state), AnimationControls())
	world.get(AnimationControls)[entities].assign(speed=[1, 0], upper_weight=[1, 0])
	control = EcsAnimationStateAdapter(world, graph.machine, graph.state_clips)
	animation = EcsAnimationAdapter(world, {0: rig}, graph.clips)
	return world, entities, rig, graph.clips, control, animation, graph.plan


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
