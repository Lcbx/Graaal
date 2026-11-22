"""Standalone batched blend-state playback with a masked upper-body layer.

Run: python -m scenes.animation_graph_batch_example
"""

import numpy as np

from graaal.Animation.AnimationEvents import EventTrack
from graaal.Animation.AnimationGraph import AnimationGraphDefinition
from graaal.Animation.AnimationGraphPlayback import AnimationGraphPlayback
from graaal.Animation.SkeletalAnimation import Clip, Skeleton, model_matrices, sample_clip_pairs, skin_matrices


def create_scene():
	translation = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
	rotation, scale = np.tile([0, 0, 0, 1], (2, 1)), np.ones((2, 3))
	rig = Skeleton([-1, 0], translation, rotation, scale, np.tile(np.eye(4), (2, 1, 1)))
	clips = {}
	for name, distance, fps in (("idle", 0, 1), ("walk", 1, 1), ("run", 3, 2), ("upper", 0, 1)):
		poses = np.tile(translation, (3, 1, 1))
		poses[:, 0, 0] = np.linspace(0, distance, 3)
		if name == "upper": poses[:, 1, 1] = 3
		clips[name] = Clip(poses, np.tile(rotation, (3, 1, 1)), np.tile(scale, (3, 1, 1)), fps=fps)
	definition = AnimationGraphDefinition({
		"version": 1, "parameters": {"move": 0, "pace": 0.5, "upper_weight": 0},
		"states": {"idle": {"clip": "idle"}, "moving": {"blend_space": "locomotion"}},
		"initial_state": "idle",
		"blend_spaces": {"locomotion": {"parameter": "pace", "points": [{"position": 0, "clip": "walk"}, {"position": 1, "clip": "run"}], "sync_reference": "walk"}},
		"transitions": [
			{"source": source, "target": target, "conditions": [{"parameter": "move", "comparison": comparison, "threshold": 0.5}], "duration": 0.5}
			for source, target, comparison in (("idle", "moving", ">"), ("moving", "idle", "<"))
		],
		"sample_slots": ["locomotion", "upper"], "weight_slots": ["upper_weight"],
		"masks": {"upper_body": {"root": 1}},
		"tasks": [
			{"name": "base", "op": "sample", "slot": "locomotion"},
			{"name": "upper", "op": "sample", "slot": "upper"},
			{"name": "output", "op": "blend", "base": "base", "overlay": "upper", "weight": "upper_weight", "mask": "upper_body"},
		],
	})
	tracks = {name: EventTrack(clips[name].duration, [0, clips[name].duration / 2], ["left", "right"]) for name in ("walk", "run")}
	graph = definition.compile(rig, clips, sync_tracks=tracks)
	player = AnimationGraphPlayback(graph, 3)
	parameters = np.array([[1, 0, 0], [1, 1, 1], [0, 0.5, 0.5]])
	return player, parameters


def step(player, parameters, dt):
	"""Advance controls, compose a layer and return an independent skin palette."""
	player.advance(dt, parameters)
	graph, count = player.graph, player.count
	ids = np.full((count, 2), graph.clip_ids["upper"], dtype=np.int64)
	upper = sample_clip_pairs(graph.plan.skeleton, graph.clips, ids, np.zeros((count, 2)), np.tile([1, 0], (count, 1)))
	pose = graph.input_plan.evaluate(graph.clips, (), parameters[:, graph.weight_columns], poses=[player.pose(parameters), upper])
	return skin_matrices(graph.plan.skeleton, model_matrices(graph.plan.skeleton, *pose))


def run_scene():
	player, parameters = create_scene()
	for tick in range(4):
		if tick == 2: parameters[:, 0] = [0, 1, 1]
		matrices = step(player, parameters, 0.25)
		print(f"tick={tick} states={player.states} roots={matrices[:, 0, :, 3]} upper={matrices[:, 1, 1, 3]}")


if __name__ == "__main__":
	run_scene()
