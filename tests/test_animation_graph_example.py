import numpy as np

from scenes.animation_graph_example import create_scene, step


def test_state_driven_layer_uses_owner_controls_and_outputs_skin_matrices():
	world, entities, rig, clips, control, animation, plan = create_scene()
	outputs = step(world, rig, clips, control, animation, plan, 0.25)
	matrices = {int(entity): matrix for owners, batch in outputs for entity, matrix in zip(owners, batch)}
	moving, idle = (matrices[int(entity)] for entity in entities)
	np.testing.assert_allclose(moving[:, :, 3], [[1, 0, 0], [1, 3, 0], [1, -1, 0]])
	np.testing.assert_allclose(idle[:, :, 3], [[0, 0, 0], [0, 1, 0], [0, -1, 0]])
	outputs = step(world, rig, clips, control, animation, plan, 0.25)
	matrices = {int(entity): matrix for owners, batch in outputs for entity, matrix in zip(owners, batch)}
	np.testing.assert_allclose(matrices[int(entities[0])][0, :, 3], [2, 0, 0])
