"""Merged animation graph, accepted motion, collision IK and representation hand-off."""

import numpy as np
import pytest

from graaal.EcsPhysics import CharacterState
from graaal.EcsAnimation import ClipPlayer
from graaal.MaterialTransform import MaterialTransform
from graaal.Math import Quaternion
from graaal.SkeletalAnimation import model_matrices


def advance(schedule, count=1):
	for _ in range(count): schedule.run(1 / 120, fixed_steps=1, fixed_dt=1 / 120)


def test_graph_character_cycle_uses_collision_motion_foot_ik_and_one_representation():
	pj = pytest.importorskip("pyjoltphysics")
	from scenes.character_example import create_demo
	ecs, owner, environment, physics, schedule, state, asset = create_demo()
	stages = set()
	jumped, cleared_hurdle = False, False
	with physics:
		for _ in range(960):
			advance(schedule)
			stages.add(state["stage"])
			assert (int(owner) in physics.characters) != (int(owner) in physics.ragdolls)
			assert len(state["batches"]) == 1
			assert state["batches"][0].entities.tolist() == [owner]
			assert np.isfinite(state["batches"][0].matrices).all()
			jumped |= np.any((physics.events["kind"] == pj.EventKind.CHARACTER) & (physics.events["phase"] == pj.EventPhase.JUMP))
			if state["stage"] == "controller":
				position = ecs.get(MaterialTransform)[[owner]].position[0]
				cleared_hurdle |= abs(position[0] - 3) < 0.05 and position[1] > 0.75
				ground = ecs.get(CharacterState)[[owner]]
				assert not (ground.grounded[0] and ground.ground[0] == environment[1])
		assert stages == {"controller", "settling", "ragdoll"}
		assert state["stage"] == "controller"
		assert state["requested_speed"] == pytest.approx(2, abs=0.01)
		assert state["accepted_speed"] > 1.9
		assert ecs.get(MaterialTransform)[[owner]].position[0, 0] > 4
		assert jumped and cleared_hurdle
		assert state["trigger_begins"] > 0 and state["trigger_ends"] > 0
		assert np.all(state["foot_hits"])
		model = model_matrices(asset.skeleton, *state["final_pose"])
		feet = model[0, [9, 12], :3, 3] + ecs.get(MaterialTransform)[[owner]].position[0]
		np.testing.assert_allclose(feet[:, 1], 0.06, atol=0.01)
	assert physics.world.reserved_bodies == 0


@pytest.mark.parametrize("direction", ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (1, 0, -1)))
def test_character_faces_movement_and_keeps_foot_ik_in_local_space(direction):
	pytest.importorskip("pyjoltphysics")
	from scenes.character_example import create_demo
	ecs, owner, environment, physics, schedule, state, asset = create_demo()
	state["auto"], state["direction"] = False, np.array(direction)
	ecs.get(MaterialTransform)[[owner]].position = (-8, 0, 0)
	with physics:
		advance(schedule, 5)
		frame = ecs.get(MaterialTransform)[[owner]]
		turn = Quaternion.matrix_batch(frame.rotation)[0]
		np.testing.assert_allclose(turn @ [0, 0, 1], np.array(direction) / np.linalg.norm(direction), atol=1e-6)
		assert np.all(state["foot_hits"])
		model = model_matrices(asset.skeleton, *state["final_pose"])
		feet = model[0, [9, 12], :3, 3] @ turn.T + frame.position[0]
		np.testing.assert_allclose(feet[:, 1], 0.06, atol=0.01)
		rotation = frame.rotation.copy()
		state["direction"] = np.zeros(3)
		advance(schedule)
		np.testing.assert_allclose(ecs.get(MaterialTransform)[[owner]].rotation, rotation)


def test_manual_handoff_preserves_motion_and_recovers_near_physical_root():
	pytest.importorskip("pyjoltphysics")
	from scenes.character_example import create_demo
	ecs, owner, environment, physics, schedule, state, asset = create_demo()
	state["auto"] = False
	with physics:
		advance(schedule, 30)
		velocity = physics.characters[int(owner)].velocity.copy()
		assert velocity[0] > 1.9
		state["fall"] = True
		advance(schedule)
		assert state["stage"] == "settling"
		np.testing.assert_allclose(state["launch"], velocity)
		advance(schedule, 2)
		assert state["stage"] == "ragdoll"
		assert not physics.characters
		handle = physics.ragdolls[int(owner)]
		native_velocity = physics.world.bodies_state(handle.bones[:1]).linear_velocity[0]
		assert native_velocity[0] > 1.5
		advance(schedule, 20)
		position, _ = handle.pose()
		velocity = physics.world.bodies_state(handle.bones[:1]).linear_velocity[0].copy()
		state["recover"] = True
		advance(schedule)
		assert state["stage"] == "controller"
		assert not physics.ragdolls
		feet = ecs.get(MaterialTransform)[[owner]].position[0]
		np.testing.assert_allclose(feet[[0, 2]], position[0, [0, 2]], atol=0.08)
		accepted = ecs.get(CharacterState)[[owner]].velocity[0]
		assert np.linalg.norm(accepted[[0, 2]] - velocity[[0, 2]]) < 0.5


def test_merged_graph_crossfades_between_idle_and_walk_without_a_second_clock():
	pytest.importorskip("pyjoltphysics")
	from scenes.character_example import create_demo
	ecs, owner, environment, physics, schedule, state, asset = create_demo()
	state["auto"] = False
	with physics:
		advance(schedule, 30)
		state["direction"] = np.zeros(3)
		advance(schedule, 30)
		assert ecs.get(ClipPlayer)[[owner]].clip_id[0] == 0
		assert not ecs.get(ClipPlayer)[[owner]].transitioning[0]
		assert state["accepted_speed"] < 0.01
		state["direction"] = np.array([1, 0, 0])
		advance(schedule, 30)
		assert ecs.get(ClipPlayer)[[owner]].clip_id[0] == 1
		assert state["requested_speed"] > 1.9
		assert state["accepted_speed"] > 1.9


def test_recovery_waits_until_the_standing_capsule_has_clearance():
	pytest.importorskip("pyjoltphysics")
	from graaal.EcsPhysics import Collider
	from scenes.character_example import create_demo
	ecs, owner, environment, physics, schedule, state, asset = create_demo()
	state["auto"] = False
	with physics:
		advance(schedule, 30)
		state["fall"] = True
		advance(schedule, 123)
		position, _ = physics.ragdolls[int(owner)].pose()
		ceiling = ecs.create(1)[0]
		ecs.add(ceiling, Collider(0), MaterialTransform(position=(position[0, 0], 1.3, position[0, 2]), scale=(4, 0.1, 4)))
		advance(schedule)
		state["recover"] = True
		advance(schedule)
		assert state["stage"] == "ragdoll"
		assert not physics.characters
		ecs.delete(ceiling)
		advance(schedule)
		state["recover"] = True
		advance(schedule)
		assert state["stage"] == "controller"
