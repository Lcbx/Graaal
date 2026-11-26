import numpy as np
import pytest

pj = pytest.importorskip("pyjoltphysics")
mujoco = pytest.importorskip("mujoco")

from pyjoltphysics.ragdolls import forward_kinematics
from tools.ragdoll_genesis_rig import export_rig
from tools.ragdoll_project import default_project
from tools.ragdoll_genesis_train import initialize_balance, warm_start_foot_features


def test_export_preserves_rest_positions_and_masses():
	project = default_project()
	xml, actions = export_rig(project)
	model = mujoco.MjModel.from_xml_string(xml)
	data = mujoco.MjData(model)
	mujoco.mj_forward(model, data)
	positions, _ = forward_kinematics(project.settings, None, (0, 0, 0), (0, 0, 0, 1))
	for index, bone in enumerate(project.settings.bones):
		body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bone.name)
		assert body_id > 0
		np.testing.assert_allclose(data.xpos[body_id], positions[index], atol=1e-10)
		assert model.body_mass[body_id] == pytest.approx(bone.mass)
	assert len(actions) == 30
	assert model.nv == 6 + len(actions)
	assert len({action["name"] for action in actions}) == len(actions)


def test_balance_initialization_matches_feedback_near_stance():
	torch = pytest.importorskip("torch")
	from types import SimpleNamespace

	_, actions = export_rig(default_project())
	policy = SimpleNamespace(mlp=torch.nn.Sequential(torch.nn.Linear(133, 256), torch.nn.ELU(), torch.nn.Linear(256, 256), torch.nn.ELU(), torch.nn.Linear(256, 30)))
	initialize_balance(policy, actions)
	observation = torch.zeros((1, 133))
	observation[0, -4:] = torch.tensor([0.01, -0.01, 0.02, -0.02])
	result = policy.mlp(observation).detach().numpy()[0]
	for index, action in enumerate(actions):
		if action["bone"] in ("foot_l", "foot_r") and action["axis_index"] == 1:
			assert result[index] == pytest.approx(0.06, abs=0.001)
		elif action["bone"] in ("foot_l", "foot_r") and action["axis_index"] == 2:
			assert result[index] == pytest.approx(-0.06, abs=0.001)
		else:
			assert result[index] == 0


def test_foot_feature_warm_start_preserves_outputs(monkeypatch):
	torch = pytest.importorskip("torch")
	from types import SimpleNamespace

	class Model(torch.nn.Module):
		def __init__(self, inputs, outputs):
			super().__init__()
			self.mlp = torch.nn.Sequential(torch.nn.Linear(inputs, 16), torch.nn.ELU(), torch.nn.Linear(16, outputs))

	old_actor, old_critic = Model(133, 30), Model(133, 1)
	new_actor, new_critic = Model(147, 30), Model(147, 1)
	checkpoint = {"actor_state_dict": old_actor.state_dict(), "critic_state_dict": old_critic.state_dict()}
	monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
	runner = SimpleNamespace(alg=SimpleNamespace(actor=new_actor, critic=new_critic))
	warm_start_foot_features(runner, "unused")
	old = torch.randn((3, 133))
	expanded = torch.cat((old[:, :-4], torch.randn((3, 14)), old[:, -4:]), dim=1)
	torch.testing.assert_close(new_actor.mlp(expanded), old_actor.mlp(old))
	torch.testing.assert_close(new_critic.mlp(expanded), old_critic.mlp(old))
