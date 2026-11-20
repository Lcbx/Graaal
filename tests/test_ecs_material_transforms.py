"""Render transform sources operate in ECS owner order without simulation writes."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from graaal.ECS import ECS
from graaal.EcsMaterials import EcsMaterialAdapter, MaterialInstance
from graaal.MaterialTransform import MaterialTransform


def test_transform_source_refreshes_between_ticks_without_repacking_materials():
	ecs = ECS()
	ecs.register(MaterialInstance, MaterialTransform)
	owners = ecs.create(2)
	mesh, shader = object(), object()
	ecs.add(owners, MaterialInstance(mesh, {}, shader=shader), MaterialTransform())
	ecs.add(owners[0], MaterialInstance(mesh, {}, shader=shader))
	renderer = SimpleNamespace(mesh_ids={mesh: 0}, update=Mock(), update_transforms=Mock())
	alpha = 0.0
	def source(entities):
		position = ecs.get(MaterialTransform)[entities].position
		position[:, 0] = np.arange(len(entities)) + alpha
		return position, np.tile((0, 0, 0, 1), (len(entities), 1)), np.ones_like(position)
	adapter = EcsMaterialAdapter(renderer, ecs, transform_source=source)
	assert adapter.sync()
	np.testing.assert_array_equal(renderer.update.call_args.kwargs["owners"], [owners[0], owners[0], owners[1]])
	token = ecs.get(MaterialTransform).version()
	alpha = 0.5
	assert adapter.sync()
	assert renderer.update.call_count == 1
	np.testing.assert_allclose(renderer.update_transforms.call_args.args[0]["iPosition"][:, 0], [0.5, 1.5, 2.5])
	assert ecs.get(MaterialTransform).version() == token
	adapter.transform_source = None
	assert adapter.sync()
	np.testing.assert_allclose(renderer.update_transforms.call_args.args[0]["iPosition"], 0)
	assert adapter.sync() is False


def test_transform_source_rejects_broadcast_rows():
	ecs = ECS()
	ecs.register(MaterialInstance, MaterialTransform)
	owners = ecs.create(2)
	mesh, shader = object(), object()
	ecs.add(owners, MaterialInstance(mesh, {}, shader=shader), MaterialTransform())
	renderer = SimpleNamespace(mesh_ids={mesh: 0}, update=Mock())
	adapter = EcsMaterialAdapter(renderer, ecs, transform_source=lambda owners: (np.zeros((1, 3)), np.zeros((2, 4)), np.ones((2, 3))))
	with pytest.raises(ValueError, match="owner order"): adapter.sync()
	renderer.update.assert_not_called()
