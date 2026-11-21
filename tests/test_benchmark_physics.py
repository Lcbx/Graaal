"""Native workloads must measure collisions, routing and the intended presentation path."""

import json

import numpy as np
import pytest

from graaal.EcsAdapters.EcsPhysics import Collider, EcsPhysicsAdapter
from scenes.benchmarks.benchmark_physics import benchmark, make_workload, measure_ragdoll_inputs


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


@pytest.mark.parametrize("workload", ("isolated", "colliding", "compounds", "ragdolls"))
@pytest.mark.parametrize("interpolate", (False, True))
def test_workload_contacts_native_window_and_presentation(pj, workload, interpolate):
	count = 2 if workload == "ragdolls" else 8
	result = benchmark(count, 1, interpolate, 30, 16, workload=workload)
	assert result["workload"] == workload
	assert result["instances"] == count
	assert result["ticks"] == 46
	assert result["native"]["stats"]["timed_ticks"] == 16
	assert result["events_dropped"] == 0
	assert result["creation_ms"] >= 0
	assert "dispatch_events" in result["stages"]
	assert all(stage["median_ms"] >= 0 and stage["p95_ms"] >= stage["median_ms"] for stage in result["stages"].values())
	assert result["sync"]["median_ms"] <= result["fixed_tick"]["median_ms"]
	if workload == "isolated":
		assert result["contact_events"] == result["contact_ticks"] == 0
		assert result["native"]["stats"]["bodies"] == count
	else:
		assert result["contact_events"] > 0
		assert 0 < result["contact_ticks"] <= 16
	if workload == "ragdolls":
		assert result["ragdolls"] == count
		assert result["native"]["stats"]["bodies"] == count * result["bones_per_ragdoll"] + 1
		assert result["skinning"]["median_ms"] > 0
		assert set(result["ragdoll_inputs"]) == {"conversion_per_owner", "conversion_batched", "native_submission", "set_pose", "prepare_updated", "update_and_prepare", "prepare_held"}
		assert all(stage["median_ms"] >= 0 and stage["p95_ms"] >= stage["median_ms"] for stage in result["ragdoll_inputs"].values())
		assert "io" not in result and "render_transforms" not in result
	else:
		assert result["bodies"] == count
		assert set(result["io"]) == {"uid_lookup", "gather", "float64_to_float32", "scatter"}
		assert ("render_transforms" in result) == interpolate
		assert "ragdoll_inputs" not in result
	json.dumps(result, allow_nan=False)


def test_compound_workload_has_two_local_shapes_per_body(pj):
	ecs, owners, shapes, ragdolls = make_workload(8, "compounds", pj)
	assert ragdolls is None
	assert len(shapes) == 2
	assert len(ecs.get(Collider)[owners]) == 16


def test_zero_warmup_keeps_all_measured_ticks(pj):
	result = benchmark(2, 2, False, 0, 2)
	assert result["ticks"] == result["native"]["stats"]["timed_ticks"] == 2
	assert result["native"]["stats"]["events_held"] == 0


def test_ragdoll_input_probe_bounds_queues_without_stepping_or_draining_events(pj):
	ecs, owners, shapes, ragdolls = make_workload(2, "ragdolls", pj)
	with EcsPhysicsAdapter(ecs, shapes, settings=pj.WorldSettings(threads=1, allow_sleeping=False), ragdolls=ragdolls) as physics:
		for _ in range(30):
			physics.reconcile()
			physics.prepare_inputs()
			physics.step()
			physics.pull()
			physics.dispatch_events()
		stats = physics.world.stats()
		positions = physics.world.latest().position.copy()
		events = physics.events.copy()
		result = measure_ragdoll_inputs(physics, owners, ragdolls[0], 3)
		assert result["update_and_prepare"]["median_ms"] >= result["set_pose"]["median_ms"]
		after = physics.world.stats()
		assert physics.world.tick == 30
		for name in ("timed_ticks", "events_held", "events_dropped", "rejected"):
			assert after[name] == stats[name]
		np.testing.assert_array_equal(physics.world.latest().position, positions)
		np.testing.assert_array_equal(physics.events, events)
		assert physics.world.latest().applied_seq == physics.last_seq
