"""Native workloads must measure collisions, routing and the intended presentation path."""

import json

import pytest

from graaal.EcsPhysics import Collider
from scenes.benchmarks.benchmark_physics import benchmark, make_workload


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
		assert "io" not in result and "render_transforms" not in result
	else:
		assert result["bodies"] == count
		assert set(result["io"]) == {"uid_lookup", "gather", "float64_to_float32", "scatter"}
		assert ("render_transforms" in result) == interpolate
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
