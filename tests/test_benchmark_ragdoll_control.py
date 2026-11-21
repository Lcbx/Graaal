"""The native control probe reports its real workload and copies observation state."""

import json

import numpy as np
import pytest

from scenes.benchmarks.benchmark_ragdoll_control import benchmark, observations


@pytest.fixture
def pj():
	return pytest.importorskip("pyjoltphysics")


@pytest.mark.parametrize("passive", (False, True))
@pytest.mark.parametrize("mode", ("limp", "powered"))
@pytest.mark.parametrize("soft_limits", (False, True))
def test_control_probe_counts_and_native_timing_window(pj, passive, mode, soft_limits):
	result = benchmark(2, 2, 3, 8, passive=passive, mode=mode, soft_limits=soft_limits)
	assert result["ticks"] == 13
	assert result["native"]["stats"]["timed_ticks"] == 8
	assert result["native"]["stats"]["bodies"] == 2 * result["bones_per_ragdoll"]
	assert result["native"]["stats"]["rejected"] == result["events_dropped"] == 0
	assert result["motor_strength_nm"] == 0
	assert (result["peak_body_torque_nm"] > 0) == passive
	assert result["soft_limits"] == bool(passive and soft_limits)
	assert (result["peak_limit_body_torque_nm"] > 0) == bool(passive and soft_limits)
	assert result["mode"] == mode
	assert all(stage["median_ms"] >= 0 and stage["p95_ms"] >= stage["median_ms"] for stage in result["stages"].values())
	json.dumps(result, allow_nan=False)


def test_observations_preserve_uid_order_duplicates_compaction_and_owned_storage(pj):
	with pj.World(pj.WorldSettings(threads=1, gravity=(0, 0, 0))) as world:
		unrelated, _ = world.add_bodies(pj.BodyBatch.boxes([[100, 100, 100]]))
		settings = pj.RagdollSettings.humanoid()
		a = world.add_ragdoll(settings, (0, 0, 0))
		b = world.add_ragdoll(settings, (4, 0, 0))
		world.run_ticks(1)
		world.remove_bodies(unrelated)
		world.run_ticks(1)
		uids = np.stack([b.bones, a.bones, b.bones])
		state = observations(world, uids)
		direct = world.bodies_state(uids.ravel())
		for name, value in state.items():
			np.testing.assert_allclose(value.reshape(-1, value.shape[-1]), getattr(direct, name), atol=1e-6)
			assert not np.shares_memory(value, getattr(world.latest(), name))
		before = state["position"].copy()
		world.add_impulses(b.bones[:1], [[1, 0, 0]])
		world.run_ticks(1)
		np.testing.assert_array_equal(state["position"], before)
		world.remove_ragdolls([b.id])
		world.flush()
		with pytest.raises(RuntimeError, match="missing bone"): observations(world, uids)


def test_zero_warmup_retains_all_control_probe_ticks(pj):
	result = benchmark(1, 1, 0, 2)
	assert result["ticks"] == 4
	assert result["native"]["stats"]["timed_ticks"] == 2
