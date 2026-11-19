import numpy as np

from scenes.benchmarks.benchmark_animation import benchmark, make_workload


def test_benchmark_workloads_exercise_single_clip_and_crossfade():
	for crossfade in (False, True):
		adapter = make_workload(3, 4, crossfade)
		poses = adapter.update(1 / 60)
		assert poses[0].matrices.shape == (3, 4, 3, 4)
		weights = adapter.sample_batches[0].weights[:, 1]
		if crossfade: np.testing.assert_allclose(weights, 0.5, atol=1e-6)
		else: np.testing.assert_array_equal(weights, 0)
		result = benchmark(3, 4, crossfade, warmup=0, repeat=2)
		assert result["mode"] == ("crossfade" if crossfade else "single")
		assert all(result[name]["median_ms"] >= 0 for name in ("control", "pose", "combined"))
