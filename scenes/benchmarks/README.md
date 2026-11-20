# Benchmarks and diagnostics

Run these commands from the repository root after `pip install -e .` (and the
`render`, `assets`, `ui`, and `examples` extras for GPU tests). CPU scripts use
the project's core dependencies. GPU scripts need a working WGPU backend; `benchmark_ecs_uploads`
opens a window, whereas the transparency diagnostics render offscreen.
Add `--help` to an entry point to see all supported options. Output files are
chosen by the caller; benchmarking does not modify the engine's defaults.

## ECS and CPU data access

```powershell
python -m scenes.benchmarks.benchmark_ecs --entities 100000 --repeat 11 --loops 50 --output ecs_cpu.json
python -m scenes.benchmarks.benchmark_ecs_uploads --frames 120 --output ecs_uploads.json
python tests/benchmark_array_access.py
```

`benchmark_ecs` reports query and mutation timings, gather/scatter costs and
array sizes. `benchmark_ecs_uploads` measures CPU-side uploads in a bounded
`direct_draw_example` run; its durations are not GPU frame times.
`benchmark_array_access` compares ordinary reads, reusable output arrays and
dense edits. Use `--implementation /path/to/compatible/ECS.py` with
`benchmark_ecs` to compare against a *standalone* ECS implementation. The
`benchmark_ecs_query_pairs` script similarly accepts a required
`--implementation` and `--output`; a standalone historical snapshot is needed
because the current `ECS.py` has package-relative imports.

`python -m scenes.benchmarks.benchmark_animation --entities 64 256 1024 --bones 12 60 --repeat 10`
measures controller advancement, pose evaluation and the combined update for
single clips and crossfades. It reports median and p95 wall time, plus peak
traced memory from a separate invocation; tracing is not an allocation count or
a complete process/GPU memory measurement. Use `--output animation_cpu.json`
to save the report. These synthetic short-chain rigs do not measure GPU upload,
root motion, events or graph blending; timings depend on the machine and workload.

## Physics synchronization

```powershell
python -m scenes.benchmarks.benchmark_physics --bodies 100 1000 --threads 1 --warmup 30 --repeat 240 --output physics_cpu.json
```

Install the `physics` extra. This headless workload uses isolated unit boxes,
gravity disabled, sleeping disabled, shared linear/angular velocities, and
120 Hz scheduler ticks. It compares interpolation disabled and enabled, with
median/p95 wall times for reconciliation, input submission, native stepping,
pose readback, and render-pose preparation. Creation is measured separately;
warmup ticks are excluded. `sync` combines reconcile, inputs and pull; the
fixed-tick total includes scheduler overhead and native stepping. Render-pose
preparation is separate from that total.

The `io` probe separately measures UID lookup, NumPy gather, float64-to-float32
conversion and ECS scatter on one snapshot, releasing the snapshot afterward.
It excludes physics stepping and render packing; it reassigns identical ECS
poses after the timed simulation. Results depend on the platform and workload;
they do not establish GPU frame rate or collision-heavy physics performance.

On Windows x64, CPython 3.14.10, NumPy 2.5.5 and pyjoltphysics 0.1.0a3, one
local 240-sample run at 1,000 bodies measured median 0.55 ms per fixed tick with
interpolation enabled (0.30 ms sync, 0.24 ms native step), plus 0.26 ms for
render-pose preparation. Conversion alone measured roughly 0.001 ms. These are
observations, not CI thresholds; rerun on the target machine before tuning.

## Transparency

| Script | Output / purpose |
| --- | --- |
| `diagnose_banded_tail` | CPU band-selection bias and noise against sorted layers; JSON. |
| `profile_banded_tail` | GPU timing and image errors for proxy/subset/sample-count variants; JSON. |
| `diagnose_proxy_edges` | GPU color errors at proxy discontinuities. |
| `profile_ao` | GPU timing comparison of AO in composition and a separate pass. |

```powershell
python -m scenes.benchmarks.diagnose_banded_tail --trials 64 --seed 2026 --output banded_tail.json
python -m scenes.benchmarks.profile_banded_tail --size 1280x720 --frames 24 --variants baseline proxy subset single --output banded_profile.json
python -m scenes.benchmarks.diagnose_proxy_edges
python -m scenes.benchmarks.profile_ao --size 1280x720 --frames 30 --output ao_profile.json
```

`profile_banded_tail` accepts `--pairs`, `--samples`, `--frames`, `--warmup`
and `--rounds`; it requires GPU timestamp queries. Timestamp readbacks synchronize
frames. Diagnostic images and metrics describe static comparisons.
Results depend on GPU, drivers, resolution and scene.

## Optional legacy Raylib comparisons

`benchmark_raylib_objects.py` and `benchmark_raylib_ecs.py` share
`raylib_benchmark_loop.py`. Install `pyray` separately to use them; it is not
part of the package extras. These interactive scripts are separate from the
WGPU scene benchmarks.

For the underlying ECS and rendering APIs, see [ECS](../../graaal/ECS_API.md),
[transparency](../../graaal/stochastic_transparency.md) and
[interactive scenes](../README.md).
