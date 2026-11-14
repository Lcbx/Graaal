# Benchmarks and diagnostics

Run these commands from the repository root after `pip install -e .` (and the
`render`, `assets`, `ui`, and `examples` extras for GPU tests). CPU scripts need
NumPy only. GPU scripts need a working WGPU backend; `benchmark_ecs_uploads`
opens a window, whereas the transparency diagnostics render offscreen.
Add `--help` to an entry point to see all supported options. Output files are
chosen by the caller; benchmarking does not modify the engine's defaults.

## ECS and CPU data access

```powershell
python -m scenes.benchmarks.benchmark_ecs --entities 100000 --repeat 11 --loops 50 --output ecs_cpu.json
python -m scenes.benchmarks.benchmark_ecs_uploads --frames 120 --output ecs_uploads.json
python graaal/tests/benchmark_array_access.py
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
