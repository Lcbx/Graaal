---
name: graaal-performance
description: Use when optimizing Graaal CPU/native/GPU costs with the repository benchmarks and diagnostics.
---

# Array performance and measurement

Use the existing [benchmarks](../../../scenes/benchmarks/README.md) and [diagnostic APIs](../../../graaal/README.md#examples-and-diagnostics) for the workload in scope. Keep performance work within the requested scope; historical timings and roadmap budgets are not acceptance thresholds. Put new runnable benchmarks in [scenes/benchmarks/](../../../scenes/benchmarks/).

## Array and upload decisions

Preserve dtype, shape, ordering, ownership and aliasing contracts before reducing allocations. Prefer vectorized NumPy work over Python iteration across array data. Inspect actual strides, contiguity, writability and memory sharing where an optimization depends on them; do not assume a slice is contiguous or a reshape is a view.

Use existing reusable outputs and preallocated buffers where they help the measured path. ECS provides `selection.read(name, out=...)` and scoped `edit_dense()`; their exact-dtype and lifetime restrictions are in the [array access contract](../../../graaal/ECS_API.md#explicit-arrays-and-reusable-field-reads). Independent stores still require alignment through joins/gathers. Use `out=` only where aliasing and broadcasting preserve results.

At GPU/native boundaries, check the consumer's documented dtype/layout/contiguity requirements and existing conversion path. Avoid unconditional per-frame `ascontiguousarray` or dtype conversions: they can allocate and alter precision. WGSL padding and packed GPU dtypes must match their consumers. Preserve version-driven cache invalidation, instance ordering and explicit object/dictionary notifications when reducing gathers or uploads.

## Choose and interpret measurements

- ECS/array access: `benchmark_ecs`, `benchmark_ecs_uploads` and `tests/benchmark_array_access.py` cover different costs. Upload benchmark times are CPU durations, not GPU frame time. Historical standalone-ECS comparison options need a compatible standalone snapshot; current `ECS/core.py` has relative imports.
- Animation: `benchmark_animation` separates controller, pose and combined CPU work. Traced peak memory is measured separately and is neither an allocation count nor total process/GPU memory. Synthetic rigs do not establish root-motion, events or rendering performance.
- Physics: `benchmark_physics` separates synchronization/native stepping, render-pose preparation and the I/O probe. Select the relevant workload and inspect contacts/overflow before interpreting it as collision-heavy. Warmup and native timing reset rules are documented in the benchmark guide.
- Rendering: resource counters and encoded draw/dispatch counts do not measure GPU execution. `RenderMonitor` CPU pass timings measure recording; submit timing measures the queue call. GPU timestamps require supported timestamp-query features, synchronize on readback, and do not isolate individual draws or copies outside measured passes.
- Memory: tracked GPU bytes estimate engine resources, excluding driver/transient/untracked allocations. Process RSS, Python traced memory, CPU mirrors and GPU capacity describe different quantities; keep them distinct.

Choose bounded runs, warmups and output paths appropriate to the question; inspect each entry point's `--help`. Compare the same workload and measurement conditions before/after and verify behavior with affected subsystem tests. For rendering changes, use relevant image/error diagnostics where timing alone cannot establish correctness. Report backend limitations and variability; do not silently change defaults based on one run.
