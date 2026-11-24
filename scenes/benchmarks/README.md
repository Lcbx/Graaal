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
because the current `graaal/ECS/core.py` has package-relative imports.

`python -m scenes.benchmarks.benchmark_animation --entities 64 256 1024 --bones 12 60 --repeat 10`
measures controller advancement, pose evaluation and the combined update for
single clips and crossfades. It reports median and p95 wall time, plus peak
traced memory from a separate invocation; tracing is not an allocation count or
a complete process/GPU memory measurement. Use `--output animation_cpu.json`
to save the report. These synthetic short-chain rigs do not measure GPU upload,
root motion, events or graph blending; timings depend on the machine and workload.

## River crowd animation

```powershell
python -m scenes.benchmarks.benchmark_river_crowd --rows 12 24 --frames 30 --rounds 3 --output river_crowd.json
```

Install the `render`, `assets` and `physics` extras. The benchmark uses the
river scene's physics, graph, skin palette and material draws at 640×480
offscreen by default. Each crowd row adds 12 dummies. The three modes use exact
CPU poses, GPU poses for every eligible row with CPU fallback, or the scene's
18/22-unit distance LOD with CPU fallback. Every mode replays the same
fixed ticks and camera sweep. The bank window starts at spawn and the swim
window starts after `--swim-tick` simulation ticks. Each window discards
`--warmup` frames before sampling. Mode order rotates between rounds.

The report includes swimmer counts, actual CPU/GPU pose counts, and graph
ineligibility split into swimming, active fades and other rows. Distance LOD
reports how many otherwise eligible rows it kept on the CPU. Pose preparation
includes CPU routing, evaluation, staging and command recording. Wall frame
time includes simulation, rendering and waiting for GPU completion. GPU
timestamps span the command buffer's initial marker through its last draw pass,
including intervening palette copies. Queue staging uploads occur before the
marker. The GPU pose time measures the crowd compute pass. Timestamp readback
and fallback counting happen outside the wall timer. GPU times are `null` on
devices without timestamp-query support. Use the JSON report for per-frame data
and median and p95 timings.

## Physics synchronization

```powershell
python -m scenes.benchmarks.benchmark_physics --bodies 100 1000 --threads 1 --warmup 30 --repeat 240 --output physics_cpu.json
python -m scenes.benchmarks.benchmark_physics --workloads isolated colliding compounds ragdolls --bodies 100 1000 --ragdolls 10 20 --threads 1 2 4 --warmup 60 --repeat 240 --output physics_workloads.json
```

Install the `physics` extra. All workloads use 120 Hz lockstep scheduler ticks,
sleeping disabled and native contact persist reporting. The default workload
remains `isolated`; choose one or more with `--workloads`:

| Workload | Simulation |
| --- | --- |
| `isolated` | Spaced unit boxes, gravity disabled, shared linear/angular velocities. |
| `colliding` | Stacked unit boxes falling onto a static floor under gravity. |
| `compounds` | The same stack with two adjacent half-box colliders per body. |
| `ragdolls` | Tilted native humanoids falling limp onto a shared floor, with self collision. |

`--bodies` selects moving body counts for the first three workloads;
`--ragdolls` selects humanoid counts (default 10 and 20). Each collision workload
adds one static floor. Ragdolls use the backend's humanoid prefab; the report
records its bone count and native total body count. These are passive ragdolls,
not animation-following or learned controllers.

Every case compares interpolation disabled and enabled, reporting median/p95
CPU wall times for reconciliation, input submission, native stepping, pose
readback and event routing. `sync` combines reconcile, inputs, pull and
`dispatch_events`; `fixed_tick` also includes native stepping and scheduler
overhead. Event routing is now measured inside the fixed tick, unlike the
historical baseline below. Creation measures declaration reconciliation and
native command submission; native application happens in the first tick.
Warmup ticks are excluded from both wall-time samples and native timing means.

Render transforms for interpolated rigid bodies and final skinning for ragdolls
are measured separately from the fixed tick. Ragdoll skinning runs in both modes;
interpolation adds previous/current bone-pose sampling. No renderer or GPU is
created. `native` contains the [native diagnostics](../../graaal/physics.md#diagnostics-and-performance).
`contact_events` counts contact records, including persist/end records, and
`contact_ticks` counts measured ticks with any contact records; neither is a
unique contact-pair count. `events_dropped` counts overflow during the measured
window. Check these fields before interpreting a case as collision-heavy.

The `io` probe separately measures UID lookup, NumPy gather, float64-to-float32
conversion and ECS scatter on one snapshot, releasing the snapshot afterward.
It excludes physics stepping and render packing; it reassigns identical ECS
poses after the timed simulation. This probe applies to rigid bodies only;
ragdoll bone poses are converted through final skinning instead. Results depend
on the platform and workload and do not establish GPU frame rate.

Ragdoll reports include a separate `ragdoll_inputs` probe after the measured
simulation. It compares native-parent target conversion per owner and in one
batch, then measures per-handle native submission with precomputed targets and
frames. `set_pose` measures the adapter's full-batch copy/validation/conversion;
`prepare_updated` measures preparation immediately afterward, and
`update_and_prepare` measures their combined duration. `prepare_held` measures
another preparation without a new target. The probe repeatedly submits the
skeleton rest pose to all owners; it does not evaluate animation or measure
powered motor simulation. Native `flush()` calls outside the timed sections
bound queue/pool use without advancing ticks or draining events. The probe
replaces the held targets after simulation; the reported native diagnostics,
contacts and presentation timings are captured before it runs.

Historically, on Windows x64, CPython 3.14.10, NumPy 2.5.5 and pyjoltphysics 0.1.0a3, one
local 240-sample run at 1,000 bodies measured median 0.55 ms per fixed tick with
interpolation enabled (0.30 ms sync, 0.24 ms native step), plus 0.26 ms for
render-pose preparation. Conversion alone measured roughly 0.001 ms. This older
isolated-box result excluded event routing from the measured tick. These are
observations, not CI thresholds; rerun on the target machine before tuning.

A subsequent 48-case run on the same software versions used 60 warmup and 240
measured ticks across all four workloads, both sizes, 1/2/4 threads and both
interpolation modes. With interpolation enabled, 1,000-box fixed-tick medians
were 3.28/2.67/2.10 ms at 1/2/4 threads; compounds measured 3.35/2.69/2.07 ms.
Render transforms cost about 0.27 ms separately. All collision cases reported
contacts in every measured tick without event overflow. Twenty ragdolls were
dominated by Python input preparation, with substantial run-to-run variation;
see the [physics status](../../implementation_plans/physics_integration_plan.md#phase-6-workload-measurements)
for the measurements and limitations before making tuning decisions.

## Ragdoll control prerequisites

```powershell
python -m scenes.benchmarks.benchmark_ragdoll_control --ragdolls 10 20 --threads 1 2 4 --warmup 60 --repeat 480 --output ragdoll_control.json
python -m scenes.benchmarks.benchmark_ragdoll_control --ragdolls 10 20 --threads 1 2 4 --soft-limits --warmup 60 --repeat 480 --output ragdoll_soft_limits.json
```

Install the `physics` extra. This standalone native probe uses 120 Hz lockstep,
the backend's 11-bone humanoid, zero gravity, no floor/self collision, no body or
joint friction/damping, and no root anchoring. Ragdolls are spaced four metres
apart. Both shoulders start 0.5 rad from rest after two kinematic settling ticks;
velocities are cleared before switching to limp or powered mode. Active motor
strengths are zero in both modes throughout the measurement.

Each case compares neutral springs off/on, separating snapshot UID lookup and
detached pose/velocity observation copies, NumPy spring evaluation, one flattened
native torque batch, per-handle per-bone strength submission, stepping and event
draining. Uniform probe spring gains are 1 N m/rad and 0.05 N m s/rad, with a
0.5 N m cap per joint. Strength arrays are resubmitted every tick to measure that
binding path even though their values stay zero. `peak_body_torque_nm` can exceed
the joint cap because parent reactions accumulate. `fixed_tick` covers these
stages together. Warmup is excluded from samples and native timing means;
`ticks` also includes the initial two settling ticks. `direct_readback` separately
times native locked `bodies_state` copies after simulation, without advancing time.

`--soft-limits` adds the Python/NumPy progressive swing/twist evaluator to passive
cases, using copied authored humanoid frames and limits. Its separate
`limit_torques` stage uses margin 0.1 rad, stiffness 1 N m/rad, damping
0.05 N m s/rad and a 0.5 N m cap per joint. The no-passive baseline keeps both
force paths disabled. `soft_limits` identifies cases where the evaluator ran;
`peak_limit_body_torque_nm` isolates its accumulated body reactions. Neutral and
limit torque caps apply independently before their sums are submitted, so the
combined joint torque may exceed either individual cap.

`initial_joint_error_rad` and `final_joint_error_rad` average shortest relative
rotation error across non-root joints and instances. The uniform weak gains probe
the torque path; they do not establish natural posture quality, convergence for
arbitrary rigs or stability at another timestep. This is not a collision workload,
powered animation tracking, an RL environment or a training-reset benchmark.
Contact/application observations, mass/COM features and controller state are not
part of these pose/velocity observation copies. Check native rejection/event
overflow counters before interpreting performance. Results and thread scaling
depend on the machine; [current local measurements](../../implementation_plans/active_ragdolls.md#backend-validation-status)
do not establish runtime or GPU frame rate.

## Ragdoll push recovery

```powershell
python -m scenes.benchmarks.benchmark_ragdoll_push_recovery --checkpoint build/standing-residual-tilt025.zip --report build/ragdoll-push-recovery.json
```

Install the `physics` extra to evaluate classical ankle balance. `--checkpoint`
adds a [residual PPO policy](../../tools/README.md#ragdoll-standing-task) trained
with `--support-area`. Use `--stepping-checkpoint` and `--whole-body-checkpoint`
for policies trained with `--action-space stepping` and `--action-space whole-body`
respectively, both with `--support-area`. Checkpoint options require
Stable-Baselines3 and can be used together. The default grid applies
0.5, 1, and 2 N·s to the pelvis or chest center of mass in world ±X and ±Z at
2 seconds. Each cell uses the same 10 starts with tilt up to 0.025 rad and
spin up to 0.1 rad/s. Use `--seed`, `--episodes` and `--impulses` to set the
first seed, the number of starts per cell and the impulse magnitudes.

A fall or an incomplete one-second standing hold before the push fails the
trial. A passing recovery completes a fresh one-second standing hold within
five seconds of the push. Recovery time includes the hold. A step is a foot
losing floor contact and landing at least 2 cm horizontally from its last
planted position, measured at 120 Hz. Step counts cover the post-push period
through recovery, fall or timeout. Each counted landing records the foot,
time and world X/Z positions at takeoff and touchdown. The report includes
overall and per-cell success rates, average recovery times for successful
trials, and individual outcomes, times and step counts. With `--report`, the
full report goes to the given file and the overall comparison goes to stdout.

```powershell
python -m scenes.benchmarks.benchmark_ragdoll_push_recovery --stepping-checkpoint build/ragdoll-push-stepping-seed87.zip --whole-body-checkpoint build/ragdoll-push-whole-body-seed87.zip --impulses 2 4 8 12 --episodes 5 --seed 58000 --report build/ragdoll-push-whole-body-evaluation.json
```

`--scripted-step` compares classical balance with a timed right-foot backstep.
It uses the benchmark's push timing and activates on −Z impulses of at least
8 N·s at the chest or 12 N·s at the pelvis. The report records the duration of
continuous floor contact after each landing in physics ticks and Jolt's peak
estimated floor-contact impulse for that foot.

```powershell
python -m scenes.benchmarks.benchmark_ragdoll_push_recovery --scripted-step --impulses 2 4 8 12 --episodes 5 --seed 58200 --report build/ragdoll-scripted-backstep-evaluation.json
```

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
