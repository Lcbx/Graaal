# Active ragdoll plan

General native integration already exists; see the
[physics status](physics_integration_plan.md) and
[ragdoll API](../graaal/physics.md#native-ragdolls-and-skeletons).
This plan concerns additional passive mechanics and learned control; verified
prerequisites and remaining work are distinguished below.

Implementation stays in Python and NumPy, using the existing pyjoltphysics API.
Changes to Jolt or its native bindings, and implementations in other programming
languages, are out of scope. Missing backend capabilities remain explicit limits.

## Backend validation status

The pinned pyjoltphysics 0.1.0a3 already supports batched body torques, detached
native body-state reads, snapshot pose/velocity arrays and per-bone motor torque
limits. Added standalone NumPy [neutral joint springs](../graaal/physics.md#passive-neutral-joint-springs)
with bounded joint torques and equal opposite parent reactions. The evaluator
requires explicit gains/neutral rotations and has no ECS or native dependency.
It supplies neutral-posture mechanics; it does not implement anatomical soft
limits or a balance controller. No binding extension was needed for this path.

Native tests confirm relaxation of isolated spherical-body joints in both limp
and powered modes at zero active strength, per-bone motor weakening, repeated
one-/two-thread trajectories in independent worlds, and isolation of impacts
between worlds. Pure NumPy tests check restoring direction, relative damping,
joint caps, branching reactions, world rotation covariance, quaternion signs,
normalization and rejected invalid inputs. Snapshot observations preserve UID
order/duplicates after native compaction and own their retained arrays.

The [headless control benchmark](../scenes/benchmarks/README.md#ragdoll-control-prerequisites)
measures observation copies, neutral springs, one bulk torque submission,
per-handle strength submission, native stepping and event draining. A Windows x64
run with CPython 3.14.10, NumPy 2.5.5 and pyjoltphysics 0.1.0a3 covered 24 cases:
10/20 native humanoids, 1/2/4 threads, limp/powered modes and springs off/on, using
60 warmup and 480 measured ticks. All active strengths were zero; the environment
had zero gravity, no floor, no self collision and no root anchoring.

At 20 ragdolls with springs, median CPU timings across the matrix were:

| Stage | Median range (ms) |
| --- | --- |
| Snapshot observation copies | 0.039–0.051 |
| NumPy spring evaluation | 0.219–0.298 |
| Bulk torque submission | 0.014–0.018 |
| Per-handle per-bone strength submission | 0.055–0.088 |
| Native step | 0.165–0.206 |
| Complete control tick | 0.531–0.685 |
| Separate direct body-state readback | 0.008–0.014 |

There were no rejected commands or dropped events. With the probe's weak uniform
gains, mean joint error fell from 0.100 to roughly 0.078 rad after 4.5 seconds;
the no-spring baseline remained at 0.100 rad. This modest reduction is a mechanism
check, not convincing passive motion or a stable thread-scaling claim. Stronger
initial gains oscillated in light limbs, reinforcing the need for authored gains
and timestep/inertia validation. Local raw output is the ignored artifact
`scenes/benchmarks/results/ragdoll_control_windows.json`.

Validation: 136 focused spring, benchmark, quaternion, native ragdoll, replay and
import-boundary tests passed; `python -m graaal` passed. The boundary module ran
outside the sandbox to access pytest's temporary directory. No GPU or additional
platform validation was performed.

The authored humanoid and bounded contact scenarios below extend these isolated
mechanism checks. Broader tuning and visual validation remain prerequisites.
Full character/ragdoll replay and
selective training resets still need native control-state capture. Body pose and
velocity writes do not provide that reset contract. Expanded observations and
training/runtime parity remain future work.

## Python soft angular limits

Added standalone NumPy `joint_limit_torques` and one-time
`ragdoll_limit_parameters` preparation from authored native Bone metadata. The
evaluator uses asymmetric twist bounds and the quaternion swing ellipse with
rest frames, plane/twist axes and cone bend. It insets a comfortable range,
applies quadratic restoring torque and outward-only damping, caps each soft
joint, and accumulates equal opposite reactions. Locked/narrow axes, free-angle
wraps and antiparallel orientations are handled. Neutral and soft-limit forces
have independent caps and remain active independently of animation motors.

Pressure tests run Python control against the existing native hard constraints
in limp and powered modes at zero motor strength, with one and two threads.
Twist, swing and bent joints use non-identity rest rotations. Under 0.1 N m
pressure, the tested joints settle more than 0.1 rad before the hard boundary;
late variation stays below 0.02 rad. Pressure exceeding the 0.5 N m soft cap
still reaches the native hard backstop within the tested solver tolerance.
Pure NumPy checks cover the interior dead zone, quadratic growth, outward-only
damping, elliptic swing coupling, asymmetric/locked/free limits, per-instance
caps, frame covariance, metadata copies and invalid values.

The control benchmark's `--soft-limits` option measures the combined path.
On the same Windows x64/CPython 3.14.10/NumPy 2.5.5/pyjoltphysics 0.1.0a3 setup,
a 24-case run covered 10/20 humanoids, 1/2/4 threads, limp/powered modes and
passive forces off/on with 60 warmup and 480 measured ticks. At 20 humanoids with
both force paths, soft-limit evaluation medians were 0.559–0.575 ms, torque
submission 0.017–0.019 ms and complete control ticks 1.236–1.374 ms. Every case
retained 480 native timing samples, with no rejected commands or dropped events.
The raw local report is the ignored artifact
`scenes/benchmarks/results/ragdoll_soft_limits_windows.json`.

Validation: 179 focused physics, quaternion, replay, benchmark and optional-import
tests passed; `python -m graaal` passed. The import-boundary module used its
previously required access outside the sandbox for pytest's temporary directory.
No GPU or additional platform validation was performed. These tests validate
the force mechanism and the measured zero-gravity workload; humanoid anatomical
tuning and full passive drops, stairs and impacts were still unfinished in that run.

## Authored passive humanoid and contact scenarios

Added the standalone Python [passive humanoid example](../scenes/passive_ragdoll_example.py),
with 13 bodies in depth-first order, box feet and authored ankle swing/twist
limits. Named masses total 70 kg at 1.8 m; shapes and mass determine native
inertias. Height bakes into translations/shapes, and passive gains scale with
mass times length squared. A relaxed pose bends elbows and knees, angles the
upper arms forward and out, and uses weaker neck/ankle gains and stronger
torso/hip/knee gains. The tuned soft-limit stiffness, damping and torque caps
are 25% higher than the previous humanoid tuning. Both force paths run in
Python/NumPy and submit one combined torque batch per tick.

The example uses zero native motor strength/root anchoring, self collision,
0.05/s linear body damping, 0.3/s angular body damping, zero joint friction,
floor friction 0.8 and zero restitution. Four caller-stepped 120 Hz scenarios
cover zero-gravity relaxation, a tilted floor drop, descending stairs with an
initial lateral velocity, and an off-centre chest impulse. The stair spawn height
gives the tilted colliders 0.01 m of clearance over the highest step at the
default scale. The no-passive baseline preserves body damping/native hard
constraints and disables Python forces. The scene drains event records per
tick and retains owned copies of the results.

On Windows x64 with CPython 3.14.10, NumPy 2.5.5 and pyjoltphysics 0.1.0a3,
960-tick checks ran all four scenarios with one and two native threads, plus
a zero-gravity no-passive baseline at each thread count. Mean relaxed-pose
error fell from 0.3920 to 0.0886 rad in zero gravity. The baseline stayed at
0.3920 rad. Each elbow finished about 5.2 cm outboard of its shoulder in the
pelvis frame, up from 2.7 cm. Contact cases reported foot and floor contacts.
The stair case started above the highest step and reported step, floor and self contacts
during the fall. Peak anchor separation was 0.0514 m for the drop,
0.0533 m for stairs and 0.0162 m for the impact. The chest's immediate linear
velocity change matched impulse divided by its 22 kg mass. The peak speeds
include the immediate velocity sample after the off-centre impulse.

Across the contact cases, final linear kinetic energy ranged from roughly
0.000074 to 0.00241 J. During the final 120 ticks, maximum body speeds stayed
below 0.068 m/s and 1.53 rad/s. Transient angular speeds reached 47.124 rad/s
in the drop and stair cases. Jolt reported zero rejected commands and zero
dropped events. Raw local results are the ignored artifact
`scenes/benchmarks/results/passive_humanoid_windows.json`.

The replay viewer draws the terrain and either the colliders or a user-supplied
`dummy_rigged.glb` driven by the same simulated poses. It provides looping
playback, pause and restart without ECS.
See the [scene commands](../scenes/README.md#headless-animation-and-physics).

The tuned settings passed 130 focused passive-scene, ragdoll, spring and
soft-limit tests and the offscreen dummy replay in zero gravity. Earlier
viewer validation passed for 160 focused tests, eight GPU skinning tests and
the `python -m graaal` smoke test. With ECS imports blocked, offscreen playback
passed for all four collider scenarios and for the dummy in zero gravity and on stairs.
Broader visual review, anatomical and impact tuning, active balancing and
complete training resets remain pending.

## Overview

Implement a physically simulated humanoid with **natural passive joint behavior** and a **neural controller** capable of balancing, following animations, stepping, bracing during falls, and getting back up. Physics is handled by **Jolt Physics through pyjoltphysics**, while Python defines ragdolls, joint constraints, motor commands, simulation scenarios and batched observations using **NumPy**. A **PyTorch PPO policy** learns purposeful movement through reinforcement learning. The same physics, actuation and observation code runs during training and at runtime in **Graaal**, with NumPy-based neural inference.

The system separates three sources of joint forces:

- **Passive soft limits:** Progressive resistance and damping near joint boundaries, with hard limits as backstops.
- **Passive neutral-posture pull:** Weak springs toward a relaxed, human-like equilibrium pose, even without active control.
- **Active neural control:** Strength-limited motors that track animation targets while adapting to balance, impacts and environmental contacts.

## Implementation phases

### Phase 0 — Backend validation

- Test pyjoltphysics rigid bodies, ragdolls, joint constraints, motors, impulses and state snapshots.
- Verify independent training environments, selective resets and reproducibility.
- Benchmark batched observations, motor commands and native multithreaded stepping.
- Identify missing APIs for passive joint torques, bulk motor updates or efficient resets; keep implementation within Python and the existing binding.

**Deliverable:** Working headless ragdoll prototype and measured simulation throughput.

### Phase 1 — Passive ragdoll

- Define a humanoid with approximately 13 rigid bodies, including feet and ankle joints.
- Configure masses, inertias, collision shapes, joint frames and anatomical hard limits.
- Implement **soft angular limits**, with progressive restoring torque and damping near boundaries.
- Implement **neutral-posture pull**, using weak springs toward relaxed joint orientations inspired by human neutral body posture in microgravity.
- Keep passive forces active independently of neural motor strength.
- Support physical impulses at arbitrary body locations to generate linear and rotational impacts.

**Validation scenarios:**

- Zero gravity: limbs gradually settle into relaxed orientations.
- Passive drops and stair falls: weighted motion without excessive flailing, snapping or bouncing.
- Explosions: plausible limb rotation and whole-body reactions.
- Joint-limit stress tests: no instability or unnatural hard-limit oscillation.

**Deliverable:** A passive ragdoll with convincing movement and impacts, without any neural controller.

### Phase 2 — Active actuation and observations

- Add strength-limited joint motors that track reference poses from standing, walking or other animations.
- Allow motor targets and strengths to vary continuously, including partial weakening during impacts.
- Implement a simple classical balance and pose-tracking controller as a reference baseline.
- Define NumPy observations containing body orientation, velocities, joint states, contacts, centre of mass, reference pose and previous actions.
- Define neural outputs as joint target offsets and optionally individual motor strengths.

**Deliverable:** A shared control interface that combines passive joint mechanics with active, bounded motor control.

### Phase 3 — Parallel training environments

- Run multiple ragdolls in spatially separated regions of one or several Jolt worlds.
- Implement batched stepping, actions, observations, rewards and selective resets.
- Randomize mass, joint strength, friction, terrain and external impulses.
- Support resets into standing, stumbling, falling, prone and supine configurations.
- Benchmark throughput and optimize Python/native communication.

**Deliverable:** Reproducible, high-throughput reinforcement learning environment.

### Phase 4 — Reinforcement learning

Train a **PPO policy**, initially with a small MLP and optionally residual classical control.

| Training scenario           | Learned behavior                                  |
| --------------------------- | ------------------------------------------------- |
| Standing with perturbations | Maintain upright equilibrium                      |
| Small and moderate pushes   | Recover through ankle, hip and stepping responses |
| Severe impacts              | Stumble, use arms to brace and reduce injury      |
| Falling or airborne starts  | Protect head and torso, manage landing            |
| Prone and supine starts     | Roll, push against surfaces and stand up          |
| Walking and uneven terrain  | Follow target movement while adapting balance     |

**Reward design:**

\\[ R = w_p R\_{\text{pose}} +w_b R\_{\text{balance}} +w_g R\_{\text{goal}} -w_h C\_{\text{head}} -w_t C\_{\text{torso}} -w_e C\_{\text{effort}} -w_s C\_{\text{sliding}} \\]

- Reward proximity to the reference pose, upright standing and desired movement.
- Penalize strong head and torso impacts using contact impulses or impact severity.
- Penalize excessive joint effort, unnatural movement and foot sliding.
- Use context-sensitive reward weights: during an unavoidable fall, protecting the head should take priority over remaining upright.
- Train with randomized initial poses and velocities, including characters already falling or lying on the ground.
- Allow episodes to continue after falls so the policy can learn recovery.
- Avoid rewarding permanent crouching, rigid postures or unrealistic refusal to fall.

**Deliverable:** A policy that follows animations, balances, steps, protects itself during falls and rises from the ground.

### Phase 5 — Evaluation and refinement

- Measure recovery probability across push directions, magnitudes and application points.
- Evaluate head/torso impact severity during falls and stair tumbles.
- Measure recovery success and time from prone and supine positions.
- Validate smooth transitions from controlled movement to stumbling, collapse and recovery.
- Test behavior under reduced strength and compare against classical control.
- Add motion imitation or additional posture constraints only if movement quality requires them.

**Deliverable:** Validated natural movement, physically plausible failures and reliable recovery behavior.

### Phase 6 — Graaal integration

- Export neural weights and observation normalization to `.npz`.
- Implement batched NumPy MLP inference.
- Connect Jolt body transforms to ECS entities and the rendering skeleton.
- Run physics at a fixed timestep (initially 120 Hz) and neural control at 60 Hz.
- Support reference animation changes, external impacts and continuous motor-strength variation.
- Benchmark multiple characters and verify training/runtime behavioral parity.

**Deliverable:** Neural-controlled physical characters running in Graaal with the same behavior observed during training.

## Risks and integration boundary

Validate passive torque access, efficient motor updates/selective resets and
training throughput before extending bindings. Keep passive forces independent
of motor strength, and preserve continuous loss/recovery of control through
bounded actuation. Reference animation supplies intent; native physics owns
motion. Compare learned behavior with the classical baseline and verify runtime
parity before integrating more characters.

Documentation audit: removed repeated decisions and linked the delivered native
integration so this future control work is not confused with basic ragdoll support.
