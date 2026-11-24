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
selective training resets still need native control-state capture. Fresh isolated
worlds can reproduce the same rollout from the same spawn and inputs. Selective
replay in a live world also needs the native held control state and application
state. Training/runtime policy parity remains future work.

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
Follow-up review rendered five samples per scenario with colliders and the
dummy rig. The impact review showed the body tipping before the impulse at tick
120. The impulse now lands at tick 24 (0.20 s), after initial contact. Before
the impulse, the pelvis-to-chest axis had an upward cosine of 0.952–0.957 over
1.5 m/50 kg, 1.8 m/70 kg and 2.1 m/100 kg at one or two threads. The new impact
trials reached 0.011–0.015 m peak anchor separation and 30–35 rad/s peak body
angular speed. The immediate linear impulse response still matches impulse
divided by mass. Across all four scenarios at those sizes, zero-gravity final
errors were 0.086–0.089 rad and contact trials finished below 0.072 m/s peak
body speed over their final 120 ticks. Other platforms and broader anatomical
tuning remain open.

Use the [ragdoll tuner](../tools/ragdoll_tuner.md) to edit colliders, masses,
joint limits and passive gains in drop, stair and zero-gravity previews and
save the tuned settings as JSON. Pass `--tuning PATH` to the passive example
to run reports or playback with the saved rig and passive gains. The repository
tuning file was checked in all four headless scenarios and in offscreen
collider and dummy playback.

## Overview

Implement a physically simulated humanoid with **natural passive joint behavior** and a **whole-body neural controller** capable of balancing, following animations, recovering from pushes with steps, arm and torso motion or environmental support, bracing during falls, and getting back up. Physics is handled by **Jolt Physics through pyjoltphysics**, while Python defines ragdolls, joint constraints, motor commands, simulation scenarios and batched observations using **NumPy**. A **PyTorch PPO policy** learns purposeful movement through reinforcement learning. The same physics, actuation and observation code runs during training and at runtime in **Graaal**, with NumPy-based neural inference.

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

`PhysicsRagdollControl` provides detached observation rows in bone UID order
and bounded per-joint motor actions. The [active example](../scenes/active_ragdoll_example.py)
combines those actions with the passive torques and a classical standing probe.
Its controller applies an upright root spring and a contact-gated horizontal
pelvis force of up to 200 N. With a 3 N s chest impulse over 480 ticks on the
default body, minimum pelvis height was 0.983 m and final tracking error was
0.004 rad. With root assistance disabled, tracking alone reached 0.101 m and
0.098 rad. Fresh isolated runs reproduce the sampled result. Sustained
free-root balance and walking reference evaluation remain open.

- Add strength-limited joint motors that track reference poses from standing, walking or other animations.
- Allow motor targets and strengths to vary continuously, including partial weakening during impacts.
- Implement a simple classical balance and pose-tracking controller as a reference baseline.
- Define NumPy observations containing body orientation, velocities, joint states, contacts, centre of mass, reference pose and previous actions.
- Define neural outputs as joint target offsets and optionally individual motor strengths.

**Deliverable:** A shared control interface that combines passive joint mechanics with active, bounded motor control.

### Phase 3 — Parallel training environments

The [standing task](../tools/ragdoll_standing.py) runs one ragdoll per episode
on a flat floor, rebuilding the native world at reset. It runs joint actions
at 60 Hz and checks for falls at the 120 Hz physics rate. Success requires
pelvis height, upright pelvis and chest, and no floor contact from non-foot
bodies. The task provides passive and fixed-pose motor baselines, a seeded
starting angular perturbation, and an optional PPO runner.

The PPO runner supports independent subprocess workers with one Jolt world each.
It rebuilds only the world belonging to the episode that ended, preserving the
state of other episodes. Resuming a PPO checkpoint restores the model, optimizer,
and training step count. Each worker starts with a fresh physics episode.

At 0.1 rad/s seeded initial pelvis spin, the passive and fixed-pose baselines
lasted 0.38 s and 3.39 s on average across eight five-second evaluations.
Three PPO trials of 32,768 to 65,536 policy steps completed zero evaluation
episodes. The best trial averaged 3.04 s.

A four-worker resume collected another 131,072 transitions at 987 policy
steps/s, ending at 180,224 total steps. The resumed policy averaged 2.72 s
over 12 five-second evaluations and completed zero. Fixed-pose tracking
averaged 3.40 s on the same starts. Worker isolation and reset reproducibility
are covered by the standing tests.

- Compare independent training workers with spatially batched ragdolls in one Jolt world.
- Measure batched stepping and observations. In-place ragdoll resets still need native control-state support.
- Add the reset poses, scheduled disturbances and per-episode randomization used
  by the curriculum below. Keep fresh-world resets until native control-state
  support makes in-place resets viable.
- Benchmark throughput and optimize Python/native communication.

**Deliverable:** Reproducible, high-throughput reinforcement learning environment.

### Phase 4 — Reinforcement learning

#### Research findings

- Ferigo et al. (2021) trained PPO to recover from pushes using joint reference
  velocities, foot contacts and center-of-mass feedback relative to the support
  area. A single policy used ankle corrections, upper-body momentum and steps.
  Their trials ran to 20 million agent steps on a different humanoid. Graaal's
  180,224-step run is an early result for its current task and action space.
  [Standing and push recovery](https://arxiv.org/html/2104.14534)
- DeepMimic (2018) used joint angle targets, reference pose and velocity rewards,
  and resets sampled from the reference clip. Its ablations found that these
  resets and early termination mattered especially for dynamic skills. For
  game-directed motion, DReCon feeds the physical controller a reference stream
  from a separate kinematic controller driven by player input.
  [DeepMimic](https://arxiv.org/html/1804.02717) · [DReCon](https://static-wordpress.ubisoft.com/montreal.ubisoft.com/wp-content/uploads/2019/11/13214229/DReCon.pdf)
- AMP learns a style reward from unaligned clips. PHC combines this with
  explicit tracking and a dedicated recovery objective, demonstrating get-up
  and return to tracking without external stabilizing forces. Consider an AMP
  experiment when there is a varied motion library and measured tracking or
  style shortcomings. Direct clip tracking is sufficient for the first test.
  [AMP](https://arxiv.org/abs/2104.02180) · [PHC](https://arxiv.org/html/2305.06456)

#### First standing experiment

Use the existing PPO MLP, one Jolt world per worker, 120 Hz physics and 60 Hz
control. Keep gravity, contacts, passive springs and soft limits active. The
policy acts through bounded joint motors with zero pelvis assistance throughout
training and evaluation. See the [standing tool guide](../tools/README.md#ragdoll-standing-task)
for the current runner.

1. Author a standing reference separately from the passive neutral pose. Check
   flat, nonpenetrating soles, usable knee and ankle travel, and a projected
   center of mass inside the foot support area. Run the fixed-pose motors on
   this stance and record drift and joint target errors before training.
2. On the same rig, compare the existing 48-channel action against a compact
   policy that offsets only usable joint axes and holds each motor's configured
   torque cap fixed. Start both controllers near the standing reference. Keep
   reset distributions, reward, seeds and sample budgets matched for this
   comparison. Test the support-area features and reward as a separate change.
   [Environment design ablations](https://arxiv.org/html/2010.04304)
3. Train first from quiet starts, then add small seeded pose and velocity
   perturbations, including the current 0.1 rad/s pelvis spin. Checkpoint and
   evaluate at 250,000 and 1 million policy steps per variant. Extend promising
   runs to 3 million steps if held-out survival is still improving. Confirm
   the selected setup with three independent training seeds. One policy step
   commands two physics ticks unless the episode ends during the first tick.
   If neither policy improves on fixed-pose tracking, inspect contact geometry,
   target errors near joint limits and center-of-mass drift before spending more
   samples. A joint-only ankle and hip feedback controller can serve as another
   baseline and, in a separate trial, supply targets for PPO to adjust.

#### Standing pilot results

The tuned reference adds 0.05 rad of knee bend and compensating hip and ankle
rotations. The passive neutral pose and joint limits retain their authored
values. Both soles start flat with 1 mm clearance and the projected center of
mass starts inside the estimated support area. On the built-in rig, selecting
constraint axes with at least 0.22 rad of total travel reduces the policy from
48 outputs to 30 with fixed motor caps.

With eight Jolt workers and PPO seed 77, each action-space variant collected
65,536 quiet-start steps, then resumed for 185,344 steps with starting tilt up
to 0.025 rad and pelvis spin up to 0.1 rad/s. On 30 paired held-out ten-second
starts (seeds 50,000–50,029), all trials fell before the time limit.

| Controller | Mean time upright | Settled successes |
| --- | ---: | ---: |
| Passive | 0.35 s | 0/30 |
| Fixed-pose motors | 2.05 s | 0/30 |
| PPO, compact actions | 2.04 s | 0/30 |
| PPO, full actions | 2.04 s | 0/30 |

Quiet-start fixed-pose tracking lasted 3.46 s. The quiet PPO checkpoints
lasted 2.84 s with compact actions and 2.78 s with full actions. A separate
65,536-step compact run using the estimated sole support area reached 3.08 s
from the same quiet start. These are single-training-seed pilot measurements.
Both policies remain below the standing acceptance gate after 250,880 steps.

In one quiet fixed-pose rollout, both feet still reported floor contact when
the estimated support margin fell from +0.065 m at 2 s to −0.159 m at 3 s.
Mean joint reference error through 3 s was 0.038 rad. On the 30 held-out seeds,
fixed-pose tracking lasted 3.46 s with 0.1 rad/s starting spin alone and 2.06 s
with starting tilt alone. Tilt plus spin gave 2.05 s. This points to initial
tilt as the larger disturbance for this rig.

Checkpoints and per-episode results are local artifacts under `build/`. The
held-out results are in `build/standing-regimen-evaluation.json`.

#### Joint-only ankle balance baseline

A two-start tuning sweep tried center-of-mass position and velocity feedback
through ankle and hip motor targets. Those trials fell. The ankle torque
baseline uses the tuned stance and keeps fixed-pose motors on the other joints.
At each physics tick it estimates a center of pressure beyond the center of
mass in the direction of drift. It limits the pressure offset to the reference
sole bounds and estimates the vertical load by dividing body weight evenly
among the contacting feet. It applies equal and opposite torque to each
supported foot and its shin, bounded by that ankle's existing 100 N m cap.
The pelvis receives zero root assistance. This pressure estimate uses the sole
geometry and contact flags.

Each row uses the same 30 held-out seeds (52,000–52,029), with pelvis spin
sampled up to 0.1 rad/s and an independently sampled starting tilt. Seeds
41,001 and 41,002 were used to tune the controller. Trials lasted up to ten
seconds and used the Phase 5 settled stance criteria. Passive control fell
after 0.35–0.36 s on average and passed zero trials at every tilt setting.

| Maximum start tilt | Fixed-pose mean survival | Fixed-pose settled | Ankle feedback settled | Ankle feedback mean peak COM drift |
| ---: | ---: | ---: | ---: | ---: |
| 0 rad | 3.46 s | 0/30 | 30/30 | 0.80 cm |
| 0.005 rad | 2.70 s | 0/30 | 30/30 | 0.78 cm |
| 0.015 rad | 2.32 s | 0/30 | 30/30 | 0.96 cm |
| 0.025 rad | 2.14 s | 0/30 | 30/30 | 1.28 cm |

At the highest tilt, fixed-pose tracking crossed the estimated support
boundary at 1.39 s on average. Ankle feedback recorded no crossings, with a
mean minimum margin of 6.74 cm. Per-episode results are in
`build/standing-ankle-balance-tilt-*.json`. See the
[repeatable evaluation command](../tools/README.md#ragdoll-standing-task).

The classical baseline meets the quiet-standing gate on the nominal rig.
The [ragdoll example](../scenes/ragdoll_example.py) shares the authored rig,
standing reference, motor caps and ankle feedback with the standing tool.
Its `--standing` replay shows ten seconds of joint-only balance on flat ground.

#### Learned ankle torque corrections

`--action-space residual` learns world-space X/Z torque corrections for each
ankle. The correction norm is capped at 10 N m per ankle. The controller
ignores corrections for airborne feet and clips the combined classical and
learned command to the existing sole bounds and ankle torque cap. The other
joints follow the tuned pose with their configured caps. The PPO action head
starts at zero for deterministic inference, which reproduces the classical
controller exactly. The residual trial uses the support-area reward and adds a
small penalty on changes to the normalized correction between policy ticks.

One PPO run used eight independent worlds, seed 77, ten-second episodes, and
flat-sole starts with pelvis spin sampled up to 0.1 rad/s. Later stages resumed
the policy and optimizer with successively larger starting tilt.

| Maximum tilt | New policy steps | Total policy steps | Local checkpoint in `build/` |
| ---: | ---: | ---: | --- |
| 0 rad | 65,536 | 65,536 | `standing-residual-flat.zip` |
| 0.005 rad | 40,960 | 106,496 | `standing-residual-tilt005.zip` |
| 0.015 rad | 40,960 | 147,456 | `standing-residual-tilt015.zip` |
| 0.025 rad | 65,536 | 212,992 | `standing-residual-tilt025.zip` |

Throughput was approximately 1,070–1,150 policy steps/s. On the 30 held-out
flat-sole starts (seeds 54,000–54,029), both the classical and residual
controllers passed every ten-second trial. Mean peak center-of-mass drift
was 0.80 cm for classical balance and 1.27 cm with the learned correction.

At the 0.025 rad tilt setting, all controllers used the same 30 held-out
starts (seeds 54,100–54,129).

| Controller | Settled trials | Mean survival | Mean peak COM drift |
| --- | ---: | ---: | ---: |
| Classical ankle balance | 30/30 | 10.00 s | 1.40 cm |
| Ankle balance with PPO correction | 30/30 | 10.00 s | 1.83 cm |
| Earlier compact motor-target PPO | 0/30 | 2.09 s | 80.38 cm |
| Earlier full motor-target PPO | 0/30 | 2.10 s | 81.48 cm |

The residual passed the standing gate. The classical controller had lower
drift at both settings and less accumulated foot slip at full tilt
(0.59 cm versus 1.20 cm per foot). The final learned correction averaged
1.62 N m per ankle before contact gating and pressure clipping. The older
motor-target policies used the foot-midpoint reward. These results describe
one residual training seed. The classical controller is the reference for
the push-recovery grid, with the learned correction available for comparison.
Per-episode reports are in `build/standing-residual-evaluation.json`.

#### Training curriculum

Advance when a validation set meets the Phase 5 criteria. Keep some easier
starts in later training stages and recheck earlier scenarios at each advance.

| Stage | Resets and disturbances | Behavior to learn |
| --- | --- | --- |
| Quiet standing | Flat floor, tuned stance, then small pose and velocity variation | Stand with low drift, a near-reference height and settled contacts. |
| Push recovery | Apply small horizontal impulses after an initial settle period. Grow direction, magnitude and application-point coverage as recovery improves. | Recover in place from small pushes and allow steps when needed. Episodes continue through valid steps. |
| Whole-body push recovery | Start with stronger flat-floor pushes, then vary stance and push direction near a platform edge with limited stepping room. | Use the torso and arms to counter rotation, and place a recovery step when there is room. |
| Wall-assisted recovery | Start beside a reachable wall. Vary side, distance, angle and impulse. Include starts with a wall just out of reach. | Reach with a hand or forearm, bear load through contact and release after regaining a stable stance. |
| Motion tracking | Start with a single retargeted walk cycle. Sample valid pose and velocity pairs along the clip and pass the current and upcoming reference pose to the policy. | Track the clip on flat ground, then add turning, perturbations and uneven surfaces. |
| Bracing and get-up | Start from controlled stumbles, falls, and valid prone or supine poses. Train bracing and ground recovery as distinct objectives before testing transitions. | Limit head and torso impact, regain foot support and return to a stable stance or resume the reference. Hand and knee contact is allowed. |

Vary friction, mass distribution and available motor strength after the nominal
rig meets the standing and push criteria. Introduce one variation at a time to
identify the source of regressions. Add uneven terrain to the tracking task
after flat-ground tracking.
Standing and tracking may end early on a fall. Bracing and get-up continue
through ground contact until success, the time limit or an invalid state. Treat
external time limits as truncations. A quiet-standing trial can pass at its
time limit. A get-up trial that reaches its time limit without getting up fails.

#### Whole-body recovery before a fall

1. Extend the push task to expose bounded motor targets on the usable axes of
   the full articulated rig, including the torso, shoulders and elbows. Keep
   the pelvis free and use the existing joint limits and torque caps. Compare
   the leg-only and whole-body policies on the same strong flat-floor pushes
   first, including the three cells where the hip and knee experiment fell.
2. Give the policy a local view of the support surfaces. Start with downward
   probes near the feet and possible landing positions, and short probes around
   the upper body for reachable walls. At the policy rate, encode each hit as a
   hit flag, scaled distance and surface normal in a heading-aligned frame.
   Exclude the ragdoll from its own queries, mask misses and use the same
   sampling in training and runtime. Include pelvis-relative hand and foot
   positions and velocities, and distinguish foot support and hand or forearm
   contacts by surface. On the 13-body rig, use the distal forearm position
   as the hand endpoint. [Physics queries](../graaal/physics.md#queries-filters-and-materials)
3. Use native collision and friction for wall support. Allow deliberate hand
   or forearm contact with a designated brace surface to continue the episode.
   Count head or torso impacts that indicate a fall, loss of upright posture,
   and dropping off the platform as failures. Train toward restoring quiet foot
   support after releasing the brace. Keep the foot support-area metric for
   floor balance and track wall load separately, since the wall can apply a
   horizontal reaction. Score fall protection and getting up in their own tasks.

#### Reward signals

Normalize distance terms by rig height or foot size and rate terms by fixed
physical scales. Log each component before weighting it so reward changes can
be checked against the behavior. Start with the task terms below and keep
smoothness and effort penalties small enough to allow corrective movement.

| Term | Signal and applicable stages |
| --- | --- |
| Upright stance | Reward pelvis and chest alignment with up and pelvis height near the standing reference. Penalize excessive crouching through the height term. Uses the full stance target for standing and for the final part of get-up. |
| Balance | Use the signed margin of projected center of mass to the estimated support area. Penalize approaching or crossing the boundary. Initially build the area from projected sole bounds of feet reporting ground contact and log native contact points to verify the approximation, especially during foot tilt or single support. Include horizontal center-of-mass speed near equilibrium. Mask the margin when airborne and allow brief support changes during steps. |
| Control quality | Penalize tangential foot motion during stance contact and changes in commanded joint targets. Use measured joint torque or mechanical work if available. Log tracking error and time near motor or joint limits as separate diagnostics. |
| Wall-assisted recovery | Reward restoring a stable stance after a push. Track brace contact, load and release as diagnostics. Penalize hard head or torso impacts and sustained reliance on the wall after balance returns. |
| Animation tracking | Reward local joint rotations and velocities, and pelvis-relative foot and hand positions near the reference. For locomotion also measure heading and speed error. Tune foot contact timing and slip against the clip. |
| Impact | Penalize new head and torso contacts by their estimated impulse, normalized by body mass. Track peak impact and approach speed separately. Give this term priority over upright posture during bracing. [Contact event fields](../graaal/physics.md#events-and-entity-identity) |
| Get-up | Reward progress toward a valid stance and give a completion reward after foot support, height and uprightness remain stable for a hold period. |

For the support-area experiment, keep the current posture reward as the
positive term. Replace the foot-midpoint penalty with a bounded penalty that
ramps up near the support boundary and reaches full weight outside it. Keep the
existing speed, slip and action-change penalties in the first comparison so
the effect of changing the support signal is visible in the learning curves.

**Deliverable** · Repeatable standing and whole-body push recovery on flat
ground, at platform edges and near walls, followed by reference tracking and
separate fall bracing and get-up capabilities.

### Phase 5 — Evaluation and refinement

Use paired starts and disturbances for the passive, fixed-pose, classical
joint-only and learned controllers as they become available. Keep held-out test
scenarios separate from training and checkpoint selection, and report episode
counts and per-seed results. Preserve the existing five-second test for
comparison with recorded baselines and use a longer horizon to check stability.

| Scenario | Acceptance check and reported metrics |
| --- | --- |
| Quiet standing | Initial target is at least 27 of 30 held-out ten-second trials on the nominal rig, including small seeded starting perturbations. Require no fall and a settled stance during the final second. Initial settle thresholds are foot support, horizontal center-of-mass speed below 0.05 m/s, pelvis height at least 90% of reference, and pelvis and chest up cosines of at least 0.95. Report survival time, minimum pelvis height, peak displacement from spawn, foot slip and support margin. |
| Push recovery | Use a fixed grid of direction, impulse magnitude and application point. For the nominal rig, start with 0.5, 1 and 2 N·s at the pelvis and chest in four cardinal directions at 2 s. Count trials that fall or fail to settle before the push as failures. Success means returning to the standing criteria for one continuous second within 5 s of the push. Target 90% success on the small-impulse grid, and report recovery time, step count and results for each cell, including larger impulses as the grid expands. |
| Platform-edge recovery | Compare leg-only and whole-body policies on paired pushes toward and along an edge, across held-out edge clearances and headings. Use the standing hold and recovery deadline from the push test. Report falls off the platform, landing positions and time to regain stance. Run a matched arm-and-torso ablation and review replays for counter-rotation when a step is constrained. |
| Wall-assisted recovery | Pair trials with a reachable wall, a wall out of reach and open floor. Vary wall side and orientation. Report recovery rate and time, hand or forearm contact duration and estimated impulse, head and torso impacts, and how often a supported hold ends in one second of stable foot support after release. |
| Tracking | Report completion rate, joint and end-effector tracking error, speed and heading error, and contact-phase foot slip on held-out starts and transitions. Review replays for sustained crouch, foot skating and jitter. |
| Bracing and get-up | Compare estimated peak head and torso impulses and speeds at contact with the passive baseline on identical falls. Report prone and supine success separately, time to stand and hold, and rate of successful return to tracking. Include stair falls after baseline flat-floor cases. |

Record the rig and tuning, policy and reward configuration, disturbance range,
training steps, seeds and throughput with each result. Save replay inputs or
recorded poses for representative successes and failures. Treat thresholds as
initial acceptance targets until measured on Graaal's rig.

**Deliverable** · Reproducible comparisons on held-out standing, recovery and
motion-tracking scenarios.

#### Push-recovery benchmark

The [push-recovery benchmark](../scenes/benchmarks/README.md#ragdoll-push-recovery) applies the
Phase 5 grid at the pelvis or chest center of mass on the tuned humanoid. Each
cell uses 10 held-out starts (seeds 55,000–55,009) with tilt up to 0.025 rad
and spin up to 0.1 rad/s. Both controllers must settle for one second
before the push. Recovery time includes the new one-second hold. A step
counts when a foot loses floor contact and lands at least 2 cm horizontally
from its last planted position.

| Controller | Recovered | Mean recovery | Steps |
| --- | ---: | ---: | ---: |
| Classical ankle balance | 240/240 | 1.000 s | 0 |
| Ankle balance with PPO correction | 240/240 | 1.006 s | 0 |

Both passed all 24 cells and met the 90% gate. The PPO policy took 2.342 s
after a 2 N·s chest push in world −Z on seed 55,008. Every other trial met
the standing criteria throughout the first second after the push. The
comparison used the 212,992-step `standing-residual-tilt025.zip` checkpoint.
The replay inputs and individual results are in
`build/ragdoll-push-recovery.json`.

In a separate one-seed probe, classical ankle balance recovered in 7 of 8
trials at 8 N·s and 5 of 8 at 16 N·s. A chest push in world −Z caused a fall
at both magnitudes. The larger-push probe is in `build/push-deadline-probe.json`.

#### Hip and knee control experiment

The stepping action space adds X/Z offsets at both hips and a flexion offset
at each knee to the four ankle corrections. Both PPO policies use the same
134 observations and start with zero deterministic output. They run at 60 Hz
and use the same tuned stance, motor caps, reward terms and sampling settings.
The hip and knee offsets use the existing motor targets, with caps of 0.45 rad
at each hip and offsets in [−0.05, 0.55] rad at each knee.

Both policies trained independently with seed 87 and eight workers for
196,608 policy steps each. Seven-second episodes used the standing tilt and
spin distributions. The impulse at two seconds sampled from 0, 4, 8 and 12 N·s,
with additional practice on chest pushes toward world −Z. See the
[training commands](../tools/README.md#ragdoll-standing-task).

The held-out grid used 2, 4, 8 and 12 N·s and five seeds (58,000–58,004) per
body, direction and magnitude. Each controller faced the same 160 trials.

| Controller | Recovered | Mean time among recoveries | Steps |
| --- | ---: | ---: | ---: |
| Classical ankle balance | 145/160 | 1.204 s | 0 |
| PPO ankle corrections | 145/160 | 1.190 s | 0 |
| PPO ankle corrections with hip and knee targets | 145/160 | 1.173 s | 0 |

All three recovered on the same trials. The fifteen falls came from −Z chest
pushes at 8 and 12 N·s and −Z pelvis pushes at 12 N·s. No foot left the floor
and landed at least 2 cm from its takeoff position. The extra motor targets
shaved about 17 ms off mean recovery versus PPO ankle corrections in this
one-training-seed experiment. The matched comparison used
`build/ragdoll-push-residual-seed87.zip` and
`build/ragdoll-push-stepping-seed87.zip`. Per-trial outcomes and foot landing
records are in `build/ragdoll-push-stepping-evaluation.json`.

#### Whole-body flat-floor experiment

The `whole-body` action space adds 16 constraint-axis targets for the chest,
neck, shoulders and elbows to the ten leg commands. Target offsets are
bounded to 0.22 rad at the chest and neck and 0.55 rad at the shoulders and
elbows. The pelvis stays free, and ankle control uses the same contact-gated
torque feedback and learned correction as the stepping policy. Zero upper-body
outputs reproduce the leg-only simulation and reward through a push.

The whole-body policy used the same 134 observations, reward, disturbances,
PPO seed 87, eight workers and 196,608 steps as the hip and knee experiment.
Its checkpoint is `build/ragdoll-push-whole-body-seed87.zip`. The [training](../tools/README.md#ragdoll-standing-task)
and [benchmark](../scenes/benchmarks/README.md#ragdoll-push-recovery) commands
reproduce the setup.

The paired benchmark repeated the 160 held-out starts from the hip and knee
experiment for each controller. The classical and stepping trial records
matched their previous run exactly.

| Controller | Recovered | Mean time among recoveries | Steps |
| --- | ---: | ---: | ---: |
| Classical ankle balance | 145/160 | 1.204 s | 0 |
| PPO ankle corrections with hip and knee targets | 145/160 | 1.173 s | 0 |
| PPO whole-body targets with ankle corrections | 145/160 | 1.145 s | 0 |

All three recovered on the same trials. The whole-body policy fell on the
same 15 −Z pushes and completed no recovery steps. Among the 145 paired
recoveries, it was faster than the leg policy on 33, slower on 25 and tied on
87. The mean was 29 ms shorter in this one-training-seed comparison. The
per-trial results are in `build/ragdoll-push-whole-body-evaluation.json`.
Validation covered 70 focused ragdoll, benchmark and library-boundary tests,
plus the CPU smoke test.

#### Scripted backward step

The [scripted benchmark](../scenes/benchmarks/README.md#ragdoll-push-recovery)
uses the push schedule to trigger a right-foot step for the three failed
conditions. The controller lifts and sweeps the ankle along a timed arc,
presses down at the landing, then blends back to the standing targets.
The foot-space commands become paired hip, knee and ankle torques under
the standing torque caps. The pelvis stays free and all movement comes
from joint actuation and native contacts. The report tracks contact duration
and estimated floor-contact impulse after each counted landing.

The paired grid used 2, 4, 8 and 12 N·s with five fresh starts per cell
(58,200–58,204). Classical balance and the scripted controller recovered on
the same 145 of 160 trials. The script stepped on all 15 target trials. The
first right-foot landing moved backward by a median of 8.4 cm and stayed in
continuous floor contact for a median of 1.73 s. Fourteen trials ended in
falls and one timed out. The 145 trials outside the activation conditions
matched classical balance exactly. The report is
`build/ragdoll-scripted-backstep-evaluation.json`.

The current script demonstrates foot clearance and a sustained landing.
Make the transition back to standing respond to the planted foot, projected
center of mass and pelvis velocity before using it as a PPO recovery reference.
Use the same held-out recovery criteria for that transition. The scripted
step and the unchanged standing paths passed 71 focused tests and the CPU
smoke test.

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
