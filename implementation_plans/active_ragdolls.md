# Active ragdoll plan

## Scope and architecture

Build a physically simulated humanoid that can stand, recover from pushes with
whole-body motion or steps, brace against its environment, follow animation
references, protect itself during falls and get up. [Native physics integration](physics_integration_plan.md)
and the [ragdoll API](../graaal/physics.md#native-ragdolls-and-skeletons)
already exist; this file tracks **passive mechanics and learned control**.

- **Bulk training:** Use GPU-batched Genesis (`genesis-world`) simulation and
  RSL-RL PPO with a PyTorch MLP, without mocap or motion priors. Train pose
  matching and push recovery against the authored humanoid.
- **Fine-tuning and runtime:** Retain the existing Jolt standing task,
  workers, benchmarks and viewer for final fine-tuning and evaluation. Jolt
  remains the runtime physics backend through pinned `pyjoltphysics 0.1.0a3`.
  Export the final actor and its observation normalization to ONNX for Python
  runtime inference with ONNX Runtime.
- **Control:** The policy proposes bounded residual joint targets around an
  authored pose. Genesis PD motors and Jolt motors must use a shared, explicitly
  mapped observation/action contract. Preserve the existing Jolt passive
  springs and soft limits when measuring or fine-tuning against that setup.
- **Rates:** Start at 120 Hz physics and 60 Hz policy actions in both backends.
  The pelvis is free: **no root/pelvis assistance** in learned training or
  evaluation.
- **Boundary:** Calibrate the existing Jolt frequency/damping motors against
  Genesis PD response; the pinned ragdoll API does not expose direct
  stiffness/damping motor settings. Native binding changes are out of scope.

An experimental Genesis environment, approximate rig export and RSL-RL
trainer are implemented; the cross-backend policy bridge, Jolt fine-tuning
path and ONNX runtime are **planned**, not implemented. The existing Jolt
training path below continues to work independently. On the current
Windows machine, Genesis 1.5.0 and PyTorch 2.14.1+cu130 were installed in
the regular Python user packages; PyTorch detected the RTX 5060 Ti and ran a
CUDA tensor operation, and a Genesis GPU physics step succeeded. RSL-RL 6.0.0
and ONNX are also installed in that interpreter. This is local setup evidence,
not a universal performance measurement.

## Implementation history and findings

The measurements below come from recorded Windows x64 Jolt runs. They provide
baselines and diagnostics for the new training and transfer pipeline.

<a name="backend-validation-status"></a>
<a name="authored-passive-humanoid-and-contact-scenarios"></a>

### 1. Native validation and passive mechanics — implemented

- Confirmed batched body torques, owned snapshots preserving bone UID order,
  detached body reads, per-bone motor limits, independent Jolt worlds and
  reproducible fresh-world rollouts. Selective in-place ragdoll reset and full
  live-world replay still lack native held-control-state capture.
- Implemented standalone NumPy
  [neutral joint springs](../graaal/physics.md#passive-neutral-joint-springs)
  and `joint_limit_torques` with authored asymmetric twist/swing bounds,
  quadratic restoring forces, outward-only damping, independent caps and
  equal/opposite parent reactions. Native hard limits remain backstops.
- On the 20-ragdoll zero-gravity benchmark, a full control tick measured
  **0.531–0.685 ms** with neutral springs, or **1.236–1.374 ms** with both
  passive force paths (median ranges across configurations). The latter's
  soft-limit evaluation alone was 0.559–0.575 ms. See the
  [control benchmark](../scenes/benchmarks/README.md#ragdoll-control-prerequisites)
  and local `scenes/benchmarks/results/ragdoll_*_windows.json` reports.
- Built the 13-body, 1.8 m/70 kg
  [passive humanoid](../scenes/passive_ragdoll_example.py), authored relaxed
  posture, ankle joints and contact scenarios (zero gravity, tilted drop,
  stairs and chest impulse). Mean zero-gravity pose error fell from **0.3920
  to 0.0886 rad**; no-passive control remained at 0.3920 rad. Contact and
  replay checks passed on the recorded configurations.
- Added collider/dummy replay and the [ragdoll tuner](../tools/ragdoll_tuner.md)
  for masses, colliders, joint limits and passive gains. Passive drops and
  impacts were checked, but anatomical tuning and other platforms remain open.

### 2. Actuation, standing and training workers — implemented

- `PhysicsRagdollControl` exposes detached observations and bounded joint
  targets. The [active example](../scenes/active_ragdoll_example.py) tests
  reference tracking and a **root-assisted classical probe**; its assisted
  outcome is **not** a free-root balance result.
- The [standing task](../tools/ragdoll_standing.py) uses one native world per
  subprocess worker, rebuilds that world on reset, supports deterministic
  seeds and resumable PPO checkpoints. With eight workers, later runs reached
  roughly **1,070–1,150 policy steps/s**, including fresh-world resets.
- A tuned stance has flat soles, slight knee bend and projected COM within
  the estimated support area. The early 30- or 48-output direct motor-target
  PPO pilots reached **0/30** settled ten-second starts after 250,880 steps;
  fixed-pose motors also reached **0/30** under the paired tilted starts.
- Classical **ankle torque balance** (contact-aware pressure feedback, other
  joints following the stance, no root assistance) reached **30/30** settled
  trials with tilt up to 0.025 rad and spin up to 0.1 rad/s. Its ankle cap is
  100 N m. A PPO ankle-torque residual also reached **30/30**, but with
  **higher peak COM drift** (1.83 vs 1.40 cm at maximum tilt) after 212,992
  steps. Retain classical balance as the reference and protect these easier
  cases from regressions.

### 3. Push-recovery experiments — implemented, hard cases unsolved

| Experiment | Paired result | Finding |
| --- | --- | --- |
| Small-push grid, classical / PPO ankle residual | Both **240/240** (0.5–2 N·s) | Stable in-place recovery; no steps needed |
| 2–12 N·s grid, classical / PPO ankle / PPO hip-knee | All **145/160** | Same 15 falls; no steps; hip/knee targets saved only ~17 ms on recoveries |
| Same grid, PPO whole-body targets | **145/160** | Adding torso, shoulder and elbow targets saved ~29 ms on recoveries but did not fix any fall |
| Scripted right-foot backward step | **145/160** | Triggered steps on all 15 difficult trials, but no hard-push recovery; landing alone was insufficient |
| Contact-based recovery PPO, 196,608 steps | **111/120**, same as ankle baseline | All nine hard backward pushes still failed; **zero learned steps** |

The 15 failures in the 160-trial grid were specifically **world `−Z`
chest pushes of 8 and 12 N·s** and **`−Z` pelvis pushes of 12 N·s**.
The 120-trial recovery grid repeated nine hard backward failures. The
whole-body policy used 26 motor outputs; the later recovery experiment added
foot selection/request, landing offsets, swing timing and loading control to
those outputs. These are **historical implementations, not the intended new
control architecture**.

The recovery pilot used eight workers, PPO seed 87 and 196,608 collected policy
steps. KL early stopping shortened **optimizer updates** on **180/192
rollouts**; it did not shorten collection of the recorded rollout steps. The
pilot checkpoint is `build/ragdoll-recovery-seed87.zip`.

Forced-request diagnostics explain the two separate failure modes:

- The policy's maximum absolute step request was **0.088**, below the **0.18**
  activation threshold; no learned swing could start.
- Forcing a left-foot step yielded a counted landing in **9/9** difficult
  trials, but **0/9** recovered. Horizontal COM speed **increased** during
  the first 0.2 s after every left-foot touchdown. The contact-and-loading
  script therefore demonstrated landing, **not effective braking**.

Relevant reports: `build/standing-regimen-evaluation.json`,
`build/standing-residual-evaluation.json`,
`build/ragdoll-push-stepping-evaluation.json`,
`build/ragdoll-push-whole-body-evaluation.json`,
`build/ragdoll-scripted-backstep-evaluation.json`,
`build/ragdoll-recovery-evaluation.json`, and
`build/ragdoll-recovery-step-trace.json`. See the
[training guide](../tools/README.md#ragdoll-standing-task) and
[push benchmark](../scenes/benchmarks/README.md#ragdoll-push-recovery).

### 4. Direct continuous motor control

`--action-space direct` gives the MLP 30 bounded joint target offsets at
60 Hz on the built-in humanoid. The free pelvis, passive springs, soft limits
and native hard limits remain active. The 187 inputs cover heading-relative
COM and body motion, foot and lower-arm motion relative to the pelvis, joint
state, foot clearance and tilt, contact and support, and previous commands.
Contact informs the observations and reward. The motor targets are active
throughout the episode. The request-based policy remains a comparison baseline.

The direct push benchmark accepts a saved rig and checkpoint. It reports the
three historical hard cells separately and includes settling, post-push COM
speed, foot clearance, target saturation and touchdown measurements.

The first direct PPO run used the default 10 Hz joint motors. Seed 87
completed 196,608 policy steps over 192 rollouts. KL triggered early stopping
in 161 rollouts, and the resulting checkpoint settled in 0/30 held-out quiet
starts. It fell after 2.07 s on average and used less than 0.98 of the target
range on every measured quiet control tick.

A motor-response probe found a stable COM feedback law using the same
continuous ankle targets at 20 Hz motor frequency and 0.8 damping. The
`--direct-balance-init` option seeds the MLP weights from that feedback law.
PPO then controls the same outputs while collecting a mix of quiet, light
and hard episodes. Each environment starts from stance and runs at 120/60 Hz.

### 5. Warm-start direct control

Paired held-out starts begin at seed 58300 with up to 0.025 rad tilt and
0.1 rad/s spin. Recovery time includes the one-second settled hold. The
original grid combines 2/4/8/12 N·s, two application bodies and four
directions. The hard grid repeats the three historically failing −Z cells
with ten starts per cell.

| Controller | Quiet 10 s | 0.5/1/2 N·s grid | Original grid | Hard −Z | Mean hard recovery | 14/16/18 N·s grid |
| --- | --- | --- | --- | --- | --- | --- |
| Classical ankle feedback | 30/30 | 240/240 | 145/160 | 0/30 | — | 80/120 |
| Direct initialization | 30/30 | 240/240 | 160/160 | 30/30 | 1.425 s | 60/120 |
| Direct PPO seed 87, 65,536 steps | 30/30 | 240/240 | 160/160 | 30/30 | 1.242 s | 60/120 |
| Direct PPO seed 88, 65,536 steps | 30/30 | 240/240 | 160/160 | 30/30 | 1.422 s | 57/120 |

The paired ankle residual and request-based recovery policies also scored
0/30 on the hard grid. Seed 87 reduced mean recovery time from 1.876 s
to 1.335 s for the 12 N·s chest case. All these recoveries occurred in place.
The 14–18 N·s grid still exposes a narrower lateral and backward recovery
range with direct control. The 20 Hz tuning and checkpoint initialization
account for the original recovery-count improvement.

The saved tuning is `build/ragdoll-direct-20hz-rig.json`. Checkpoints and
reports share the prefix `build/ragdoll-direct-warm-` and identify the
training steps, seed and evaluation grid in their filenames. The
[training guide](../tools/README.md#ragdoll-standing-task) and
[benchmark guide](../scenes/benchmarks/README.md#ragdoll-push-recovery)
show how to run the same paths.

### 6. Stronger pushes and optimizer diagnostics

Two matched continuations of seed 87 collected 65,536 additional steps
each with training seed 89 and a 3e-6 learning rate. The disturbance
list was `0 0 0 1 2 4 8 12 14 14 16 16 18 18` N·s. One run used the
braking, support and landing compliance terms. The other disabled all
three. Both settled on 30/30 quiet starts, recovered on 240/240 light
pushes and 160/160 original-grid pushes, and reached 60/120 at 14–18 N·s
with zero relocated recoveries. Mean hard-cell times were 1.379 s
with shaping and 1.294 s with the ablation. The training logs recorded
zero compliance reward, so the difference does not isolate landing
compliance.

At held-out seed 58300, the traced 14 and 16 N·s backward chest
failures and the 14 N·s lateral chest failure kept both feet in
contact throughout the first 0.5 s. At least one ankle target reached
98% of its range in 29 of 30 control samples. Every other target
stayed below 1.2% of its range. See
`build/ragdoll-direct-strong-control-traces.json` and the
`build/ragdoll-direct-strong-*` reports.

A third matched continuation raised initial non-ankle exploration from
about 0.015 to 0.10 in normalized action units. It retained 30/30 quiet
and 30/30 hard backward results. It scored 58/120 on the stronger grid,
with zero relocated recoveries. This increased exploration during
training while the evaluated policy continued to recover in place.

With the original push curriculum and the same training seed 87,
reducing the learning rate from 1e-5 to 3e-6 reduced KL early stops
from 63/64 to 17/64 rollouts. Both recovered 30/30 hard pushes with mean
times of 1.242 s and 1.237 s respectively. The instrumented eight-rollout
comparison accepted 2 full epochs at 1e-5 and 33 at 3e-6, out of 40
available in each run. The KL target remained 0.03. The traces are
`build/ragdoll-direct-optimizer-*.jsonl`.

### 7. Stepped recovery from backward chest pushes

The fitted right-foot reference produces relocated landings during training.
Its optional stronger-push variant uses a farther landing target starting at 14 N·s. The
backstep policy learns joint residuals after swing. It sees elapsed time
since the push, and the benchmark uses the scheduled impulse to choose and
trigger the reference. The pelvis remains freely simulated. These runs
measure how well the policy loads the relocated foot and settles under that
reference.

The replayable grid pairs ten starts (60700–60709) at each magnitude from
11 through 16 N·s. The swing ablation keeps the same checkpoint, reference
timing and body motor targets and removes the swing and loading commands.
The braking-step metric requires 5 cm of landing displacement along the
push, 0.05 s of maintained contact and a speed drop of at least 0.02 m/s
over the first 0.2 s after landing, in a trial that subsequently settles.
The [training guide](../tools/README.md#ragdoll-standing-task) and
[benchmark guide](../scenes/benchmarks/README.md#ragdoll-push-recovery)
include commands to reproduce and browse the trials.

| Controller | 11–13 N·s | 14 N·s | 15 N·s | 16 N·s | Recovered with braking step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ankle balance | 0/30 | 0/10 | 0/10 | 0/10 | 0 |
| Direct PPO seed 87 | 30/30 | 0/10 | 0/10 | 0/10 | 0 |
| Fitted reference | 2/30 | 2/10 | 2/10 | 0/10 | 0 |
| Backstep PPO, 33,024 steps | 3/30 | 2/10 | 2/10 | 0/10 | 2 |
| Backstep PPO, 82,176 steps | 2/30 | 2/10 | 3/10 | 0/10 | 1 |
| Strong-reference PPO, 57,600 steps | 4/30 | 0/10 | 0/10 | 1/10 | 1 |

Every PPO backstep recovery included a relocated foot. Removing swing and
loading reduced recovery to 0/60 for each checkpoint. The 82,176-step run added
one recovery at 15 N·s while losing one at 11 N·s. The strong-impulse
continuation recovered two-step seed 60700 at 16 N·s, with the first foot
landing 8.5 cm farther back, and had fewer successful strong trials overall.
The 33,024-step checkpoint also settled on all 30 ten-second quiet starts.
The braking criterion was met only on the 11–12 N·s cells.

The main paired report is `build/ragdoll-backstep-holdout.json`. The
strong-reference report is
`build/ragdoll-backstep-heavy-57344-seed93-holdout.json`. Training used seed 93
for the first checkpoint and seeds 94 and 95 for the two continuations.

### 8. Impact costs and episode length

Backstep PPO now penalizes peak COM acceleration and estimated floor impulse
above calibrated thresholds. The first physics tick of the scripted push is
excluded. Supported posture earns the training hold bonus. The benchmark
also requires slow horizontal motion for the settled hold.

Two continuations from the 82,176-step checkpoint each collected 33,024 policy
steps with seed 96 and the same push curriculum. They used five- and ten-second
limits after the push. All policies were evaluated on the same 60 trials with
ten seconds after the push. A sustained step here means a relocated landing
after at least 0.05 s out of contact.

| Policy | Recoveries | Trials with at least two sustained steps | Latest follow-up landing |
| --- | ---: | ---: | ---: |
| Prior checkpoint | 7/60 | 29/60 | 0.958 s |
| Impact reward, 5 s after push | 9/60 | 30/60 | 0.992 s |
| Impact reward, 10 s after push | 6/60 | 30/60 | 1.008 s |

Every successful trial had at least two sustained steps, and all recoveries
finished within five seconds. In the final training logs the last 100
episodes averaged 2.72 and 2.78 seconds from the end of the scripted swing.
The longer window offered little extra practice on this grid. All three
runs used a fixed first swing and scripted hip pitch through and after
loading. Impact peaks stayed similar across the three checkpoints.
The full reports are `build/ragdoll-backstep-impact-{prior,7s,12s}-holdout.json`.

### 9. Target-pose hold and post-landing hip control

The pose-recovery mode ends on a full second near the target posture with foot
support, slow COM motion and low pelvis spin. The five-second window allows an
active hold to finish within one extra second. Early failure requires a persistently
low or tilted pelvis while the trunk contacts the floor or motion has slowed.
The dense impact costs still use acceleration and estimated floor impulse.

Two continuations from the 115,200-step impact checkpoint collected 12,288
policy steps each with seed 97 and the same push curriculum. The hip-release
run blends both pitch targets back to PPO over 0.25 seconds after loading.
All three policies used pose recovery on 60 paired held-out chest pushes.

| Policy | Recoveries | Holds finished during grace | Recoveries at 14–15 N·s |
| --- | ---: | ---: | ---: |
| Starting checkpoint | 5/60 | 2 | 3/20 |
| Continued with scripted hips | 3/60 | 2 | 2/20 |
| Continued with released hips | 11/60 | 2 | 6/20 |

Every recovery included at least two relocated landings with at least 0.05 s
of airborne time and 0.05 s of contact each. One recovery with released hips
met the benchmark braking-step threshold. All three policies failed at 16 N·s.
The relaxed failure rule exposed hard floor impacts during failed trials.
Across the 60 trials, mean peak COM acceleration and estimated floor impulse
were 129 m/s² and 39.5 N·s at the start and 103 m/s² and 26.9 N·s with released
hips. This is a single-seed comparison. The viewer discovers the three paired
reports in `build/ragdoll-backstep-pose-{start,locked,hips}-holdout.json`.

Standing CLI baseline comparisons use the Backstep push sampler for matching
disturbances under both push sources. Training checks that the configured
pushes can activate the reference. A resumed PPO rollout verified pose recovery
with curriculum-generated pushes. The benchmark records landing contact duration
for every controller so its braking-step criterion applies across comparisons.

### 10. Touchdown support signal

An optional Backstep reward measures the change in COM support margin during
the first 0.2 seconds after each foot touchdown. It evaluates both COM positions
against the current support polygon. The existing acceleration and impact
costs remain active.

Two 12,288-step continuations from the 127,488-step released-hip checkpoint
used seed 98 and the same push curriculum. On the 60 paired held-out pushes,
the reward run recovered 10 and the control recovered 12. Both recovered
seed 60703 at 16 N·s. Neither had a recovery meeting the braking-step criterion.
Mean peak COM acceleration and estimated floor impulse were 105.79 m/s² and
28.01 N·s with the signal, and 103.66 m/s² and 28.91 N·s for the control.
The starting checkpoint recovered 11/60 with one braking step. This single
seed provides no evidence that the new signal improves recovery. Reports are
`build/ragdoll-backstep-touchdown-holdout.json` and
`build/ragdoll-backstep-touchdown-control-holdout.json`.

### 11. Scaling the Backstep training budget

The 139,776-step control checkpoint continued for 360,448 policy steps in
three stages with rollout seeds 99, 100 and 101. Training used the released
hip targets, the same backward chest push curriculum and a learning rate of
0.00005. Eight workers collected 1,024 steps per rollout with five PPO epochs
scheduled per rollout. The optimizer completed 1,759 of 1,760 full epochs.

Each checkpoint ran the same 11–16 N·s grid on the original starts
60700–60709 and on fresh starts 61700–61709.

| Cumulative steps | Original recoveries | Fresh recoveries | 16 N·s recoveries across both sets |
| ---: | ---: | ---: | ---: |
| 139,776 | 12/60 | 9/60 | 1/20 |
| 205,312 | 10/60 | 10/60 | 0/20 |
| 336,384 | 14/60 | 13/60 | 2/20 |
| 500,224 | 13/60 | 13/60 | 1/20 |

Fresh-start mean peak COM acceleration and estimated floor impulse fell
from 120.59 m/s² and 36.96 N·s to 100.11 m/s² and 28.88 N·s. The highest
combined recovery count was 27/120 at 336,384 steps in
`build/ragdoll-backstep-scale-336384-seed100.zip`. The latest checkpoint,
`build/ragdoll-backstep-scale-500224-seed101.zip`, recovered 26/120. These
stages form one checkpoint lineage. The viewer discovers the paired reports as
`build/ragdoll-backstep-scale-*-holdout.json` and
`build/ragdoll-backstep-scale-*-fresh.json`.

### 12. First-touchdown analysis

Across both benchmark start sets for the 336,384-step checkpoint, all 60
trials at 14–16 N·s recorded a landing. Median first touchdown was 0.325 s
after the push. In 5/60 trials, horizontal COM speed dropped by at least
0.02 m/s between first touchdown and 0.2 s later. This counts the speed
criterion at the first landing across all outcomes. The benchmark's braking-step
recovery metric also requires displacement, continuous contact and a later
settled hold, and can qualify a subsequent landing.

Backstep schedules the first swing from the known push and fixes its timing,
swing side and landing offset for each reference variant. PPO first changes
the commands after the swing phase. Separating a poorly placed foot from a
foot that fails to take load calls for traces of the predicted capture point,
planned versus actual touchdown, per-foot load and slip.
[Backstep implementation](../tools/ragdoll_recovery_step.py)

### 13. Adaptive stepping baseline

The benchmark's `--adaptive-step` controller uses measured COM movement,
torso lean and foot contact. Once the balance estimate stays outside planted
support while the body is moving, it evaluates both possible stance feet and
three swing durations. It scores the predicted capture point against the
proposed support area, limits foot travel, and reassesses after loading to
allow another step. The torso and stance leg use bounded motor targets and
both feet can contribute ankle feedback after touchdown.

The report records the planned and actual touchdown, swing torque demand
relative to the cap, signed COM velocity along the push, planted drift and
estimated floor impulse. The viewer shows the projected support boundary,
capture point and touchdown markers in **View → Balance overlay**, with
per-foot estimated load and horizontal contact speed at the playhead.
[Benchmark and replay commands](../scenes/benchmarks/README.md#ragdoll-push-recovery)

The paired comparison reused the 11–16 N·s chest pushes and both start sets
from section 11. The quiet-standing regression settled in 30/30 trials and
all 240 light pushes recovered without a step request. A motion check in the
trigger resolved quiet false positives found in the first sweep.

| Controller | Original starts | Fresh starts | 14–16 N·s | Recovered with a braking step |
| --- | ---: | ---: | ---: | ---: |
| Ankle balance | 0/60 | 0/60 | 0/60 | 0/120 |
| Fixed Backstep | 2/60 | 6/60 | 4/60 | 1/120 |
| Backstep PPO, 336,384 steps | 14/60 | 13/60 | 16/60 | 4/120 |
| Adaptive step | 3/60 | 2/60 | 3/60 | 2/120 |

The adaptive controller completed at least two planned touchdowns in 58/60
strong-push trials. Its median horizontal error at the first planned touchdown
was 8.6 cm, and median planted drift over the first 0.2 s of the first relocated
landing reached 10.1 cm. Median peak swing torque demand was 3.33 times the
cap at the first planned step. Only 2/60 strong trials met the speed-drop
criterion after the first relocated touchdown. Seed 60707 recovered with two
planned steps at 15 N·s. Both touchdown placement and load transfer remain
limiting factors before using this controller to seed PPO.

The viewer discovers `build/ragdoll-adaptive-recovery-{holdout,fresh}.json`
and the `light` and `quiet-replay` reports under the same prefix.

### 14. Initial Genesis training prototype

The [Genesis trainer](../tools/README.md#experimental-genesis-ragdoll-training)
exports the 13-body rig to MJCF and runs RSL-RL PPO with 129 observations and
30 residual hinge targets at 60 Hz, with 120 Hz physics. The pelvis is free.
The exporter preserves rest joint positions and body masses in a MuJoCo check;
independent hinge limits approximate the Jolt swing/twist constraints. Genesis
also filtered two neutral self-collision pairs. Motor calibration, passive
forces, exact collision/limit parity, randomization, Jolt policy transfer and
ONNX runtime integration remain open.

An initial seed-1 run with 64 environments, 1 N·s scheduled chest pushes and
100 PPO iterations collected 153,600 policy steps. A 64-environment comparison
using seed 10001, the same initial pose and a five-second limit recorded 0/64
completed episodes for both the trained actor and zero residual actions; mean
survival was 2.30 s and 2.00 s respectively. These are survival measurements,
not settled standing or successful push recoveries, and do not establish that
each episode received a push. A quiet run gave the same completion and mean
survival values, so the result does not demonstrate recovery from the
scheduled pushes. A zero-action quiet trace showed the pelvis drifting
backward with increasing tilt while both feet remained near their initial
positions; it fell at approximately two seconds. The checkpoint is
`build/genesis-standing-seed1/model_99.pt`. Diagnose the quiet stance and rig
discrepancies before scaling the training budget.

### 15. Genesis balance initialization and push training

The original Genesis motor gains let the rig tip backward during quiet
standing. A GPU gain sweep found that 500 N·m/rad stiffness, 20 N·m·s/rad
damping and COM position/velocity ankle feedback could maintain standing for
ten seconds in the tested cases. The trainer can add four COM balance features,
initialize its actor near that feedback law, vary starting tilt and spin,
sample quiet and nonzero pushes, and resume compatible PPO checkpoints.
Selective randomized resets were checked on a four-environment GPU scene.

With the feedback-initialized actor, seed 3 collected 153,600 policy steps
using mixed quiet events and chest pushes up to 4 N·s. A paired held-out
evaluation used 64 initial states with up to 0.025 rad tilt and 0.1 rad/s
spin, a single chest push at two seconds and a ten-second episode. A success
requires a full final second with foot support, near-reference pelvis height,
upright pelvis/chest and horizontal COM speed below 0.05 m/s. Push directions
are sampled across world ±X and ±Z. The evaluation does not yet enforce
per-direction counts, measure foot slip or verify controlled stepping.

| Policy | Quiet | 12 N·s | 16 N·s | 20 N·s |
| --- | ---: | ---: | ---: | ---: |
| Feedback-initialized, one PPO update | 64/64 | 64/64 | 49/64 | 49/64 |
| PPO seed 3, 153,600 steps | 64/64 | 64/64 | 49/64 | 18/64 |
| Seed 3 plus 307,200 stronger-push steps | 64/64 | 49/64 | 49/64 | 35/64 |
| Initialized actor plus 307,200 stronger-push steps | 64/64 | 49/64 | 49/64 | 49/64 |
| Initialized actor plus 307,200 varied-start, 0–12 N·s steps | 64/64 | 64/64 | 49/64 | 49/64 |
| Varied-start actor plus 307,200 backward-focused steps | 64/64 | 49/64 | 49/64 | 49/64 |
| Varied-start actor plus 614,400 broad-limb exploration steps | 64/64 | 49/64 | 49/64 | 49/64 |
| Foot-observing actor plus 614,400 hard-push steps | 64/64 | 49/64 | 49/64 | 49/64 |

All 16 N·s failures were the 15 sampled −Z trials. The 20 N·s outcomes show
that additional PPO training did not improve on initialization on this grid,
and some continuations regressed at 12 N·s. The varied-start checkpoint is
`build/genesis-balance-varied/model_199.pt`. The table uses one evaluation seed
and a narrow chest-push scenario, not evidence of general recovery or Jolt
transfer. On a second set of 64 starts, the varied-start checkpoint again
settled 64/64 through 12 N·s and 45/64 at 16 and 20 N·s; all 19 failures
were backward pushes. A third 64-start set with pelvis pushes settled 64/64
at 4, 8, 12 and 16 N·s. On 32 further starts with fixed −Z chest pushes,
the varied-start policy settled 32/32 at 12 N·s, 28/32 at 13 N·s and
0/32 at 14–16 N·s. A 16 N·s trace showed ankle actions saturating while
other joint actions remained near zero and neither foot made a meaningful
recovery step. A 614,400-step continuation with 0.5-rad non-ankle target
range, 0.12 initial non-ankle exploration, a 3e-6 learning rate and the
backward-focused push curriculum still failed the strong backward cell and
regressed at 12 N·s. Another 614,400-step run added foot position, velocity
and contact observations and also scored 49/64 at 12, 14, 16 and 20 N·s on
the paired grid; it did not improve the backward cases. A separate 64-case
probe varied a scripted right-leg hip/knee target, timing and duration after
a 16 N·s backward push. It produced up to 0.53 m of horizontal foot movement,
but 0/64 trials reached the final settled hold. These are single-start,
oracle-timed diagnostics; foot movement alone does not establish a step or
effective load transfer. Next investigate contact-aware support, reachable
step placement and landing load transfer rather than assuming additional
exploration alone will solve the failure. Reports are in the corresponding
`build/genesis-balance-*/holdout.json` and `fresh.json` files. Keep the
Genesis-only workflow until a policy has demonstrated idle
standing and push recovery on broader held-out cases; do not begin Jolt
fine-tuning merely because these initial tests pass.

### 16. Genesis step-focused reward

The evaluator now tracks post-push airborne motion, relocated landings and COM
braking; its thresholds and contact limitations are in the
[training guide](../tools/README.md#experimental-genesis-ragdoll-training).
The standing checkpoint recorded no qualifying steps on the 12–16 N·s
backward chest grid.

Two 614,400-step runs used the same seed, randomized starts, a single mixed
quiet/12–18 N·s push curriculum, foot observations, 0.5-rad limb targets and
0.12 non-ankle exploration. One added the experimental movement, landing,
braking and settled-hold rewards; the other was a no-reward control. On 32
paired held-out starts, both settled 32/32 quiet trials. At 12 N·s the reward
run settled 29/32 and the control 0/32; both settled 0/32 at 14 and 16 N·s.
Neither produced a qualifying held-out landing or braking step. The reward
run logged occasional qualifying landings during training but no braking or
settled-hold reward. This one-seed comparison does not establish a reliable
step policy. Reports are `build/genesis-step-reward/holdout.json` and
`build/genesis-step-control/holdout.json`; retain the earlier standing
checkpoint while investigating reachable landing and load transfer.

### 17. Oracle-timed Genesis step reference and imitation

**Correction:** The reference benchmark used the standing actor's non-ankle
actions at the trial's 0.5-rad scale instead of the actor's 0.22-rad scale.
The reference counts and trajectory labels below describe that earlier,
mismatched setup and must not be used as evidence for the corrected reference.
With proper scaling, the selected 16 N·s candidate and its swing ablation both
settled 0/64 on the same paired starts, with floor-attributed contacts. See
`build/genesis-step-corrected-candidate223.json` and
`build/genesis-step-corrected-candidate223-noswing.json`. Previously trained
actors were evaluated independently, but were trained on affected labels;
regenerate reference datasets before further imitation training.

With corrected scaling, a new 1,024-candidate search found 21 braking-step
holds from its fixed starting stance. Candidate 12 then recovered 32/64 varied
starts at 16 N·s with floor-attributed contact and a final settled hold;
removing its swing commands recovered 0/64 on the same starts. A separate
128-start collection yielded 70 corrected braking-step demonstrations. These
are scripted with known push timing and are not learned-policy results. See
`build/genesis-step-corrected-candidate12.json` and
`build/genesis-step-corrected-teacher.json`.
Imitation trained only on these corrected demonstrations held 32/32 quiet
starts when motion-gated. On a fresh 32-start backward-push grid, its epoch-30
checkpoint settled 32/32 at 12 N·s and 2/32 at 14 N·s without a qualifying
step, and 0/32 at 16 N·s. The epoch-90 and final checkpoints settled none of
the push trials. These results do not demonstrate learned recovery stepping;
see `build/genesis-step-corrected-imitation/holdout.json`.

The [joint-target search](../scenes/benchmarks/README.md#genesis-recovery-step-search)
uses the known push time to probe swing, stance and landing commands without
root assistance. A 1,024-candidate local search produced five trials with
both a measured braking step and a final settled hold. One selected reference
recovered with a braking step in 38/64 varied starts under a 16 N·s backward
chest push; removing its swing commands gave 0/64 on the same starts. A fresh
128-start collection yielded 76 braking-step recoveries. Repeating the 64-start
evaluation with foot-to-floor contact pairs gave the same 38/64 result. Slip
and impact severity remain unmeasured.

Those successful trajectories train a 147-observation neural actor without
push-time input. A 30-epoch imitation actor settled in 32/32 held-out quiet
and 12, 14 and 16 N·s backward trials, but produced no qualifying recovery
steps. A more heavily weighted 200-epoch actor failed quiet standing before
the push. These are imitation experiments, not evidence that the actor has
learned to step. Reference search, ablation, trajectory and actor artifacts
are under `build/genesis-step-*`; continue Genesis training and evaluate
quiet standing, held-out pushes and actual braking steps without oracle timing.

A blend triggered by measured backward COM velocity preserved 32/32 quiet
starts without false triggers, but the initial step actor did not settle on
any of 32 held-out 16 N·s pushes. A 307,200-step PPO continuation with step
rewards also settled none of 32 trials at either 12 or 16 N·s, despite
relocated landings. It logged some braking events during training but no
settled-hold rewards. Reports are in `build/genesis-step-gated-train/`.
The trigger uses measured motion, whereas the reference labels still use
the known push time. Student-state collection produced 128 additional failed
trajectories with oracle labels; these are not successful demonstrations.
Continuing imitation from the student on both datasets for 60 epochs yielded
32/32 quiet holds and 13/32 settled recoveries with a relocated step at 14 N·s
on the first 32-start set; one met the braking threshold. It settled 0/32 at
12 and 16 N·s. Earlier checkpoints settled none of these push trials. These
are motion-triggered results with floor-attributed contacts, not oracle-timed
inference; they remain a narrow single-seed result. See
`build/genesis-step-dagger/holdout.json`. On a fresh 64-start set the same
blend held 64/64 quiet starts and recovered 16/64 at 14 N·s with a relocated
landing, none meeting the braking threshold; it recovered 0/64 at 12 and
16 N·s. See `build/genesis-step-dagger/fresh.json`.

A 307,200-step PPO continuation from that student used the same motion gate
and mixed pushes. It logged some braking events, but no full settled-hold
rewards during training. On the 32-start holdout, its final checkpoint held
32/32 quiet trials and settled 4/32 at both 12 and 14 N·s, with relocated
steps but no braking steps; it settled 0/32 at 16 N·s. Intermediate checkpoints
also failed to improve consistently across the push grid. Keep the original
imitation and standing checkpoints for further work; see
`build/genesis-step-dagger-ppo/holdout.json`.

Another imitation round added 128 failed student trajectories from 16 N·s
pushes, labeled by the same scripted reference. On the 32-start holdout its
final actor held 32/32 quiet trials, recovered 2/32 at 12 N·s (one with a
braking step), and recovered 0/32 at 14 and 16 N·s. Earlier checkpoints
similarly failed at 14 and 16 N·s. The reduced action error did not translate
to robust closed-loop recovery; see `build/genesis-step-dagger-round2/`.

## Remaining work — priorities

### P1 — Matched rig and cross-backend control contract

1. Validate and refine the prototype MJCF export of the authored rig and tuner
   projects. Map body order, collider dimensions and offsets, mass/inertia, rest poses,
   joint frames, axes, limits, collision filtering and torque caps. Document
   approximations: Jolt swing/twist constraints and scalar joint coordinates
   need not have equivalent limits or rotations. Test rest and articulated
   poses, isolated joint motion and contacts in both solvers.
2. Define a versioned policy contract covering coordinate frames, joint and
   action order, target pose, normalization, scaling, clipping, timing and
   reset state. Start with projected gravity, root angular velocity, joint
   positions and velocities, target angles and previous actions; add COM,
   foot or contact features if recovery needs them. Derive the observation
   size from the rig rather than assuming 45–60 values. Map scalar joint
   commands to Jolt's parent-local quaternion targets and validate signs and
   composition on every controlled axis.
3. Calibrate the existing Jolt frequency/damping motors against Genesis
   stiffness/damping PD using matched single-joint and whole-rig response
   tests, including symmetric torque caps, passive forces, saturation and
   contacts. Record residual mismatch; do not assume the gains map directly.
   Keep root assistance disabled.

### P2 — Genesis training and Jolt fine-tuning

1. Extend the prototype batched Genesis environment and RSL-RL PPO runner to
   use the validated shared contract. Start from target-pose residual control and a standing
   curriculum; reward pose tracking, uprightness, height and appropriate COM
   support, with action-rate, torque and acceleration penalties. Allow single
   support and do not penalize COM outside a nonexistent contact polygon.
2. Add seeded torso pushes with varied timing, direction and magnitude, and
   evaluate friction and link-mass randomization around ±15% as a transfer
   aid. Begin with small batches and measure scaling toward 2,048–4,096
   environments where GPU memory and throughput permit. Verify independent
   resets and disturbances and distinguish timeouts from falls.
3. Make the Genesis-trained actor trainable in the existing Jolt apparatus.
   RSL-RL and Stable-Baselines3 checkpoints, network architecture,
   normalization and observation/action spaces are not interchangeable by
   default. Implement and test an explicit weight/normalizer transfer or a
   Jolt environment compatible with the same RSL-RL actor and optimizer;
   preserve existing Jolt PPO checkpoints and baseline workflows. Check actor
   output parity before and after transfer, then fine-tune with fresh-world
   resets and paired held-out evaluations.
4. Export the final actor and normalization to ONNX and integrate ONNX Runtime
   inference with Jolt observations and motor targets at 60 Hz. Verify outputs
   against the training actor and compare complete closed-loop Jolt rollouts.
   Keep dependencies optional and runtime physics independent of the trainer.

### P3 — Recovery quality and measured scaling

- Use the replay traces to investigate reachable touchdown, placement error,
  load transfer, planted drift and follow-up steps under strong pushes. Keep
  impact costs, directed COM-velocity diagnostics and the one-second settled
  hold; use the existing quiet and light cases as regressions.
- Compare Genesis, transferred and Jolt-fine-tuned policies on paired held-out
  starts and disturbances. Run at least three independent training seeds and
  report the 14–16 N·s cells and braking steps separately. Measure throughput,
  wall time, memory and recovery outcomes; neither training time nor inference
  latency is assumed from another machine.
- Profile the bottlenecks before optimizing. Retain Jolt fresh-world resets
  unless a replacement restores native held-control and application state and
  demonstrates equivalent seeded behavior without cross-episode leakage.

## Subsequent capability roadmap

Work below follows reliable hard-push recovery rather than expanding an
unverified learning setup.

1. **Constrained support and walls:** Add downward landing probes and nearby
   wall probes (hit, distance, normal in a heading-aligned frame), excluding
   self-collisions. Train edge recovery, arm/forearm support, controlled release
   and return to foot balance. Native friction/contact supplies actual support.
2. **Animation following:** Track a retargeted walk reference with joint pose,
   velocity and end-effector rewards. Sample valid reference frames on resets;
   add heading changes, pushes and uneven ground after flat-floor tracking.
3. **Bracing and getting up:** Train fall protection and prone/supine recovery
   separately, then transitions back to standing/tracking. Keep episodes active
   through hand/knee contact; prioritize limiting head and torso impacts.
4. **Graaal integration:** Connect the verified ONNX/Jolt controller to
   ECS/skin. Support animation changes, impulses and continuously varying motor
   strength after training/runtime parity has been established.

## Evaluation rules

Use paired starts and a held-out disturbance grid for the classical, historical
learned, Genesis-trained, transferred and Jolt-fine-tuned policies. Report
training budgets and simulator-specific throughput alongside outcomes. Report failures
by **direction × application body × impulse**, not only aggregate success.
The existing [push benchmark](../scenes/benchmarks/README.md#ragdoll-push-recovery)
is the reference; keep its recorded 2/4/8/12 N·s comparison and extend the
hard `−Z` cases with multiple starts and training seeds.
Use the [3D recovery viewer](../scenes/benchmarks/README.md#view-recovery-trials-in-3d)
to inspect quiet-standing or push cases and compare controllers on the same seed.
`python -m scenes.ragdoll_recovery_viewer` discovers replayable reports in `build/`.
Use **Scenario group** to select a report, then pick the controller, case and
seed. The 16 N·s recoveries at seed 60700 for the 336,384-step checkpoint and
seed 61701 for the 500,224-step checkpoint matched their recorded recovery times
on replay.

| Scenario | Initial acceptance / diagnostics |
| --- | --- |
| Quiet standing | At least **27/30** held-out ten-second trials; stable foot support, COM speed <0.05 m/s, pelvis height ≥90% of reference and pelvis/chest up cosine ≥0.95 in the final second. Report drift, support margin and slip. |
| Push recovery | Target **≥90%** on the original 0.5/1/2 N·s grid. For Backstep pose recovery, require a one-second settled hold within **5 s** after the push, with up to one extra second to finish a hold already underway. Report hard `−Z` success separately, time to arrest COM velocity, landing displacement, contact duration, touchdown margin and fall/step counts. |
| Platform edges / walls | Report falls, landing positions, return to foot support, arm/forearm contacts, wall load and release; compare whole-body with leg-only ablations. |
| Motion tracking | Completion, joint and end-effector errors, heading/speed errors, contact timing and slip. |
| Bracing / get-up | Peak head/torso impact vs passive baseline; prone/supine success, time to stand and return to tracking. |

During training, use dense signals; the one-second hold remains an objective
success check. Standing/tracking may terminate on falls; fall and get-up tasks
continue through contact. Distinguish external time-limit truncation from
failure. Save rig/tuning, reward and action configurations, seeds, training
steps, optimizer/KL statistics, throughput, per-cell outcomes and replay traces.

## Research references

- Egle et al. use a dynamics model to plan reachable footholds while RL adapts
  timing and the reachable step region. Their simulated humanoid used crossover
  steps under stronger lateral pushes. This supports testing foot placement and
  timing together. [Step adaptation for push recovery](https://arxiv.org/html/2411.01000)
- Poddar et al. use capture point, COM state and centroidal momentum for training.
  Their policy learned ankle, hip and stepping responses in simulation, with
  preliminary hardware tests. Capture point gives Graaal a way to measure how
  motion relates to usable support before and after a landing.
  [Balance-informed humanoid recovery](https://arxiv.org/html/2603.08619v1)
- Ferigo et al. used whole-body PPO with randomized disturbances and foot-load
  observations on iCub in simulation. Each training run collected 20 million
  agent steps, about 40 times the current Backstep budget in policy steps.
  [Whole-body push-recovery learning](https://arxiv.org/html/2104.14534)
- [DeepMimic](https://arxiv.org/html/1804.02717) and [DReCon](https://static-wordpress.ubisoft.com/montreal.ubisoft.com/wp-content/uploads/2019/11/13214229/DReCon.pdf) — pose/velocity tracking, reference initialization and game-directed animation control.
- [Environment design ablations](https://arxiv.org/html/2010.04304) — controlled comparisons of observations, rewards and action design.
- [AMP](https://arxiv.org/abs/2104.02180) and [PHC](https://arxiv.org/html/2305.06456) — potential later style and recovery techniques, **not** first prerequisites.
