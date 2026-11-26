# Active ragdoll plan

## Scope and architecture

Build a physically simulated humanoid that can stand, recover from pushes with
whole-body motion or steps, brace against its environment, follow animation
references, protect itself during falls and get up. [Native physics integration](physics_integration_plan.md)
and the [ragdoll API](../graaal/physics.md#native-ragdolls-and-skeletons)
already exist; this file tracks **passive mechanics and learned control**.

- **Physics:** Jolt via the pinned `pyjoltphysics 0.1.0a3`; native constraints,
  contacts, collisions and rigid-body motion.
- **Control:** Python/NumPy computes observations and passive joint torques;
  bounded joint motors execute the policy's continuous targets. PyTorch PPO
  trains an MLP; Graaal will run NumPy inference from exported weights.
- **Forces:** Independent passive neutral-posture springs, progressive soft
  angular limits (native hard limits as backstops), and active motor commands.
- **Rates:** 120 Hz physics, 60 Hz neural actions. **No root/pelvis assistance**
  in learned training or evaluation.
- **Boundary:** Stay within Python/NumPy and the existing pyjoltphysics API.
  Native binding changes and other implementation languages are out of scope.

## Implementation history and findings

Completed work is summarized here, rather than repeated as future tasks. The
numbers below describe the recorded Windows x64 runs, not cross-platform or
full-character performance guarantees.

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

### 4. Direct continuous motor control (P0 implemented)

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

### 5. Warm-start direct control (P1 gate met)

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
The longer window offered little extra practice on this grid. The reference
still prescribes one swing and overwrites both hip-pitch targets after its
loading phase. This comparison shows the timing of subsequent landings for
that controller. Impact peaks stayed similar across the three checkpoints.
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

## Remaining work — priorities

### Extend the recovery range

Keep the original quiet and push grids as regression checks. Oversample
14–18 N·s pushes while keeping quiet, light and other-direction starts in
training. Learn swing timing and foot placement from contact and COM state.
Seek reliable relocations with effective braking in the stronger backward
and lateral cells. Use paired checkpoints and inspect target
saturation, foot clearance, action ranges, contact geometry and COM/capture
traces when the stronger cases plateau.

### P2 — Touchdown braking and dense learning signal

Direct control now measures these terms every control step and logs their
episode totals. An initial ablation compared the three new terms as a group.
The next comparison needs enough relocated touchdowns to exercise the
compliance term and measure braking through landing.

- **Horizontal COM braking:** reward reductions in horizontal COM speed after
  the impulse, particularly over the first 0.2 s after a new touchdown;
  penalize continuing or increasing speed away from feasible support.
- **Compliant landing:** reward controlled hip/knee yielding on impact and
  subsequent stabilization; avoid encouraging an unrecoverable crouch. Track
  foot slip, contact impulse, peak deceleration and excessive joint motion.
- **Support recovery:** reward improving signed COM-to-support-area margin
  immediately after foot contact and reducing predicted capture-point deficit.
  Use real contact state to form a changing support area; mask unsupported
  margins while airborne, and allow temporary single support during a step.
- **Keep** uprightness, appropriate standing height, action smoothness and
  small effort penalties. Retain the continuous one-second settled hold as
  **evaluation / terminal success**, not the principal training signal.

Inspect pre-impact, touchdown, +0.2 s and later recovery traces. Ablate
braking, compliance and margin individually once the policy attempts
relocations. Compare their effects on strong pushes and quiet and light starts.

### P3 — Diagnose PPO KL stopping, then increase training budget

1. Use `--optimizer-trace` for minibatch KL, accepted epochs per rollout,
   clip fraction, entropy loss, action variance, value loss and explained
   variance. Training logs include the reward components. Compare the effects
   of changes on both accepted updates and held-out recovery.
2. Compare learning rate, target KL, clip range, optimizer epochs and batch
   size in controlled runs. Do **not** simply disable KL protection; verify
   that updates become effective and hard-push validation improves.
3. With the P1 gate met, scale to **500,000–1,000,000+ policy steps** with
   **at least three training seeds**, using paired held-out evaluations and
   checkpoints. Extend toward multi-million-step runs while success and
   stability continue to improve. Report per-seed spread.

### P4 — Reset and throughput optimization, after learning is verified

- Profile fresh-world creation, physics stepping, observation copies, NumPy
  torque evaluation and PPO transfer/update separately at the larger budgets.
- Compare the existing isolated subprocess worlds with **faster resets** or
  **spatially batched ragdolls/worlds**, checking no cross-episode contact,
  state leakage or changes in seeded results. A safe in-place reset requires
  restoring native held control state **and** application state, not just pose
  and velocity; if unavailable, retain fresh-world resets.
- Optimize only if equivalent held-out behavior is maintained and measured
  throughput makes multi-million-step experiments materially more practical.

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
4. **Graaal runtime:** Export policy weights and observation normalization to
   `.npz`, implement batched NumPy MLP inference, connect Jolt transforms to
   ECS/skin, and check training/runtime parity at 120/60 Hz. Support animation
   changes, impulses and continuously varying motor strength.

## Evaluation rules

Use paired starts, a held-out disturbance grid and matched budgets for the
classical, historical learned and new direct-control policies. Report failures
by **direction × application body × impulse**, not only aggregate success.
The existing [push benchmark](../scenes/benchmarks/README.md#ragdoll-push-recovery)
is the reference; keep its recorded 2/4/8/12 N·s comparison and extend the
hard `−Z` cases with multiple starts and training seeds.
Use the [3D recovery viewer](../scenes/benchmarks/README.md#view-recovery-trials-in-3d)
to inspect quiet-standing or push cases and compare controllers on the same seed.
`python -m scenes.ragdoll_recovery_viewer` discovers replayable reports in `build/`.
Automatic discovery found both impact-reward runs, and each initial replay
matched its recorded trial.

| Scenario | Initial acceptance / diagnostics |
| --- | --- |
| Quiet standing | At least **27/30** held-out ten-second trials; stable foot support, COM speed <0.05 m/s, pelvis height ≥90% of reference and pelvis/chest up cosine ≥0.95 in the final second. Report drift, support margin and slip. |
| Push recovery | For the original 0.5/1/2 N·s grid, target **≥90%** recovery. A recovery requires one continuous second of settled stance within **5 s** after the push. Also report hard `−Z` success separately, time to arrest COM velocity, touchdown margin, landing displacement and fall/step counts. |
| Platform edges / walls | Report falls, landing positions, return to foot support, arm/forearm contacts, wall load and release; compare whole-body with leg-only ablations. |
| Motion tracking | Completion, joint and end-effector errors, heading/speed errors, contact timing and slip. |
| Bracing / get-up | Peak head/torso impact vs passive baseline; prone/supine success, time to stand and return to tracking. |

During training, use dense signals; the one-second hold remains an objective
success check. Standing/tracking may terminate on falls; fall and get-up tasks
continue through contact. Distinguish external time-limit truncation from
failure. Save rig/tuning, reward and action configurations, seeds, training
steps, optimizer/KL statistics, throughput, per-cell outcomes and replay traces.

## Research references

- [Ferigo et al. (2021): standing and push recovery](https://arxiv.org/html/2104.14534) — whole-body PPO, COM/contact feedback and stepping; substantially larger reported training budget than these pilots.
- [DeepMimic](https://arxiv.org/html/1804.02717) and [DReCon](https://static-wordpress.ubisoft.com/montreal.ubisoft.com/wp-content/uploads/2019/11/13214229/DReCon.pdf) — pose/velocity tracking, reference initialization and game-directed animation control.
- [Environment design ablations](https://arxiv.org/html/2010.04304) — controlled comparisons of observations, rewards and action design.
- [AMP](https://arxiv.org/abs/2104.02180) and [PHC](https://arxiv.org/html/2305.06456) — potential later style and recovery techniques, **not** first prerequisites.
