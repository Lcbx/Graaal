# Repository tools

## Module inspection

`module_inspection.py` prints a public API outline with signatures, annotations,
docstrings and comments immediately preceding declarations. It accepts a Python
file, package directory or directory of modules. Package imports retain their
qualified names so relative imports work.

```sh
python tools/module_inspection.py graaal/ECS/core.py
python tools/module_inspection.py graaal/ECS/core.py --only classes methods properties --no-comments --no-docstrings
python tools/module_inspection.py graaal/ECS/core.py --include-imports
python tools/module_inspection.py graaal/ECS/core.py --only imports --no-docstrings
python tools/module_inspection.py graaal/ECS --source-ranges --no-comments --no-docstrings
python tools/module_inspection.py graaal/ECS --filter '*Query*' --source-ranges
python tools/module_inspection.py graaal/ECS --regex '^ECS\.(query|register)$' --no-docstrings
python tools/module_inspection.py graaal -r -o api.txt
```

`--only` accepts one or more of `classes`, `functions`, `methods`, `properties`,
`attributes` and `imports`. Classes remain as containers for selected members; selecting
only `classes` omits their methods and attributes. Properties include getters,
setters and deleters. Class members are limited to those defined on the class,
including `__init__`; other private names are omitted. Module `__all__`, when
present, controls exports and includes re-exported classes and functions.

`--include-imports` adds import statements to the outline; `--only imports`
selects just imports. In Python, use `dump_module(module, include_imports=True)`
or `dump_module(module, only={"imports"})`. Imports are collected from available
source in source order, preserving aliases and relative import syntax, with
multiline statements normalized to one line. This includes conditional and local
imports regardless of `__all__`; it does not detect dynamic imports such as
`importlib.import_module()`. Import listing is omitted by default.

`--source-ranges` adds `# source: /absolute/path/module.py:START-END` before
classes, functions, methods and each property accessor. Line numbers are
one-based and inclusive, covering decorators through the end of the definition.
Re-exports and aliases point to the definition's file; wrapped functions follow
`__wrapped__`. Locations remain visible with `--no-comments` and are omitted
when source is unavailable, such as for built-ins. In Python, use
`dump_module(module, source_ranges=True)`.

`--filter PATTERN` matches definition names using case-sensitive shell wildcards
(`*`, `?`, `[abc]`); a plain string matches an exact name, and `*text*` matches
names containing `text`. `--regex PATTERN` uses case-sensitive regular expression
search instead; use `^` and `$` for exact matches. These options are mutually
exclusive. Quote patterns to prevent shell expansion.

Both filters check bare names (`query`), class-qualified names (`ECS.query`),
and module-qualified names (`graaal.ECS.core.ECS.query`), including exported
aliases and attributes. A matching class includes its members, subject to
`--only`; a matching member keeps its containing classes. Private names remain
excluded unless exported by module `__all__`. Comments, docstrings and source
ranges stay with the retained definitions; module headings and docstrings remain.
Import listings are independent of definition filters. In Python, use
`dump_module(module, name_filter="*Query*")` or
`dump_module(module, regex=r"^ECS\.query$")`.

`-r` includes subdirectories when given a directory or `__init__.py`.
`--no-comments` and `--no-docstrings` work independently. The same filters are
available through `tools.module_inspection.dump_module()`:

```python
from graaal import ECS
from tools.module_inspection import dump_module

print(dump_module(ECS, only={"classes", "methods"}, include_comments=False, include_docstrings=False))
```

This is runtime inspection: target modules are imported, their top-level code
runs, and their dependencies must be available. Import failures stop the dump
and identify the affected module. The output is an outline, not an executable
stub or a list of references and call sites. Regression tests are in
[`tests/test_module_inspection.py`](../tests/test_module_inspection.py).

## Animation tools

- [Animation timeline](animation_timeline.md): inspect tracks and author events and sync markers.
- [Animation graph](animation_graph.md): configure and preview a graph for a supported skinned asset.
- [Authoring projects](animation_project.md): explicitly save/load graph definitions and markers in shared JSON sidecar files.

Both tools use the shared asset loader and CPU preview; neither requires the other to run.

## Physics tools

- [Ragdoll tuner](ragdoll_tuner.md): edit per-bone colliders, mass, hard and soft limits, and neutral springs in a live physics preview.
- [Ragdoll standing task](#ragdoll-standing-task): compare passive, fixed-pose and ankle feedback baselines, train PPO, and evaluate saved policies.

### Ragdoll standing task

The passive and joint-tracking baselines require the `physics` extra.

```sh
python -m tools.ragdoll_standing --initial-angular-speed 0.1 --episodes 8
```

The task advances Jolt at 120 Hz and holds each joint action for two ticks.
The pelvis is free of root springs and applied balance forces. By default the
policy controls a bounded rotation offset and motor strength per joint.
`--action-space compact` keeps the motor caps fixed and controls only joint
axes with sufficient travel, giving the built-in rig 30 outputs instead of 48.
`--stance tuned` starts in a shallow knee bend with flat soles and uses that
pose as the motor reference. An episode succeeds if it reaches the time limit
without a non-foot contact with the floor, a drop below 70% of its starting
pelvis height, or a tilt more than 60 degrees from upright in the pelvis or chest.
Settled success also requires continuous foot support, near-reference height,
an upright pelvis and chest, and low center-of-mass speed for the final second.
`--tuning PATH` accepts a ragdoll tuner project using the built-in humanoid's
bone names, parent relationships and order.

Training and checkpoint evaluation also require
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
To use your regular Python, install it into that interpreter's user packages
with `python -m pip install --user "stable-baselines3>=2.7,<3"`.

```sh
python -m tools.ragdoll_standing --stance tuned --action-space compact --seconds 10 --workers 8 --seed 77 --steps 65536 --output standing-quiet.zip
python -m tools.ragdoll_standing --stance tuned --action-space compact --seconds 10 --workers 8 --seed 77 --steps 185344 --vary-start --initial-angular-speed 0.1 --resume standing-quiet.zip --output standing-varied.zip
python -m tools.ragdoll_standing --stance tuned --action-space compact --seconds 10 --vary-start --initial-angular-speed 0.1 --checkpoint standing-varied.zip --eval-seed 50000 --episodes 30
```

With `--workers`, each parallel worker owns a separate Jolt world and resets
its own episode when it falls or reaches the time limit. `--steps` counts
policy steps across all workers. `--resume` restores the PPO model, optimizer,
and training step count, then collects new rollouts from freshly seeded worlds.

The starting spin direction is seeded. `--vary-start` also samples the pose
tilt and spin magnitude. Use `--max-start-tilt` with `--vary-start` to set the
upper tilt in radians (default 0.025). A value of zero keeps the reference
stance and still samples spin. The same seed gives the same spin at each tilt
setting. All controllers use the same evaluation seeds, starting at
`--eval-seed` or 10,000 beyond the training seed by default.
The report includes completion and settled-success rates, elapsed time,
minimum pelvis height, drift, contact foot slip, pose error and reward.

`--support-area` adds a signed support margin observation and replaces the
foot-midpoint penalty with a penalty near or outside the estimated support
area. It uses projected soles of contacting box feet and reports the mean
first exit time across episodes with a recorded exit. Use the same action
space and support-area setting when resuming or evaluating a checkpoint.

`--ankle-balance` evaluates a third baseline for rigs with box feet. It
keeps fixed-pose motor targets on the other joints and uses bounded,
equal-and-opposite foot and shin torques to shift estimated foot pressure.
The feedback uses current center-of-mass position and velocity, the ankle
positions, and foot contacts. The model assumes the contacting feet share
the rig's weight evenly and uses the reference sole bounds to limit the
pressure shift. This comparison uses the same unassisted pelvis as the other
baselines.

```sh
python -m tools.ragdoll_standing --ankle-balance --stance tuned --support-area --seconds 10 --vary-start --max-start-tilt 0 --initial-angular-speed 0.1 --eval-seed 52000 --episodes 30
```

`--action-space residual` trains or evaluates four PPO outputs for horizontal
torque on the two ankles. Each foot's correction is capped at 10 N m.
It applies only with floor contact. The combined correction and classical
feedback go through the same sole bounds and ankle torque caps. Zero outputs
reproduce the classical controller, which is included in the comparison.
The report includes mean commanded residual torque per ankle before contact
gating and pressure clipping.

```sh
python -m tools.ragdoll_standing --stance tuned --action-space residual --support-area --seconds 10 --vary-start --max-start-tilt 0 --initial-angular-speed 0.1 --workers 8 --seed 77 --steps 65536 --output standing-residual-flat.zip
python -m tools.ragdoll_standing --stance tuned --action-space residual --support-area --seconds 10 --vary-start --max-start-tilt 0.005 --initial-angular-speed 0.1 --workers 8 --seed 77 --steps 40960 --resume standing-residual-flat.zip --output standing-residual-tilt005.zip
```

Use the same action space and support-area setting when loading a residual
checkpoint. For the final curriculum stage, evaluate with its maximum tilt.

```sh
python -m tools.ragdoll_standing --stance tuned --action-space residual --support-area --seconds 10 --vary-start --max-start-tilt 0.025 --initial-angular-speed 0.1 --eval-seed 54100 --episodes 30 --checkpoint build/standing-residual-tilt025.zip
```

Use the [push-recovery benchmark](../scenes/benchmarks/README.md#ragdoll-push-recovery)
to compare a residual checkpoint with classical ankle balance after a push.

Train on flat-floor disturbances with `--push-impulses`. At two seconds the
task applies one center-of-mass impulse. It samples a magnitude from the list,
where zero gives a quiet episode. For nonzero impulses it picks a −Z chest
push half the time and otherwise samples a body and cardinal direction
uniformly. The reset seed controls the initial tilt, spin and impulse.

`--action-space stepping` keeps the four ankle corrections and adds six
normalized outputs. The next four are local X/Z target-angle offsets for the
left and right hips, capped at 0.45 rad per hip. The last two are left and
right knee flexion offsets in [−0.05, 0.55] rad. Use `--stance tuned` with
this action space. The observation layout matches `residual` with the same
`--support-area` setting, including the previous four ankle commands.

`--action-space whole-body` appends 16 chest, neck, shoulder and elbow outputs
on the built-in rig. Channels follow bone order, then twist and swing axes
with at least 0.22 rad of travel. Combined target offsets are bounded to
0.22 rad at the chest and neck and 0.55 rad at each shoulder and elbow.
This mode uses the same observations, leg controls and joint torque caps as
`stepping`.

```sh
python -m tools.ragdoll_standing --stance tuned --action-space stepping --support-area --seconds 7 --vary-start --initial-angular-speed 0.1 --push-impulses 0 4 8 12 --seed 87 --workers 8 --steps 196608 --episodes 3 --output build/ragdoll-push-stepping-seed87.zip
python -m tools.ragdoll_standing --stance tuned --action-space residual --support-area --seconds 7 --vary-start --initial-angular-speed 0.1 --push-impulses 0 4 8 12 --seed 87 --workers 8 --steps 196608 --episodes 3 --output build/ragdoll-push-residual-seed87.zip
python -m tools.ragdoll_standing --stance tuned --action-space whole-body --support-area --seconds 7 --vary-start --initial-angular-speed 0.1 --push-impulses 0 4 8 12 --seed 87 --workers 8 --steps 196608 --episodes 3 --output build/ragdoll-push-whole-body-seed87.zip
```

`--action-space recovery` enables support-area observations and extends the
whole-body layout. On the built-in humanoid it uses 165 inputs and 32 outputs.
The additional observations describe the heading-relative capture point
estimate, signed support margin at that point, foot sole tilt and clearance,
contact age, filtered Jolt contact impulse estimates, and chest angular
velocity. The policy also observes the current step phase, target and last
step command.

Six outputs follow the 26 whole-body commands. A signed request selects a
swing foot, two outputs offset the capture-based landing target in the
ragdoll's heading frame, and the remaining outputs adjust clearance, swing
duration and loading after touchdown. The controller keeps a stance foot
and uses bounded, paired joint torques for the swing and landing. The task
rewards braking and regaining support over the 0.2 seconds after a relocated
landing and awards one bonus for a fresh one-second standing hold after a
push. It keeps the fall and slip costs and relaxes the height and upright
costs during the step.

`--recovery-curriculum` samples quiet, light, near-threshold and hard
episodes with probabilities 20%, 35%, 25% and 20% before optional replay
sampling. `--replay-start-fraction` attempts to start that fraction of
training episodes in swing or shortly after touchdown. Each attempt
replays the push and a guided step in a fresh world. If the requested state
is unavailable, the episode starts from the regular stance. Evaluation
starts from stance. `--epochs` caps the optimizer passes per rollout
(default 10 for a new policy), with KL early stopping still enabled.

```sh
python -m tools.ragdoll_standing --stance tuned --action-space recovery --seconds 7 --vary-start --initial-angular-speed 0.1 --recovery-curriculum --replay-start-fraction 0.15 --seed 87 --workers 8 --steps 196608 --epochs 10 --episodes 3 --output build/ragdoll-recovery-seed87.zip
```

Use [`--recovery-checkpoint` in the push benchmark](../scenes/benchmarks/README.md#ragdoll-push-recovery)
to measure recovery from paired starts.

`--action-space direct` gives the policy 30 continuous target offsets on the
built-in rig, in bone and constraint-axis order. It uses the tuned stance as
the motor reference and enables motor targets at the ankles, hips, knees,
torso and arms on every control tick. The pelvis stays free. Its 187 observations
include heading-relative body motion, COM and capture-point estimates, foot
and lower-arm motion relative to the pelvis, joint state, foot contact and
clearance, and the previous target commands. Support margin is estimated
from the soles of the contacting feet.

With `direct`, `--recovery-curriculum` samples 20% quiet episodes, 35% mixed
0.5–4 N·s pushes, 25% backward 8–12 N·s pushes, and 20% from the three
historically failing backward cells. Every scheduled push occurs at two seconds.
The direct task tracks braking, support, landing compliance and effort
rewards. Braking has higher weight in the first 0.2 seconds after foot
contact resumes. Compliance rewards controlled leg flexion after a relocated
foot lands. Training logs each component separately. For a reward ablation,
pass `--without-direct-reward` with any of `braking support compliance`.

```sh
python -m tools.ragdoll_standing --stance tuned --action-space direct --seconds 7 --vary-start --initial-angular-speed 0.1 --recovery-curriculum --seed 87 --eval-seed 58100 --workers 8 --steps 196608 --epochs 6 --episodes 3 --output build/ragdoll-direct-seed87.zip
```

`--direct-balance-init` initializes the PPO network to use COM feedback at
both ankle motors. Training and evaluation then run the MLP's motor targets
at every tick. The warm-start pilot used a rig with 20 Hz joint motors and
0.8 damping. Save the rig and use the same project with the checkpoint.
`--direct-limb-std` sets the initial exploration noise on the non-ankle
outputs for a new or resumed run. `--optimizer-trace` appends per-minibatch
approximate KL, accepted epochs and optimizer metrics to a JSON Lines file.

```sh
python -c "from tools.ragdoll_project import default_project, write_project; project = default_project(); project.settings.motor_frequency = 20; project.settings.motor_damping = 0.8; write_project('build/ragdoll-direct-20hz-rig.json', project)"
python -m tools.ragdoll_standing --tuning build/ragdoll-direct-20hz-rig.json --stance tuned --action-space direct --direct-balance-init --seconds 7 --vary-start --initial-angular-speed 0.1 --recovery-curriculum --seed 88 --workers 8 --steps 65536 --epochs 5 --learning-rate 0.000003 --output build/ragdoll-direct-warm-65536-seed88.zip
python -m tools.ragdoll_standing --tuning build/ragdoll-direct-20hz-rig.json --stance tuned --action-space direct --seconds 10 --vary-start --initial-angular-speed 0.1 --eval-seed 58300 --episodes 30 --checkpoint build/ragdoll-direct-warm-65536-seed88.zip
```

`--action-space backstep` trains joint residuals to load the foot and settle
after a reference right-foot swing. It uses the tuned stance, ankle feedback
and whole-body action layout with an extra elapsed-push observation. The
backstep curriculum samples 11–13 N·s backward chest pushes. Training replays
the reference up to the end of swing in a fresh world, so the PPO rollout
starts where its residuals can affect the motion. The benchmark replays each
trial from the original standing start.

The backstep reward penalizes sudden center-of-mass acceleration above
8 m/s² and large estimated floor impulses. On the built-in rig the impulse
threshold is about 1.26 N·s. Each control step uses the peak of each
measurement across its two physics ticks. The tick that receives the
external push is excluded. A one-time bonus rewards a full second of
upright, supported posture. The benchmark's settled-stance test also
checks horizontal speed.

Use `--pose-recovery` with pushed backstep training to end an episode after
a continuous one-second hold near the target pose. The regular time limit
allows up to one extra second for a hold that has already started. A sustained
topple ends the episode early. Speed gates the hold, while the per-action impact
costs still use COM acceleration and floor impulse. The success reward includes
the discounted standing reward for the unused portion of the regular window.
`--release-backstep-hips` blends hip pitch from the script to the learned motor
targets over 0.25 seconds after loading. Pass both options to the benchmark
when evaluating a checkpoint trained with them.

`--touchdown-support-reward` adds a training signal for changes in COM support
margin during the first 0.2 seconds after a foot lands. It measures both COM
positions against the current planted-foot support polygon, and logs the
accumulated signal as `touchdown_support_reward`. The signal is zero without
foot contact and does not penalize speed directly.

Backstep training accepts `--recovery-curriculum` or `--push-impulses` with at
least one backward chest push of 10 N·s or more. Both sources support
`--pose-recovery`. The standing CLI uses the same seeded disturbances for
Backstep PPO and the passive, tracking and ankle-balance baselines.
`--strong-backstep-reference` uses a farther
landing target at 14 N·s and above. Pass the same option to the benchmark
when evaluating a checkpoint trained with it.

The following continuations use equal PPO step budgets, with five and ten
seconds after the push respectively. `--seconds` includes two seconds before
the push. Omit `--resume` to train from scratch.

```sh
python -m tools.ragdoll_standing --stance tuned --action-space backstep --seconds 7 --vary-start --initial-angular-speed 0.1 --push-impulses 11 12 13 14 14 15 15 16 --steps 32768 --resume build/ragdoll-backstep-81920-seed93.zip --workers 6 --epochs 5 --learning-rate 0.00005 --seed 96 --episodes 3 --output build/ragdoll-backstep-impact-7s.zip
python -m tools.ragdoll_standing --stance tuned --action-space backstep --seconds 12 --vary-start --initial-angular-speed 0.1 --push-impulses 11 12 13 14 14 15 15 16 --steps 32768 --resume build/ragdoll-backstep-81920-seed93.zip --workers 6 --epochs 5 --learning-rate 0.00005 --seed 96 --episodes 3 --output build/ragdoll-backstep-impact-12s.zip
```

```sh
python -m tools.ragdoll_standing --stance tuned --action-space backstep --pose-recovery --release-backstep-hips --seconds 7 --vary-start --initial-angular-speed 0.1 --push-impulses 11 12 13 14 14 15 15 16 --steps 12288 --resume build/ragdoll-backstep-impact-7s.zip --workers 4 --epochs 5 --learning-rate 0.00005 --seed 97 --episodes 1 --output build/ragdoll-backstep-pose-hips-seed97.zip
```
