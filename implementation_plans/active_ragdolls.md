# **Active Ragdoll — Implementation Plan**

## **Overview**

Implement a physically simulated humanoid with **natural passive joint behavior** and a **neural controller** capable of balancing, following animations, stepping, bracing during falls, and getting back up. Physics is handled by **Jolt Physics through pyjoltphysics**, while Python defines ragdolls, joint constraints, motor commands, simulation scenarios and batched observations using **NumPy**. A **PyTorch PPO policy** learns purposeful movement through reinforcement learning. The same physics, actuation and observation code runs during training and at runtime in **Graaal**, with NumPy-based neural inference.

The system separates three sources of joint forces:

- **Passive soft limits:** Progressive resistance and damping near joint boundaries, with hard limits as backstops.
- **Passive neutral-posture pull:** Weak springs toward a relaxed, human-like equilibrium pose, even without active control.
- **Active neural control:** Strength-limited motors that track animation targets while adapting to balance, impacts and environmental contacts.

The passive system produces natural motion independently of the neural policy. The active system adds purposeful behavior without replacing physical simulation.

## **Implementation phases**

### **Phase 0 — Backend validation**

- Test pyjoltphysics rigid bodies, ragdolls, joint constraints, motors, impulses and state snapshots.
- Verify independent training environments, selective resets and reproducibility.
- Benchmark batched observations, motor commands and native multithreaded stepping.
- Identify missing APIs for passive joint torques, bulk motor updates or efficient resets; extend bindings if necessary.

**Deliverable:** Working headless ragdoll prototype and measured simulation throughput.

### **Phase 1 — Passive ragdoll**

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

### **Phase 2 — Active actuation and observations**

- Add strength-limited joint motors that track reference poses from standing, walking or other animations.
- Allow motor targets and strengths to vary continuously, including partial weakening during impacts.
- Implement a simple classical balance and pose-tracking controller as a reference baseline.
- Define NumPy observations containing body orientation, velocities, joint states, contacts, centre of mass, reference pose and previous actions.
- Define neural outputs as joint target offsets and optionally individual motor strengths.

**Deliverable:** A shared control interface that combines passive joint mechanics with active, bounded motor control.

### **Phase 3 — Parallel training environments**

- Run multiple ragdolls in spatially separated regions of one or several Jolt worlds.
- Implement batched stepping, actions, observations, rewards and selective resets.
- Randomize mass, joint strength, friction, terrain and external impulses.
- Support resets into standing, stumbling, falling, prone and supine configurations.
- Benchmark throughput and optimize Python/native communication.

**Deliverable:** Reproducible, high-throughput reinforcement learning environment.

### **Phase 4 — Reinforcement learning**

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

### **Phase 5 — Evaluation and refinement**

- Measure recovery probability across push directions, magnitudes and application points.
- Evaluate head/torso impact severity during falls and stair tumbles.
- Measure recovery success and time from prone and supine positions.
- Validate smooth transitions from controlled movement to stumbling, collapse and recovery.
- Test behavior under reduced strength and compare against classical control.
- Add motion imitation or additional posture constraints only if movement quality requires them.

**Deliverable:** Validated natural movement, physically plausible failures and reliable recovery behavior.

### **Phase 6 — Graaal integration**

- Export neural weights and observation normalization to `.npz`.
- Implement batched NumPy MLP inference.
- Connect Jolt body transforms to ECS entities and the rendering skeleton.
- Run physics at a fixed timestep (initially 120 Hz) and neural control at 60 Hz.
- Support reference animation changes, external impacts and continuous motor-strength variation.
- Benchmark multiple characters and verify training/runtime behavioral parity.

**Deliverable:** Neural-controlled physical characters running in Graaal with the same behavior observed during training.

## **Key technical decisions**

- **Physics:** pyjoltphysics/Jolt; no custom rigid-body or collision solver.
- **Passive behavior:** Soft angular limits, joint damping and weak neutral-posture springs, always active.
- **Active behavior:** Neural-controlled motor targets and strengths, bounded by physical capabilities.
- **Animation:** Reference poses provide movement intent; neural control adapts them to physical circumstances.
- **Learning:** PPO with a curriculum covering balance, impacts, falling, bracing and standing up.
- **Transitions:** Continuous loss and recovery of control through finite motor strength and physical constraints, rather than abrupt ragdoll mode switches.
- **Parallelism:** Native Jolt threads with batched NumPy observations and commands.
- **Main risks:** Implementing passive joint forces through available bindings, efficient motor updates and resets, training throughput, and avoiding unnatural learned behavior.

**Core principle:** Passive joint mechanics make the body move naturally; neural control makes it behave purposefully. Both operate continuously, so stumbling, falling and recovery emerge from the interaction between physical forces and limited active control.