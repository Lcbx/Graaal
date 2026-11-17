Active ragdoll

## Core Concepts & Tweaks

Standard ragdoll physics rely on **hard limits**—joints move freely within a range and instantly stop or snap back at the boundary. The presenter introduces two main tweaks to improve ragdoll realism:

1. **Soft Limits:** Joint rotation slows down as it approaches maximum and minimum boundaries, simulating the natural resistance of muscles, tendons, and ligaments.
2. **Natural search for equilibrium:** When pushed, humans tend to attempt to remain upright and standing. A loss of balance can result in steps to get feet under it or raised hands to brace against walls or the ground.
3. **Neutral Body Posture Pull:** Joints experience a subtle pull toward a relaxed equilibrium state based on NASA’s research on human posture in microgravity.

---

## Visual & Motion Differences

* **Zero Gravity Testing:** Hard limits leave limbs floating in arbitrary positions when moved. Soft limits cause limbs to naturally drift into a relaxed, human-like pose (slightly bent knees, forward-rolled shoulders, bent elbows).
* **Ground Impacts & Falling:**
* Hard limits cause joints to snap abruptly, bounce unnaturally, or settle into awkward poses (e.g., arms stiffly above the head).
* Soft limits absorb impacts gradually, giving the body inertia, perceived weight, and realistic landing positions.


* **Stair Falls & Explosions:**
* Soft limits remove the typical "flail and bounce" look when falling down stairs, adding weight and fluidity.
* Adding **rotational impact** to shoulders and legs during explosions (e.g., arms forced back, legs forced forward due to foot resistance on the ground) creates dynamic, realistic mid-air motion.



---

## Active Ragdolls & Procedural Locomotion

* Applying soft limits and posture pulling to **active ragdolls** (simulated characters driven by procedural animation) allows characters to dynamically balance.
* Depending on the torso's orientation and the environment, some target poses or animation can be set.
* As simulation forces increase, the character transitions smoothly from an upright stance to upper-body simulation, eventually stumbling or collapsing naturally if balance cannot be maintained.
