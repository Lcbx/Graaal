# SDF penumbra plan

Proposed shadow pipeline; values below are starting targets for measurement.

## Frame layout

**CPU, start of frame:** animate and skin (including off-screen characters within a distance) to produce capsule poses, update rigid transforms, select shadow lights into global slots, and queue clipmap updates.

1. **Clipmap update** (GPU, no depth dependency): scrolled slabs and dirty regions, reading the full static instance list.
2. **Caster cull:** view frustum extruded toward the sun, plus light-radius spheres, with no HZB. Output is the capsule and dynamic-caster buffers.
3. **Depth prepass, then HZB build** (existing).
4. **Depth reduction:** half-res depth, per-tile min/max, a depth pyramid matching the blur chain, and normals from depth.
5. **Receiver cull** (existing, frustum then HZB). It also outputs the list of visible Probe-flagged instances. This is independent of step 4.
6. **Tile culling:** per-tile light mask, caster lists, skip flags and indirect dispatch args.
7. **Half-res trace** (indirect): clipmap plus capsules into an RGBA8 mask, one channel per global slot. **Probe pass** (indirect over visible Probe instances) can overlap it.
8. **Depth-aware dual Kawase** at half-res.
9. **Bilateral upsample** to the full-res mask.
10. **Forward opaque:** Mask, Probe or Trace per material.
11. **Deferred transparency:** None by default, or Probe/Trace per material.

## Decisions to lock early

- **Global shadow-light slots:** the sun is always slot 0 and the best local lights take slots 1-3, so the mask channel is the global slot. Per-tile masks only decide which slots get traced. Changing slot assignments fades strength over about 0.2 s. That's a scalar lerp with no history buffer, so it stays consistent with "no temporal accumulation." Raising the cap above 4 is what would create tile seams, so test that deliberately.
- **Receiver modes** (a per-material enum):

| Mode | Default for | Notes |
|---|---|---|
| Mask | Anything in the prepass | Valid only if the object is in the prepass. Tooling errors otherwise. |
| Probe | Opaque objects not in the prepass | Per-instance samples, opt-in for transparents |
| None | Transparents | Fully lit |
| Trace | Nothing | Per-fragment full-res trace for hero cases only |

- **Prepass cutoff:** add a per-material "force into prepass" flag for small receivers, since that's the cheapest fix and gives them Mask. Alpha-tested materials need the same decision. Either they alpha-test in the prepass, or they fall into Probe, which suits foliage poorly.
- **Trace as a shared include**, callable from compute and fragment shaders. The probe pass, Trace mode and the half-res trace all use it.
- **Capsules** are evaluated inside the step loop at first, with the same `min(vis, k*d/t)` formula as the world. Switch to closed-form ray-vs-capsule evaluation only if profiling demands it.
- **Softness:** each light exposes an angular radius, and `k` is derived from it.
- **Blur halos:** depth rejection in the Kawase chain needs a depth value at each level, hence the pyramid in step 4. If halos persist, a depth-aware two-pass blur at half-res is the fallback.

## Mesh and material processing (import time)

1. **Classify** each mesh as static caster, rigid dynamic caster, skeletal (capsule rig) or non-caster. Alpha-tested foliage and hair default to non-caster or a proxy.
2. **Clean the mesh:** weld vertices, fix winding, and detect non-watertight geometry.
3. **Inside/outside** comes from a generalized winding number or multi-direction ray voting. Open or thin meshes get an unsigned distance minus an authored thickness.
4. **Bake the SDF:** 32-64³ per mesh with a minimum voxel size, bounds padded by the max influence distance, R8/R16 distances, and 2-3 mips.
5. **Capsule rig** for skeletal meshes: one tapered capsule per bone fitted from skin weights, a manual editor with a gizmo view, and a single body capsule for far LOD.
6. **Leak report** listing features thinner than a voxel.
7. **Receiver audit:** assign default receiver modes from prepass membership and flag Mask on non-prepass materials as errors.

## Scene processing

1. **Instance gather:** the full static caster list with transforms. Non-uniform scale is restricted or compensated by multiplying distance by the smallest scale axis.
2. **Clipmap build:** gather-style, with a per-brick instance list and a min-composite per voxel. R8 textures don't support atomic min, so avoid stamping with atomics. Add a heightmap-to-SDF source for terrain. Cascades clamp distance to about 8 voxels.
3. **Scrolling:** toroidal addressing with origins snapped to the voxel size, updating only the newly exposed slabs.
4. **Optional brick cache** on disk for shipping builds.
5. **Per-frame dynamic data:** capsule transforms, rigid props and light selection.

## Phases

**Phase 0: groundwork**
- GPU timers and debug views (clipmap slice, mask channels, tile heatmap, capsule gizmos, receiver-mode overlay).
- A reference render (high-res shadow map) for A/B comparison.
- A placeholder budget of about 2 ms at 1080p on your lowest GPU.
- Counters for how many visible instances fall into each receiver mode.

**Phase 1: mesh SDF baker**
- Mesh steps 1-4, 6 and 7, plus a sphere-trace viewer.
- *Exit:* baked SDFs match the meshes and the leak report catches your thin test assets.

**Phase 2: static world and naive sun**
- Scene steps 1-2 on a fixed test level, and the shared trace include running full-res in the forward pass.
- *Exit:* plausible soft shadows, per-cascade bias tuned, cascade blending and distance fade working.

**Phase 3: half-res pipeline**
- Depth reduction, normals from depth, half-res trace, depth-aware Kawase, bilateral upsample and forward consumption.
- Non-prepass small objects temporarily use Trace mode, so the pipeline is testable before probes exist.
- *Exit:* no dark halos around silhouettes, no lit fringes in contact shadows, the sun is in budget, and a small prop in front of a distant wall shows no mask leakage.

**Phase 4: tiles and local lights**
- Global slots, light selection with fades, per-tile depth min/max, light masks, indirect dispatch, early-outs, rays capped at light radius, and step count scaling with distance.
- *Exit:* a worst-case tile (4 lights) stays in budget with no tile seams.

**Phase 5: characters**
- Mesh step 5, the extruded-frustum caster cull, the capsule buffer, per-tile caster lists (overflow flag in debug), the in-loop evaluation and LOD cross-fade by distance.
- *Exit:* a walking character grounds correctly with no style mismatch, LOD doesn't pop, and an off-screen or occluded character still casts a correct shadow into view.

**Phase 5b: probe pass**
- Compute over visible Probe instances: 1 sample if tiny, otherwise 2×2×2 at the oriented bounds corners.
- Samples inside the static SDF are pushed outward along the gradient by `-d + epsilon`. The instance's own SDF and capsules are excluded from its samples.
- The fragment shader trilinearly interpolates by object-space position, then multiplies per light like the mask.
- Distant instances drop to 1 sample, and very distant ones inherit the mask.
- *Exit:* a small prop on the ground in a penumbra has no dark bottom face.

**Phase 6: streaming and dynamic props**
- Clipmap scrolling and brick streaming. Rigid props start as capsules or boxes, then get per-object SDF volumes in the caster lists if needed.
- *Exit:* a flythrough shows no shimmer or cascade shifts.

**Phase 7: transparents**
- Default None. Per-material Probe, with one probe per emitter for particles. The deferred pass stores the instance ID.
- Use 4³ samples for large surfaces (panes, water, smoke volumes). A low-res froxel grid is the upgrade path if that fails.
- *Exit:* smoke in sun versus shade, and a glass pane crossing a shadow boundary.

**Phase 8: polish**
- Contact shadows (about 8 steps, sun only, combined with `min`). These can run inside the forward shader, which has the true fragment normal and little overdraw behind a prepass.
- Penumbra-driven variable blur from the Kawase pyramid, and SDF mip selection by cone width.
- Presets: Low is quarter-res, sun plus 1 light and 12 steps. Medium is half-res, 2 lights and 16 steps. High is half-res, 4 lights and 24 steps.

## Starting values

| Item | Start with |
|---|---|
| Clipmap | 4 cascades × 128³ R8 (~8 MB), voxels 0.25 / 0.5 / 1 / 2 m |
| Trace tile | 16×16 at half-res |
| Mask | RGBA8 half-res (~2 MB at 1080p) |
| Steps | 16-24 |
| Capsule | 32 B, ≤24 per character |
| Per-tile caster cap | 32 (overflow flagged) |
| Probe | 8 samples × 4 B = 32 B per instance |

## Test scenes

- A thin wall and fence for leaks.
- A character on flat ground in sun and under a local light.
- A flythrough across cascade boundaries.
- A tile with 4 lights plus several characters.
- A high-contrast silhouette against a distant background for blur halos.
- A small prop in front of a distant wall.
- A prop in a penumbra.
- A character behind the camera and one behind a wall, both casting into view.
- Smoke in sun versus shade.
- A glass pane crossing a shadow boundary.

## Open items to settle by measurement

- **Prepass cutoff:** where it sits and how many instances land in Probe mode. If there are only a few, forcing them into the prepass may beat building the probe path.
- **Transparent mix:** particles and smoke, or surfaces like glass and water. This decides whether per-emitter probes suffice or the 4³ grid is needed.
- **The 4-light cap:** confirm with the worst-case tile test in Phase 4.