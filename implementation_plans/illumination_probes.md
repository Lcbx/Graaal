# Dynamic SH probe clipmap lighting plan

Proposed dynamic, low-frequency diffuse GI for the forward renderer. Target
Steam Deck-class hardware, deterministic lighting, moving geometry/lights and
emissives, without baking or temporal accumulation. Direct shadows remain a
separate pipeline. These are design targets, not implemented APIs or measured budgets.

## Data and ownership

Use camera-centered nested grids with geometrically increasing spacing. Snap
each origin to `floor(camera_position / spacing) * spacing` so probes remain
world anchored within a cell. Blend into coarser levels at borders; toroidal
addressing can later limit updates to newly exposed slices.

| Resource per level | Representation |
| --- | --- |
| Lighting | L1 RGB SH: 4 coefficients × RGB in three `rgba16float` 3D textures |
| Propagation | Two lighting sets, SH_A/SH_B, used in ping-pong passes |
| Occupancy scratch | Binary `atomic<u32>` per cell; initially rebuilt each frame |
| Persistent visibility | Six axis-aligned free paths as u8 cell counts, plus validity, packed into two u32 values per probe |
| Injection scratch | Fixed-point `atomic<i32>` per SH component, resolved into SH_A |

A 32³ SH field costs 0.75 MiB before ping-pong duplication. Example spacings
are 0.5, 2 and 8 m; begin with two 32³ levels and tune coverage on target hardware.
Textures hold lighting, compact buffers hold topology, and atomic buffers are
temporary scatter/occupancy storage.

## Geometry visibility

Conservatively voxelize simplified GI proxies: transformed bounds, explicit
boxes or collision geometry. A proxy intersecting a cell marks it occupied.
Allow about 5–10% spacing expansion; avoid whole-voxel dilation.

Occupied probe cells are invalid: exclude them from injection, propagation and
geometry-aware interpolation. Generate six free paths with forward/backward
sweeps over X, Y and Z lines. Encode 0 for an immediately adjacent blocker,
1 for one empty cell and 255 for no blocker found. Pack ±X/±Y into the first
u32 and ±Z/flags into the second. Release occupancy scratch after these sweeps.

## Sources and injection

| Source | Injection data |
| --- | --- |
| Sky | SH sky/ground hemispheres, attenuated by directional visibility |
| Important directional lights | Reflective shadow maps (depth, normal, albedo, reflected flux); reconstruct position from depth |
| Local lights | Analytic injection for small lights; RSMs for important spots; point-light RSMs only where their projection cost is justified |
| Emissives | Bounded surfels containing position, normal, radiance and surface weight; rigid meshes can retain local-space samples |

Keep `casts_shadow`, `gi_enabled` and `indirect_intensity` independent.
Approximate reflected flux as radiance × albedo × `max(N·L, 0)`.
Project source contributions into L1 SH, scatter through fixed-point atomic
adds, then resolve to floating-point textures.

## Propagation and shading

Each propagation pass reads six neighbors. Transfer energy only when both
probes are valid and their opposing free paths are open. Combine local energy
with weighted visible-neighbor energy; start with two passes (target range 1–3).
This reconstructs smooth irradiance rather than many physical bounces.

The initial forward lookup uses three hardware-trilinear texture samples,
evaluates SH at the surface normal and multiplies irradiance by albedo.
If this leaks through walls, add a boundary path that loads eight probes and
normalizes trilinear weights multiplied by visibility and validity. For each
axis, compare the signed surface-to-probe offset with approximate clearance
`(free_cells + 0.5) * spacing`; soften the cutoff to avoid abrupt boundaries.

Direct directional shadows use CSM + ESM (depth, exponential transform,
separable blur, optional artistic visibility ramp). Spots use 2D ESM maps,
preferably atlased; reserve point shadows for important lights. GTAO/SSAO
adds contact/crease detail to indirect light and must not mask large-scale GI
leakage: `indirect *= mix(1, AO, ao_strength)`.

## Frame dependencies and initial scope

1. Update transforms/origins, voxelize proxies and generate packed visibility.
2. Prepare direct shadow maps and GI sources; these can run independently of camera visibility.
3. Clear/inject/resolve SH, then propagate SH_A → SH_B → SH_A.
4. Build camera depth/normal prepass and Hi-Z, then frustum/Hi-Z visible draw streams.
5. Forward shading consumes visible draws, direct shadows and final SH. Apply AO to the indirect contribution.

Start with sun + sky, a sun RSM, emissive surfels, conservative proxies,
fixed-point injection and hardware-trilinear lookup. Add boundary lookup and
incremental clipmap updates after measuring leakage and cost. Validate moving
occluders, enclosed rooms, thin walls, clipmap borders and deterministic static
frames before expanding light counts.
