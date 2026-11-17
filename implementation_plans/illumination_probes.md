Dynamic SH Probe Clipmap Lighting System


## 1. Goals

The renderer targets:

* Forward opaque rendering with GPU indirect draws.
* Depth prepass + Hi-Z culling.
* Fully dynamic geometry, lights, and emissive materials.
* No baked lighting.
* No TAA, temporal GI accumulation, temporal denoising, or stochastic temporal sampling.
* Deterministic frame-to-frame lighting.
* Multiple direct lights.
* Dynamic diffuse indirect lighting from:
  * sky;
  * directional lights;
  * local lights;
  * emissive surfaces.
* Soft, low-frequency, stylized bounce lighting.
* Low-resolution filtered direct shadows.
* Steam Deck-class hardware as a lower target.

The goal is stable and plausible diffuse GI rather than physically exact light transport.

---

# 2. High-level architecture

Direct and indirect lighting are mostly independent.

```text
DIRECT LIGHTING

lights
  ↓
shadow maps
  ↓
ESM filtering
  ↓
forward shading
```

```text
INDIRECT LIGHTING

sky / lights / RSMs / emissives
              ↓
        SH injection
              ↓
     SH 3D probe textures
              ↓
occlusion-aware propagation
              ↓
final SH clipmap textures
              ↓
forward probe lookup
```

Geometry visibility is stored separately:

```text
GI occluder proxies
        ↓
occupancy buffer
        ↓
six-direction free paths
        ↓
visibility / validity buffer
```

---

# 3. Probe clipmaps

Use several camera-centered nested grids.

Example:

```text
Level 0
32³ probes
0.5 m spacing

Level 1
32³ probes
2 m spacing

Level 2
32³ probes
8 m spacing
```

Spacing follows a geometric progression.

Each clipmap origin is snapped to its probe spacing:

```text
origin =
    floor(camera_position / spacing)
    * spacing
```

Probe positions therefore remain world anchored while the camera moves inside a cell.

Toroidal addressing may be used so that crossing a grid boundary only changes newly exposed slices.

Near clipmap borders, blend spatially into the next coarser level.

---

# 4. Probe lighting representation

Each probe stores first-order RGB spherical harmonics:

```text
L1 SH
=
4 coefficients × RGB
=
12 scalar values
```

L1 preserves low-frequency directional effects such as:

```text
sky from above
ground bounce from below
colored wall bounce from the side
```

The SH field is stored directly in sampled 3D textures.

A practical packing is:

```text
3 × RGBA16Float 3D textures
```

per clipmap.

For `32³`:

```text
32³ × 24 bytes
≈ 0.75 MiB
```

per SH field.

Propagation ping-pongs between two SH texture sets:

```text
SH_A
SH_B
```

---

# 5. Geometry occupancy

Each probe level also has a temporary binary occupancy grid:

```text
0 = empty
1 = occupied
```

The occupancy resolution initially matches the probe grid.

Geometry is represented using simplified GI occluder proxies rather than full visible triangles.

Typical proxies:

```text
transformed mesh bounds
explicit GI occluder boxes
simple collision geometry
```

Voxelization is conservative:

```text
proxy intersects cell
→ occupied
```

A small expansion such as:

```text
5–10% of probe spacing
```

may be used.

Do not dilate by a complete voxel.

The occupancy buffer may simply be rebuilt every frame initially.

A temporary:

```text
atomic<u32>[probe_count]
```

buffer is sufficient.

---

# 6. Probe validity

If the probe cell itself is occupied:

```text
probe = invalid
```

Invalid probes:

* receive no normal lighting injection;
* do not propagate GI;
* are excluded from geometry-aware interpolation.

Validity is packed into the persistent visibility metadata.

After visibility generation, the temporary occupancy buffer can be reused.

---

# 7. Six-direction visibility

Each probe stores free-path distance along:

```text
+X
-X
+Y
-Y
+Z
-Z
```

Distances are stored as numbers of empty cells rather than world-space floats.

Use `u8` values:

```text
0   = occupied immediately next to probe
1   = one empty cell
2   = two empty cells
...
255 = no blocker found
```

Pack visibility into:

```text
visibility0 : u32
    +X -X +Y -Y

visibility1 : u32
    +Z -Z flags
```

Total:

```text
8 bytes / probe
```

The flags contain at least probe validity.

---

# 8. Visibility generation

Free paths are generated from the occupancy grid using three directional sweep passes.

For X:

```text
one invocation per YZ line

left → right
right → left
```

Repeat for Y and Z.

This produces all six distances.

For a `32³` grid, the total work is very small.

Afterward:

```text
occupancy buffer
→ no longer needed by normal GI propagation/shading
```

Only the compact visibility buffer remains.

---

# 9. Indirect lighting sources

Probe GI can receive energy from:

```text
sky
directional lights
local lights
emissive surfaces
```

Each source produces low-frequency directional energy that is projected into L1 SH.

---

# 10. Sky lighting

Sky lighting can be represented directly in SH.

A simple stylized model may use:

```text
upper hemisphere = sky color
lower hemisphere = ground color
```

Directional visibility reduces sky contribution for enclosed probes.

---

# 11. Directional-light GI

Important directional lights use Reflective Shadow Maps.

The light-space pass stores:

```text
depth
normal
albedo
reflected flux
```

World position is reconstructed from depth.

Approximate reflected energy is:

```text
flux ∝
    light_radiance
    × albedo
    × max(N·L, 0)
```

This energy is projected into SH and injected into nearby probes.

---

# 12. Local-light GI

Direct-light participation and GI participation remain separate.

A light may expose:

```text
casts_shadow
gi_enabled
indirect_intensity
```

Small local lights may inject analytically into nearby probes.

Important spot lights may use RSMs.

Point-light RSMs should be reserved for important lights because several projections are required.

---

# 13. Emissive geometry

Emissive surfaces produce a bounded number of surfels:

```text
world position
normal
radiance
surface weight
```

These inject directional SH energy into nearby probes.

Rigid meshes may store local-space surfels and transform them per instance.

The emitted lighting remains fully dynamic.

---

# 14. SH injection

Injection behaves as a scatter operation:

```text
light sample
   ↓
several probes
```

Because SH textures are floating-point textures, scatter injection uses a temporary fixed-point atomic buffer.

For example:

```text
atomic<i32>
```

per SH component.

Injection:

```text
float SH contribution
→ fixed-point scale
→ integer
→ atomicAdd
```

Then resolve:

```text
atomic accumulator
→ floating-point SH
→ SH_A 3D textures
```

This temporary accumulator is used by:

* RSM samples;
* emissive surfels;
* analytic local-light injection.

---

# 15. Spatial propagation

Propagation is performed directly between SH 3D textures.

Example:

```text
SH_A
  ↓
propagation pass
  ↓
SH_B
  ↓
propagation pass
  ↓
SH_A
```

Each probe reads its six immediate neighbors.

Propagation is allowed only if:

```text
current probe valid
neighbor valid
current free path toward neighbor is open
neighbor free path toward current probe is open
```

Otherwise the contribution is zero.

Conceptually:

```text
new_SH =
    local_SH
    + k × visible_neighbor_SH
```

Use only a small fixed number of passes:

```text
1–3
```

Two passes are a good initial target.

Propagation intentionally reconstructs smooth local irradiance rather than simulating many physical bounces.

---

# 16. Forward GI lookup

The final propagated SH textures are directly sampled by the forward shader.

The cheapest path uses hardware trilinear filtering:

```text
3 × trilinear 3D texture samples
```

to retrieve the 12 SH values.

Then:

```text
irradiance =
    evaluate_SH(SH, surface_normal)

indirect_diffuse =
    albedo × irradiance
```

This makes normal per-pixel GI lookup very cheap.

---

# 17. Geometry-boundary lookup

Ordinary hardware trilinear filtering can interpolate lighting through a wall.

The visibility buffer therefore remains available for a more conservative lookup near geometry boundaries.

Two lookup modes can be supported.

## Fast path

In clearly open regions:

```text
hardware trilinear SH sampling
```

## Boundary path

Near occupied regions:

```text
load 8 surrounding SH texels
        ↓
read probe visibility metadata
        ↓
visibility-aware weights
        ↓
manual trilinear interpolation
```

For probe `P` and surface `X`:

```text
delta = X - P
```

The appropriate signed free path is selected independently for each axis.

Approximate world-space clearance:

```text
clearance =
    (free_cells + 0.5)
    × probe_spacing
```

Probe weight becomes:

```text
weight =
    trilinear_weight
    × visibility
    × validity
```

then weights are normalized.

A small smooth transition near the visibility boundary avoids abrupt changes.

This boundary-aware path can be added only if simple trilinear sampling produces visible leakage.

---

# 18. Direct shadows

Direct shadows remain separate from probe GI.

## Directional lights

Use:

```text
CSM
+
ESM
```

Pipeline:

```text
shadow depth
→ exponential transform
→ separable blur
→ filtered visibility
```

The visibility may be remapped using a smooth artistic ramp.

## Spot lights

Use ordinary 2D ESM maps, preferably atlas-packed.

## Point lights

Only important point lights should cast full dynamic shadows.

---

# 19. Ambient occlusion

GTAO/SSAO handles only high-frequency local occlusion.

Its role is:

* contact darkening;
* creases;
* reinforcing missing small-scale GI occlusion.

It should not be responsible for stopping large-scale probe leakage.

Typical application:

```text
indirect *= mix(1.0, AO, ao_strength)
```

---

# Frame timelines

The renderer contains several mostly independent GPU pipelines.

---

## A. Scene and GI visibility

```text
scene transforms updated
        ↓
clipmap origins updated
        ↓
occupancy scratch cleared
        ↓
GI proxies voxelized
        ↓
probe validity determined
        ↓
X free-path sweeps
        ↓
Y free-path sweeps
        ↓
Z free-path sweeps
        ↓
packed visibility buffer ready
```

The occupancy scratch buffer may now be reused.

---

## B. Direct shadows

```text
shadow-casting lights selected
        ↓
directional CSM
spot / point shadow maps
        ↓
ESM conversion
        ↓
separable blur
        ↓
direct-shadow data ready
```

---

## C. GI source generation

```text
GI lights selected
        ↓
RSMs rendered for important lights

emissive surfels updated

analytic local-light data prepared
        ↓
GI source data ready
```

These operations can largely run independently of the camera visibility pipeline.

---

## D. Probe GI generation

Requires GI source data and probe visibility metadata.

```text
atomic SH accumulator cleared
        ↓
sky injection
RSM injection
local-light injection
emissive injection
        ↓
resolve accumulator → SH_A
        ↓
propagation SH_A → SH_B
        ↓
propagation SH_B → SH_A
        ↓
final SH clipmap textures ready
```

---

## E. Camera visibility

```text
opaque depth / normal prepass
        ↓
Hi-Z generation
        ↓
GPU frustum culling
        ↓
GPU Hi-Z culling
        ↓
visible indirect draw streams ready
```

This pipeline is largely independent of probe GI generation.

---

## F. Final opaque shading

Requires:

```text
visible draw streams
direct-shadow textures
final SH clipmap textures
```

Then:

```text
forward opaque pass
    ├─ direct lighting
    ├─ ESM shadows
    └─ SH probe GI
        ↓
main color
        ↓
GTAO
        ↓
AO modulation of indirect lighting
        ↓
final opaque image
```

---

# Initial implementation

A minimal useful version is:

```text
2 × 32³ probe clipmaps

L1 RGB SH
3 × RGBA16Float textures per SH field

SH_A / SH_B propagation ping-pong

binary occupancy scratch buffer

6 × u8 free paths
packed into 2 × u32 / probe

conservative proxy voxelization

sun + sky GI
sun RSM

emissive surfels

fixed-point atomic injection

2 propagation passes

hardware-trilinear SH lookup initially

optional visibility-aware boundary lookup later

CSM + ESM direct shadows

GTAO
```

The resulting architecture keeps responsibilities simple:

```text
3D textures
    → lighting field

compact buffers
    → visibility / topology

atomic buffers
    → temporary scatter accumulation
```

This preserves the main design goals: deterministic lighting, fully dynamic GI, low-frequency directional bounce, low per-pixel cost, and no temporal reconstruction.
