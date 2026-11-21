# Stateless grass scattering plan

Proposed GPU generation for roughly 20–50 spawners with about 200 instances
each. Generate from stable identity and current transforms; a missing spawner
produces no instances, and recreating its authored GUID reproduces its scatter.
The original C++/HLSL sketches are summarized here as algorithms; implementation
must use Graaal's Python/WGSL interfaces and explicit draw/visibility contracts.

## Contracts and data

- Seed from the authored GUID and instance index, never entity index, draw ID,
  buffer slot or frame. Hash GUID bytes once at load; regenerate duplicate
  placement GUIDs. Runtime children can derive identity from parent GUID + index.
- Use a fixed, low-poly scattering proxy independent of rendered ground LOD.
  Include only grass-bearing faces and keep it close to the visual surface.
- Support uniform positive parent scale only. Density uses proxy area × scale²;
  blade size stays independent of parent scale unless explicitly requested.
- Generate object-space samples separately from world-space resolution so a
  later cache can retain samples while parents move.
- Keep generation independent of ECS; an adapter collects ECS spawners. The
  existing culler needs an explicit integration for generated buffers rather
  than assuming a second source is already supported.

Spawner inputs: GUID/hash, proxy and grass mesh keys, optional normal map,
maximum count, density (zero selects fixed count), scale range, normal/up blend,
root sink and maximum blade height. GPU descriptors add output offsets and
shared proxy vertex/index/CDF offsets. CPU mirrors must match WGSL alignment.
Start with 256 instances per spawner and up to 1,024 proxy triangles; any
larger count requires matching capacity and dispatch handling.

## Proxy import and sampling

Build a normalized inclusive triangle-area CDF in float64, then store float32
entries with the final value exactly 1. Reject empty/zero-total-area proxies.
Keep total area for density counts; painted triangle weights may multiply areas
before normalization. Zero-area triangles repeat the preceding CDF value;
select the first entry strictly greater than the sample so they are skipped.
Tiny contributions can disappear at float32 precision.

Use an R3 additive recurrence with a GUID-derived three-axis offset:
`u = fract(offset + (0.8191725134, 0.6710436067, 0.5497004779) * (index + 1))`.
Choose a triangle from `u.x`. For uniform triangle sampling, set
`s = sqrt(u.y)` and barycentrics `(1-s, s*(1-u.z), s*u.z)`.
Hash GUID + index independently for yaw, scale and tint. This preserves every
prefix when density/LOD changes. For counts beyond a few thousand, consider a
wrapping fixed-point recurrence instead of float32.

Interpolate proxy position/normal. Optional normal mapping requires proxy UVs
and tangents, orthogonalized tangent frames and explicit texture LOD in compute.
Decode the map's format correctly, including reconstructed Z for two-channel maps.
Transform position/normal, blend the normal toward world +Y, sink the root,
then construct an orthonormal frame and apply yaw. Reject downward faces that
could produce a zero up vector. Preserve handedness and winding.

## Frame integration

1. Collect current spawners after transforms. Count is
   `min(max_count, round(density * total_area * scale²))`, or `max_count` for fixed density.
2. Prefix-sum contiguous output offsets; validate capacity and upload descriptors
   with lifetime appropriate to frames in flight. Skip empty dispatches.
3. Dispatch one workgroup per spawner, cooperatively loading its CDF before sampling.
4. Record generation before culling reads its output. Express ordering through
   WGPU commands; native D3D/Vulkan barrier recipes are not application APIs here.
5. Route generated transforms, mesh keys and bounds through the same visibility
   and indirect-draw rules as CPU instances. Keep all metadata in packed instance order.

Optional parent culling uses bounds expanded by blade height, scale and wind.
Include every consuming view, including shadow views, or offscreen grass may
lose visible shadows. Distance LOD reduces the prefix count; shrinking the last
few blades can soften changes. Profile before adding either optimization.

## Optional cache and baked sampling

If generation becomes significant, cache local samples (position, octahedral
normal, yaw, scale, tint; about 32 bytes) in fixed 256-entry slots. A 128-slot
pool is about 1 MiB. Track GUID → slot, free slots, parameter hash and last use.
Headers contain current transform/up blend/root sink/mesh key and resident count.

Allocate/regenerate on appearance or parameter changes; free on destruction.
For streaming, evict distant slots with hysteresis. The cache key includes GUID,
proxy/normal-map versions, count, scale range and generator version. Exclude
transform, up blend, root sink and mesh key: apply these during resolution in
the culler. Skip nonresident slots and entries beyond their active count.
Reuse slots only when prior consuming GPU work is safe; multiple queues require
explicit synchronization. Cached and stateless paths must share the resolver.

If R3 coverage is insufficient, bake prefix-ordered Poisson/sample-elimination
points as `(triangle, u, v)`. Runtime sampling becomes a table lookup; optionally
bake normals too. Store per proxy, with 4–8 GUID-selected variants to reduce
repeated patterns. Preserve the same resolver and prefix-LOD contract.

## Validation

Check flat-quad coverage, area-weighted counts on unequal triangles, skipped
degenerates, static-frame determinism, distinct GUIDs and recreation, transform
attachment, density scaling and unchanged prefixes. Debug coloring by triangle
helps find offset errors. Verify normal-map UVs, handedness, output capacity,
empty dispatches and buffer lifetime before performance tuning.
