# Stateless Grass Scattering: Implementation Guide

Design and build notes for ECS-driven grass spawners (about 20 to 50 spawners, about 200 instances each) feeding an existing GPU cull pass. Stateless generation comes first. Caching is covered in Addendum A, and a baked blue-noise variant in Addendum B.

## 1. Overview

Every frame, a small compute pass regenerates every grass instance from three inputs: the spawner’s GUID, the instance index and the spawner’s current transform. Nothing is remembered between frames, so there are no has-spawned flags and no despawn logic. A spawner that is destroyed or culled simply isn’t dispatched, and one that comes back with the same GUID reproduces the identical scatter.

Decisions this guide commits to:

- **Seed:** a hash of the spawner’s authored GUID plus the instance index. Never the draw ID, entity index, frame number or buffer slot.
- **Distribution:** area-weighted triangle choice through a CDF, driven by an R3 low-discrepancy sequence with a per-spawner offset instead of independent random numbers. At 200 samples this gives visibly more even coverage than white noise.
- **Proxy mesh:** a fixed, low-poly, single-LOD scattering mesh, separate from the rendered ground. Its CDF is a few hundred floats and fits in groupshared memory.
- **Scale:** uniform only. Areas scale by s² and normals need no inverse-transpose.
- **Output:** world-space instances written to a GPU buffer that your existing flat cull pass reads as a second source, so culling stays in one place.
- **Orientation:** sampled from the proxy’s normal map, blended toward world up, with a random yaw.

Assumptions: HLSL-style shaders, column-vector matrices, Y-up, and a cull pass that dispatches flat over an instance array whose elements carry whatever batch or mesh ID the pass needs. Adjust where yours differ.

## 2. Frame flow and build order

Per frame:

1. **Collect (CPU, ECS system):** query spawners, compute instance counts, write one descriptor per spawner, and compute each spawner’s output offset with a running prefix sum.
2. **Upload:** descriptors into a ring-buffered buffer (one copy per frame in flight, exactly like your existing instance upload). CPU instances upload as today.
3. **Prepass and HZB build:** unchanged.
4. **Generate:** one workgroup per spawner. This only has to finish before the cull pass, so it can run anywhere earlier in the frame, including on an async compute queue alongside the prepass.
5. **Barrier:** transition the generated buffer from UAV to SRV (D3D12), or a compute-to-compute pipeline barrier from shader write to shader read (Vulkan).
6. **Cull:** the existing flat dispatch, now over CPU instances followed by generated instances. Frustum, HZB, compaction and indirect args are unchanged.
7. **Draw:** indirect draws as today.

Suggested build order, testing each step before the next:

1. Proxy import with CDF, plus a debug view of the proxy triangles.
2. Generator writing positions only on a flat quad proxy. Check the distribution visually.
3. Wire the generated buffer into the cull pass.
4. Add the normal map, up-blend and yaw.
5. Add scale and tint variation, density and LOD.
6. Run the validation checklist in section 9.

## 3. Data model

### 3.1 ECS component

```cpp
struct GrassSpawner {
    Guid     id;              // authored, unique per placement, serialized with the scene
    uint32_t idHash;          // derived from id at load time, never serialized
    AssetRef proxyMesh;       // scattering proxy, imported together with its CDF
    AssetRef grassMesh;       // the mesh that gets instanced
    AssetRef normalMap;       // baked on the proxy UVs, optional
    uint32_t maxCount;        // must be <= 256 (one workgroup pass per 256 threads)
    float    density;         // instances per square meter; 0 means use maxCount as a fixed count
    float    scaleMin, scaleMax;
    float    upBlend;         // 0 = follow surface normal, 1 = world up
    float    rootSink;        // push roots below the surface, in meters
    float    maxBladeHeight;  // used to expand bounds for culling
};
```

### 3.2 From GUID to seed

Hash the 16 GUID bytes down to 32 bits once, when the component is created or loaded (any decent byte hash works: murmur3, xxHash, FNV-1a). Do not hash the string form and do not re-hash every frame.

Folding 128 bits into 32 is safe here. With 50 spawners the chance of any collision is about 3 in 10 million, and the only consequence would be two spawners sharing a pattern.

The real risk is duplicate GUIDs. A copy-pasted scene object or a prefab instanced several times will share a GUID and therefore a scatter. Add editor validation that regenerates the GUID on duplicate or instantiate. For spawners created at runtime, derive the GUID from the parent’s GUID plus a child index so it is still reproducible.

### 3.3 GPU structs

Proxies for all spawners are concatenated at load time into shared vertex, index and CDF buffers. Descriptors hold offsets into them.

```hlsl
struct ProxyVertex {
    float3 pos;
    float3 normal;
    float4 tangent;   // w = handedness sign
    float2 uv;
};

// The part that is used after sampling. Kept separate so the cache in Addendum A can reuse it.
struct ResolveParams {
    float4x4 localToWorld;   // uniform scale only
    float    upBlend;
    float    rootSink;
    uint     meshID;         // whatever the cull pass needs to route the instance
    uint     pad;
};

struct SpawnerDesc {
    ResolveParams resolve;
    uint  idHash;
    uint  instanceCount;     // after density and LOD, computed on the CPU
    uint  outputOffset;      // first slot in the generated buffer (CPU prefix sum)
    uint  cdfOffset;
    uint  triCount;
    uint  indexOffset;       // into the shared proxy index buffer (indices are proxy-local)
    uint  vertexOffset;      // into the shared proxy vertex buffer
    int   normalMapSlice;    // -1 if none
    float scaleMin;
    float scaleMax;
    uint2 pad2;
};
```

Mind packing and alignment on the CPU mirror of these structs, especially around float4x4 and the trailing padding.

## 4. Proxy asset import

Run this once at import time, on the CPU, in double precision.

```cpp
struct ProxyCdf { std::vector<float> cdf; double totalArea; };

ProxyCdf BuildCdf(std::span<const double3> pos, std::span<const uint32_t> idx)
{
    const size_t triCount = idx.size() / 3;
    assert(triCount > 0 && triCount <= 1024);          // matches MAX_PROXY_TRIS in the shader

    std::vector<double> area(triCount);
    double total = 0.0;
    for (size_t t = 0; t < triCount; ++t) {
        const double3 a = pos[idx[3*t]], b = pos[idx[3*t+1]], c = pos[idx[3*t+2]];
        area[t] = 0.5 * length(cross(b - a, c - a));
        total  += area[t];
    }
    assert(total > 0.0);

    ProxyCdf out;
    out.totalArea = total;
    out.cdf.resize(triCount);
    double running = 0.0;
    for (size_t t = 0; t < triCount; ++t) {
        running += area[t];
        out.cdf[t] = (float)(running / total);         // inclusive prefix sum, normalized
    }
    out.cdf[triCount - 1] = 1.0f;                      // exact, so the search always terminates
    return out;
}
```

Notes:

- **Zero-area triangles** repeat the previous cumulative value. The shader’s search returns the first entry strictly greater than r, so such triangles can never be picked.
- **Tiny triangles** whose area share is below roughly 1e-7 can round to the same float as their neighbours and become unselectable. Irrelevant for a proxy.
- **Store totalArea** in the asset. The CPU uses totalArea × s² to turn a density into a count.
- **Authored density (optional):** multiply each triangle’s area by a painted weight, for example the average vertex color, before building the CDF. The CDF then encodes both area and the density mask. Store the weighted total in place of totalArea.
- **Proxy only contains grass-bearing faces.** Leave out walls and undersides.
- **The proxy must hug the visual ground.** Where the two diverge, roots float or sink. Add an editor overlay drawing the proxy over the ground, and use rootSink to bury roots slightly.
- **Tangents and UVs** are required on the proxy if it uses a normal map, and the normal map must have been baked on that proxy’s UVs.

## 5. Generator shader

### 5.1 Helpers

```hlsl
static const uint   GROUP_SIZE     = 256;
static const uint   MAX_PROXY_TRIS = 1024;
static const float  TWO_PI         = 6.2831853;

// R3 additive recurrence: alpha_j = 1 / phi3^(j+1), phi3 = 1.2207440846
static const float3 R3_ALPHA = float3(0.8191725134, 0.6710436067, 0.5497004779);

uint PCGHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
uint  HashCombine(uint a, uint b) { return PCGHash(a ^ PCGHash(b)); }
float Hash01(uint v)              { return (PCGHash(v) >> 8) * (1.0 / 16777216.0); }   // [0,1)
```

float32 is precise enough for the R3 recurrence up to a few thousand instances per spawner. Beyond that, switch to a 32-bit fixed-point recurrence that wraps on overflow.

### 5.2 Surface sampling (object space, pure function of descriptor and index)

```hlsl
struct SurfaceSample {
    float3 posL;      // object-space position on the proxy
    float3 normalL;   // object-space normal, after the normal map
    float  yaw;
    float  scale;
    float  tint;      // 0..1 per-instance variation
};

groupshared float sCdf[MAX_PROXY_TRIS];

StructuredBuffer<float>        gCdf;
StructuredBuffer<uint>         gProxyIndices;
StructuredBuffer<ProxyVertex>  gProxyVerts;
Texture2DArray<float4>         gNormalMaps;
SamplerState                   gLinearClamp;

uint PickTriangle(float r, uint triCount)
{
    // First index whose cumulative value is strictly greater than r.
    uint lo = 0, hi = triCount - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi) >> 1;
        if (sCdf[mid] > r) hi = mid; else lo = mid + 1;
    }
    return lo;
}

SurfaceSample SampleSpawner(SpawnerDesc d, uint i)
{
    // Cranley-Patterson offset: spawners that share a proxy get different patterns.
    float3 offset = float3(Hash01(HashCombine(d.idHash, 1u)),
                           Hash01(HashCombine(d.idHash, 2u)),
                           Hash01(HashCombine(d.idHash, 3u)));
    float3 u = frac(offset + R3_ALPHA * (float)(i + 1u));

    uint tri = PickTriangle(u.x, d.triCount);
    uint i0  = gProxyIndices[d.indexOffset + tri * 3u + 0u];
    uint i1  = gProxyIndices[d.indexOffset + tri * 3u + 1u];
    uint i2  = gProxyIndices[d.indexOffset + tri * 3u + 2u];
    ProxyVertex a = gProxyVerts[d.vertexOffset + i0];
    ProxyVertex b = gProxyVerts[d.vertexOffset + i1];
    ProxyVertex c = gProxyVerts[d.vertexOffset + i2];

    // Uniform point in a triangle. Two independent uniforms used directly would be biased.
    float  s    = sqrt(u.y);
    float3 bary = float3(1.0 - s, s * (1.0 - u.z), s * u.z);

    float3 pos = a.pos * bary.x + b.pos * bary.y + c.pos * bary.z;
    float3 nrm = normalize(a.normal * bary.x + b.normal * bary.y + c.normal * bary.z);

    if (d.normalMapSlice >= 0)
    {
        float2 uv       = a.uv * bary.x + b.uv * bary.y + c.uv * bary.z;
        float3 tangent3 = a.tangent.xyz * bary.x + b.tangent.xyz * bary.y + c.tangent.xyz * bary.z;
        float  handed   = a.tangent.w < 0.0 ? -1.0 : 1.0;
        float3 t        = normalize(tangent3 - nrm * dot(nrm, tangent3));
        float3 bt       = cross(nrm, t) * handed;
        // SampleLevel, because compute shaders have no derivatives. Decode for your format (BC5 needs z rebuilt).
        float3 nts = gNormalMaps.SampleLevel(gLinearClamp, float3(uv, (float)d.normalMapSlice), 0).xyz * 2.0 - 1.0;
        nrm = normalize(t * nts.x + bt * nts.y + nrm * nts.z);
    }

    uint h = HashCombine(d.idHash, i);
    SurfaceSample o;
    o.posL    = pos;
    o.normalL = nrm;
    o.yaw     = Hash01(HashCombine(h, 11u)) * TWO_PI;
    o.scale   = lerp(d.scaleMin, d.scaleMax, Hash01(HashCombine(h, 12u)));
    o.tint    = Hash01(HashCombine(h, 13u));
    return o;
}
```

Why this is stable: every value depends only on the GUID hash, the index and static assets. The R3 sequence is also prefix-stable, so asking for 120 instead of 200 returns exactly the first 120 of the same layout.

### 5.3 Instance resolution (applies the parent transform and orientation)

```hlsl
InstanceData ResolveInstance(SurfaceSample s, ResolveParams p)
{
    float3 posW = mul(p.localToWorld, float4(s.posL, 1.0)).xyz;
    float3 nrmW = normalize(mul((float3x3)p.localToWorld, s.normalL));   // uniform scale: no inverse-transpose

    // Grass tends to grow upward regardless of slope. upBlend = 0 follows the surface, 1 is world up.
    float3 up = normalize(lerp(nrmW, float3(0, 1, 0), p.upBlend));
    posW -= up * p.rootSink;

    // Orthonormal frame around up, rotated by the random yaw.
    float3 helper = abs(up.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t0 = normalize(cross(helper, up));
    float3 b0 = cross(up, t0);
    float3 x  = t0 * cos(s.yaw) + b0 * sin(s.yaw);
    float3 z  = cross(x, up);   // right-handed; flip if your engine is left-handed, or blades mirror and winding inverts

    InstanceData inst;
    inst.SetTransform(x * s.scale, up * s.scale, z * s.scale, posW);   // pack into your own layout
    inst.meshID = p.meshID;
    inst.custom = s.tint;                                              // or wherever per-instance data lives
    return inst;
}
```

InstanceData is your existing instance struct. Blade scale deliberately ignores the parent’s scale. Multiply by it only if you want grass to grow with the object.

The proxy should have no downward-facing faces. A normal exactly opposite to world up combined with an upBlend of 0.5 would normalize a zero vector.

### 5.4 Kernel

```hlsl
StructuredBuffer<SpawnerDesc>  gSpawners;
RWStructuredBuffer<InstanceData> gOut;

[numthreads(GROUP_SIZE, 1, 1)]
void CSGenerate(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    SpawnerDesc d = gSpawners[gid.x];

    for (uint k = tid; k < d.triCount; k += GROUP_SIZE)
        sCdf[k] = gCdf[d.cdfOffset + k];
    GroupMemoryBarrierWithGroupSync();

    for (uint i = tid; i < d.instanceCount; i += GROUP_SIZE)
        gOut[d.outputOffset + i] = ResolveInstance(SampleSpawner(d, i), d.resolve);
}
```

The loop handles counts above 256 without changes. Dispatch is (descriptorCount, 1, 1), and skipped entirely when there are no spawners.

## 6. CPU side

One ECS system, run after transforms are final for the frame.

```cpp
void CollectGrassSpawners(World& world, GrassFrameData& out)
{
    uint32_t cursor = 0;
    for (auto [e, sp, xf] : world.Query<GrassSpawner, WorldTransform>())
    {
        const ProxyAsset& proxy = assets.Get(sp.proxyMesh);
        const float s = xf.UniformScale();                 // assert the three axis lengths match
        const float worldArea = proxy.totalArea * s * s;

        uint32_t count = sp.density > 0.0f
            ? std::min(sp.maxCount, (uint32_t)std::lround(sp.density * worldArea))
            : sp.maxCount;
        // count = ApplyDistanceLod(count, distance)       // optional, see section 8
        if (count == 0) continue;

        SpawnerDesc d{};
        d.resolve.localToWorld = xf.matrix;
        d.resolve.upBlend      = sp.upBlend;
        d.resolve.rootSink     = sp.rootSink;
        d.resolve.meshID       = assets.GpuMeshID(sp.grassMesh);
        d.idHash               = sp.idHash;
        d.instanceCount        = count;
        d.outputOffset         = cursor;
        d.cdfOffset            = proxy.gpuCdfOffset;
        d.triCount             = proxy.triCount;
        d.indexOffset          = proxy.gpuIndexOffset;
        d.vertexOffset         = proxy.gpuVertexOffset;
        d.normalMapSlice       = assets.NormalMapSlice(sp.normalMap);
        d.scaleMin             = sp.scaleMin;
        d.scaleMax             = sp.scaleMax;

        out.descs.push_back(d);
        cursor += count;
    }
    out.generatedCount = cursor;     // the cull pass needs this
    assert(cursor <= out.generatedCapacity);
}
```

Capacity of the generated buffer is the number of spawners times 256 times sizeof(InstanceData). With 50 spawners and 64-byte instances that is about 800 KB.

The ECS owns lifecycle completely: destroy the entity and the spawner disappears from the next frame’s list.

## 7. Cull pass integration

Your cull pass is a flat dispatch over a CPU-uploaded instance array, so the generated instances become a second source behind it. The dispatch size is now cpuCount plus generatedCount, both known on the CPU.

```hlsl
[numthreads(64, 1, 1)]
void CSCull(uint3 dtid : SV_DispatchThreadID)
{
    uint id = dtid.x;
    if (id >= cb.cpuCount + cb.generatedCount) return;

    InstanceData inst;
    if (id < cb.cpuCount)
        inst = gCpuInstances[id];
    else
        inst = gGenInstances[id - cb.cpuCount];

    // ... existing frustum and HZB tests, compaction append, indirect arg increment
}
```

Use an if/else, not a ternary, so only the valid buffer is read. Nothing else in the cull pass changes. Generated instances flow through the same visibility tests, compacted buffer and indirect draws as everything else.

If your cull pass needs per-instance bounds, they come from the mesh ID as for CPU instances.

## 8. Optional: parent culling and distance LOD

Both are optimizations. At 10,000 instances per frame the generator costs less than the dispatch overhead around it, so start without them.

**Parent culling.** In the collect system, test each spawner’s world bounds against the frustum and skip it if it fails. Expand the bounds by maxBladeHeight × scale plus your wind amplitude. If grass casts shadows, test against the union of every view that consumes these instances (main view and shadow cascades), otherwise off-screen spawners will drop shadows that fall into view. Parent culling is purely a CPU decision, so the output offsets stay contiguous.

**Distance LOD.** Because the sequence is prefix-stable, LOD is just a smaller instanceCount on the descriptor. The surviving instances are always a well-distributed subset of the full layout, and nothing reshuffles when the count changes. Each integer change adds or removes one blade, which is effectively invisible at 200 blades. To soften it further, shrink the last few blades toward zero scale.

## 9. Validation checklist

1. **Flat-quad test.** Use a single-quad proxy and draw the points. Coverage should look even with no streaks along an edge. Temporarily replace the sqrt mapping with raw (u.y, u.z) to see the bias the formula removes.
2. **Area weighting.** In a debug build, generate about 100,000 samples on a multi-triangle proxy with very different triangle sizes, read back per-triangle counts and compare to area share.
3. **Determinism.** In a static scene, capture the generated buffer on two different frames and compare. They must be bit-identical.
4. **GUID behaviour.** Two spawners with the same proxy and different GUIDs must have different patterns. Destroy and recreate one with the same GUID and the pattern must match.
5. **Transform behaviour.** Move, rotate and uniformly scale a spawner. Roots must stay attached and count must follow the density × s² rule.
6. **Prefix stability.** Halve the count and confirm the first half of the instances is unchanged.
7. **Degenerate triangles.** Add a zero-area triangle to a test proxy and confirm it never receives an instance.
8. **Debug color.** Tint blades by triangle index to catch indexing mistakes and offset errors quickly.

## 10. Common mistakes

- Seeding with time, frame number, draw ID, entity index or slot index. Only the authored GUID plus the instance index is stable.
- Duplicate GUIDs from copy-paste or prefab instancing, which silently produce identical scatter.
- Using Sample instead of SampleLevel in a compute shader.
- Missing the UAV to SRV barrier between the generator and the cull pass.
- A proxy with more than MAX\_PROXY\_TRIS triangles, or a proxy that no longer matches the visual ground.
- Normal map baked on the visual mesh UVs instead of the proxy UVs.
- Handedness or up-axis mismatch, which shows up as mirrored or sideways blades.
- Dispatching with zero descriptors.
- Forgetting to ring-buffer the descriptor upload per frame in flight.
- Non-uniform scale on a spawner. Assert on it in the collect system.

## Addendum A: Adding a cache

Only worth doing if profiling shows generation matters, or the instance count grows by about two orders of magnitude. The generator is already written as a pure function, so caching changes when it runs, not what it computes.

### A.1 Two splits that are already in the code

- **SampleSpawner versus ResolveInstance.** Sampling is object space and independent of the parent transform. Resolving applies the transform, the up-blend and the frame. Cache the sample, resolve in the cull pass.
- **The up-blend must stay in resolve.** It depends on the parent’s world rotation, so storing it would invalidate the cache whenever a parent rotates. This is why ResolveParams exists as its own struct.

### A.2 Storage

```hlsl
struct LocalInstance {         // about 32 bytes
    float3 posL;
    uint   normalOct;          // octahedral-encoded object-space normal
    float  yaw;
    float  scale;
    float  tint;
    uint   pad;
};

struct SlotHeader {
    ResolveParams resolve;     // transform, upBlend, rootSink, meshID
    uint resident;
    uint count;
    uint2 pad;
};
```

- A persistent GPU-only pool of LocalInstance, divided into fixed slots of 256 entries. 64 to 128 slots is plenty for 20 to 50 spawners, and 128 × 256 × 32 bytes is about 1 MB.
- A SlotHeader array of the same slot count, uploaded every frame next to the descriptors (ring-buffered, tiny).
- A CPU table mapping GUID to slot index, plus a free list. Per slot it stores a paramsHash and a last-used frame.

### A.3 Lifecycle

Tie residency to the ECS, not to the frustum. Frustum visibility is already handled per instance in the cull pass.

- **Spawner appears:** allocate a slot and add it to this frame’s regenerate list.
- **Parameters change:** if paramsHash differs from the stored one, add it to the regenerate list. Keep the slot.
- **Spawner destroyed:** free the slot. The header goes non-resident.
- **Pool full or streaming worlds:** evict least-recently-used spawners outside a radius (spawn at R, evict at about 1.25 R for hysteresis). Because generation is pure, a miss costs one dispatch and correctness never depends on cache state, so eviction can be sloppy.

### A.4 The cache key

paramsHash must cover everything that changes the sampled output:

- authored GUID (via idHash), proxy asset version, instance count, scaleMin and scaleMax, normal map version
- a GENERATOR\_VERSION constant that you bump whenever the sampling shader changes. Without this, editing the shader leaves stale cached data on screen.

It must not cover the transform, upBlend, rootSink or meshID, since those are applied at resolve time.

### A.5 Generation dispatch

Each frame, build descriptors only for slots on the regenerate list, with outputOffset set to slot × 256, and dispatch one workgroup per entry using a second kernel that writes LocalInstance instead of resolved InstanceData:

```hlsl
[numthreads(GROUP_SIZE, 1, 1)]
void CSGenerateLocal(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex)
{
    SpawnerDesc d = gSpawners[gid.x];
    // ... same CDF load into groupshared ...
    for (uint i = tid; i < d.instanceCount; i += GROUP_SIZE)
        gPool[d.outputOffset + i] = PackLocal(SampleSpawner(d, i));
}
```

Mark the slot resident on the CPU as soon as it is queued. On a single in-order queue, a UAV to SRV barrier before the cull pass is enough. No readback and no wait state is needed.

### A.6 Cull pass change (flat dispatch)

Add the pool as a third region in the same flat dispatch. Threads for empty slots return immediately:

```hlsl
uint total = cb.cpuCount + cb.slotCount * SLOT_CAPACITY;   // slotCount * 256
if (id >= total) return;

InstanceData inst;
if (id < cb.cpuCount)
{
    inst = gCpuInstances[id];
}
else
{
    uint p     = id - cb.cpuCount;
    uint slot  = p / SLOT_CAPACITY;
    uint local = p % SLOT_CAPACITY;
    SlotHeader h = gSlots[slot];
    if (h.resident == 0 || local >= h.count) return;
    inst = ResolveInstance(UnpackLocal(gPool[p]), h.resolve);
}
```

The same ResolveInstance function from section 5.3 is reused, so the stateless and cached paths cannot drift apart. Costs: one matrix multiply per live instance in the cull pass, and about 32,000 mostly idle threads, which is negligible.

Distance LOD stays free: compare local against count × densityFactor(distance) and return early.

### A.7 Gotchas

- **Frames in flight:** on a single in-order queue, reusing a freed slot is safe immediately. With async compute or multiple queues, hold a freed slot back for a few frames.
- **Hot reload:** bumping the asset version in paramsHash handles proxy and normal map reloads automatically.
- **Slot capacity:** assert maxCount ≤ 256 at authoring time.
- **Header upload:** it is rewritten every frame because transforms change; only the pool is persistent.
- **Parents that never move:** you can skip the resolve cost by resolving once into world space and invalidating on transform change, but this only pays off for fully static scenes.

## Addendum B: Baked blue-noise points (upgrade path)

If R3 coverage isn’t even enough, swap the sampler without touching anything else.

- **Offline:** run Poisson-disk or sample elimination over the proxy. Store the ordered points as (triangle, u, v), using sample elimination ordering so that every prefix is well distributed. Optionally bake normal map orientation into each point, which removes the runtime texture sample.
- **Runtime:** SampleSpawner becomes a table lookup by index. The signature and the prefix-stable LOD behaviour stay the same, and the CDF disappears.
- **Trade-off:** every spawner using the same proxy gets the same pattern. Bake several variants (4 to 8) and pick one per spawner with the GUID hash.
- **Memory:** per unique proxy, not per instance.
