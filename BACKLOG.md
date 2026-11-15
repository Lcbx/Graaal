# Engine backlog

Grouped by estimated implementation difficulty, not priority.
Estimates assume the current engine architecture; items with open design questions may move between groups.

## Small — investigation, fixes, and local changes

- Check that alpha mask textures use a low-precision format for alpha.
- Grazing fade is not taken into account for alpha mask (use dithering).
- Ensure nearest material is not recorded in the stochastic banded tail.
- See how banded / noise is used (read doc).
- Check out old anchored blue implementation; see if it worked well / can be reused.
- Add setting for Hi-Z level to stop culling at (memory vs leniency).
- Use squared distances and FoV for LOD; take FoV into account.

## Medium — contained rendering and tooling features

- Add proper monitoring and logging.
  - Time different passes so performance can be assessed.
- Make SSAO respect normals.
- Implement ramp / toon shading.
- Experiment with a Kuwahara material.
- Add post-process depth/normal edge detection outlines.
- Ensure depth prepass only has close / big surfaces, not alpha-tested surfaces.
- Make wgpu-py Canvas use MSAA for main buffer (?).
- Implement parallax texture mapping.
- Implement triplanar mapping.
- Add sphere frustum & Hi-Z check before (instead of ?) AABB.
  - Use it for region binning on CPU ?
- Can we use different shaders for different LOD levels?
- Ensure transparency pass uses a stencil to avoid dispatch on unaffected pixels.
- Support decals (uses transparency pass).
  - Implementation can be based on Material, MaterialRenderer, TransparencyPass and the related shaders.
- Implement a simple irradiance map based on global directional map.
- Peripheral motion blur (take camera motion for intensity multiplied with `1 - vignette mask`).
- Godrays: https://blog.cclarke-magrab.me/blogs/graphics/god-rays-02

## Large — new subsystems and integration work

- Implement shadow maps.
- Implement Forward+ / tiled shading.
  - Add compute light binning.
- Implement GPU animation using custom vertex attributes and animation texture atlas.
- Use OpenAL for binaural audio.
- Use Bullet for physics (at first; maybe migrate to ctypes & Jolt or Box3D later).
  - Add Box3D support?
- Add a particle system, using GPU particles:
  - Particles are mesh instances whose creation, updates and death are managed by GPU batches.
  - Creation batch takes an array of positions, a number of particles to spawn at each position, a `lod_group_id`, and a particle material. This avoids polluting regular materials.
  - Default `ParticleMaterial` contains emission shape (initially point; later sphere, disc, rectangle, box), velocity (angle range, acceleration, friction), and lifetime (randomizable range).
  - Update batch destroys a particle if its material sets `should_die` to true.
  - Future improvements: explosiveness / respawning particles. In theory any mesh can be a particle.
- Systems pipeline (multithreaded). Use Culverin?
- Custom Blender scene export.
  - Maybe just use a glTF export per asset, save scenes with `struct.pack` (do not trust indices / sizes!).
- Make a material that spawns objects on its surface (see implementation plan).
- Networking: eventually add multiplayer suppport (spacetimedb ?)
	or SteamPeer
	or use UDP hole punching (users must exchange IP)

## Very large / experimental — significant design and validation

- Implement grid-based ambient probes.
  - Retry illumination probes.
- Eventually, use SDF for penumbra mapping (see implementation plan).

## Not doing

- Meshlets / task shaders
- Dynamic GI / radiance cascades
- Inverted-hull outlines
- TAA, DLSS/FSR upscaling
