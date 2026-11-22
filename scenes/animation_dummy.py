"""The shared textured character and baked locomotion clips for the animation scenes."""

from io import BytesIO
from pathlib import Path

import numpy as np

from graaal.Animation.SkeletalAnimation import Clip, Skeleton
from graaal.Animation.SkeletalGltf import load_skinned_gltf


ASSET = Path(__file__).parent / "resources" / "dummy_rigged.glb"
ANIMATIONS = ASSET.with_name("dummy_animations.npz")
UNIT_SCALE = 0.1


def load_dummy():
	from pygltflib import GLTF2

	asset = load_skinned_gltf(ASSET)
	if len(asset.primitives) != 1: raise ValueError("The dummy example expects one skinned primitive")
	primitive = asset.primitives[0]
	primitive.vertices["position"] *= UNIT_SCALE
	bind = asset.skeleton.inverse_bind.copy()
	bind[:, :3, 3] *= UNIT_SCALE
	rig = Skeleton(asset.skeleton.parents, asset.skeleton.translation * UNIT_SCALE, asset.skeleton.rotation, asset.skeleton.scale, bind)
	clips = [("Rest pose", Clip(rig.translation[None], rig.rotation[None], rig.scale[None], 30))]
	gltf = GLTF2().load(str(ASSET))
	joint_names = [gltf.nodes[i].name for i in asset.joint_nodes]
	with np.load(ANIMATIONS) as data:
		if data["joints"].tolist() != joint_names: raise ValueError("Baked animations do not match the dummy rig")
		for name in ("Walking", "Running"):
			rotation = data[f"{name}_rotation"]
			translation = np.tile(rig.translation, (len(rotation), 1, 1))
			translation[:, 0] = data[f"{name}_hips"] * UNIT_SCALE
			scale = np.broadcast_to(rig.scale, translation.shape)
			clips.append((name, Clip(translation, rotation, scale, float(data[f"{name}_fps"]))))
	return rig, primitive, clips


def dummy_material_mesh(primitive):
	from PIL import Image
	from pygltflib import GLTF2
	from graaal.Assets import gltf_image_bytes
	from graaal.Materials.DefaultMaterial import DefaultMaterialShader
	from graaal.Materials.MaterialRenderer import MaterialMesh
	from graaal.RenderPasses.TextureProcessing import build_material_texture_array
	from graaal.RenderPasses.TransparencyData import MATERIAL_DTYPE

	gltf = GLTF2().load(str(ASSET))
	pbr = gltf.materials[primitive.material_id].pbrMetallicRoughness
	image_id = gltf.textures[pbr.baseColorTexture.index].source
	with Image.open(BytesIO(gltf_image_bytes(gltf, gltf.images[image_id], ASSET.parent))) as image:
		image.thumbnail((1024, 1024))
		pixels = np.array(image.convert("RGBA"))
	records = np.zeros(1, dtype=MATERIAL_DTYPE)
	records["base_color"] = pbr.baseColorFactor
	textures = build_material_texture_array([pixels], records, (1024, 1024))
	shader = DefaultMaterialShader(default_material={"base_color": pbr.baseColorFactor, "roughness": pbr.roughnessFactor})
	mesh = MaterialMesh(primitive.vertices, shader, indices=primitive.indices, bounds=((0, 0.9, 0), (2.5, 2.5, 2.5)))
	return mesh, textures
